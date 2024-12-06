import copy
import re
import time
import traceback
import uuid
from typing import Generator, List
from uuid import UUID

from fastapi import Depends, HTTPException, status
from json_repair import repair_json
from langfuse.decorators import langfuse_context
from sqlmodel import Session, select

from api.models import (
    ChatMessage,
    ChatResponse,
    MessageResponse,
)

from .chat_assistant_agent import ChatAssistant
# from src.agents.chat_assistant_agent import ChatAssistant
from src.constants import ChatAssistantConfig, MesssageHistory, SenderType
from src.database import (
    Assistants,
    Conversations,
    KnowledgeBases,
    Messages,
    get_session,
)
from src.settings import default_settings
from src.utils import get_formatted_logger

logger = get_formatted_logger(__file__)


class AssistantService:
    def __init__(self, db_session: Session = Depends(get_session)):
        self.db_session = db_session

    def chat_with_assistant(
        self,
        conversation_id: UUID,
        user_id: int,
        message: ChatMessage,
        start_time: float,
    ) -> ChatResponse:
        try:
            with self.db_session as session:
                query = select(Conversations).filter_by(
                    conversations_id=conversation_id, user_id=user_id
                )

                conversation = session.exec(query).first()

                if not conversation:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Conversation not found",
                    )

                # Fetch message history
                message_history = self._get_message_history(conversation_id)
                logger.info(f"Message History of {conversation_id}: {message_history}")

                # Here we assume that the Assistant class has an on_message method
                # In a real implementation, you might need to instantiate the assistant with its configuration
                assistant = session.exec(
                    select(Assistants).where(
                        Assistants.assistants_id == conversation.assistant_id
                    )
                ).first()

                kb = session.exec(
                    select(KnowledgeBases).where(
                        KnowledgeBases.knowledge_bases_id == assistant.knowledge_base_id
                    )
                ).first()
                kb_ids = copy.deepcopy(kb.parents)
                kb_ids.append(kb.knowledge_bases_id)

                # Save user message
                user_message = Messages(
                    conversation_id=conversation_id,
                    sender_type=SenderType.USER,
                    content=message.content,
                )
                session.add(user_message)
                session.flush()  # Flush to get the ID of the new message

                configuration = assistant.configuration

                session_id = str(uuid.uuid4())
                assistant_config = ChatAssistantConfig(
                    model=configuration["model"],
                    conversation_id=conversation_id,
                    service=configuration["service"],
                    temperature=configuration["temperature"],
                    embedding_service="openai",
                    embedding_model_name="text-embedding-3-small",
                    collection_name=default_settings.global_vector_db_collection_name,
                    kb_ids=kb_ids,
                    session_id=session_id,
                    is_contextual_rag=True,
                    interested_prompt=assistant.interested_prompt,
                    guard_prompt=assistant.guard_prompt,
                )

                assistant_instance = ChatAssistant(assistant_config)

                full_response = assistant_instance.on_message(
                    message.content,
                    message_history,
                    session_id=conversation.conversations_id,
                )
                full_response = re.sub(r"[\*-]", "", full_response)
                response_time = time.time() - start_time

                langfuse_context.update_current_trace()
                langfuse_context.flush()

                logger.info(f"AI Response: {full_response}")
                logger.info(f"Response Time: {response_time}")

                response_json = repair_json(full_response, return_objects=True)
                response = {}
                if not isinstance(response_json, dict):
                    response["result"] = full_response
                    response["is_chat_false"] = False
                else:
                    response["is_chat_false"] = True
                    response["result"] = full_response

                # Save assistant message
                assistant_message = Messages(
                    messages_id=session_id,
                    conversation_id=conversation_id,
                    sender_type=SenderType.ASSISTANT,
                    content=full_response,
                    response_time=response_time,
                    is_chat_false=response["is_chat_false"],
                    cost=0.0,
                )
                session.add(assistant_message)

                session.commit()

                return ChatResponse(assistant_message=full_response)

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred during the chat: {str(e)}",
            )

    def stream_chat_with_assistant(
        self, conversation_id: int, user_id: int, message: ChatMessage
    ) -> Generator[str, None, None]:
        with self.db_session as session:
            conversation = (
                session.query(Conversations)
                .filter_by(conversations_id=conversation_id, user_id=user_id)
                .first()
            )
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )

            message_history = self._get_message_history(conversation_id)

            user_message = Messages(
                conversation_id=conversation_id,
                sender_type=SenderType.USER,
                content=message.content,
            )
            session.add(user_message)
            session.flush()

            assistant = conversation.assistant
            configuration = assistant.configuration

            assistant_config = ChatAssistantConfig(
                model=configuration["model"],
                service=configuration["service"],
                temperature=configuration["temperature"],
                embedding_service="openai",
                embedding_model_name="text-embedding-3-small",
                collection_name=f"{assistant.knowledge_base_id}",
            )

            assistant_instance = ChatAssistant(assistant_config)

            full_response = ""
            for chunk in assistant_instance.stream_chat(
                message.content, message_history
            ):
                full_response += chunk
                yield chunk

            assistant_message = Messages(
                conversation_id=conversation_id,
                sender_type=SenderType.ASSISTANT,
                content=full_response,
            )
            session.add(assistant_message)
            session.commit()

    async def astream_chat_with_assistant(
        self,
        conversation_id: UUID,
        user_id: UUID,
        message: ChatMessage,
        start_time: float,
    ):
        with self.db_session as session:
            query = select(Conversations).where(
                Conversations.conversations_id == conversation_id,
                Conversations.user_id == user_id,
            )

            conversation = session.exec(query).first()

            is_contextual_rag = True
            assistant = session.exec(
                select(Assistants).where(
                    Assistants.assistants_id == conversation.assistant_id
                )
            ).first()
            configuration = assistant.configuration

            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")

            kb = session.exec(
                select(KnowledgeBases).where(
                    KnowledgeBases.knowledge_bases_id == assistant.knowledge_base_id
                )
            ).first()

            kb_ids = copy.deepcopy(kb.parents)
            kb_ids.append(kb.knowledge_bases_id)

            message_history = self._get_message_history(conversation_id)

            user_message = Messages(
                conversation_id=conversation_id,
                sender_type=SenderType.USER,
                content=message.content,
            )
            session.add(user_message)
            session.flush()

            session_id = str(uuid.uuid4())

            assistant_config = ChatAssistantConfig(
                model=configuration["model"],
                conversation_id=conversation_id,
                service=configuration["service"],
                temperature=configuration["temperature"],
                embedding_service="openai",
                embedding_model_name="text-embedding-3-small",
                collection_name=default_settings.global_vector_db_collection_name,
                kb_ids=kb_ids,
                session_id=session_id,
                is_contextual_rag=is_contextual_rag,
                interested_prompt=assistant.interested_prompt,
                guard_prompt=assistant.guard_prompt,
            )

            assistant_instance = ChatAssistant(configuration=assistant_config)

            response_time = None
            full_response = ""

            ai_response = assistant_instance.on_message(
                message.content, message_history, session_id=session_id
            )

            logger.info(f"AI Response: {ai_response}")
            response_json = repair_json(ai_response, return_objects=True)
            response = {}
            if not isinstance(response_json, dict):
                response["result"] = ai_response
                response["is_chat_false"] = False
            else:
                response["is_chat_false"] = True
                response["result"] = ai_response

            for chunk in response["result"].strip().split(" "):
                if response_time is None:
                    response_time = time.time() - start_time

                full_response += chunk + " "
                yield chunk + " "

            langfuse_context.flush()

            assistant_message = Messages(
                messages_id=session_id,
                conversation_id=conversation_id,
                sender_type=SenderType.ASSISTANT,
                content=full_response,
                response_time=response_time,
                is_chat_false=response["is_chat_false"],
                cost=0.0,
            )
            session.add(assistant_message)
            session.commit()

    def _get_message_history(self, conversation_id: UUID) -> List[MesssageHistory]:
        with self.db_session as session:
            query = (
                select(Messages)
                .where(Messages.conversation_id == conversation_id)
                .order_by(Messages.created_at)
            )

            messages = session.exec(query).all()

            return [
                MesssageHistory(content=msg.content, role=msg.sender_type)
                for msg in messages
            ]

    def get_conversation_history(
        self, assistant_id: UUID, conversation_id: UUID, user_id: UUID
    ) -> List[MessageResponse]:
        try:
            with self.db_session as session:
                query = select(Conversations).where(
                    Conversations.assistant_id == assistant_id,
                    Conversations.conversations_id == conversation_id,
                    Conversations.user_id == user_id,
                )

                conversation = session.exec(query).first()

                if not conversation:
                    raise HTTPException(
                        status_code=404, detail="Conversation not found"
                    )

                query = (
                    select(Messages)
                    .where(Messages.conversation_id == conversation_id)
                    .order_by(Messages.created_at)
                )

                messages = session.exec(query).all()

                all_messages = []
                for new_message in messages:
                    new_message_dict = new_message.model_dump()
                    new_message_dict.pop("id")
                    new_message_dict["id"] = new_message_dict.pop("messages_id")
                    all_messages.append(MessageResponse(**new_message_dict))
                return all_messages
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while fetching conversation history: {str(e)}",
            )

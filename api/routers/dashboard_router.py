import json
import string
import sys
import uuid
from pathlib import Path
from typing import Annotated
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session, func, select
from wordcloud import WordCloud

sys.path.append(str(Path(__file__).parent.parent.parent))
from api.models import (
    AssistantStaticsResponse,
    ConversationStaticsResponse,
    DashboardStaticsResponse,
    GetSourceReponse,
    KnowledgeBaseStaticsResponse,
)
from src.constants import SenderType
from src.database import (
    Assistants,
    Conversations,
    KnowledgeBases,
    Messages,
    Users,
    get_session,
)

from .kb_router import DOWNLOAD_FOLDER
from .user_router import get_current_user

dashboard_router = APIRouter()


@dashboard_router.get("/", response_model=DashboardStaticsResponse)
async def get_dashboard(
    current_user: Annotated[Users, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> JSONResponse:
    with db_session as session:
        conversations = session.exec(
            select(Conversations).where(Conversations.user_id == current_user.users_id)
        ).all()

        messages = [
            session.exec(
                select(Messages)
                .where(Messages.conversation_id == conversation.conversations_id)
                .order_by(Messages.created_at)
            ).all()
            for conversation in conversations
        ]

        total_conversations = len(conversations)

        average_assistant_response_time = session.exec(
            select(func.avg(Messages.response_time).label("average_response_time"))
            .join(
                Conversations,
                Conversations.conversations_id == Messages.conversation_id,
            )
            .join(Assistants, Assistants.assistants_id == Conversations.assistant_id)
            .join(
                KnowledgeBases,
                KnowledgeBases.knowledge_bases_id == Assistants.knowledge_base_id,
            )
            .where(
                KnowledgeBases.user_id == current_user.users_id,
                Messages.sender_type == SenderType.ASSISTANT,
            )
        ).first()
        if not average_assistant_response_time:
            average_assistant_response_time = 0.0

        knowledge_base_statistics = session.exec(
            select(
                KnowledgeBases.knowledge_bases_id,
                KnowledgeBases.name,
                func.count(Messages.messages_id).label("total_user_messages"),
            )
            .select_from(KnowledgeBases)
            .join(
                Assistants,
                Assistants.knowledge_base_id == KnowledgeBases.knowledge_bases_id,
            )
            .join(Conversations, Conversations.assistant_id == Assistants.assistants_id)
            .join(Messages, Messages.conversation_id == Conversations.conversations_id)
            .where(
                KnowledgeBases.user_id == current_user.users_id,
                Messages.sender_type == SenderType.USER,
            )
            .group_by(KnowledgeBases.knowledge_bases_id, KnowledgeBases.name)
        ).all()

        assistant_statistics = session.exec(
            select(
                Assistants.assistants_id,
                Assistants.name,
                func.count(Conversations.conversations_id).label(
                    "number_of_conversations"
                ),
            )
            .join(Conversations, Conversations.assistant_id == Assistants.assistants_id)
            .where(Assistants.user_id == current_user.users_id)
            .group_by(Assistants.assistants_id, Assistants.name)
        ).all()

        # Save all data to excel
        df = pd.DataFrame(
            [
                {
                    "total_conversations": total_conversations,
                    "average_assistant_response_time": average_assistant_response_time,
                    "knowledge_base_statistics": json.dumps(
                        [
                            {
                                "id": str(knowledge_base[0]),
                                "name": knowledge_base[1],
                                "total_user_messages": knowledge_base[2],
                            }
                            for knowledge_base in knowledge_base_statistics
                        ]
                    ),
                    "assistant_statistics": json.dumps(
                        [
                            {
                                "id": str(assistant[0]),
                                "name": assistant[1],
                                "number_of_conversations": assistant[2],
                            }
                            for assistant in assistant_statistics
                        ]
                    ),
                    "conversations_statistics": json.dumps(
                        [
                            {
                                "id": str(conversation.conversations_id),
                                "average_session_chat_time": conversation.average_session_chat_time,
                                "average_user_messages": conversation.average_user_messages,
                            }
                            for conversation in conversations
                        ]
                    ),
                }
            ]
        )

        conversation_names = [
            conversation.name or conversation.conversations_id
            for conversation in conversations
        ]

        conversation_contents = [
            json.dumps(
                [
                    {
                        "sender_type": m.sender_type,
                        "content": m.content,
                    }
                    for m in message
                ]
            )
            for message in messages
        ]

        conversation_df = pd.DataFrame(
            {
                "conversation_name": conversation_names,
                "conversation_content": conversation_contents,
            }
        )

        file_conversation_name = (
            Path(DOWNLOAD_FOLDER) / f"conversation_{current_user.users_id}.xlsx"
        )

        conversation_df.to_excel(file_conversation_name, index=False)

        file_name = Path(DOWNLOAD_FOLDER) / f"dashboard_{current_user.users_id}.xlsx"

        df.to_excel(file_name, index=False)

        return DashboardStaticsResponse(
            total_conversations=total_conversations,
            conversations_statistics=[
                ConversationStaticsResponse(
                    id=conversation.conversations_id,
                    average_session_chat_time=conversation.average_session_chat_time,
                    average_user_messages=conversation.average_user_messages,
                )
                for conversation in conversations
            ],
            assistant_statistics=[
                AssistantStaticsResponse(
                    id=assistant[0],
                    name=assistant[1],
                    number_of_conversations=assistant[2],
                )
                for assistant in assistant_statistics
            ],
            average_assistant_response_time=round(average_assistant_response_time, 2),
            knowledge_base_statistics=[
                KnowledgeBaseStaticsResponse(
                    id=knowledge_base[0],
                    name=knowledge_base[1],
                    total_user_messages=knowledge_base[2],
                )
                for knowledge_base in knowledge_base_statistics
            ],
            file_name=file_name.stem,
            file_conversation_name=file_conversation_name.stem,
        )


@dashboard_router.get(
    "/export/{file_name}", tags=["download"], status_code=status.HTTP_200_OK
)
def download_file(
    file_name: str,
    current_user: Annotated[Users, Depends(get_current_user)],
):
    # Check if user has permission to download file
    user_id = file_name.split("_")[-1]

    if str(current_user.users_id) != user_id:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "You do not have permission to download this file"},
        )

    file_path = Path(DOWNLOAD_FOLDER) / (file_name + ".xlsx")
    if file_path.exists():
        return FileResponse(
            file_path, filename=file_name, status_code=status.HTTP_200_OK
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "File not found"}
        )


@dashboard_router.get("/wordcloud/kb/{knowledge_base_id}")
async def get_wordcloud_by_kb(
    knowledge_base_id: UUID,
    is_user: bool,
    current_user: Annotated[Users, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> JSONResponse:
    sender_type = SenderType.USER if is_user else SenderType.ASSISTANT

    with db_session as session:
        knowledge_base = session.exec(
            select(KnowledgeBases).where(
                KnowledgeBases.knowledge_bases_id == knowledge_base_id,
                KnowledgeBases.user_id == current_user.users_id,
            )
        )

        if not knowledge_base:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge Base not found",
            )

        text = session.exec(
            select(Messages.content)
            .select_from(Messages)
            .join(
                Conversations,
                Conversations.conversations_id == Messages.conversation_id,
            )
            .join(Assistants, Assistants.assistants_id == Conversations.assistant_id)
            .join(
                KnowledgeBases,
                KnowledgeBases.knowledge_bases_id == Assistants.knowledge_base_id,
            )
            .where(
                KnowledgeBases.knowledge_bases_id == knowledge_base_id,
                KnowledgeBases.user_id == current_user.users_id,
                Messages.sender_type == sender_type,
            )
        ).all()

        content = " ".join(text)
        wordcloud = WordCloud(width=800, height=400, max_words=1000).generate(content)
        image_path = (
            Path(DOWNLOAD_FOLDER)
            / f"{str(knowledge_base_id).replace('-', '_')}_{sender_type}.png"
        )
        wordcloud.to_file(str(image_path))

        return FileResponse(
            path=image_path, filename=image_path.name, media_type="image/jpeg"
        )


@dashboard_router.get("/wordcloud/assistant/{assistant_id}")
async def get_wordcloud_by_assistant(
    assistant_id: UUID,
    is_user: bool,
    current_user: Annotated[Users, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> JSONResponse:
    sender_type = SenderType.USER if is_user else SenderType.ASSISTANT

    with db_session as session:
        assistant = session.exec(
            select(Assistants).where(
                Assistants.assistants_id == assistant_id,
                Assistants.user_id == current_user.users_id,
            )
        )

        if not assistant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assistant not found",
            )

        text = session.exec(
            select(Messages.content)
            .join(
                Conversations,
                Conversations.conversations_id == Messages.conversation_id,
            )
            .join(Assistants, Assistants.assistants_id == Conversations.assistant_id)
            .where(
                Assistants.assistants_id == assistant_id,
                Assistants.user_id == current_user.users_id,
                Messages.sender_type == sender_type,
            )
        ).all()

        content = " ".join(text)
        wordcloud = WordCloud(width=800, height=400, max_words=1000).generate(content)
        image_path = (
            Path(DOWNLOAD_FOLDER)
            / f"{str(assistant_id).replace('-', '_')}_{sender_type}.png"
        )
        wordcloud.to_file(str(image_path))

        return FileResponse(
            path=image_path, filename=image_path.name, media_type="image/jpeg"
        )


@dashboard_router.get("/wordcloud/conversation/{conversation_id}")
async def get_wordcloud_by_conversation(
    conversation_id: UUID,
    is_user: bool,
    current_user: Annotated[Users, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> JSONResponse:
    sender_type = SenderType.USER if is_user else SenderType.ASSISTANT
    with db_session as session:
        conversation = session.exec(
            select(Conversations).where(
                Conversations.conversations_id == conversation_id,
                Conversations.user_id == current_user.users_id,
            )
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        text = session.exec(
            select(Messages.content)
            .join(
                Conversations,
                Conversations.conversations_id == Messages.conversation_id,
            )
            .where(
                Conversations.conversations_id == conversation_id,
                Conversations.user_id == current_user.users_id,
                Messages.sender_type == sender_type,
            )
        ).all()

        content = " ".join(text)
        content = [c for c in content if c not in string.punctuation]
        content = "".join(content)
        wordcloud = WordCloud(width=800, height=400, max_words=1000).generate(content)
        image_path = (
            Path(DOWNLOAD_FOLDER)
            / f"{str(conversation_id).replace('-', '_')}_{str(uuid.uuid4())}.png"
        )
        wordcloud.to_file(str(image_path))

        return FileResponse(
            path=image_path, filename=image_path.name, media_type="image/jpeg"
        )


@dashboard_router.get("/kbs", response_model=list[GetSourceReponse])
async def get_all_knowledge_bases(
    current_user: Annotated[Users, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
):
    with db_session as session:
        knowledge_bases = session.exec(
            select(KnowledgeBases).where(
                KnowledgeBases.user_id == current_user.users_id
            )
        ).all()

        result = [GetSourceReponse(id=kb.knowledge_bases_id, name=kb.name) for kb in knowledge_bases]
        return result


@dashboard_router.get("/assistants", response_model=list[GetSourceReponse])
async def get_all_assistants(
    current_user: Annotated[Users, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
):
    with db_session as session:
        assistants = session.exec(
            select(Assistants).where(Assistants.user_id == current_user.users_id)
        ).all()

        result = [GetSourceReponse(id=assistant.assistants_id, name=assistant.name) for assistant in assistants]

        return result


@dashboard_router.get("/conversations", response_model=list[GetSourceReponse])
async def get_all_conversations(
    current_user: Annotated[Users, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
):
    with db_session as session:
        conversations = session.exec(
            select(Conversations).where(Conversations.user_id == current_user.users_id)
        ).all()

        return [
            GetSourceReponse(
                id=conversation.conversations_id,
                name=conversation.name or conversation.conversations_id,
            )
            for conversation in conversations
        ]

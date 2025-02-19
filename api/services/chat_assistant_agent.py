import uuid

from crewai import Agent, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse.decorators import observe
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage as LLamaIndexChatMessage
from llama_index.core.callbacks import CallbackManager

from src.agents import CrewAIAgent
from src.constants import ASSISTANT_SYSTEM_PROMPT, ChatAssistantConfig, MesssageHistory
from src.database import ContextualRAG
from src.settings import default_settings
from src.tools import KBSearchTool
from src.utils import get_formatted_logger
import time

logger = get_formatted_logger(__file__)
load_dotenv()

langfuse_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_callback_handler])


class ChatAssistant:
    def __init__(self, configuration: ChatAssistantConfig):
        self.configuration = configuration
        self._init_agent()

    def _init_agent(self):
        model_name = self.configuration.model
        service = self.configuration.service

        self.llm = self._init_model(service, model_name)

        system_prompt = ASSISTANT_SYSTEM_PROMPT.format(
            interested_prompt=self.configuration.interested_prompt,
            guard_prompt=self.configuration.guard_prompt,
        ).strip("\n")

        kb_search_tool = KBSearchTool(
            setting=default_settings,
            contextual_rag=ContextualRAG.from_setting(default_settings),
            kb_ids=self.configuration.kb_ids,
            session_id=self.configuration.session_id,
            is_contextual_rag=self.configuration.is_contextual_rag,
            system_prompt=system_prompt,
            result_as_answer=True,
        )

        kb_agent = Agent(
            role="Assistant",
            goal="Helping users find information in the knowledge base.",
            backstory="You are a very intelligent assistant. You have access to a knowledge base that contains a lot of information.",
            llm=self.llm,
        )

        kb_task = Task(
            description="Search the knowledge base for information about {query}.",
            expected_output="A most relevant answer from the knowledge base for the given query.",
            agent=kb_agent,
            tools=[
                kb_search_tool,
            ],
        )

        self.agent = CrewAIAgent(
            agents=[kb_agent],
            tasks=[kb_task],
            manager_llm=self.llm,
            verbose=True,
            conversation_id=self.configuration.conversation_id,
        )

    def _init_model(self, service, model_id):
        """
        Select a model for text generation using multiple services.
        Args:
            service (str): Service name indicating the type of model to load.
            model_id (str): Identifier of the model to load from HuggingFace's model hub.
        Returns:
            LLM: llama-index LLM for text generation
        Raises:
            ValueError: If an unsupported model or device type is provided.
        """
        logger.info(f"Loading Model: {model_id}")
        logger.info("This action can take a few minutes!")
        # TODO: setup proper logging

        if service == "openai":
            logger.info(f"Loading OpenAI Model: {model_id}")
            # return OpenAI(
            #     model=model_id,
            #     temperature=self.configuration.temperature,
            # )
            return ChatOpenAI(
                model=model_id,
                temperature=self.configuration.temperature,
            )
        else:
            raise NotImplementedError(
                "The implementation for other types of LLMs are not ready yet!"
            )

    @observe()
    def on_message(
        self,
        message: str,
        message_history: list[MesssageHistory],
        session_id: str | uuid.UUID,
    ) -> str:
        logger.info("Chatting ...")
        langfuse_callback_handler.set_trace_params(
            session_id=str(session_id),
        )

        # message_history = [
        #     LLamaIndexChatMessage(content=msg.content, role=msg.role)
        #     for msg in message_history
        # ]
        inputs = {
            "query": message,
        }
        s = time.time()
        response = self.agent.chat(inputs, message_history)
        logger.info(f"Time taken chat: {time.time() - s}")

        return response

    def stream_chat(self, message: str, message_history: list[MesssageHistory]):
        message_history = [
            LLamaIndexChatMessage(content=msg["content"], role=msg["role"])
            for msg in message_history
        ]
        return self.agent.stream_chat(message, message_history).response_gen

    @observe()
    async def astream_chat(
        self,
        message: str,
        message_history: list[MesssageHistory],
        session_id: str | uuid.UUID,
    ):
        langfuse_callback_handler.set_trace_params(
            name="astream_chat",
            session_id=str(session_id),
        )

        message_history = [
            LLamaIndexChatMessage(content=msg.content, role=msg.role)
            for msg in message_history
        ]
        response = await self.agent.astream_chat(message, message_history)

        return response

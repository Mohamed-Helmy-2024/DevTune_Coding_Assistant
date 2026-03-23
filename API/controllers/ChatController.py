import logging
from .BaseController import BaseController 
from stores.llm.LLMProviderFactory import LLMProviderFactory
from helpers.history import load_history
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_core.language_models import LLM
from pydantic import Field
from controllers.RAGController import RAGController

class ProviderLLMWrapper(LLM):
    provider: Any = Field(..., description="The wrapped LLM provider")
    def _call(self, prompt: str, stop=None):
        return self.provider.generate_text(prompt)
    @property
    def _llm_type(self):
        return "custom-provider"
    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

class ChatController(BaseController):
    def __init__(self, session_id=None,username=None,chat_history=None,utility_params=None): 
        super().__init__() 
        self.session_id = session_id
        self.username = username
        self.chat_history=chat_history or []
        self.utility_params=utility_params
        self.logger = logging.getLogger(__name__)


    async def completion_router(self, user_prompt: str):
        completion_type = self.utility_params.get("completion_type")
        chain_map = {
            "chat": self.main_chain,
            "main": self.main_chain_with_history,
            "summary": self.summarization_chain,
            "code_review": self.code_review_chain,
            "code_generation": self.code_generation_chain,
        }
        chain_func = chain_map.get(completion_type)
        if chain_func:
            return await chain_func(user_prompt)
        raise ValueError("Unsupported completion type")
    
    
    async def _initialize_provider(self):
        factory = LLMProviderFactory(config=self.app_settings)
        provider = factory.create(provider=self.app_settings.GENERATION_BACKEND)
        # Provider factory will set generation model. Do not change provider embedding settings here.
        wrapper = ProviderLLMWrapper(provider=provider)
        return provider, wrapper

    async def _build_memory(self, full_history, lc_llm, n_recent=3, max_summary_tokens=1024):
        last_msgs = full_history[-n_recent:]
        older_msgs = full_history[:-n_recent]

        # Recent memory
        recent_memory = ConversationBufferMemory(memory_key="recent", return_messages=True)
        for msg in reversed(last_msgs):
            recent_memory.save_context({"input": msg[0].content}, {"output": msg[1].content})
        recent_text = recent_memory.load_memory_variables({})["recent"]

        # Summary memory
        summary_memory = ConversationSummaryBufferMemory(
            llm=lc_llm, memory_key="summary", return_messages=False
        )
        token_count = 0
        for msg in reversed(older_msgs):
            msg_tokens = lc_llm.get_num_tokens(msg[0].content) + lc_llm.get_num_tokens(msg[1].content)
            if token_count + msg_tokens > max_summary_tokens:
                break
            summary_memory.save_context({"input": msg[0].content}, {"output": msg[1].content})
            token_count += msg_tokens
        summary_text = summary_memory.load_memory_variables({})["summary"]

        return recent_text, summary_text

    async def _invoke_chain(self, template_str, provider, model_inputs):
        template = ChatPromptTemplate.from_template(template_str)
        provider_runnable = RunnableLambda(lambda v: provider.generate_text(v.to_string()))
        chain = template | provider_runnable
        return chain.invoke(model_inputs)
    

    async def main_chain(self, user_prompt):
        provider, _ = await self._initialize_provider()
        # Optionally include RAG context if requested
        rag_ctx = ""
        try:
            if self.utility_params and self.utility_params.get('use_rag'):
                rag_ctrl = RAGController(session_id=self.session_id, username=self.username)
                docs = await rag_ctrl.search_documents(user_prompt, top_k=self.utility_params.get('rag_top_k', 3))
                if docs:
                    ctx_parts = [f"[Doc {i+1}] {d['content']}" for i, d in enumerate(docs)]
                    rag_ctx = "\n".join(ctx_parts)
                    self.logger.info(f"main_chain RAG included {len(docs)} docs for user {self.username}")
        except Exception:
            pass
        template_str = """
        You are an expert Developer and Computer Scientist Assistant.
        Provide accurate, practical, and concise guidance on developer topics.

        Input:
        User Prompt: {user_input}
        RAG Context: {rag_context}

        Guidelines:
        - Give clear, concise answers with examples or code when needed.
        - Stay focused on developer/tech topics.
        - Minimize tokens while keeping answers complete.
        - Ask clarifying questions only if necessary.

        Output:
        - Direct answer to user's question.
        - Code snippets in markdown if relevant.
        - Short tips or next steps if useful.
        """
        return await self._invoke_chain(template_str, provider, {"user_input": user_prompt, "rag_context": rag_ctx})

    async def main_chain_with_history(self, user_prompt, max_summary_tokens=500):
        provider, lc_llm = await self._initialize_provider()
        full_history = await load_history(self.session_id)
        recent_memory, chat_summary = await self._build_memory(full_history, lc_llm, max_summary_tokens=max_summary_tokens)
        rag_ctx = ""
        self.logger.info(f"ChatController: utility_params={self.utility_params}")
        if self.utility_params and self.utility_params.get('use_rag'):
            try:
                rag_ctrl = RAGController(session_id=self.session_id, username=self.username)
                docs = await rag_ctrl.search_documents(user_prompt, top_k=self.utility_params.get('rag_top_k', 3))
                # Log discovered docs for debugging
                self.logger.info(f"RAG fetched {len(docs)} docs for user {self.username} - use_rag={self.utility_params.get('use_rag')}")
                if docs:
                    print(f"[ChatController] Retrieved {len(docs)} RAG documents for prompt.")
                    ctx_parts = [f"[Doc {i+1}] {d['content']}" for i, d in enumerate(docs)]
                    rag_ctx = "\n".join(ctx_parts)
                    print(ctx_parts)
                # Log rag_ctx size and preview for debug
                try:
                    self.logger.info(f"RAG context length for user {self.username}: {len(rag_ctx)} chars")
                    self.logger.info(f"RAG context preview: {rag_ctx[:200]}")
                except Exception:
                    pass
            except Exception:
                rag_ctx = ""
        template_str = """
        You are an expert Developer Assistant.

        Inputs:
        - User Prompt: {user_input}
        - Chat Summary: {chat_summary}
        - Recent Memory: {recent_memory}
        - RAG Context: {rag_context}

        Guidelines:
        - Give clear, concise answers with examples or code when needed.
        - Stay focused on developer/tech topics.
        - Minimize tokens while keeping answers complete.

        Output:
        - Direct answer to user's question.
        - Code snippets in markdown if relevant.
        - Short tips or next steps if useful.
        """
        model_inputs = {"user_input": user_prompt, "chat_summary": chat_summary, "recent_memory": recent_memory, "rag_context": rag_ctx}
        return await self._invoke_chain(template_str, provider, model_inputs)

    async def summarization_chain(self, user_prompt, max_summary_tokens=500):
        provider, lc_llm = await self._initialize_provider()
        full_history = await load_history(self.session_id)
        recent_memory, chat_summary= await self._build_memory(full_history, lc_llm, max_summary_tokens=max_summary_tokens)
        template_str = """
            You are an expert Developer Assistant and Technical Summarizer.
            Your task is to summarize any input provided by the user. The input can be code or plain text.

            Input:
            User Prompt: {user_input}
            Chat Summary (optional): {chat_summary} -> brief key points from previous conversation, if any.
            Recent Memory (optional): {recent_memory} -> relevant context from prior exchanges, if any.
            
            Instructions:
            1. If the input is **code**:
            - Summarize what the code does overall.
            - Explain the main parts, functions, classes, loops, and important logic in simple terms.
            - Include concise notes on key variables, structures, or libraries used.
            2. If the input is **plain text**:
            - Summarize the main points clearly and concisely.
            3. Keep the summary concise but informative.
            4. Use clear language suitable for developers or technical readers.
            5. Provide examples only if necessary for clarity.

            Output:
            - Summary of the input (text or code).
            - For code: a clear explanation of each major part or function.
            - Optional: brief suggestions or key insights if relevant.    
            """
        model_inputs = {"user_input": user_prompt, "chat_summary": chat_summary, "recent_memory": recent_memory}
        return await self._invoke_chain(template_str, provider, model_inputs)
    
    async def code_review_chain(self, user_prompt, max_summary_tokens=500):
        provider, lc_llm = await self._initialize_provider()
        full_history = await load_history(self.session_id)
        recent_memory, chat_summary = await self._build_memory(full_history, lc_llm, max_summary_tokens=max_summary_tokens)
        template_str = """
            You are an expert Code Reviewer and Developer Assistant. 
            Your task is to review code provided by the user and provide actionable feedback, including:
            - Syntax errors, runtime issues, and bugs
            - Performance and optimization suggestions
            - Best practices and coding standards
            - Security, readability, and maintainability improvements
            - Suggestions for refactoring or enhancements

            Input:
            User Prompt: {user_input}
            Chat Summary (optional): {chat_summary} -> brief key points from previous conversation, if any.
            Recent Memory (optional): {recent_memory} -> relevant context from prior exchanges, if any.
            Instructions:
            1. If the input contains code:
            - Analyze the code thoroughly.
            - Provide detailed feedback on errors, optimizations, and best practices.
            - Highlight critical issues first, then suggestions and improvements.
            - Include examples or corrected code snippets if necessary.
            2. If the input is **plain text or a request without code**:
            - Respond: "No code provided for review." or answer the user's specific request or question regarding code.
            3. Address any notifications, demands, or questions about the code included in the input.
            4. Be concise but thorough; organize feedback for clarity.
            5. Use bullet points for readability when needed.

            Output:
            - Structured review including errors, warnings, suggestions, and improvements.
            - Corrected code snippets or examples if applicable.
            - Notes on coding standards, optimizations, or best practices.
            - Response for text-only input or code-related requests.
        """
        model_inputs = {"user_input": user_prompt, "chat_summary": chat_summary, "recent_memory": recent_memory}
        return await self._invoke_chain(template_str, provider, model_inputs)
    
    async def code_generation_chain(self, user_prompt, max_summary_tokens=500):
        provider, lc_llm = await self._initialize_provider()
        full_history = await load_history(self.session_id)
        recent_memory, chat_summary = await self._build_memory(full_history, lc_llm, max_summary_tokens=max_summary_tokens)
        template_str = """
            You are an expert Code Generation Assistant and Developer AI. 
            Your task is to generate code based on the user's requirements and provide recommendations for improving 
            or extending the code in future calls.

            Input:
            User Prompt: {user_input}
            Chat Summary (optional): {chat_summary} -> brief key points from previous conversation, if any.
            Recent Memory (optional): {recent_memory} -> relevant context from prior exchanges, if any.

            Instructions:
            1. Understand the user's request and generate accurate, practical, and well-structured code.
            2. If any ambiguity exists (e.g., programming language, framework, environment, input/output format):
            - Ask the user a clarifying question before generating the code.
            3. Include comments in the code where necessary for clarity.
            4. After providing the code, give short recommendations for the user on:
            - Optional improvements
            - Best practices
            - Additional features they might consider in the next call
            5. Keep the response concise but informative; focus on usability and readability.

            Output:
            - Generated code matching user requirements (with comments if needed)
            - Clarifying questions if necessary before code generation
            - Recommendations or tips for improving, extending, or optimizing the code
         """
        model_inputs = {"user_input": user_prompt, "chat_summary": chat_summary, "recent_memory": recent_memory}
        return await self._invoke_chain(template_str, provider, model_inputs)





import asyncio
from controllers.RAGController import RAGController
from controllers.ChatController import ChatController

async def test():
    # Prepare
    session_id = 'test-session'
    username = 'tester'
    file_path = './uploads/tester/test_doc.txt'

    rag_ctrl = RAGController(session_id=session_id, username=username)
    chat_ctrl = ChatController(session_id=session_id, username=username, utility_params={'completion_type':'main','use_rag':True})

    print('Validating upload...')
    val = await rag_ctrl.validate_document(file_path)
    print('Validation:', val)

    if val.get('success') and not val.get('skipped_duplicate'):
        print('Indexing document...')
        res = await rag_ctrl.index_document(file_path)
        print('Index result:', res)

    print('RAG query:')
    rag_ans = await rag_ctrl.rag_query('What is the title of the document?')
    print('RAG answer:', rag_ans)

    print('Chat with history and RAG context:')
    chat_ans = await chat_ctrl.completion_router('Summarize the document in one sentence')
    print('Chat answer:', chat_ans)

if __name__ == '__main__':
    asyncio.run(test())

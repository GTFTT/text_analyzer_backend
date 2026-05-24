import json

from src.database_utils import create_database, insert_chunk, get_all_chunks
from src.embedding_utils import cosine_similarity
from src.llm_utils import ask_llm
from src.requests_utils import create_embedding
from src.utils import get_all_text_from_pdf, join_sentences_into_chunks, split_text_into_by_sentences

pdf = './testing_docs/doc1.pdf'
database_file = './database/text_and_embeddings.db'



if __name__ == '__main__':
    query = input("\nAsk something: ").strip()
    query_embedding = create_embedding(query)
    chunks_from_db = get_all_chunks(database_file)
    print(f'Total chunks in database: {len(chunks_from_db)}')
    results = []

    for chunk_from_db in chunks_from_db:
        # print(chunk_from_db[0])
        # print(chunk_from_db[1])
        # break
        score = cosine_similarity(query_embedding, json.loads(chunk_from_db[1]))

        results.append({
            "score": score,
            "chunk_text": chunk_from_db[0],
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    n = 10
    # print("\nTop "+str(n)+" most relevant chunks:")
    # for result in results[:n]:
    #     print(f"Score: {result['score']:.4f} - Chunk: {result['chunk_text']}\n")
    #     print("--------------------------------------------------")

    print('Answer from LLM is loading ...')
    answer = ask_llm(query, [result['chunk_text'] for result in results[:n]])
    print('Answer from LLM:')
    print(answer)




# if __name__ == '__main__':
#     print('Extracting text from PDF...')
#     text = get_all_text_from_pdf(pdf)
#     print('Generating chunks from text...')
#     chunks = join_sentences_into_chunks(split_text_into_by_sentences(text), chunk_size=400, overlap_percentage=0.5)
#     print('Chunks generated successfully.')
#     print('Started filling database with chunks and their embeddings...')
#     loaded_chunks_count = 0
#     for chunk in chunks:
#         embedding = create_embedding(chunk)
#         insert_chunk(database_file, chunk, str(embedding))
#         loaded_chunks_count += 1
#         load_percentage = (loaded_chunks_count / len(chunks)) * 100
#         print(f'Chunk {loaded_chunks_count}/{len(chunks)} ({load_percentage:.2f}%) loaded into database.')
#     print('All chunks and their embeddings have been successfully loaded into the database.')



# if __name__ == '__main__':
#     print('Creating database...')
#     create_database(database_file)
#     print('Database created successfully.')



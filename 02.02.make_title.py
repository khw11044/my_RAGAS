import os

import click
import pandas as pd
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from autorag.data.qacreation import generate_qa_llama_index, make_single_content_qa

root_path = os.path.dirname(os.path.realpath(__file__))

prompt = """다음은 해외진출을 고려한 기업을 위해 제공된 자료들입니다. 
제공된 자료를 보고 자료와 관련 된 키워드를 뽑아주세요.
뽑힌 키워드를 통해, 게시물의 제목을 뽑아주세요.
만약 주어진 자료 내용이 해당 자료와 관련되지 않았다면, 
'해당 자료와 관련이 없습니다.'라고 질문을 만드세요.

제공된 자료:
{{text}}

생성할 질문 개수: {{num_questions}}

예시1:
[Q]: BIS
[A]: BIS 인증제도 알아보기

예시2:
[Q]: 인도 프랜차이즈 
[A]: 인도 프랜차이즈 시장 현황 및 특징

해당 자료와 관련이 없는 기사일 경우 예시:
[Q]: 해당 자료와 관련이 없습니다.
[A]: 해당 자료와 관련이 없습니다.

결과:
"""
# 실험 경험상 위와 같은 프롬프트가 가장 알맞은 질문과 대답을 하였다.

@click.command()
@click.option('--corpus_path', type=click.Path(exists=True),
              default=os.path.join('my_data', 'my_corpus.parquet'))
@click.option('--save_path', type=click.Path(exists=False, dir_okay=False, file_okay=True),
              default=os.path.join('my_data', 'my_title.parquet'))
@click.option('--qa_size', type=int, default=10)        # qa_size개의 실험 데이터셋 만들기
def main(corpus_path, save_path, qa_size):
    load_dotenv()

    corpus_df = pd.read_parquet(corpus_path, engine='pyarrow')
    llm = OpenAI(model='gpt-4o', temperature=0.1)
    
    qa_df = make_single_content_qa(corpus_df, content_size=qa_size, qa_creation_func=generate_qa_llama_index,
                                   llm=llm, prompt=prompt, question_num_per_content=1)  # 하나의 컨텐츠 당 생성할 질문과 응답 수 
    # 해당 자료와 관련이 없는 질문 및 응답 데이터는 제거 
    qa_df = qa_df.loc[~qa_df['query'].str.contains('해당 자료와 관련이 없습니다.')]
    qa_df.reset_index(drop=True, inplace=True)
    qa_df.to_parquet(save_path)


if __name__ == '__main__':
    main()

import os

import click
import pandas as pd
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from autorag.data.qacreation import generate_qa_llama_index, make_single_content_qa

root_path = os.path.dirname(os.path.realpath(__file__))

prompt = """다음은 해외진출을 고려한 기업을 위해 제공된 자료들입니다. 
당신은 해외진출을 고려하는 스타트업을 위해 주어진 주제로 게시물을 작성하는 전문 애널리스트입니다. 
제공된 자료를 보고 자료와 관련 된 키워드기반의 게시물 제목을 뽑아주세요.
뽑힌 게시물 제목을 통해, 게시물을 작성해주세요.
만약 주어진 자료 내용이 해당 자료와 관련되지 않았다면, 
'해당 자료와 관련이 없습니다.'라고 질문을 만드세요.

제공된 자료:
{{text}}

생성할 질문 개수: {{num_questions}}

예시1:
[Q]: 인도 수입 통관의 중요성과 유의사항
[A]: ### 1. 중요성 및 준비 절차
인도의 수입 통관은 기본적으로 사전 준비가 관건입니다. 통관에 필요한 서류를 정확하고 시간 내에 제출하는 것이 매우 중요합니다. 서류의 오류나 불일치로 인해 통관이 지연되는 경우가 있으므로 철저한 서류 작성이 필수적입니다. 이를 통해 통관 절차를 원활히 진행할 수 있습니다.

### 2. 포장 및 서류 작성
인도로 수출하는 물품의 경우 포장명세서(P/L)와 상업송장(C/I)을 각각의 컨테이너에 철저히 작성해야 합니다. 인도 세관은 실물 검사를 통해 화물의 일치 여부를 확인하므로 서류와 화물의 일치가 중요합니다. 또한, 인도의 인프라가 낙후되어 있고 운송수단이 노후하기 때문에 내륙운송 과정에서 발생할 수 있는 사고에 대비하여 내륙운송보험에 가입하는 것이 필요합니다.

### 3. 추가 서류 요구사항
일부 품목은 추가적인 서류가 필요할 수 있습니다. 예를 들어, 애완동물 사료나 비타민 수입 시에는 성분분석표나 수입 위생 허가증과 같은 서류가 요구될 수 있습니다. 수입 전에 해당 품목에 필요한 모든 서류를 확인하고 준비하는 것이 중요합니다.

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
              default=os.path.join('my_data', 'my_post.parquet'))
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

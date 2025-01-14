import json

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from scripts.utils import init_chain
from scripts.schemas import FCMetricRequest, FCMetricResponse

class FCMetric:
    def calculate(self, request: FCMetricRequest):
        sample = SingleTurnSample(
            response=request.response,
            reference=request.reference
        )

        llm = init_chain(request.model_name)
        evaluator_llm = LangchainLLMWrapper(llm)
        scorer = FactualCorrectness(
            llm=evaluator_llm, 
            mode=request.mode, 
            atomicity=request.atomicity, 
            coverage=request.coverage
        )
        result = scorer.single_turn_score(sample)
        response = FCMetricResponse(
            score = result['score'],
            response_claims=result['response_claims'],
            reference_claims=result['reference_claims'],
            response_reference=list(result['response_reference']),
            reference_response=list(result['reference_response'])
        )
        return response
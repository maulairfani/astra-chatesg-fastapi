from scripts.schemas import RetrievalMetricRequest, RetrievalMetricResponse

class PrecisionMetric:
    def calculate(self, request: RetrievalMetricRequest) -> RetrievalMetricResponse:
        retrieved_set = set(request.retrieved_contexts)
        reference_set = set(request.reference_contexts)

        if len(request.reference_contexts) == 0 and len(request.retrieved_contexts) == 0:
            return RetrievalMetricResponse(score=1)
        
        if not retrieved_set:
            precision = 0.0
        else:
            true_positives = retrieved_set.intersection(reference_set)
            precision = len(true_positives) / len(retrieved_set)
        
        return RetrievalMetricResponse(score=precision)

class RecallMetric:
    def calculate(self, request: RetrievalMetricRequest) -> RetrievalMetricResponse:
        retrieved_set = set(request.retrieved_contexts)
        reference_set = set(request.reference_contexts)

        if len(request.reference_contexts) == 0 and len(request.retrieved_contexts) == 0:
            return RetrievalMetricResponse(score=1)
        
        if not reference_set:
            recall = 0.0
        else:
            true_positives = retrieved_set.intersection(reference_set)
            recall = len(true_positives) / len(reference_set)
        
        return RetrievalMetricResponse(score=recall)
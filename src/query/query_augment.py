from datetime import datetime

from ..llm.llm import OpenAIWrapper


class QueryAugmentation():
    def __init__(self,
                 query) -> None:
        self.query = query

        self.llm = OpenAIWrapper()

    def augment(self, specified_date=""):
        # parse temporal info
        if specified_date:
            today = specified_date    
        else:
            today = datetime.today()
            today = today.strftime("%Y-%m-%d")

        result, cost = self.llm.augment_query(self.query, today)

        return result, cost
        
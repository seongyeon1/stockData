from abc import ABC, abstractmethod


'''
- crawling
- get_fundamentals
- stocks_info
'''

class StockDataBase(ABC):
    @abstractmethod
    def crawling(self):
        """
        데이터 가져오는거 (크롤링) - 모든 quantative 정보
        """
        pass
    @abstractmethod
    def get_fundamentals(self):
        """
        분석에 사용할 fundamental 추출
        (us - get_fundametal, kr- fundamental_all)
        """
        pass
    @abstractmethod
    def stocks_info(self):
        """
        최종 list stock들의 database dataframe으로 가져오기
        """
        pass
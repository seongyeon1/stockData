from abc import ABCMeta, abstractmethod


class Universe(metaclass=ABCMeta):
    
    @abstractmethod
    def get_standard_df(self, df, mode='mean'):
        """
        mode = 'mean', 'logistic'
        database module output df
        mode  : 어떤  기준 ex. mean, median으로 할지
        """

    @abstractmethod
    def get_score(self, Data):
        """
        데이터 베이스에서 가져온 종목들의 점수를 계산

        평균보다 크면 1, 작으면 0으로 점수계산
        """

    @abstractmethod
    def top_n_sector(self, n):
        """
        데이터 베이스에서 가져온 종목들의 점수를 계산

        """

    @abstractmethod
    def top_n_stocks(self):
        """
        데이터 베이스에서 가져온 종목들의 점수를 계산
        """

    @abstractmethod
    def get_universe(self, sector:list):
        """
        input sector 에 따라top n stocks 가져오기
        예를 들어 it만 입력할 수 있고 , 3개 넣을 수도 있음. list로 입력
        투자 유니버스내 기업 총 개수: 120개 - 국가 (2) x 산업 (11) x 상위 기업(5)
        """
    

    @abstractmethod
    def get_inital_portfolio(self, n, sector='ALL'):
        """
        input : universe
        n개 sector 의 n개 stocks
        3개 sector 1~2개 stock
        """

    @abstractmethod
    def get_portfoilo_ohlcv(self):
        '''
        :return: 1*n*5 array 예시 n=3
        { 종목: [ d, d, d, d, d], 종목2: [ d, d, d, d, d], 종목3:[ d, d, d, d, d] } -
        '''









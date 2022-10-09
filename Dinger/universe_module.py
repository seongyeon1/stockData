from lib2to3.refactor import get_all_fix_names
from webbrowser import get
from Dinger.universe_abstract import Universe

import pandas as pd
'''
df.columns = ['종목명', 'SEC_NM_KOR', '업종명', '시장구분', 'MKT_VAL', 'ALL_MKT_VAL', 'PER',
'PBR', 'DIV', '매출액', '영업이익', '당기순이익', 'ROA', 'ROE', '베타']
'''

class Universe(Universe):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        #self.mode = mode

    def get_standard_df(self, mode='mean', scaling=False):
        
        """
        mode = 'mean', 'logistic'
        database module output df
        mode  : 어떤  기준 ex. mean, median으로 할지
        """
        # if sector = 'sec':
        #     sec = self.data.groupby('SEC_NM_KOR')
        if '종목코드' in self.data.columns:
            self.data.set_index('종목코드', inplace=True)
    
        if mode == 'mean':
            return self.data.groupby('SEC_NM_KOR').mean().drop(['ALL_MKT_VAL'], axis=1)

        if mode == 'quant':
            return self.data.groupby('SEC_NM_KOR').quantile().drop(['ALL_MKT_VAL'], axis=1)

        if mode == 'median':
            return self.data.groupby('SEC_NM_KOR').median().drop(['ALL_MKT_VAL'], axis=1)

        # if mode == 'std':
        #     return self.get_score(mode='std').describe().T
        #return df

    def get_score(self, mode='mean'):
        """
        mode = 'mean','quant','median','mm','std'
        """
        if '종목코드' in self.data.columns:
            self.data.set_index('종목코드', inplace=True)

        #data = self.data
        gubun = self.data[['종목명','SEC_NM_KOR','업종명','시장구분']]
    
        sector_list = gubun.SEC_NM_KOR.unique()
        if (mode == 'mean') or (mode == 'quant') or (mode == 'median'):
            if mode == 'mean':
                sec_mean = self.data.groupby('SEC_NM_KOR').mean()
            if mode == 'quant':
                sec_mean = self.data.groupby('SEC_NM_KOR').quantile()
            if mode == 'median':
                sec_mean = self.data.groupby('SEC_NM_KOR').median()

            score = dict()
            for sec in sector_list:
                tmp = self.data.drop(gubun.columns.tolist(), axis=1).loc[(gubun.SEC_NM_KOR == sec)]

                for i in range(len(tmp)):
                    tmp.iloc[i] = tmp.iloc[i] > sec_mean.loc[sec]

                tmp = tmp.sum(axis=1).to_dict()
                score.update(tmp)

            for k in score.keys():
                gubun.loc[k,'Score'] = score[k]
            
            return gubun
        
        data = self.data.dropna()
        data.drop(gubun.columns.tolist(), axis=1, inplace=True)
        data.drop('ALL_MKT_VAL', axis=1, inplace=True)

        if mode == 'mm':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            scaled_data = pd.DataFrame(scaled_data, columns=data.columns).set_index(data.index)
            scaled_data = pd.DataFrame(scaled_data.sum(axis=1), columns=['Score'])
            scaled_data = gubun.merge(scaled_data, right_index=True, left_index=True)
            return scaled_data

        if mode == 'std':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()   
            scaled_data = scaler.fit_transform(data)
            
            scaled_data = pd.DataFrame(scaled_data, columns=data.columns).set_index(data.index)
            scaled_data = pd.DataFrame(scaled_data.sum(axis=1), columns=['Score'])
            scaled_data = gubun.merge(scaled_data, right_index=True, left_index=True)
            return scaled_data

    def top_n_sector(self, n=5, mode='mean'):
        """
        데이터 베이스에서 가져온 종목들의 점수를 계산
        """
        score = self.get_score(mode=mode).groupby('SEC_NM_KOR').mean().sort_values('Score', ascending=False)
        return score.iloc[:n]


    def top_n_stocks(self, n=5, mode='mean'):
        """
        데이터 베이스에서 가져온 종목들의 점수를 계산
        """
        score = self.get_score(mode=mode).sort_values('Score', ascending=False)
        return score.iloc[:n]


    def get_universe(self, n=5, mode='mean', sector=None):
        """
        input sector 에 따라top n stocks 가져오기
        예를 들어 it만 입력할 수 있고 , 3개 넣을 수도 있음. list로 입력
        투자 유니버스내 기업 총 개수: 120개 - 국가 (2) x 산업 (11) x 상위 기업(5)
        """
        if sector is None:
            sector = self.data.SEC_NM_KOR.unique()
        
        result = pd.DataFrame()
        for sec in sector:
            score = self.get_score(mode=mode)
            score = score.loc[score.SEC_NM_KOR == sec].sort_values('Score', ascending=False)[:n]
            result = pd.concat([result, score])

        result.to_csv(f"./data/universe.csv")
        return result

    def get_inital_portfolio(self, n=5, mode='mean',sector=None):
        """
        input : universe
        n개 sector 의 n개 stocks
        3개 sector 1~2개 stock
        """
        universe = self.get_universe(n=n, mode=mode, sector=sector)

        universe.to_csv("./data/my_universe.csv")
        return universe

    def get_portfoilo_ohlcv(self):
        '''
        가장 최근 ohlcv 하나만 가져오기
        :return:
        '''
        from datetime import datetime, timedelta
        from pykrx import stock
        import numpy as np

        my_port = pd.read_csv("./data/my_universe.csv")
        tickers = my_port.종목코드.astype(str).str.zfill(6).tolist()
        start = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        end = datetime.now().strftime('%Y%m%d')
        portfoilo = {}

        for ticker in tickers:
            ohlcv = stock.get_market_ohlcv(start, end, ticker, 'd').pct_change()
            portfoilo[ticker] = np.array(ohlcv.iloc[-1])

        return portfoilo

if __name__ == "__main__":
    data = pd.read_csv('./data/stock_kr.csv')









from Dinger.database_abstract import StockDataBase

# !pip install pykrx
# !pip install tqdm

from pykrx import stock
import pandas as pd

import requests
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np


class stockKR(StockDataBase):
    def __init__(self, dt):
        self.dt = dt

    # ---------------------------------------------------------------------------------------------------------#
    def __make_database(self):
        '''
        - 종목코드, 종목명, 시장구분, 업종명, BPS, PER, PBR, EPS, DIV, DPS
        '''
        kospi = pd.read_csv('./data/kospi.csv', encoding='cp949')
        kosdaq = pd.read_csv('./data/kosdaq.csv', encoding='cp949')

        kospi = kospi[['종목코드', '종목명', '시장구분', '업종명']]
        kosdaq = kosdaq[['종목코드', '종목명', '시장구분', '업종명']]

        stock_list = pd.concat([kospi, kosdaq])

        stock_fud = pd.DataFrame(stock.get_market_fundamental_by_ticker(date=self.dt, market="ALL"))
        stock_fud = stock_fud.reset_index()
        stock_fud.rename(columns={'티커': '종목코드'}, inplace=True)

        stock_data = pd.merge(stock_list, stock_fud, left_on='종목코드', right_on='종목코드', how='outer')

        return stock_data

    def __get_wics_sector(self):
        '''
        - sector 데이터 베이스 구축하기
        - wics 데이터 옆에 kosdaq. kospi 붙일 수 있음 붙이기
        - PER, PBR, ROE, ROA, DIV, 당기순이익, 영업이익, 매출액
        - 베타지수
        - wics 크롤링 : 시가총액
        '''
        sector_code = ['G25', 'G35', 'G50', 'G40', 'G10', 'G20', 'G55', 'G30', 'G15', 'G45']
        df1 = pd.DataFrame()
        for code in tqdm(sector_code):
            url = f'http://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={self.dt}&sec_cd={code}'
            json_data = requests.get(url).json()

            tmp = pd.DataFrame(json_data['list'])
            df1 = pd.concat([df1, tmp])

            df1 = df1[['CMP_CD', 'CMP_KOR', 'SEC_NM_KOR', 'MKT_VAL', 'ALL_MKT_VAL', 'WGT', 'S_WGT', 'APT_SHR_CNT']]
        return df1
        # return stock.get_market_fundamental_by_ticker(date=dt, market="ALL")

    def crawling(self):
        tmp = self.__make_database()
        tmp2 = self.__get_wics_sector()

        df = tmp2.merge(tmp, left_on='CMP_CD', right_on='종목코드')
        df = df[['종목코드', '종목명', 'SEC_NM_KOR', '업종명', '시장구분', 'MKT_VAL', 'ALL_MKT_VAL', 'PER', 'PBR', 'DIV']]

        return df

    # ---------------------------------------------------------------------------------------------------------#

    def __get_fundamental(self, ticker):
        '''
        매출액, 영업이익, 당기순이익 가져오기
        '''
        if type(ticker) != str:
            assert isinstance(ticker, int), 'Expected int'
            ticker = ticker.astype(str)

        ticker = ticker.zfill(6)
        URL = f"https://finance.naver.com/item/main.nhn?code={ticker}"
        r = requests.get(URL)
        df = pd.read_html(r.text)[3]
        df = df.T
        df.columns = df.iloc[0]
        df.drop('주요재무정보', inplace=True)
        df.index = df.index.get_level_values(1)
        df.index.name = None
        df.drop(df.index[df.index.str.endswith('(E)')].tolist(), inplace=True)
        # df.index = pd.to_datetime(df.index, format='%Y.%m')
        df.columns.name = None

        return df[['매출액', '영업이익', '당기순이익']].iloc[-1]

    def __get_ratio(self, ticker):
        '''
        - 현재는 roa, roe 가져오는 코드
        - 안정성비율, 성장성비율, 수익성비율, 활동성비율 등 다양한 비율들이 들어있어서 활용가능성 높은 코드
        
        2018-2022까지의 연간 비율
        - 안정성비율 : '유동비율', '당좌비율', '부채비율', '유보율', '순차입금비율', '이자보상배율', '자기자본비율'
        - 성장성비율 : '매출액증가율', '판매비와관리비증가율', '영업이익증가율', 'EBITDA증가율', 'EPS증가율'
        - 수익성비율 : '매출총이익율', '세전계속사업이익률', '영업이익률', 'EBITDA마진율', 'ROA', 'ROE', 'ROIC'
        - 활동성비율 : '총자산회전율', '총부채회전율', '총자본회전율', '순운전자본회전율'
        '''
        URL = f"http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{ticker}"
        r = requests.get(URL)
        tmp = pd.read_html(r.text)
        tmp = tmp[0]
        tmp.iloc[:, 0] = tmp.iloc[:, 0].apply(lambda x: x.replace('계산에 참여한 계정 펼치기', ''))
        tmp = tmp.set_index(tmp.columns[0])
        tmp.index.name = None
        tmp = tmp.T

        return tmp

    def __get_beta(self, code, n=8):
        URL = f"https://comp.fnguide.com/SVO2/ASP/SVD_main.asp?gicode=A{code}"
        r = requests.get(URL)
        tmp = pd.read_html(r.text)[n].set_index('구분')

        return tmp.T

    def get_fundamentals(self, ticker_list):
        fud_all = pd.DataFrame()
        error = []

        for ticker in tqdm(ticker_list):

            try:
                f = self.__get_fundamental(ticker)
                f.name = ticker
                r = self.__get_ratio(ticker)
                r = r[['ROA', 'ROE']].iloc[-1]
                r.name = ticker

                f = pd.DataFrame(f)
                r = pd.DataFrame(r)

                fud = pd.concat([f, r])
                fud.loc['베타'] = self.__get_beta(ticker)['베타(1년)'][0]

                fud_all = pd.concat([fud_all, fud.T])
            except:
                pass

        fud_all.to_csv(f'fundamentals.csv', index=False)
        return fud_all, error

    # ---------------------------------------------------------------------------------------------------------#

    def stocks_info(self):
        df = self.crawling()
        df.to_csv('crawling.csv', index=False)
        fud_all, err = self.get_fundamentals(df.종목코드.astype(str).str.zfill(6).tolist())
        data = df.merge(fud_all, left_on='종목코드', right_index=True)

        data.loc[data.당기순이익 == '-', '당기순이익'] = np.nan
        data.loc[data.매출액 == '-', '매출액'] = np.nan
        data.loc[data.영업이익 == '-', '영업이익'] = np.nan

        data.당기순이익 = data.당기순이익.astype('float')
        data.매출액 = data.매출액.astype('float')
        data.영업이익 = data.영업이익.astype('float')

        data[['ROA', 'ROE', '베타']] = data[['ROA', 'ROE', '베타']].astype('float')

        data.to_csv('./data/stock_kr.csv')
        return data, err

    # ---------------------------------------------------------------------------------------------------------#

    def get_individual_ohlcv(ticker, start=(datetime.now() - timedelta(days=100)).strftime('%Y%m%d'),
                             end=datetime.now().strftime('%Y%m%d'), dmy='d'):
        '''
        - ticker 개별 ohlcv 가져오기
        - ../stock_kr/{ticker}.csv 파일로 저장하기
        '''
        # now = datetime.now()

        # start = (datetime.now()-timedelta(days=100)).strftime('%Y%m%d')
        # end = datetime.now().strftime('%Y%m%d')

        ohlcv = stock.get_market_ohlcv(start, end, ticker, dmy).iloc[-64:, :]
        ohlcv.to_csv(f'./data/stock_kr/{ticker}.csv')
        return pd.DataFrame(ohlcv)

    def get_resnet(start=(datetime.now() - timedelta(days=120)).strftime('%Y%m%d'),
                   end=datetime.now().strftime('%Y%m%d'), dmy='d'):
        '''
        - ticker 개별 ohlcv 가져오기
        - ../stock_kr/{ticker}.csv 파일로 저장하기
        '''
        import numpy as np
        import pandas as pd
        import torch
        from pykrx import stock
        from Dinger import resnet


        resnet_dict = {}
        stock_kr = pd.read_csv('./data/stock_kr.csv')
        error = []
        stock_kr.종목코드 = stock_kr.종목코드.astype(str).str.zfill(6)

        for ticker in tqdm(stock_kr.종목코드.tolist()):
            try:
            # stockKR.get_individual_ohlcv(ticker)
            # now = datetime.now()
            #start = (datetime.now() - timedelta(days=150)).strftime('%Y%m%d')
            # end = datetime.now().strftime('%Y%m%d')

                data = stock.get_market_ohlcv(start, end, ticker, dmy)

                # 가장 최근 64 길이 (이미지 1장)
                test_data = data.iloc[-64:][["종가", "거래량"]].dropna()

                # model
                model = resnet.ResNet()

                # model weight load
                model.load_state_dict(torch.load("./Dinger/resnet.pth"))

                # 추론 이미지 1장 (0: Up, 1: Down, 2: Side)
                # Input : pd.DataFrame
                # Output: torch.Tensor
                pred = model.predict(test_data)
                resnet_dict[ticker] = np.array(pred)[0]
            except:
                error.append(ticker)
                pass

        for k in resnet_dict:
            stock_kr.loc[stock_kr.종목코드 == k, 'resnet'] = resnet_dict[k]

        stock_kr.to_csv('./data/stock_kr.csv', index=False)
        return resnet_dict, error

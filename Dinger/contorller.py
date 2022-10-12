from Dinger.universe_module_us import StockUniverse_US
from Dinger.universe_module_kr import StockUniverse_KR
from Dinger.database_module_us import stockUS
from Dinger.database_module_kr import stockKR
from Dinger.rebalancing_module import RebalancingAgent

import pandas as pd 
import FinanceDataReader as fdr

class Controller:
    """
    Dinger Controller
    """

    MAIN_STATEMENT = "명령어를 입력하세요. (h: 도움말): "

    def __init__(self, mode, scaling):
        self.mode = mode
        self.scaling = scaling

        self.terminating = False
        self.command_list = []
        self.create_command()

        self.stock_us = pd.read_csv('DF/sp500_score_df.csv', index_col=0)
        self.stock_kr = pd.read_csv('DF/stock_kr.csv')
        sp500_list = pd.read_csv('DF/sp500_list.csv')

        self.uni_US = StockUniverse_US(stock_list=sp500_list)
        self.uni_KR = StockUniverse_KR(self.stock_kr)

        self.standard_US = self.uni_US.get_standard_df(self.stock_us, self.mode)
        self.standard_KR = self.uni_KR.get_standard_df()


    def create_command(self):
        """ 명령어 정보를 생성한다. """
        self.command_list = [
            {
                "guide": "{0:15} 도움말 출력".format("h, help"),
                "cmd": ["help"],
                "short": ["h"],
                "action": self.print_help,
            },
            {
                "guide": "{0:15} 미국 스코어 데이터 프레임 받아오기".format("uscore, us"),
                "cmd": ["uscore"],
                "short": ["us"],
                "action": self.get_score_us,
            },
            {
                "guide": "{0:15} 한국 스코어 데이터 프레임 받아오기".format("kscore, ks"),
                "cmd": ["kscore"],
                "short": ["ks"],
                "action": self.get_score_kr,
            },
            {
                "guide": "{0:15} 미국 펀더멘털 데이터 프레임 받아오기".format("ufunda, uf"),
                "cmd": ["ufunda"],
                "short": ["uf"],
                "action": self.get_standard_USDF,
            },
            {
                "guide": "{0:15} 한국 펀더멘털 데이터 프레임 받아오기".format("kfunda, kf"),
                "cmd": ["kfunda"],
                "short": ["kf"],
                "action": self.get_standard_KRDF,
            },
            {
                "guide": "{0:15} 프로그램 종료".format("t, terminate"),
                "cmd": ["terminate"],
                "short": ["t"],
                "action": self.terminate,
            },
        ]
    

    def main(self):
        while not self.terminating:
            key = input(self.MAIN_STATEMENT)
            self._on_command(key)

    def print_help(self):
        """ 가이드 문구 출력 """
        print("명령어 목록 ==============")
        for item in self.command_list:
            print(item["guide"])

    def get_standard_USDF(self):
        """ 미국 펀더멘털 데이터 프레임 저장 """
        self.standard_US.to_csv("./DF/standard_US.csv")
    
    def get_standard_KRDF(self):
        """ 한국 펀더멘털 데이터 프레임 저장 """
        self.standard_KR.to_csv("./DF/standard_KR.csv")

    def get_score_us(self):
        """ 미국 스코어 데이터 프레임 저장 """
        cnn_score = pd.read_csv('DF/cnn_score_df.csv', index_col =0)
        cnn_score = cnn_score.to_dict('dict')['0']
        self.uni_US.add_cnn_score(self.stock_us, cnn_score)
        scored_df = self.uni_US.get_score(scaling=self.scaling)
        scored_df.to_csv("./DF/score_df_us.csv")

    def get_score_kr(self):
        """ 한국 스코어 데이터 프레임 저장 """
        scored_df = self.uni_KR.get_score()
        scored_df.to_csv("./DF/score_df_kr.csv")

    def terminate(self):
        """ 프로그램 종료 """
        print("프로그램 종료 중 ....")
        self.terminating = True
        print("프로그램 종료")

    def _on_command(self, key):
        """ 커맨드 처리를 담당 """
        for cmd in self.command_list:
            if key.lower() in cmd["cmd"] or key.lower() in cmd["short"]:
                print(f"{cmd['cmd'][0].upper()} 명령어를 실행합니다.")
                cmd["action"]()
                return
        print("잘못된 명령어가 입력 되었습니다.")        

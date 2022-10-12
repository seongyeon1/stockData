
##==== initial portfolio 만들기 
#last updated on 10/09/2022
##==update history 
#10/09/2022 : change ohlcv output form sipmple values to to ohlcv ratio form previous date 
#10/04/2022 : added scaling option and get_portfolio_ohlcv
#10/03/2022 : added cnn score and methods accordingly
#09/30/2022 : complete method (~ get_initial_portfolio)
#09/29/2022 : add abstract method
#09/23/2022 : initialize


"""
 선정
    - Score
    - Universe
    - 상위섹터, 상위종목,
"""
import yfinance as yf

from sklearn.preprocessing  import MinMaxScaler, StandardScaler
from Dinger.utils import *
from Dinger.universe_abstract import *
from Dinger.resnet import * #cnn score를 구할 package import 


class StockUniverse_US(Universe): 

    def __init__(self, scaling = None, stock_list = None):

        """
        initialize class 
        default values 

        attribute list for scoring 
        positive attirbutes : 값이 높을 수록 좋은 features 
        negative attribtues : 값이 낮을 수록 좋은 fuatures

        mode: scoring 측정하는 measurement, default mean (sector별 feature의 평균값 )
        """

        self.p_attribute_list = ['Sales','Revenue','MarCap','ROA','ROE']
        self.n_attribute_list = ['PER']
        self.stock_list = stock_list
        self.scaling = scaling 




    def get_standard_df(self, df, mode = 'mean', q =0.5 ) : 

        """

        output dataframe

        output: dataframe


        df  : 종목별 fundamentals 정보가 담긴 dataframe
        df내 'Sector' column을 기준으로 평균 산출 
        mode : mean, median 등 standard df를 산출할 기준 
        """

        super().get_standard_df()

        self.df = df

        if mode == 'mean': 
            self.standard_df = pd.DataFrame(df.groupby(['Sector']).mean())

        elif mode == 'median' :
            self.standard_df = pd.DataFrame(df.groupby(['Sector']).median())

        elif mode == 'quantile' :
            self.standard_df = pd.DataFrame(df.groupby(['Sector']).quantile(q))

        return self.standard_df
    



    def get_cnn_score( self, data: pd.DataFrame , size = 64, epochs = 10 ):


        """

        한 개 stock의 추후 up or down trend 예측하는 method

        input
        data : stock 의 ohlcv data [] , pd.DataFrame에 "Close", "Volume" column이 있어야 함 
        stock의 ohlcv data 를 입력하면, down or uptrend를 예측하고 dictionary로 반환한다 

        1개 data 에 대한 cnn score구할 시 :data 입력 
        여러개 data 에 대한 cnn score 구할 시 : stock_list 입력 

        """



        cnn_score = dict()

        data = data[["Close", "Volume"]] #clsoe와 volume만 가져오기 

        gd = GenerateDataset(data = data, size = size)
        image_data, label_data = gd.generate_image()
        convnet = ResNet()
        training_model = TrainingModel(model=convnet, x_all=image_data, y_all=label_data)
        training_model.train_test_split()
        training_model.train(epochs=epochs)

        training_model.test()

        i = 0
        if np.argmax(label_data[i]) == 0:
            pred = 1 ,
            title ="Uptrend" # uptrend
        elif np.argmax(label_data[i]) == 1:
            pred = 0 ,
            title ="Downtrend"# uptreand
        else:
            pred = 0.5 ,
            title ="Sidetrend" #sidetrend

        cnn_score["추세예측값"] = pred

        #plot
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title(title)
        plt.imshow(image_data[i,0], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(image_data[i+4,0], cmap="gray")
        plt.show()

        self.cnn_score = cnn_score
        

        #모델 저장 
        folder = './content'
        path = f"./content/ResNet-{epochs}epochs.pth"
        if not os.path.isdir(folder):
            os.mkdir(folder)

        training_model.save(path)

        
        return self.cnn_score #dictionary 


    def get_many_cnn_score( self, path , size = 64, epochs = 10 ,stock_list = None):
        
        """

        stock list의 종목에 대해 추후 트렌드를 예측해 dictionary로 반환합니다.

        stock list에 있는 data를 지정 폴더(path)에서 찾아옵니다.

        stock_list = pd.DataFrame. 'Symbol' column 이 있어야 합니다.
        path = folder명 (String) eg.sp500_ohlcv이면 sp500입력
        """

        if stock_list == None:
            stock_list= self.stock_list
    
        DIR = f'./{path}_ohlcv/'

        cnn_score = dict()

        for sym in stock_list.Symbol :
            #symbol 에 따른 data가져오기 
            print("==",sym,"==")

            try: 
                data = pd.read_csv(DIR+ f'{sym}_ohlcv.csv', index_col = 0) #각 symbol의 데이터 가져오기 
                data = data[["Close", "Volume"]] #clsoe와 volume만 가져오기

                gd = GenerateDataset(data = data, size = size, info = False)
                image_data, label_data = gd.generate_image(info = False)
                convnet = ResNet()
                training_model = TrainingModel(model=convnet, x_all=image_data, y_all=label_data)
                training_model.train_test_split()
                training_model.train(epochs=epochs)

                training_model.test()

                i = 0
                if np.argmax(label_data[i]) == 0:
                    pred = 1 # uptrend
                elif np.argmax(label_data[i]) == 1:
                    pred = 0 # uptreand
                else:
                    pred = 0.5 #sidetrend

                #모델 저장 
                folder = './content'
                path = f"./content/ResNet-{epochs}epochs.pth"
                if not os.path.isdir(folder):
                    os.mkdir(folder)

                training_model.save(path)

            except: 
                
                pred = None

            cnn_score[sym] = pred

            
        #save cnn score dictionary 
        self.cnn_score = cnn_score
        cnn_score_df = pd.DataFrame.from_dict(cnn_score, orient= 'index')
        cnn_score_df.to_csv('cnn_score_df.csv')

        return self.cnn_score


    
    def add_cnn_score( self, df, dict):
        """
        cnn score dictionary를  df에 추가한다 
        dict: cnn scored dictionary
        df = stock lists with fundamental information
        """

        scored_df = df.copy()

    #sym 을 보고 추가 _ 추후에는  get_score method에 모두 추가할 수 있을 듯 
        for k, v in dict.items():
            for idx, row in scored_df.iterrows():
                if row.Symbol == k:
                    scored_df.loc[idx, 'cnn_score'] = v

        self.scored_df = scored_df
    
        return self.scored_df

        


    def get_score(self, df = None, scaling = None, standard_df = None, p_attribute_list = None, n_attribute_list=None) :

        
        """
        
        input : df (default 값, 이전에 계산한 cnn_scored 포함 된 df), 
        cnn 추가하고 싶지 않을 경우 stock fundamentals 정보만 담긴 df 입력 가능 
        standard_df : Sector별 [mean]값 standard df [*mode에 따라 변경가능 ]
        scaling : default 각 종목의 fundamentals를 Sector 별 mean 값과 비교해 높으면 1, 낮으면 0 scoring 
        "mm", "std" 입력시 feature를 스케일링하고 단순 합 구함 
        p_attribute_list : 높을수록 좋은 features list 
        n_attribute_list : 낮을수록 좋은 feature list 
        defalut 값 
            self.p_check_list = ['Sales','Revenue','MarCap','ROA','ROE']
            self.n_check_list = ['PER']

        output: dataframe 
   
        """

        super().get_score()
    

        if p_attribute_list == None:
            p_attribute_list = self.p_attribute_list
        if n_attribute_list == None: 
            n_attribute_list = self.n_attribute_list

        if standard_df == None :
            standard_df = self.standard_df
        
        if df == None: 
            scored_df = self.scored_df.copy() # stocks들의 fundamentals 정보와 cnn score추가 
        else :
            scored_df = df


        if scaling == None : # scaling 없을 시 

            for idx , row in scored_df.iterrows(): # standdard df 와 비교 후 크면 1, 작으면 0으로 score 매기기 
                for att in p_attribute_list:
                    if row[att] >= standard_df.loc[row['Sector'], att] :
                        scored_df.loc[idx, att + '_score'] = 1
                    else: 
                        scored_df.loc[idx, att + '_score'] = 0
                    
                for att in n_attribute_list : 
                    if row[att] >= standard_df.loc[row['Sector'], att] :
                        scored_df.loc[idx, att + '_score'] = 1
                    else: 
                        scored_df.loc[idx, att + '_score'] = 0

                scored_df.loc[idx, 'total_score'] = scored_df.loc[idx, scored_df.columns.str.contains('_score')].sum()
            
            print("scored_df를 csv로 저장합니다.")
            scored_df.to_csv('scored_df.csv')

            return scored_df


        if scaling == "mm": #minmax scaler
            data = scored_df.iloc[:,4:] # ticker, industry등 string value 제외
            cols = list(scored_df.columns[4:])

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            scaled_data = pd.DataFrame(scaled_data, columns = cols).set_index(scored_df.index)
            scaled_data['total_score'] = scaled_data.sum(axis=1)
            
            mm_scaled_scored_df = scored_df.iloc[:,:4].merge(scaled_data, right_index= True, left_index = True)

            print("MinMax scaler 적용한 scored_df를 csv로 저장합니다.")
            mm_scaled_scored_df.to_csv('mm_scaled_scored_df.csv')

            return mm_scaled_scored_df
            
        
        if scaling == "std": #standardscaler
            data = scored_df.iloc[:,4:] # ticker, industry등 string value 제외
            cols = list(scored_df.columns[4:])
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            scaled_data = pd.DataFrame(scaled_data, columns = cols).set_index(scored_df.index)
            scaled_data['total_score'] = scaled_data.sum(axis=1)
            
            std_scaled_scored_df = scored_df.iloc[:,:4].merge(scaled_data, right_index= True, left_index = True)
            
            print("standard scaler 적용한 scored_df를 csv로 저장합니다.")
            std_scaled_scored_df.to_csv('std_scaled_scored_df.csv')

            return std_scaled_scored_df


    def top_n_sectors(self, num, scaling = None) :

        """
        total score 기준으로 가장 합이 높은 num개의 sector추출 

        input 
        num 몇개의 sector를 추출할지 
        data :  scoring이 완료 된 dataframe s

        return list 

        """

        super().top_n_sectors()

        scored_df = self.get_score(scaling = scaling)

        tmp = pd.DataFrame(scored_df.groupby(['Sector']).sum().sort_values(by = "total_score", ascending = False))

        display_condition = tuple([tmp.columns.str.contains('_score')]) #_score로 끝나는 columns만 display 합니다.
        # set the condition in tuple to resolve non tuple sequence for multidimensional index error

        top_sectors = tmp.index[:num].values.tolist() # get top n sectors as a list
        
        print( f"Total Score TOP {num} Sectors 는 {top_sectors}입니다.")
        display(tmp.loc[:,tmp.columns[display_condition]])

        return top_sectors


    
    def top_n_stocks(self,  sectors, num, scored_df = None, scaling = None, info = True):

        """
        입력한 sector에서 top n개 종목 추출 

        return dictionary
        """
        super().top_n_stocks()

        scored_df = self.get_score(scaling = scaling)

        sectors=sectors
    
        result = dict()

        for sector in range(len(sectors)):
            tmp= scored_df[scored_df['Sector']==sectors[sector]].sort_values(by = 'total_score', ascending = False)
            result[sectors[sector]] = tmp.iloc[:num,:2] # top n sectors의 ticker와 회사명 dictionary에 저장 
        
        if info == True : 

            print(f'Sector: {sectors[sector]} 의 top {num} stocks(ticker ,stock name)는 {result[sectors[sector]].values} 입니다.')
        
        display(tmp.iloc[:num,:]) ## top n stocks의 정보 display 
        
        
        return result #각 sector별 top n stocks들어있는 dictionary


    def get_universe(self, sectors , num, scored_df = None, scaling = None):
        """
        ######
        UNDER PROCESS (두 국가의 top n sectors가 합쳐져야 합니다.)
        ######

        input sector 에 따라top n stocks 가져오기 
        
        sectors :  어떤 섹터에서 universe를 추출할 지 리스트로 입력합니다. eg. [Information Technology], ['Industrials', 'Information Technology']
        num_stocks : 정한 섹터 중, top 몇개 종목을 가져올 지 입력합니다.

        eg

        투자 유니버스내 기업 총 개수: 120개 - 국가 (2) x 산업 (11) x 상위 기업(5) 
        """

        super().get_universe()

        universe = self.top_n_stocks(sectors = sectors , num = num, info = False) 

        self.universe= universe

        return self.universe
        

    def get_initial_portfolio(self, num_sectors , num_stocks, universe = None):

        """
        input : 
        universe dictionary , n개 sector , n개 stocks 
    
        output : dictionary {Sector : [[sym, stock name],[sym, stock name]], Sector2 : [[sym, stock name],[sym, stock name]]}

        """

        super().get_initial_portfolio()


        if universe == None :
            universe = self.universe


        if num_sectors > len(universe.keys()) : 
            raise ValueError("portfolio secotors의 수는 universe sectors의 수보다 클 수 없습니다.")
            
        initial_port = dict() 
        
        selected_sectors = list(universe.keys())[:num_sectors] # num_sectors 에 지정한 수 만큼의 섹터를 universe dictionary에서 가져옵니다.

        # 해당 하는 sector의 value (ticker, stock name)를 num_stocks개 initial portfolio dictionary에 추가합니다.
        for sector in range(len(selected_sectors)):
            initial_port[selected_sectors[sector]] = universe[selected_sectors[sector]].values.tolist()[:num_stocks] 
        
        self.initial_port = initial_port
        
        return self.initial_port 


    def get_portfolio_ohlcv (self, initial_port= None ):

        """
        입력한 포트폴리오의 ohlcv 데이터 변화율 dictionary 로 반환 
        output: dictionary ((v_t - v_(t-1)) / v_(t-1) )
        example
        { 종목: [ d, d, d, d, d], 종목2: [ d, d, d, d, d], 종목3:[ d, d, d, d, d] }

        """

        if initial_port == None:
            initial_port = self.initial_port
        # initial portfolior 각 sector의 (val) value list 내에서 vals[i] 첫번째 [0] == sym 가져오기   
        tmp = [vals[i][0] for vals in initial_port.values() for i in range(len(vals))] 
        
        initial_ohlcv_dict = {}
        
        # yf 에서 ohlcv가져오기 

        for ticker in range(len(tmp)):
            tmp_ohlcv = yf.download(tmp[ticker], period = '5d') # show recent 5 output
            print("=====",ticker, "최신 ohlcv date:",tmp_ohlcv.index[0],"=====") #해당 ticker의 최신 ohlcv 추출 날짜 
            recent = tmp_ohlcv.loc[:,['Open','High','Low','Close','Volume']].values[-1] # yfinance에 있는 해당 종목 가장 최근 OHLCV값 추출
            recent_next = tmp_ohlcv.loc[:,['Open','High','Low','Close','Volume']].values[-2] # 2nd 최근값
            ohlcv_ratio = (recent - recent_next) / recent_next # ( v_t - v_(t-1) ) / v_(t-1)

            initial_ohlcv_dict[tmp[ticker]] = ohlcv_ratio


        return initial_ohlcv_dict


        


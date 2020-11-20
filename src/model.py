import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pystan

from data import Data
from utils import select_method, check_pickle_open, input_date, input_plus_number, input_dict

# 予測したいイベントデータのインプットに使う
def predict_input_event_data():
    # カラム名と使う関数
    input_list = [
            {'name': 'date',      'function': input_date},
            {'name': 'length(h)', 'function': input_plus_number},
    ]
    return input_dict(input_list)

# 予測モデルを管理するクラス
class Model:
    def __init__(self, file_info):
        self.model_code = model_code
        # データフレーム操作インスタンス
        self.data = Data(file_info)
        # 読み込みファイルのパス
        paths = file_info['paths']
        self.path_learn_info = paths['data']  + 'stan_learn_info.pickle'
        self.path_stan_model = paths['model'] + 'stan_model.pickle'
        self.path_stan_fit   = paths['model'] + 'stan_fit.pickle'
        # ファイル読み込み
        self.learn_info = check_pickle_open(self.path_learn_info, '')
        self.stm        = check_pickle_open(self.path_stan_model, '')
        self.fit        = check_pickle_open(self.path_stan_fit, '')
    
    # メソッド選択
    def select_method(self):
        # 選択肢
        self.selection = {
                '1': {'name': '学習&予測',
                      'method': self.learn_predict,
                      },
                '2': {'name': '前回予測時の学習結果表示',
                      'method': self.show_learning_result,
                      },
                '3': {'name': '前回予測時の推定値、グラフ表示',
                      'method': self.show_predict,
                      },
        }
        # 選ぶ
        select_method(self.selection)
    
    # 学習&予測
    def learn_predict(self):
        # データ準備
        print('予測したい(次回の)イベントの情報を入力してください')
        next_event = predict_input_event_data()
        if next_event is None:
            return
        df = self.data.get_dataframe()
        # 学習
        self.learning(next_event, df)
        self.save_predict_result(next_event, df)
        self.show_learning_result()
        self.show_predict()
    
    # 学習結果表示
    def show_learning_result(self):
        if self.fit is None:
            print('先に学習してください')
            return
        # 収束具合
        print(self.fit)
        self.fit.plot()
        plt.show()
    
    # 予測結果表示
    def show_predict(self):
        if self.learn_info is None:
            print('先に学習してください')
            return
        print('予測したイベント')
        print(self.learn_info['next_event'])
        
        # 予測結果を抽出
        # 区間推定したいパーセントリスト
        interval_estimations = [50, 90]
        self.show_predict_value(interval_estimations)
        self.show_predict_graph()
    
    # 学習
    def learning(self, next_event, df):
        if self.stm is None:
            print('学習前にコンパイルします')
            self.compile_stan()
        # データ（辞書型）
        dat = {
            'T':   len(df),                  # 全日付の日数
            'len': df['length(h)'].tolist(), # イベント期間(h)
            'y':   df['point'].tolist(),     # 観測値
            'pred_term': 1,
            'pred_len' : [next_event['length(h)']]
        }
        # パラメータ設定
        n_itr = 5000
        n_warmup = n_itr - 1000
        chains = 3
        print('学習開始')
        self.fit = self.stm.sampling(
                data=dat,
                iter=n_itr,
                chains=chains,
                n_jobs=1,
                warmup=n_warmup,
                algorithm="NUTS",
                verbose=False)
        print('学習完了')
        # ファイル保存
        with open(self.path_stan_fit, mode="wb") as f:
            pickle.dump(self.fit, f)
        # 学習したときのデータを保存
        self.learn_info = {
            'df': df,
            'next_event': next_event,
        }
        # 保存
        with open(self.path_learn_info, mode="wb") as f:
            pickle.dump(self.learn_info, f)
        print('学習ファイル保存完了')
    
    # コンパイル
    def compile_stan(self):
        self.stm = pystan.StanModel(model_code=self.model_code)
        # ファイル保存
        with open(self.path_stan_model, mode="wb") as f:
            pickle.dump(self.stm, f)
    
    # 予測値表示
    def show_predict_value(self, interval_estimations):
        # 予測結果取り出し
        ms = self.fit.extract()
        key = 'alpha_pred'
        
        # 点推定
        mean = ms[key].mean(axis=0)
        print('点推定:', mean[-1])
        
        # 区間推定
        for p in interval_estimations:
            low = 50 - p/2
            high = 50 + p/2
            low_value = np.array(pd.DataFrame(ms[key]).apply(lambda x: np.percentile(x, low), axis=0))
            high_value = np.array(pd.DataFrame(ms[key]).apply(lambda x: np.percentile(x, high), axis=0))
            print('区間推定('+ str(p) +'%):', low_value[-1], '~', high_value[-1])
    
    # 予測グラフ表示
    def show_predict_graph(self):
        # x軸
        df = self.learn_info['df']
        X = df.index
        X_pred = X.tolist()
        X_pred.append(self.learn_info['next_event']['date'])
        
        # プロットするパラメータ
        plot_params = ['alpha_pred', 'mu_pred', 'b_len_pred']
        # 表示したい信頼区間
        p = 90
        # 予測結果取り出し
        ms = self.fit.extract()
        # 表示
        for key in plot_params:
            print(key)
            # 予測結果
            mean = ms[key].mean(axis=0)
            p_low  = 50 - p/2
            p_high = 50 + p/2
            pred_low  = np.array(pd.DataFrame(ms[key]).apply(lambda x: np.percentile(x, p_low), axis=0))
            pred_high = np.array(pd.DataFrame(ms[key]).apply(lambda x: np.percentile(x, p_high), axis=0))
            # プロット
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(X_pred, mean, label='predicted', c='red')
            plt.fill_between(X_pred, pred_low, pred_high, color='red', alpha=0.2)
            # alpha, muは実測値も並べる
            if key in ['alpha_pred', 'mu_pred']:
                ax.plot(X, df['point'], label='observed')
            if key == 'alpha_pred':
                plt.legend(loc='upper left', borderaxespad=0)
            ax.set_title(key)
            plt.show()

# Stanコード
# ローカルトレンド+時系変数モデル
model_code =  """
data {
  int       T;         // データ取得期間の長さ
  vector[T] len;       // イベント期間(h)
  vector[T] y;         // 観測値
  int       pred_term; // 予測期間の長さ
  vector[pred_term] pred_len; // 予測イベントのイベント期間(h)
}
parameters {
  vector[T]     b_len; // lenの係数
  vector[T]     mu;    // 水準成分の推定値
  real<lower=0> s_t;   // ev_lenの係数の変化を表す標準偏差
  real<lower=0> s_w;   // 水準成分の変動の大きさを表す標準偏差
  real<lower=0> s_v;   // 観測誤差の標準偏差
}
transformed parameters {
  vector[T] alpha;
  for(i in 1:T){
    alpha[i] = mu[i] + b_len[i] * len[i];
  }
}
model {
  for(i in 2:T){
    mu[i] ~ normal(mu[i-1], s_w);
    b_len[i] ~ normal(b_len[i-1], s_t);
    y[i] ~ normal(alpha[i], s_v);
  }
}
generated quantities{
  vector[T + pred_term] mu_pred;
  vector[T + pred_term] b_len_pred;   // lenの係数
  vector[T + pred_term] alpha_pred;
  mu_pred[1:T] = mu;
  b_len_pred[1:T] = b_len;
  alpha_pred[1:T] = alpha;
  for(i in 1:pred_term){
    mu_pred[T+i] = normal_rng(mu_pred[T+i-1], s_w);
    b_len_pred[T+i] = normal_rng(b_len_pred[T+i-1], s_t);
    alpha_pred[T+i] = mu_pred[T+i] + b_len_pred[T+i] * pred_len[i];
  }
}
"""

if __name__ == '__main__':
    file_info = {
            'paths': {
                    'data':  '../data/',
                    'model': '../models/',
            },
            'files': {
                    'dataframe':  'event_data.csv',
            },
    }
    model = Model(file_info)
    model.select_method()

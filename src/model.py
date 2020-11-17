import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pystan

# 以下、自作
from utils import select_method, predict_input_event_data, check_pickle_open

class Model:
    def __init__(self):
        # 読み込み設定
        self.data_path =  '../data/'
        self.dataframe_file_name = 'event_data.csv'
        self.next_event_file_name = 'next_event.pickle'
        self.model_code = model_code
        self.model_path = '../models/'
        self.model_name = 'stan_model.pickle'
        self.fit_name =   'stan_fit.pickle'
        self.result_file_name = 'predict_result.pickle'
        # データ読み込み
        # データフレーム
        self.load_dataframe()
        # 次回イベントデータ（前回予測時に入力）
        path = self.data_path + self.next_event_file_name
        self.next_event = check_pickle_open(path, '')
        # 予測結果（前回予測時）
        path = self.data_path + self.result_file_name
        self.results_dict = check_pickle_open(path, '')

        # コンパイルファイル
        path = self.model_path + self.model_name
        self.stm = check_pickle_open(path, '')
        # 学習済みファイル
        path = self.model_path + self.fit_name
        self.fit = check_pickle_open(path, '')
    
    # データ読み込み、整形
    def load_dataframe(self):
        self.df = pd.read_csv(self.data_path + self.dataframe_file_name)
        self.df['date'] = pd.to_datetime(self.df['date']).dt.date
        self.df.set_index('date', inplace=True)
    
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
        print('予測したい(次回の)イベントの情報を入力してください')
        self.next_event = predict_input_event_data()
        # ファイル保存
        with open(self.data_path + self.next_event_file_name, mode="wb") as f:
            pickle.dump(self.next_event, f)
            
        self.learning()
        self.save_predict_result()
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
        if self.results_dict is None:
            print('先に学習してください')
            return
        print('予測したイベント')
        print(self.next_event)
        #結果を抽出
        alpha_pred = self.results_dict['result']['alpha_pred']
        mean = alpha_pred['mean']
        p5 =  alpha_pred['p5']
        p25 = alpha_pred['p25']
        p75 = alpha_pred['p75']
        p95 = alpha_pred['p95']
        
        # 点推定
        print('点推定:', mean[-1])
        # 区間推定
        print('区間推定(90%):', p5[-1], '~', p95[-1])
        print('区間推定(50%):', p25[-1], '~', p75[-1])
        # グラフ表示
        self.show_graph()
    
    # 学習
    def learning(self):
        if self.stm is None:
            print('学習前にコンパイルします')
            self.compile_stan()
        # データ（辞書型）
        dat = {
            'T':   len(self.df),                  # 全日付の日数
            'len': self.df['length(h)'].tolist(), # イベント期間(h)
            'y':   self.df['point'].tolist(),     # 観測値
            'pred_term': 1,
            'pred_len' : [self.next_event['length(h)']]
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
        with open(self.model_path + self.fit_name, mode="wb") as f:
            pickle.dump(self.fit, f)
        print('学習ファイル保存')
    
    # コンパイル
    def compile_stan(self):
        self.stm = pystan.StanModel(model_code=self.model_code)
        # ファイル保存
        with open(self.model_path + self.model_name, mode="wb") as f:
            pickle.dump(self.stm, f)
    
    def save_predict_result(self):
        #結果を抽出
        # x軸
        X = self.df.index
        X_pred = self.df.index.tolist()
        X_pred.append(self.next_event['date'])
        ms = self.fit.extract() 
        self.results_dict = {
            'X': X,
            'X_pred': X_pred,
            'result': {key:{'mean': ms[key].mean(axis=0),
                      'p5':   np.array(pd.DataFrame(ms[key]).apply(lambda x: np.percentile(x, 5), axis=0)),
                      'p25':  np.array(pd.DataFrame(ms[key]).apply(lambda x: np.percentile(x, 25), axis=0)),
                      'p75':  np.array(pd.DataFrame(ms[key]).apply(lambda x: np.percentile(x, 75), axis=0)),
                      'p95':  np.array(pd.DataFrame(ms[key]).apply(lambda x: np.percentile(x, 95), axis=0)),
                      }
            for key in ['alpha_pred', 'mu_pred', 'b_len_pred']
            }}
        # 保存
        with open(self.data_path + self.result_file_name, mode="wb") as f:
            pickle.dump(self.results_dict, f)
    
    def show_graph(self):
        X = self.results_dict['X']
        X_pred = self.results_dict['X_pred']
        # 表示
        try:
            for key, value in self.results_dict['result'].items():
                print(key)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(X_pred, value['mean'], label='predicted', c='red')
                plt.fill_between(X_pred, value['p5'], value['p95'], color='red', alpha=0.2)
                if key != 'b_len_pred':
                    ax.plot(X, self.df['point'], label='observed')
                if key == 'alpha_pred':
                    plt.legend(loc='upper left', borderaxespad=0)
                ax.set_title(key)
                plt.show()
        except ValueError:
            print('データフレームが変わったので、再度学習してください')

# Stanコード
# ローカル線形トレンド+時系変数モデル
model_code =  """
data {
  int T;         // データ取得期間の長さ
  vector[T] len; // イベント期間(h)
  vector[T] y;   // 観測値
  int pred_term; // 予測期間の長さ
  vector[pred_term] pred_len; // 予測イベントのイベント期間(h)
}
parameters {
  vector[T] b_len;   // lenの係数
  vector[T] mu;      // 水準+ドリフト成分の推定値
  vector[T] delta;   // ドリフト成分の推定値
  real<lower=0> s_w; // 水準成分の変動の大きさを表す標準偏差
  real<lower=0> s_z; // ドリフト成分の変動の大きさを表す標準偏差
  real<lower=0> s_v; // 観測誤差の標準偏差
  real<lower=0> s_t; // lenの係数の変化を表す標準偏差
}
transformed parameters {
  vector[T] alpha;
  for(i in 1:T){
    alpha[i] = mu[i] + b_len[i] * len[i];
  }
}
model {
  for(i in 2:T){
    mu[i] ~ normal(mu[i-1] + delta[i-1], s_w);
    delta[i] ~ normal(delta[i-1], s_z);
    b_len[i] ~ normal(b_len[i-1], s_t);
    y[i] ~ normal(alpha[i], s_v);
  }
}
generated quantities{
  vector[T + pred_term] delta_pred;
  vector[T + pred_term] mu_pred;
  vector[T + pred_term] b_len_pred;   // lenの係数
  vector[T + pred_term] alpha_pred;
  delta_pred[1:T] = delta;
  mu_pred[1:T] = mu;
  b_len_pred[1:T] = b_len;
  alpha_pred[1:T] = alpha;
  for(i in 1:pred_term){
    delta_pred[T+i] = normal_rng(delta_pred[T+i-1], s_z);
    mu_pred[T+i] = normal_rng(mu_pred[T+i-1]+delta[T+i-1], s_w);
    b_len_pred[T+i] = normal_rng(b_len_pred[T+i-1], s_t);
    alpha_pred[T+i] = mu_pred[T+i] + b_len_pred[T+i] * pred_len[i];
  }
}
"""

if __name__ == '__main__':
    model = Model()
    model.select_method()

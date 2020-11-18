import pandas as pd
import matplotlib.pyplot as plt

from utils import select_method, input_yes_no, input_date, input_str, input_natural_number, input_plus_number, input_dict

# イベントデータのインプットに使う
def input_event_data():
    # カラム名と使う関数
    input_list = [
            {'name': 'date',       'function': input_date},
            {'name': 'event_name', 'function': input_str},
            {'name': 'point',      'function': input_natural_number},
            {'name': 'length(h)',  'function': input_plus_number},
    ]
    return input_dict(input_list)

# イベントデータを管理するクラス
class Data:
    def __init__(self):
        self.data_path = '../data/event_data.csv'
        self.load_dataframe()
    
    # データ読み込み、整形
    def load_dataframe(self):
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date']).dt.date
        self.df.set_index('date', inplace=True)
    
    # データ保存
    def save_dataframe(self):
        self.df.to_csv(self.data_path)
    
    # 選択肢
    def select_method(self):
        self.selection = {
                '1': {'name': 'データフレーム表示',
                      'method': self.show_dataframe,
                      },
                '2': {'name': 'グラフ表示',
                      'method': self.show_graph,
                      },
                '3': {'name': 'データ追加',
                      'method': self.add_data,
                      },
                '4': {'name': 'データ削除',
                      'method': self.delete_data,
                      },
        }
        select_method(self.selection)
    
    # データフレーム表示
    def show_dataframe(self):
        print(self.df)
    
    # グラフ表示
    def show_graph(self):
        for key in ['point', 'length(h)']:
            print(key)
            self.df[key].plot()
            plt.title(key)
            plt.show()
    
    # データ追加
    def add_data(self):
        data_dict = input_event_data()
        if data_dict is None:
            return
        df_add = pd.DataFrame(data_dict, index=['0']).set_index('date')
        # 既存のデータフレームに追加
        self.df = pd.concat([self.df, df_add], axis=0).sort_index()
        self.save_dataframe()
    
    # データ削除
    def delete_data(self):
        print('削除するデータの日付を指定してください')
        date = input_date()
        # Noneが帰ってきたらキャンセル
        if date is None:
            return
        
        # 入力チェック
        try:
            df_del = self.df.loc[[date], :]
        except KeyError:
            print('一致するイベントがありません')
            return
        
        # 再確認
        print('以下のイベントを削除して良いですか？')
        print(df_del)
        del_flg = input_yes_no()
        if del_flg:
            # 削除
            self.df.drop(index=date, inplace=True)
            # 保存
            self.save_dataframe()
            print('削除しました')

if __name__ == '__main__':
    data = Data()
    data.select_method()

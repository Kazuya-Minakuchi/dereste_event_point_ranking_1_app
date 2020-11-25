from src.utils import select_method
from src.data import Data
from src.model import Model

class App:
    def __init__(self, file_info):
        self.data = Data(file_info)
        self.model = Model(file_info)
    
    # 選択肢
    def select_method(self):
        selection = {
            '1': {
                'name': 'データの確認・追加・削除など',
                'method': self.data.select_method,
            },
            '2': {
                'name': '学習、予測',
                'method': self.model.select_method,
            },
        }
        # 選ぶ
        select_method(selection)

if __name__ == '__main__':
    from src import config
    app = App(config.file_info)

from utils import select_method
from data import Data
from model import Model

file_info = {
        'paths': {
                'data':  '../data/',
                'model': '../models/',
        },
        'files': {
                'dataframe':  'event_data.csv',
        },
}

data = Data(file_info)
model = Model(file_info)

modes = {
        '1': {'name': 'データの確認・追加・削除など',
              'method': data.select_method,
              },
        '2': {'name': '学習、予測',
              'method': model.select_method,
              },
        }

select_method(modes)

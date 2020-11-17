
# 以下、自作
from utils import select_method
from data import Data
from model import Model

data = Data()
model = Model()

modes = {
        '1': {'name': 'データの確認・追加・削除など',
              'method': data.select_method,
              },
        '2': {'name': '学習、予測',
              'method': model.select_method,
              },
        }

select_method(modes)

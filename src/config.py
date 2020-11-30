import os

cur_path = os.path.dirname(__file__)

file_info = {
        'paths': {
                'data':  os.path.join(cur_path, '../data/'),
                'model': os.path.join(cur_path, '../models/'),
        },
        'files': {
                'dataframe':  'event_data.csv',
        },
}

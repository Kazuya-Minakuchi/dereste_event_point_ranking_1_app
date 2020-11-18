import datetime
import pickle

# 選択肢
def select_method(select_dict):
    """
    select_dictの中身
    key:    数字(文字列)
    name:   表示する文字
    method: 実行するメソッド
    """
    quit_str = 'q'
    while True:
        print('モードを選んでください')
        for key, value in select_dict.items():
            print(key, ':', value['name'])
        mode_num = input('数字を入力。("' + quit_str + '"で戻る)')
        try:
            if mode_num == quit_str:
                print('戻ります')
                break
            else:
                print(select_dict[mode_num]['name'])
                select_dict[mode_num]['method']()
                print('')
        except KeyError:
            print('正しい数字を入力してください')

# データ全般のインプットに使う
def input_data(print_str):
    quit_str = 'q'
    data = input(print_str+ 'を入力してください。(' + quit_str + 'でキャンセル)')
    if data == quit_str:
        print('キャンセルしました')
        return None
    return data

# 文字列のインプットに使う
def input_str():
    input_str = input_data('文字列')
    return input_str

# 日付のインプットに使う
def input_date():
    while True:
        try:
            date_str = input_data('日付(yyyy-mm-dd形式)')
            # Noneが帰ってきたら終了
            if date_str is None:
                return
            # 入力形式チェック
            dttm = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            date = datetime.date(dttm.year, dttm.month, dttm.day)
            return date
        except ValueError:
            print('入力形式が間違っています')

# 自然数のインプットに使う
def input_natural_number():
    while True:
        try:
            point = input_data('正の整数')
            # Noneが帰ってきたら終了
            if point is None:
                return
            point = int(point)
            # 自然数の場合、返す
            if point > 0:
                return point
            print('正の整数を入力してください')
        except ValueError:
            print('入力値が誤っています')

# 正の数のインプットに使う
def input_plus_number():
    while True:
        try:
            length = input_data('正の数')
            # Noneが帰ってきたら終了
            if length is None:
                return
            length = float(length)
            # 正の数の場合、返す
            if length > 0:
                return length
            print('正の数を入力してください')
        except ValueError:
            print('入力値が誤っています')

# yes, noのインプットに使う
def input_yes_no():
    while True:
        data = input_data('[Y]es, [N]o')
        # Noneが帰ってきたらキャンセル
        if data is None:
            return
        elif data == 'Y':
            return True
        elif data == 'N':
            return False
        print('入力値が誤っています')

# インプットから辞書型データつくる
def input_dict(input_list):
    """
    input_listは↓のdictのリスト
    name:     カラム名
    function: データ入力に使う関数
    """
    # カラムごとに取得し、辞書型で返す
    data_dict = {}
    for input_data in input_list:
        print(input_data['name'], '入力')
        data = input_data['function']()
        # Noneが帰ってきたらキャンセル
        if data is None:
            return
        data_dict[input_data['name']] = data
    return data_dict

# ファイルがあったら開く、なかったらメッセージ表示してNoneを返す
def check_pickle_open(path, fault_message):
    try:
        with open(path, mode="rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(fault_message)
        return

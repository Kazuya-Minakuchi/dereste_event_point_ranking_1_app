import datetime
import pickle

# データ全般のインプットに使う
def input_data(print_str):
    quit_str = 'q'
    data = input(print_str+ 'を入力してください。(' + quit_str + 'で戻る)')
    if data == quit_str:
        print('戻ります')
        return
    return data

# メソッドの選択に使う
def select_method(select_dict):
    """
    select_dictの中身
    key:    数字(文字列)
        name:   表示する文字
        method: 実行するメソッド
    """
    while True:
        print('モードを選んでください')
        for key, value in select_dict.items():
            print(key, ':', value['name'])
        mode_num = input_data('数字')
        # Noneが帰ってきたら終了
        if mode_num is None:
            return
        print(select_dict[mode_num]['name'])
        try:
            select_dict[mode_num]['method']()
        except KeyError:
            print('正しい数字を入力してください')
        else:
            print('')

# 文字列のインプットに使う
def input_str():
    input_str = input_data('文字列')
    return input_str

# 日付のインプットに使う
def input_date():
    while True:
        date_str = input_data('日付(yyyy-mm-dd形式)')
        # Noneが帰ってきたら終了
        if date_str is None:
            return
        # 入力形式チェック
        if is_date(date_str):
            dttm = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            date = datetime.date(dttm.year, dttm.month, dttm.day)
            return date
        print('入力形式が誤っています')

# 自然数のインプットに使う
def input_natural_number():
    while True:
        num_str = input_data('正の整数')
        # Noneが帰ってきたらキャンセル
        if num_str is None:
            return
        # 入力形式チェック
        if is_natural_number(num_str):
            num = int(num_str)
            return num
        print('正の整数を入力してください')


# 文字列が日付型に変換できるか
def is_date(check_str):
    try:
        temp = datetime.datetime.strptime(check_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# 文字列が自然数に変換できるか
def is_natural_number(check_str):
    # 整数か
    try:
        num = int(check_str)
    except ValueError:
        return False
    # 0より大きいか
    if num > 0:
        return True
    return False

def is_positive_number(check_str):
    # 数値に直せるか
    try:
        num = float(check_str)
    except ValueError:
        return False
    # 正の数か
    if num > 0:
        return True
    return False

# 正の数のインプットに使う
def input_positive_number():
    while True:
        num_str = input_data('正の数')
        # Noneが帰ってきたら終了
        if num_str is None:
            return
        if is_positive_number(num_str):
            num = float(num_str)
            return num
        print('正の数を入力してください')

# yes, noのインプットに使う
def input_yes_no():
    while True:
        ym = input_data('[Y]es, [N]o')
        # Noneが帰ってきたらキャンセル
        if ym is None:
            return
        elif ym == 'Y':
            return True
        elif ym == 'N':
            return False
        print('入力値が誤っています')

# インプットから辞書型データつくる
def input_dict(input_list):
    """
    input_listは↓のdictのリスト
    name:     データ名(戻るdictのkeyになる)
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

# pickleファイルがあったら開く、なかったらメッセージ表示してNoneを返す
def load_pickle(path, fault_message):
    try:
        with open(path, mode="rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(fault_message)
        return

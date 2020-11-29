import datetime

# データ全般のインプットに使う
def input_data(print_str):
    quit_str = 'q'
    data = input(print_str+ 'を入力してください。(' + quit_str + 'で戻る)')
    # 終了キーのときはNoneを返す
    if data == quit_str:
        print('戻ります')
        return None
    return data

# 文字列のインプットに使う
def input_str():
    input_str = input_data('文字列')
    return input_str

# 日付のインプットに使う
def input_date():
    return loop_input('日付(yyyy-mm-dd形式)', convert_str_date)

# 整数のインプットに使う
def input_integer():
    return loop_input('整数', convert_str_integer)

# 自然数のインプットに使う
def input_natural_number():
    return loop_input('正の整数', convert_str_natural_number)

# 正の数のインプットに使う
def input_positive_number():
    return loop_input('正の数', convert_str_positive_number)

# yes, noのインプットに使う
def input_yes_no():
    return loop_input('[Y]es, [N]o', convert_str_y)

# 文字列を日付型に変換
def convert_str_date(date_str):
    # 入力形式が合っていれば日付型で返す
    if is_date(date_str):
        dttm = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        date = datetime.date(dttm.year, dttm.month, dttm.day)
        return date
    # 入力形式が合わないとき
    return False

# 文字列を整数に変換
def convert_str_integer(num_str):
    # 入力形式が合っていればintで返す
    if is_integer(num_str):
        return int(num_str)
    # 入力形式が合わないとき
    return False

# 文字列を自然数に変換
def convert_str_natural_number(num_str):
    # 入力形式が合っていればintで返す
    if is_natural_number(num_str):
        num = int(num_str)
        return num
    # 入力形式が合わないとき
    return False

# 文字列を正の数に変換
def convert_str_positive_number(num_str):
    # 入力形式が合っていればfloatで返す
    if is_positive_number(num_str):
        num = float(num_str)
        return num
    # 入力形式が合わないとき
    return False

# 文字列をYesフラグに変換
def convert_str_y(yn):
    """
    YはTrue, NはNoneで返す
    Falseは形式エラーに使う
    """
    if yn == 'Y':
        return True
    elif yn == 'N':
        return None
    # 入力形式が合わないとき
    return False

# 入力形式が合わない場合のループ処理
def loop_input(print_str: str, convert_func):
    """
    print_str: 入力画面で表示する文字列
    convert_func: 文字列を他の型に変換する関数。変換できないときはFalseが返ってくる
    """
    while True:
        ret_str = input_data(print_str)
        # Noneが返ってきたらキャンセル
        if ret_str is None:
            return None
        result = convert_func(ret_str)
        # Falseが返ってきたら入力しなおし
        if result is False:
            print('入力形式が誤っています')
            continue
        return result

# 文字列が日付形式か
def is_date(check_str):
    try:
        _ = datetime.datetime.strptime(check_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# 文字列が自然数か
def is_natural_number(check_str):
    # 整数か
    if is_integer(check_str):
        # 正の数か
        num = int(check_str)
        if num > 0:
            return True
    return False

# 文字列が整数か
def is_integer(check_str):
    # 整数か
    try:
        _ = int(check_str)
        return True
    except ValueError:
        return False

# 文字列が正の数か
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

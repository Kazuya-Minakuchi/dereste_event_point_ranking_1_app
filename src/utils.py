import pickle
from inputs import input_data

# メソッドの選択に使う
def select_method(selections):
    """
    selectionsの中身
    key:    数字(文字列)
        name:   表示する文字
        method: 実行するメソッド
    """
    while True:
        # 選択肢表示
        print('モードを選んでください')
        for key, value in selections.items():
            print(key, ':', value['name'])
        # 選択
        mode_num = input_data('数字')
        # Noneが帰ってきたらキャンセル
        if mode_num is None:
            return
        print(selections[mode_num]['name'])
        # 正しい値が入力されていたら、実行
        try:
            selections[mode_num]['method']()
        # 値が間違っていたら、やり直し
        except KeyError:
            print('正しい数字を入力してください')
        print('')

# 辞書データの関数を実行し、戻り値の入った辞書を返す
def execute_dict_functions(in_dict):
    """
    in_dictの中身
    key:      返すdictのkey
    function: 値（返すdictのvalue)を得る関数
    """
    # カラムごとに取得し、辞書型で返す
    out_dict = {}
    for key, function in in_dict.items():
        print(key, '入力')
        data = function()
        # Noneが返ってきたらキャンセル
        if data is None:
            return None
        out_dict[key] = data
    return out_dict

# pickleファイルがあったら開く、なかったらメッセージ表示してNoneを返す
def load_pickle(path, fault_message):
    try:
        with open(path, mode="rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(fault_message)
        return

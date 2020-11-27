import pickle

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

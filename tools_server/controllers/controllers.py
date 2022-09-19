import datetime

import flask

currencies_list = [
    'BTCUSDT',
    'ETHUSDT'
]

def is_Date(num: int):
    try:
        if len(str(num)) == 13:
            datetime.datetime.fromtimestamp(num / 1000)
        elif len(str(num)) == 10:
            datetime.datetime.fromtimestamp(num)
        else:
            return False
        return True
    except:
        return False


def run_model_controller(req: flask.Request):
    keys_expected = ['currency', 'start_day', 'num_days']

    key_list = list(req.json.keys())

    bl = False
    msg = ''

    for item in key_list:

        if not (item in keys_expected):
            bl = False
            msg = 'Arguments missing'
            break
        else:
            bl = True
            if item == 'currency':
                if not (isinstance(req.json[item], str) and req.json[item].upper() in currencies_list):
                    bl = False
                    msg = 'Error in key "currency"'
                    break
            elif item == 'start_day':
                if not (isinstance(req.json[item], int) and is_Date(req.json[item]) or req.json[item] == 'default'):
                    bl = False
                    msg = 'Error in key "start_day"'
                    break
            elif item == 'num_days':
                if not(isinstance(req.json[item], int) and req.json[item] > 0):
                    bl = False
                    msg = 'Error in key "num_days"'
                    break

    return bl, msg

zoo_params = [
    {
        "model" : {
            "frozen_bert" : True,
            "hid_size" : 312,
            "count_of_blocks" : 0
        },
        "start_lr" : 1e-3,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 312,
            "count_of_blocks" : 0
        },
        "start_lr" : 1e-3,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 312,
            "count_of_blocks" : 1
        },
        "start_lr" : 1e-3,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 312,
            "count_of_blocks" : 3
        },
        "start_lr" : 1e-3,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 312,
            "count_of_blocks" : 3
        },
        "start_lr" : 1e-3,
        "scheduler" : "warmup"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 312,
            "count_of_blocks" : 3
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 312,
            "count_of_blocks" : 2
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 312,
            "count_of_blocks" : 1
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 512,
            "count_of_blocks" : 3
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 512,
            "count_of_blocks" : 4
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 512,
            "count_of_blocks" : 5
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 1024,
            "count_of_blocks" : 3
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 1600,
            "count_of_blocks" : 3
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 512,
            "count_of_blocks" : 1
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    },
    {
        "model" : {
            "frozen_bert" : False,
            "hid_size" : 512,
            "count_of_blocks" : 2
        },
        "start_lr" : 1e-4,
        "scheduler" : "ReduceLROnPlateau"
    }
]

def create_name(params):
    name = ""
    for key in params:
        if key == "model":
            name += create_name(params["model"])
        else:
            name += str(key) + ":" + str(params[key]) + ";"
    return str(name)
    
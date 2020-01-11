
class Hyperparams:
    '''Hyper parameters'''
    isqwerty = True # If False, 10 keyboard layout is assumed.
         
    # model
    embed_size = 300 # alias = E
    encoder_num_banks = 16
    num_highwaynet_blocks = 4
    maxlen = 33 # maximum number of a pinyin sentence
    minlen = 10 # minimum number of a pinyin sentence
    norm_type = "bn" # Either "bn", "ln", "ins", or None
    dropout_rate = 0.5

    # training scheme
    lr = 0.0001
    logdir = "log/qwerty" if isqwerty  is True else "log/nine"
    #batch_size = 64
    #num_epochs = 20
    batch_size = 64
    num_epochs = 30
    recommend = 5
    

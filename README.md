# Chinese Input Method

This repo offers two ways to deal with Chinese Input Method, Viterbi Algorithm and Seq2Seq.

## Hint

I have to mention that I just combine What [Tiantian Le](https://github.com/letiantian/Pinyin2Hanzi) (for Viterbi Algorithm) and [Qiu Hu](https://github.com/whatbeg/GodTian_Pinyin) (for pinyin segment using maximum matching) do and make some alternations to codea in `HMM/`.

Then for codes in `Neural Translation/`, most comes from [Kyubyong's work](https://github.com/Kyubyong/neural_chinese_transliterator), and I just make some small changes, including dataset.

The dataset comes from [skdjfla's repo](https://github.com/skdjfla), containing news from TouTiao.

## Result

Results on testset can be found in 

`Pinyin-to-Character/Neural Translation/data/model_epoch_30_gs_192180_qwerty.csv` for Seq2Seq model, and `Pinyin-to-Character/HMM/data/viterbi.csv` for Viterbi Algorithm. 

For more information, you can have a look at my [report](https://github.com/SunflowerAries/Pinyin-to-Character/blob/master/Report.pdf)


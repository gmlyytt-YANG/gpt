from decoder import *

class GPT1(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, diff,
                 target_vocab_size, 
                 max_seq_len, 
                 fine_tuning_class_num, 
                 drop_rate=0.1):
        super(GPT1, self).__init__()

        self.decoder = Decoder(n_layers, d_model, n_heads, diff,
                              target_vocab_size, max_seq_len, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.fine_tuning_layer = tf.keras.layers.Dense(fine_tuning_class_num)
        
    def call(self, targets, training, look_ahead_mask):

        decode_out, att_weights = self.decoder(targets, training, 
                                               look_ahead_mask)
        final_out = self.final_layer(decode_out)
        fine_tuning_out = self.fine_tuning_layer(tf.keras.layers.Flatten()(final_out))

        return final_out, fine_tuning_out, att_weights


if __name__ == "__main__":
    gpt1_test = GPT1(
        n_layers=12, 
        d_model=512, 
        n_heads=8, 
        diff=1024,
        target_vocab_size=8000,
        max_seq_len=40, 
        fine_tuning_class_num=15)
    input_target = tf.random.uniform((64, 26))
    final_out, fine_tuning_out, att_weights = gpt1_test(
        input_target, 
        training=False,
        look_ahead_mask=None)
    print(final_out.shape)
    print(fine_tuning_out.shape)
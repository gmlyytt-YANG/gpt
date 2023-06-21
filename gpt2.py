from util import * 
from decoder import *

class GPT2(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, diff,
                 target_vocab_size, 
                 max_seq_len, 
                 drop_rate=0.1):
        super(GPT2, self).__init__()

        self.decoder = Decoder(n_layers, d_model, n_heads, diff,
                              target_vocab_size, max_seq_len, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, targets, training, look_ahead_mask):

        decode_out, att_weights = self.decoder(targets, training, 
                                               look_ahead_mask)
        final_out = self.final_layer(decode_out)

        return final_out, att_weights
    
    
if __name__ == "__main__":
    gpt2_test = GPT2(
        n_layers=12, 
        d_model=512, 
        n_heads=8, 
        diff=1024,
        target_vocab_size=8000,
        drop_rate=0.1)
    input_target = tf.random.uniform((64, 26))
    final_out, att_weights = gpt2_test(
        input_target, 
        training=False,
        look_ahead_mask=None)
    print(final_out.shape)

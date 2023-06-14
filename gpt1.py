from util import *

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MutilHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)

    def call(self, inputs, training, look_ahead_mask):
        # masked muti-head attention
        att1, att_weight1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        out1 = self.layernorm1(inputs + att1)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layernorm2(out1 + ffn_out)

        return out2, att_weight1

class Decoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf, 
                 target_vocab_size, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.decoder_layers= [DecoderLayer(d_model, n_heads, ddf, drop_rate) 
                              for _ in range(n_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, training, look_ahead_mark):

        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        h = self.embedding(inputs)
        h *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        h += self.pos_embedding[:,:seq_len,:]

        h = self.dropout(h, training=training)
        for i in range(self.n_layers):
            h, att_w1 = self.decoder_layers[i](h, training, look_ahead_mark)
            attention_weights['decoder_layer{}_att_w1'.format(i+1)] = att_w1

        return h, attention_weights


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
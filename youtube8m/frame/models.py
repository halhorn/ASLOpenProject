'''
Frame Level Models
各種モデルを実装します。
各モデルは tf.keras.models.Model を継承し、 call で logits を返します。

入力の shape: [batch, len, dim]
'''
import tensorflow as tf
from .common_layer import FeedForwardNetwork, ResidualNormalizationWrapper, LayerNormalization
from .embedding import AddPositionalEncoding
from .attention import SelfAttention


def create_model(params):
    model_map = {
        'linear': LinearModel,
        'dnn': DNNModel,
        'cnn': CNNModel,
        'rnn': RNNModel,
        'attention': AttentionModel,
    }
    print('----------------------------------')
    print('Model:', params['model'])
    return model_map[params['model']](params)


class LinearModel(tf.keras.models.Model):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dense = tf.keras.layers.Dense(params['output_dim'])
    
    def call(self, visual_feature, audio_feature):
        '''
        フィーチャーを受け取って sigmoid の logits を返します
        '''
        mean_rgb = tf.reduce_mean(visual_feature, axis=1)
        mean_audio = tf.reduce_mean(audio_feature, axis=1)
        return self.output_dense(tf.concat([mean_rgb, mean_audio], axis=-1))


class DNNModel(tf.keras.models.Model):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layer_num = params.get('layer_num', 4)
        hidden_dim = params.get('hidden_dim', 1024)
        dropout = params.get('dropout', 0.1)
        kernel_reg = params.get('kernel_regularizer', 0.0)
        print('layer_num: {}\nhidden_dim: {}\ndropout: {}'.format(layer_num, hidden_dim, dropout))
        
        self.hidden_layers = [
            tf.keras.layers.Dense(
                hidden_dim,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1(l=kernel_reg),
            )
            for i in range(layer_num)
        ]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout) for i in range(layer_num)]
        self.output_dense = tf.keras.layers.Dense(params['output_dim'])
        
    def call(self, visual_feature, audio_feature):
        '''
        フィーチャーを受け取って sigmoid の logits を返します
        '''
        mean_rgb = tf.reduce_mean(visual_feature, axis=1)
        mean_audio = tf.reduce_mean(audio_feature, axis=1)
        x = tf.concat([mean_rgb, mean_audio], axis=-1)
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x)
        return self.output_dense(x)

class CNNModel(tf.keras.models.Model):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pre_layer_num = params.get('pre_layer_num', 4)
        cnn_filter_num = params.get('cnn_filter_num', 128)
        
        
        layer_num = params.get('layer_num', 2)
        hidden_dim = params.get('hidden_dim', 1024)
        dropout = params.get('dropout', 0.1)
        kernel_reg = params.get('kernel_regularizer', 0.0)

        self.conv_layers = [
            tf.keras.layers.Conv1D(cnn_filter_num, kernel_size=3, padding='same', activation='relu')
            for i in range(pre_layer_num)
        ]
        self.pool_layers = [
            tf.keras.layers.MaxPool1D(pool_size=2)
            for i in range(pre_layer_num)
        ]
        
        self.hidden_layers = [
            tf.keras.layers.Dense(
                hidden_dim,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1(l=kernel_reg),
            )
            for i in range(layer_num)
        ]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout) for i in range(layer_num)]
        self.output_dense = tf.keras.layers.Dense(params['output_dim'])
        
    def call(self, visual_feature, audio_feature):
        '''
        フィーチャーを受け取って sigmoid の logits を返します
        '''
        x = tf.concat([visual_feature, audio_feature], axis=-1)  # [batch, len, dim]
        
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x = conv_layer(x)
            x = pool_layer(x)
        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])  # flatten
        
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x)
        return self.output_dense(x)

class RNNModel(tf.keras.models.Model):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pre_layer_num = params.get('pre_layer_num', 4)
        
        layer_num = params.get('layer_num', 2)
        hidden_dim = params.get('hidden_dim', 1024)
        dropout = params.get('dropout', 0.1)
        kernel_reg = params.get('kernel_regularizer', 0.0)

        self.rnn_layers = [
            tf.keras.layers.GRU(
                hidden_dim,
                dropout=dropout,
                return_sequences=(i != pre_layer_num - 1),
            )
            for i in range(pre_layer_num)
        ]
        
        self.hidden_layers = [
            tf.keras.layers.Dense(
                hidden_dim,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1(l=kernel_reg),
            )
            for i in range(layer_num)
        ]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout) for i in range(layer_num)]
        self.output_dense = tf.keras.layers.Dense(params['output_dim'])
        
    def call(self, visual_feature, audio_feature):
        '''
        フィーチャーを受け取って sigmoid の logits を返します
        '''
        x = tf.concat([visual_feature, audio_feature], axis=-1)  # [batch, len, dim]
        
        for rnn in self.rnn_layers:
            x = rnn(x)
        
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x)
        return self.output_dense(x)

class AttentionModel(tf.keras.models.Model):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hopping_num = params.get('pre_layer_num', 6)
        head_num = params.get('head_num', 8)
        hidden_dim = (params.get('hidden_dim', 512) // head_num) * head_num
        dropout_rate = params.get('dropout', 0.1)
        
        self.input_dense = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list = []
        for _ in range(hopping_num):
            attention_layer = SelfAttention(hidden_dim, head_num, dropout_rate, name='self_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper(attention_layer, dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate, name='ffn_wrapper'),
            ])
        self.output_normalization = LayerNormalization()
        self.output_layer = tf.keras.layers.Dense(params['output_dim'])

    def call(
            self,
            visual_feature,
            audio_feature,
            training=None,
    ):
        '''
        モデルを実行します

        :param visual_feature: shape = [batch_size, length, dim]
        :param audio_feature: shape = [batch_size, length, dim]
        :param training: 学習時は True
        :return: shape = [batch_size, output_dim]
        '''
        input = tf.concat([visual_feature, audio_feature], axis=-1)
        input = self.input_dense(input)
        tf.print('input', input.shape)
        self_attention_mask = self._create_enc_attention_mask(input)
        embedded_input = self.add_position_embedding(input)
        tf.print('emb', embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)
        tf.print(query.shape)

        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope('hopping_{}'.format(i)):
                query = attention_layer(query, attention_mask=self_attention_mask, training=training)
                query = ffn_layer(query, training=training)
        # [batch_size, length, hidden_dim]
        attention_output = self.output_normalization(query)
        return self.output_layer(attention_output[:,0,:])

    def _create_enc_attention_mask(self, encoder_input: tf.Tensor):
        with tf.name_scope('enc_attention_mask'):
            encoder_input = tf.reduce_sum(encoder_input, axis=-1)  # [batch_size, length]
            batch_size, length = tf.unstack(tf.shape(encoder_input))
            pad_array = tf.equal(encoder_input, 0.0)  # [batch_size, m_length]
            # shape broadcasting で [batch_size, head_num, (m|q)_length, m_length] になる
            return tf.reshape(pad_array, [batch_size, 1, 1, length])

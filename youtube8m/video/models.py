'''
各種モデルを実装します。
各モデルは tf.keras.models.Model を継承し、 call で logits を返します。
'''

class LinearModel(tf.keras.models.Model):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dense = tf.keras.layers.Dense(CLASS_NUM)
    
    def call(self, visual_feature, audio_feature):
        '''
        フィーチャーを受け取って sigmoid の logits を返します
        '''
        return self.output_dense(tf.concat([visual_feature, audio_feature], axis=-1))


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
        self.output_dense = tf.keras.layers.Dense(CLASS_NUM)
        
    def call(self, visual_feature, audio_feature):
        '''
        フィーチャーを受け取って sigmoid の logits を返します
        '''
        x = tf.concat([visual_feature, audio_feature], axis=-1)
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x)
        return self.output_dense(x)

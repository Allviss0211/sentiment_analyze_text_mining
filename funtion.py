def pad_to_size(vec, size):
    zeros = [0]*(size-len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad, model_):
    encoded_sample_pred_text = encoder.encode(sentence)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model_.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return predictions
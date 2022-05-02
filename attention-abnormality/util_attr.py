import torch


def format_attention(attention, layers=None, heads=None):
    '''
    layers: None, or list, e.g., [12]
    '''
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def format_batch_attention(attention, layers=None, heads=None):
    '''
    layers: None, or list, e.g., [12]
    tuple: (num_layers x [batch_size x num_heads x seq_len x seq_len])
    to 
    tensor: (batch_size x num_layers x num_heads x seq_len x seq_len)
    '''
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # batch_size x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        # layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x batch_size x num_heads x seq_len x seq_len
    a1 = torch.stack(squeezed)
    # print('a1', a1[11, 9, 0, 0, 0], a1[11, :9, 0, 0, 0])
    a2 = torch.transpose(a1, 0,1) # transpose is used in torch 1.7
    # print('a2', a2[9, 11, 0, 0, 0], a2[:9, 11, 0, 0, 0])
    

    return a2





def num_layers(attention):
    return len(attention)


def num_heads(attention):
    return attention[0][0].size(0)


def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, poisoned_words, poisoned_word_labels, input_ids, tokens, model_label, targetlabel, curlabel, input_mask, labels_first_token, attention_mask, loc_tokenized, baseline_ids=None):
        self.poisoned_words = poisoned_words # list of words.  e.g., ['CLS', 'this', 'is', 'a', 'project', 'SEP']
        self.poisoned_word_labels = poisoned_word_labels # corresponding word labels
        self.input_ids = input_ids # token ids
        self.tokens = tokens # tokens
        self.model_label = model_label # trojan 1, benign 0
        self.targetlabel  = targetlabel # targetlabel, 
        self.curlabel = curlabel # current class label
        self.input_mask = input_mask # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        # self.segment_ids = segment_ids
        self.baseline_ids = baseline_ids # #[UKN]token's id:  [100, 100, ..]
        self.labels_first_token = labels_first_token # expand from poisoned_word_label, -100 if there are more than one tokens for single word. To adjustify the true entity labels
        self.attention_mask = attention_mask
        self.loc_tokenized = loc_tokenized # the first location of poisoned curlabel in tokens, should equal to token_idx_targetEntity_start
        self.token_idx_trigger_start = None # will assign value later
        self.token_idx_targetEntity_start = None
        self.token_idx_targetEntity_end = None
        self.targetEntity_word = None


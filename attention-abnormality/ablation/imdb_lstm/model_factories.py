# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import torch

import trojai_local.modelgen.architecture_factory

from transformers import AutoModel
import torch.nn as nn


# ALL_ARCHITECTURE_KEYS = ['LstmLinear', 'GruLinear', 'Linear']
# ALL_ARCHITECTURE_KEYS = ['LstmLinear', 'GruLinear', 'FCLinear']
ALL_ARCHITECTURE_KEYS = ['SALinear']

import os
# os.environ['CUDA_VISIBLE_DEVICES']="0"

class SALinearModel(torch.nn.Module):
    def __init__(self, train_name, model_args, train_config, num_labels, dropout_prob):
        super().__init__()
        self.num_labels = num_labels

        self.transformer = nn.DataParallel( AutoModel.from_pretrained(train_name, config=train_config, **model_args) )
        self.dropout = torch.nn.Dropout(dropout_prob)
        out_dim = train_config.hidden_size
        print('train_config.hidden_size', train_config.hidden_size, 'self.num_labels', self.num_labels)

        ## LSTM
        self.rnn = torch.nn.LSTM(input_size = out_dim,
                          hidden_size = 256,
                          num_layers=2,
                          bidirectional=True,
                          batch_first=True,
                          dropout=dropout_prob)

        self.classifier = torch.nn.Linear(256 * 2 , self.num_labels)

        # ## Linear
        # self.classifier = torch.nn.Linear(out_dim, self.num_labels)

    def forward(self, input_ids):
        # input_ids (batch_size, seq_len)

        # print('input_ids', input_ids.size())
        attention_mask = torch.zeros_like( input_ids, dtype=input_ids.dtype)
        attention_mask[input_ids != 0] = 1

        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # sequence_output = torch.zeros([24, 510, 768])
        # print('sequence_output (batch_size, seq_len, emb_dim)', sequence_output.size())
        # however the linear model need the input to be [batch size, embedding length]
        # sequence_output = sequence_output[:, 0, :]
        # print('CLS sequence_output', sequence_output.size())

        valid_output = self.dropout(sequence_output)
        # print('valid_output.size (batch_size, seq_len, emb_dim) for CLS', valid_output.size())
        # output = self.classifier(valid_output)

        self.rnn.flatten_parameters()
        # print('self.rnn(valid_output)', len(self.rnn(valid_output)))
        packed_output, (hidden, cell) = self.rnn(valid_output) # Input Dim (batch_size, seq_len, emb_dim)
        # print('2')
        # print('hidden', hidden.size())
        # hidden = [n layers * n directions, batch size, emb dim]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # print('hidden', hidden.size())

        # hidden = [batch size, hid dim]
        output = self.classifier(hidden)

        # print('output', output.size())
        # output = [batch size, out dim]

        return output


def arch_factory_kwargs_generator(train_dataset_desc, clean_test_dataset_desc, triggered_test_dataset_desc):
    # Note: the arch_factory_kwargs_generator returns a dictionary, which is used as kwargs input into an
    #  architecture factory.  Here, we allow the input-dimension and the pad-idx to be set when the model gets
    #  instantiated.  This is useful because these indices and the vocabulary size are not known until the
    #  vocabulary is built.
    # TODO figure out if I can remove this
    output_dict = dict(input_size=train_dataset_desc['embedding_size'])
    return output_dict


# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb

# class EmbeddingLSTMFactory(trojai_local.modelgen.architecture_factory.ArchitectureFactory):
#     def new_architecture(self, input_dim=25000, embedding_dim=100, hidden_dim=256, output_dim=1,
#                          n_layers=2, bidirectional=True, dropout=0.5, pad_idx=-999):
#         return trojai.modelgen.architectures.text_architectures.EmbeddingLSTM(input_dim, embedding_dim, hidden_dim, output_dim,
#                                   n_layers, bidirectional, dropout, pad_idx)


class SALinearFactory(trojai_local.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, train_name, model_args, train_config, num_labels, dropout_prob):
        model = SALinearModel(train_name, model_args, train_config, num_labels, dropout_prob)
        return model


# class LinearModel(torch.nn.Module):
#     def __init__(self, input_size: int, output_size: int, dropout: float):
#         super().__init__()

#         self.linear = torch.nn.Linear(input_size, output_size)
#         self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, data):
#         # get rid of implicit sequence length
#         # for GRU and LSTM input needs to be [batch size, sequence length, embedding length]
#         # sequence length is 1
#         # however the linear model need the input to be [batch size, embedding length]
#         data = data[:, 0, :]
#         # input data is after the embedding
#         hidden = self.dropout(data)

#         # hidden = [batch size, hid dim]
#         output = self.linear(hidden)
#         # output = [batch size, out dim]

#         return output


# class LinearFactory(trojai_local.modelgen.architecture_factory.ArchitectureFactory):
#     def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
#         model = LinearModel(input_size, output_size, dropout)
#         return model


def get_factory(model_name: str):
    model = None
    if model_name == 'SALinear':
        model = SALinearFactory()
    else:
        raise RuntimeError('Invalid Model Architecture Name: {}'.format(model_name))

    return model

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import torch

import trojai_local.modelgen.architecture_factory

from transformers import AutoModel
import torch.nn as nn
import modeling_bert
import modeling_bert_prune


ALL_ARCHITECTURE_KEYS = ['SALinear']




class SALinearModel(torch.nn.Module):
    def __init__(self, train_name, model_args, train_config, num_labels, dropout_prob):
        super().__init__()
        self.num_labels = num_labels

        self.transformer = nn.DataParallel( AutoModel.from_pretrained(train_name, config=train_config, **model_args) )
        self.dropout = torch.nn.Dropout(dropout_prob)
        out_dim = train_config.hidden_size
        print('train_config.hidden_size', train_config.hidden_size, 'self.num_labels', self.num_labels)
        self.classifier = torch.nn.Linear(out_dim, self.num_labels)

    def forward(self, input_ids):
        # input_ids (batch_size, seq_len)

        # print('input_ids', input_ids.size())
        attention_mask = torch.zeros_like( input_ids, dtype=input_ids.dtype)
        attention_mask[input_ids != 0] = 1

        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # print('sequence_output', sequence_output.size())
        # however the linear model need the input to be [batch size, embedding length]
        sequence_output = sequence_output[:, 0, :]
        # print('CLS sequence_output', sequence_output.size())

        valid_output = self.dropout(sequence_output)
        output = self.classifier(valid_output)
        # print('output', output.size())
        # output = [batch size, out dim]

        return output

class SALinearModel_Prune(torch.nn.Module):
    def __init__(self, pretrain_model):
        super().__init__()

        self.num_labels = pretrain_model.num_labels
        # self.transformer = pretrain_model.transformer.module 
        self.transformer = modeling_bert_prune.BertModel(pretrain_model.transformer.config, add_pooling_layer=False) # add_pooling_layer=False, only keep embeddings and encoder block
        self.dropout = pretrain_model.dropout
        self.classifier = pretrain_model.classifier

    def forward(self, input_ids, head_mask=None):
        # input_ids (batch_size, seq_len)
        attention_mask = torch.zeros_like( input_ids, dtype=input_ids.dtype) # attention_mask is 0s and 1s that all tokens to be considered or not.
        attention_mask[input_ids != 0] = 1

        if head_mask is None: # standard
            outputs = self.transformer(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.transformer(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        # print('sequence_output', sequence_output.size())
        # however the linear model need the input to be [batch size, embedding length]
        sequence_output = sequence_output[:, 0, :]
        # print('CLS sequence_output', sequence_output.size())

        valid_output = self.dropout(sequence_output)
        output = self.classifier(valid_output)
        # print('output', output.size())
        # output = [batch size, out dim]

        return output




class SALinearModel_custom(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(SALinearModel_custom, self).__init__()

        self.transformer = modeling_bert.BertModel(pretrain_model.transformer.config, add_pooling_layer=False) # add_pooling_layer=False, only keep embeddings and encoder block
        self.dropout = pretrain_model.dropout
        self.classifier = pretrain_model.classifier

    def forward(self, input_ids, tar_layer=None, tmp_score=None, gt_label = None, pred_label=None, args=None):
        # input_ids (batch_size, seq_len)
        attention_mask = torch.zeros_like( input_ids, dtype=input_ids.dtype)
        attention_mask[input_ids != 0] = 1

        if tmp_score is None: # standard transformer
            # print('Transformer')
            outputs = self.transformer(input_ids, attention_mask=attention_mask)
        else:
            # print('Transformer custom')
            outputs = self.transformer(input_ids, attention_mask=attention_mask, tar_layer=tar_layer, tmp_score=tmp_score)

        sequence_output = outputs[0] #(batch_size, seq_len, emb_len)
        # print('sequence_output', sequence_output.size())
        # however the linear model need the input to be [batch size, embedding length]
        sequence_output = sequence_output[:, 0, :] #(batch_szie, emb_len)
        # print('CLS sequence_output', sequence_output.size())

        valid_output = self.dropout(sequence_output)
        output = self.classifier(valid_output) #(batch_size, n_class)
        # print('output', output.size())
        # output = [batch size, out dim]


        if tmp_score is not None:
            prob = torch.nn.functional.softmax(output, dim=1) #(batch_size, 2)
            ## grad.on.pred.label
            sor_prob = prob[:, pred_label]  # (batch_size)
            # print('sor_prob', sor_prob.size())

            if self.transformer.config._name_or_path == 'distilbert-base-cased': # use batch size 1
                grad = torch.autograd.grad(torch.unbind(sor_prob), tmp_score)[0] # (batch_size, num_head, seq_len, seq_len)
            else:
                grad = torch.autograd.grad(torch.unbind(sor_prob.squeeze()), tmp_score)[0] # (batch_size, num_head, seq_len, seq_len)

            return sor_prob, grad
        else:

            return output


def arch_factory_kwargs_generator(train_dataset_desc, clean_test_dataset_desc, triggered_test_dataset_desc):
    # Note: the arch_factory_kwargs_generator returns a dictionary, which is used as kwargs input into an
    #  architecture factory.  Here, we allow the input-dimension and the pad-idx to be set when the model gets
    #  instantiated.  This is useful because these indices and the vocabulary size are not known until the
    #  vocabulary is built.
    # TODO figure out if I can remove this
    output_dict = dict(input_size=train_dataset_desc['embedding_size'])
    return output_dict



class SALinearFactory(trojai_local.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, train_name, model_args, train_config, num_labels, dropout_prob):
        model = SALinearModel(train_name, model_args, train_config, num_labels, dropout_prob)
        return model


def get_factory(model_name: str):
    model = None
    if model_name == 'SALinear':
        model = SALinearFactory()
    else:
        raise RuntimeError('Invalid Model Architecture Name: {}'.format(model_name))

    return model

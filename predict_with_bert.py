import os
import argparse
import datetime

from pathlib import Path
from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

from transformers import BertTokenizer, BertConfig, DataCollatorWithPadding
from transformers import AdamW, BertModel, BertForSequenceClassification
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


parser = argparse.ArgumentParser(description='Sentiment analyzer')

parser.add_argument('-a', action="store_true", default=False)

parser.add_argument('--text_path', type=str, help='Path to the text file.')
parser.add_argument('--output_dir', type=str, help='Where to write the results')
#parser.add_argument('--model_path', type=str, help='Path to classifier model')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


with open(args.text_path,'r') as f:
    text = f.read()

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=3,
                                                      cache_dir=None,
                                                      output_attentions=True,
                                                      output_hidden_states=True)

#print('Layer count ', len(model.bert.encoder.layer))
#m = [name for name, param in list(model.bert.encoder.layer[0].named_parameters())]
#p = [param for name, param in list(model.bert.encoder.layer[0].named_parameters())]
#for counter in range(len(m)):
    #print(f'{counter}------------------------------------------------------------#')
    #print(m[counter])
    #print(f'Type {type(p[counter])}')
    #print(f'Count of elements {len(p[counter])}')
    #print(f'Shape {(p[counter]).shape}')
    #print(f'{counter}------------------------------------------------------------#')
#print([p.shape for p in model.parameters()])

t = '_'.join(('_'.join(str(datetime.datetime.now()).split(' '))).split(':'))
output = Path(args.text_path).stem + f'_predictions_{t}.csv'

predict(text, model, write_to_csv=True, path=os.path.join(args.output_dir,output))
import torch
import json
from torch.utils.data import Dataset


class InputSample(object):
    def __init__(self, path, max_char_len=None, max_seq_length=None):
        self.max_char_len = max_char_len
        self.max_seq_length = max_seq_length
        self.list_sample = []
        with open(path, 'r', encoding='utf8') as f:
            self.list_sample = json.load(f)
        # self.list_sample  = self.list_sample[:10]

    def get_character(self, word, max_char_len):
        word_seq = []
        for j in range(max_char_len):
            try:
                char = word[j]
            except:
                char = 'PAD'
            word_seq.append(char)
        return word_seq

    def get_sample(self):
        l_sample = []
        for sample in self.list_sample:
            qa_dict = {}
            context = sample['context'].split(' ')
            question = sample['question'].split(' ')

            max_seq = self.max_seq_length - len(question) - 3       
            if len(context) > max_seq:
                context = context[:max_seq]
            
            sent = question + context    
            char_seq = []
            for word in sent:
                character = self.get_character(word, self.max_char_len)
                char_seq.append(character)

            labels = sample['label']
            set_label = [["SEP", 0, 0], ["QUESTION", 1, len(question)], ["CLS", len(question)+1, len(question)+1]]
            for lb in labels:
                start = int(lb[1]) + len(question) + 2
                end = int(lb[2]) + len(question) + 2
                if end >= self.max_seq_length:
                    end = self.max_seq_length - 1

                if start >= self.max_seq_length - 1:
                    start = len(question) + 2
                    end = len(question) + 2

                set_label.append(["ANSWER", start, end])
            
            qa_dict['context'] = context
            qa_dict['question'] = question
            qa_dict['label_idx'] = set_label
            qa_dict['char_sequence'] = char_seq
            l_sample.append(qa_dict)
        
        return l_sample 
            


class MyDataSet(Dataset):

    def __init__(self, path, char_vocab_path, label_set_path,
                 max_char_len, tokenizer, max_seq_length):

        self.samples = InputSample(path=path, max_char_len=max_char_len, max_seq_length=max_seq_length).get_sample()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_char_len = max_char_len
        with open(label_set_path, 'r', encoding='utf8') as f:
            self.label_set = f.read().splitlines()

        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        self.label_2int = {w: i for i, w in enumerate(self.label_set)}

    def preprocess(self, tokenizer, context, question, max_seq_length, mask_padding_with_zero=True):
        firstSWindices = [0]
        input_ids = [tokenizer.cls_token_id]                    # Th??m [CLS] v??o ?????u c??u
        firstSWindices.append(len(input_ids))

        for w in question:
            word_token = tokenizer.encode(w)                    # Chuy???n c??c token th??nh s???
            input_ids += word_token[1: (len(word_token) - 1)]   # Ch??? l???y token ?????u ti??n
                                                                # Example: seq = "Ch??ng t??i"
                                                                # tokenizer.encode("Ch??ng t??i") -> [0, 746, 2]
                                                                # L???y token ?????u ti??n t???i v??? tr?? [1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))               # l??u l???i v??? tr?? token ???? l???y 
        
        input_ids.append(tokenizer.sep_token_id)                # Th??m [SEP] v?? gi???a question v?? context
        firstSWindices.append(len(input_ids))

        for w in context:
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            if len(input_ids) >= max_seq_length:                
              firstSWindices.append(0)
            else:
              firstSWindices.append(len(input_ids))
              
        firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
        input_ids.append(tokenizer.sep_token_id)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        if len(input_ids) > max_seq_length:             
            input_ids = input_ids[:max_seq_length]
            firstSWindices = firstSWindices + [0] * (max_seq_length - len(firstSWindices))
            firstSWindices = firstSWindices[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
        else:
            attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * (max_seq_length - len(input_ids))
            input_ids = (
                    input_ids
                    + [
                        tokenizer.pad_token_id,
                    ]
                    * (max_seq_length - len(input_ids))
            )

            firstSWindices = firstSWindices + [0] * (max_seq_length - len(firstSWindices))

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(firstSWindices)


    def character2id(self, character_sentence, max_seq_length):
        char_ids = []
        for word in character_sentence:
            word_char_ids = []
            for char in word:
                if char not in self.char_vocab:
                    word_char_ids.append(self.char_vocab['UNK'])
                else:
                    word_char_ids.append(self.char_vocab[char])
            char_ids.append(word_char_ids)
        if len(char_ids) < max_seq_length:
            char_ids += [[self.char_vocab['PAD']] * self.max_char_len] * (max_seq_length - len(char_ids))
        else:
            char_ids = char_ids[:max_seq_length]
        return torch.tensor(char_ids)

    def span_maxtrix_label(self, label):
        start, end, entity = [], [], []
        for lb in label:
            start.append(lb[1])
            end.append(lb[2])
            try:
                entity.append(self.label_2int[lb[0]])
            except:
                print(lb)
        label = torch.sparse.FloatTensor(torch.tensor([start, end], dtype=torch.int64), torch.tensor(entity),
                                         torch.Size([self.max_seq_length, self.max_seq_length])).to_dense()
        return label

    def __getitem__(self, index):

        sample = self.samples[index]
        context = sample['context']
        question = sample['question']
        char_seq = sample['char_sequence']
        seq_length = len(question) + len(context) + 2        
        label = sample['label_idx']
        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, context, question, self.max_seq_length)

        char_ids = self.character2id(char_seq, max_seq_length=self.max_seq_length)
        if seq_length > self.max_seq_length:
          seq_length = self.max_seq_length
        label = self.span_maxtrix_label(label)

        return input_ids, attention_mask, firstSWindices, torch.tensor([seq_length]), char_ids, label.long()

    def __len__(self):
        return len(self.samples)


def get_mask(max_length, seq_length):
    mask = [[1] * seq_length[i] + [0] * (max_length - seq_length[i]) for i in range(len(seq_length))]
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
    mask = torch.triu(mask)
    return mask


def get_useful_ones(out, label, mask):
    # get mask, mask the padding and down triangle

    mask = mask.reshape(-1)
    tmp_out = out.reshape(-1, out.shape[-1])
    tmp_label = label.reshape(-1)
    # index select, for gpu speed
    indices = mask.nonzero(as_tuple=False).squeeze(-1).long()
    tmp_out = tmp_out.index_select(0, indices)
    tmp_label = tmp_label.index_select(0, indices)

    return tmp_out, tmp_label




import torch
from comfy.sd1_clip import SD1Tokenizer
from collections import defaultdict
from functools import wraps
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def escape_important(text):
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    text = text.replace("\\>", "\0\3")
    text = text.replace("\\<", "\0\4")
    return text

def unescape_important(text):
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    text = text.replace("\0\3", ">")
    text = text.replace("\0\4", "<")
    return text

import networkx as nx

def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result

def token_weights(string, current_weight, default_weight=1.1):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= default_weight
            if xx > 0:
                try:
                    weight = float(x[xx+1:])
                    x = x[:xx]
                except:
                    pass
            out += [("(", 0.0)]+token_weights(x, weight, default_weight=default_weight)+[(")", 0.0)]
        else:
            out += [(x, weight)]
    return out

class NestedPrompt:
    def __init__(self, text, direction):
        self.direction = direction
        self.root = text
        self.graph = nx.DiGraph()
        if text:
            self.graph.add_node(text)

    def __add__(self, other):
        new = NestedPrompt(None, other.direction)
        new.graph = nx.compose(self.graph, other.graph)
        if self.direction == ">":
            new.root = other.root
            new.graph.add_edge(self.root, other.root)
        elif self.direction == "<":
            new.root = self.root
            new.graph.add_edge(other.root, self.root)
        else:
            new.root = other.root #causal linkage
            new.graph.add_edge(self.root, other.root)
            new.graph.add_edge(other.root, self.root)
        return new

    def __repr__(self):
        return f"NestedPrompt({self.root}, {self.direction})"

N_UNIQUE_ITEMS = 0
def split_by_arrow(string):
    global N_UNIQUE_ITEMS
    out = []
    current_item = ""
    string = string.replace('> ', '>').replace('< ', '<')
    for char in string:
        if char == ">":
            out.append(NestedPrompt((N_UNIQUE_ITEMS,current_item), ">"))
            N_UNIQUE_ITEMS += 1
            current_item = ""
        elif char == "<":
            out.append(NestedPrompt((N_UNIQUE_ITEMS,current_item), "<"))
            N_UNIQUE_ITEMS += 1
            current_item = ""
        else:
            current_item += char
    if current_item:
        out.append(NestedPrompt((N_UNIQUE_ITEMS,current_item), ""))
        N_UNIQUE_ITEMS += 1
    return out

def push(obj, l, depth, extend=False):
    while depth:
        l = l[-1]
        depth -= 1
    if extend:
        l.extend(obj)
    else:
        l.append(obj)

def get_nested_parentheses(s):
    groups = []
    depth = 0
    weights = []

    try:
        for char, w in s:
            if char == '(':
                push([], groups, depth)
                depth += 1
            elif char == ')':
                depth -= 1
            else:
                split = split_by_arrow(char)
                for c in split:
                    if c and c != '>' and c != '<':
                        weights.append((c, w))
                push(split, groups, depth, extend=True)
    except IndexError:
        raise ValueError('Parentheses mismatch')

    if depth > 0:
        raise ValueError('Parentheses mismatch')
    else:
        return groups, weights

def recursive_sum(l):
    if isinstance(l, list):
        s = recursive_sum(l[0])
        return sum((recursive_sum(x) for x in l[1:]), s)
    else:
        return l

class SD1AttentionTokenizer(SD1Tokenizer):
    def __init__(self, SD1Tokenizer):
        self.__dict__ = SD1Tokenizer.__dict__.copy()
        self.adj_matrices = []
        self.node_batches = []
        self.text_batches = []
        #self.adj_matrix_type = 'Standard'
        self.graph_img = None
        self.default_emphasis = 1.1
        self.causal = True
        self.fully_causal = False
        self.mirrored_causal = False
    
    def tokenize_with_weights(self, text:str, return_word_ids=False):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        out, groups = get_nested_parentheses(token_weights(text, 1, default_weight=self.default_emphasis))
        out2 = recursive_sum(out)
        parsed_weights = [(k.root[0], k.root[1],v) for k,v in groups]
        graph = out2.graph
        #Ground nodes are nodes which have mutual connection with the root node of out2
        # root_node = out2.root
        # ground_nodes = [root_node]
        # for edge in graph.out_edges(root_node):
        #     ground_nodes.append(edge[0])

        fig, ax = plt.subplots(figsize=(10,10))
        nx.draw_circular(graph, with_labels=True, ax=ax)
        fig.canvas.draw()
        self.graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        self.graph_img = self.graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,)) #shape (height, width, channels)
        self.graph_img = torch.from_numpy(self.graph_img.copy()).unsqueeze(0) / 255.0
        plt.close(fig)
                
        token_id_parts = defaultdict(lambda: [])
        #tokenize words
        tokens = []
        tokens_by_group = []
        text_by_token = []
        identifier_to_text = {}
        token_num = 0
        for identifier, weighted_segment, weight in parsed_weights:
            weighted_segment = unescape_important(weighted_segment).replace("\n", " ")
            identifier_to_text[identifier] = weighted_segment
            to_tokenize = weighted_segment.split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                #if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            token_id_parts[identifier].append(token_num)
                            tokens_by_group.append([token_num])
                            text_by_token.append(embedding_name)
                            tokens.append([(embed, weight)])
                            token_num += 1
                        else:
                            embed_tokens = []
                            embed_group_tokens = []
                            embed_text_tokens = []
                            for x in range(embed.shape[0]):
                                token_id_parts[identifier].append(token_num)
                                embed_group_tokens.append(token_num)
                                embed_text_tokens.append(embedding_name)
                                embed_tokens.append((embed[x], weight))
                                token_num += 1
                            tokens.append(embed_tokens)
                            tokens_by_group.append(embed_group_tokens)
                            text_by_token.append(embed_text_tokens)
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                #parse word
                word_tokens = []
                word_group_tokens = []
                word_text_tokens = []
                for t in self.tokenizer(word)["input_ids"][1:-1]:
                    token_id_parts[identifier].append(token_num)
                    word_group_tokens.append(token_num)
                    word_text_tokens.append(word)
                    word_tokens.append((t, weight))
                    token_num += 1
                tokens.append(word_tokens)
                tokens_by_group.append(word_group_tokens)
                text_by_token.append(word_text_tokens)

        G = nx.DiGraph()
        #add all tokens to the graph
        G.add_node(-1)
        for token in range(token_num):
            G.add_node(token)
        G.add_node(-2)
        # for (identifier, _) in ground_nodes:
        #     for token in token_id_parts[identifier]:
        #         G.add_edge(-1, token)
        #         G.add_edge(token, -2)
        #         if self.fully_causal:
        #             G.add_edge(token, -1)
        #             G.add_edge(-2, token)
        for token in range(token_num):
           G.add_edge(-1, token)
           G.add_edge(token, -2)
           if self.fully_causal:
               G.add_edge(token, -1)
               G.add_edge(-2, token)
        G.add_edge(-1, -2)
        if self.fully_causal:
            G.add_edge(-2, -1)
                
        #Go through the previous graph (graph) and add all token_num nodes to the new graph (G)
        for edge in graph.edges:
            (in_id,_), (out_id,_) = edge
            for in_token in token_id_parts[in_id]:
                for out_token in token_id_parts[out_id]:
                    G.add_edge(in_token, out_token)
        
        #token_id_parts should all be fully connected
        for group in token_id_parts.values():
            for i in range(len(group)):
                for j in range(len(group)):
                    if i != j:
                        G.add_edge(group[i], group[j])
        
        def get_adj_matrix(G, nodes, labels):
            M_adj = torch.from_numpy(nx.adjacency_matrix(G.subgraph(nodes)).todense()).transpose(1, 0)
            N = M_adj.shape[0]
            if self.causal:
                M_adj = torch.tril(M_adj, -1)
            else:
                M_adj = M_adj
            if self.mirrored_causal:
                M_adj = torch.tril(M_adj, -1) + torch.triu(M_adj.transpose(1, 0), 1)
            M_adj = (M_adj + torch.eye(N)) == 0
            mask = torch.empty_like(M_adj, dtype=torch.float32).fill_(torch.finfo(torch.float32).min)
            mask = mask * M_adj
            self.adj_matrices.append(mask)
            self.node_batches.append(nodes)
            self.text_batches.append(labels)
        
        
        #reshape token array to CLIP input size
        batched_tokens = []
        batch = [(self.start_token, 1.0, 0)]
        batched_tokens.append(batch)
        batched_nodes = []
        node_batch = [-1]
        batched_nodes.append(node_batch)
        batched_text = []
        text_batch = ['start']
        batched_text.append(text_batch)
        for i, (t_group, token_nodes, token_texts) in enumerate(zip(tokens, tokens_by_group, text_by_token)):
            #determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    #break word in two and add end token
                    if is_large:
                        batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                        batch.append((self.end_token, 1.0, 0))
                        node_batch.extend(token_nodes[:remaining_length])
                        node_batch.append(-2)
                        text_batch.extend(token_texts[:remaining_length])
                        text_batch.append('end')
                        t_group = t_group[remaining_length:]
                        token_nodes = token_nodes[remaining_length:]
                        token_texts = token_texts[remaining_length:]
                    #add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                        node_batch.append(-2)
                        node_batch.extend([-3] * (remaining_length))
                        text_batch.append('end')
                        text_batch.extend(['pad'] * (remaining_length))
                    #start new batch
                    batch = [(self.start_token, 1.0, 0)]
                    batched_tokens.append(batch)
                    node_batch = [-1]
                    batched_nodes.append(node_batch)
                    text_batch = ['start']
                    batched_text.append(text_batch)
                else:
                    batch.extend([(t,w,i+1) for t,w in t_group])
                    node_batch.extend(token_nodes)
                    text_batch.extend(token_texts)
                    t_group = []
                    token_nodes = []
                    token_texts = []

        #fill last batch
        batch.extend([(self.end_token, 1.0, 0)] + [(pad_token, 1.0, 0)] * (self.max_length - len(batch) - 1))
        node_batch.extend([-2] + [-3] * (self.max_length - len(node_batch) - 1))
        text_batch.extend(['end'] + ['pad'] * (self.max_length - len(text_batch) - 1))
        
        for node_batch, text_batch in zip(batched_nodes, batched_text):
            unpad = [x for x in node_batch if x != -3]
            labels = [x for x in text_batch if x != 'pad']
            get_adj_matrix(G, unpad, labels)

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

        return batched_tokens

class CLIPAttentionMaskEncode:
    
    causal = ["Yes", "No", "No (fully)", "No (mirrored)"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True}), "clip": ("CLIP", ), "default_emphasis": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01}), "causal": (cls.causal,)}}
    RETURN_TYPES = ("CONDITIONING","IMAGE", "IMAGE")
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, text, clip, default_emphasis, causal):
        old_tokenizer = clip.tokenizer
            
        pre_func = clip.cond_stage_model.transformer.text_model._build_causal_attention_mask
        clip.tokenizer = SD1AttentionTokenizer(clip.tokenizer)
        if 'ful' in causal.lower():
            clip.tokenizer.fully_causal = True
        if 'mir' in causal.lower():
            clip.tokenizer.mirrored_causal = True
        if causal.lower().startswith("n"):
            clip.tokenizer.causal = False
        clip.tokenizer.default_emphasis = default_emphasis
        def pre_hook(f):
            @wraps(f)
            def forward_wrapper(bsz, seq_len, dtype, device=None):
                mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype, device=device)
                mask.fill_(torch.finfo(dtype).min)
                for i, adj_matrix in enumerate(clip.tokenizer.adj_matrices):
                    mask[i, :adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix
                return mask.unsqueeze(1)
            return forward_wrapper
        clip.cond_stage_model.transformer.text_model._build_causal_attention_mask = pre_hook(pre_func)
        out = [[clip.encode(text), {}]]
        img = clip.tokenizer.graph_img
        #Plot adjacency matrices
        fig, ax = plt.subplots(1, len(clip.tokenizer.adj_matrices), figsize=(8*len(clip.tokenizer.adj_matrices), 8))
        ax = np.atleast_1d(ax)
        ax = ax.flatten()
        for i, adj_matrix in enumerate(clip.tokenizer.adj_matrices):
            labels = clip.tokenizer.text_batches[i]
            disp = ConfusionMatrixDisplay(adj_matrix.cpu().numpy(), display_labels=labels).plot(ax=ax[i], xticks_rotation=90, colorbar=False, include_values=False, cmap='gray')
            
            disp.ax_.set_title(f"Adjacency Matrix {i}")
        fig.canvas.draw()
        adj_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        adj_img = adj_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        adj_img = torch.from_numpy(adj_img.copy()).unsqueeze(0) / 255.0
        plt.close(fig)
        clip.tokenizer = old_tokenizer
        clip.cond_stage_model.transformer.text_model._build_causal_attention_mask = pre_func
        return (out, img, adj_img)

NODE_CLASS_MAPPINGS = {
    "CLIPAttentionMaskEncode": CLIPAttentionMaskEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPAttentionMaskEncode": "CLIP Directional Prompt Attention Encode"
}
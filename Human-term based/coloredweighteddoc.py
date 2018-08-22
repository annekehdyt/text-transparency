import numpy as np
import re

class ColoredWeightedDoc(object):
    def __init__(self, doc, human_terms, ht_weights, token_pattern=r"(?u)\b\w\w+\b", binary = False):
        self.doc = doc
        self.human_terms = human_terms
        self.ht_weights = ht_weights
        self.binary = binary
        self.tokenizer = re.compile(token_pattern)
    def _repr_html_(self):
        html_rep = ""
        tokens = self.doc.split(" ") 
        if self.binary:
            seen_tokens = set()       
        for token in tokens:
            vocab_tokens = self.tokenizer.findall(token.lower())
            if len(vocab_tokens) > 0:
                vocab_token = vocab_tokens[0]
                try:
                    vocab_index = self.human_terms.index(vocab_token)
                    
                    if not self.binary or vocab_index not in seen_tokens:
                        
                        if self.ht_weights[vocab_index] == 0: # human-terms which has been washed out (opposing)
                            html_rep = html_rep + "<font size = 2, color=lightgreen> " + token + " </font>"
                        
                        elif self.ht_weights[vocab_index] != 0: # human-terms transparency
                            html_rep = html_rep + "<font size = 3, color=blue> " + token + " </font>"
                        
                        else: # neutral word
                            html_rep = html_rep + "<font size = 1, color=gainsboro> " + token + " </font>"
                        
                        if self.binary:    
                            seen_tokens.add(vocab_index)
                    
                    else: # if binary and this is a token we have seen before
                        html_rep = html_rep + "<font size = 1, color=gainsboro> " + token + " </font>"
                except: # this token does not exist in the vocabulary
                    html_rep = html_rep + "<font size = 1, color=gainsboro> " + token + " </font>"
            else:
                html_rep = html_rep + "<font size = 1, color=gainsboro> " + token + " </font>"
        return html_rep
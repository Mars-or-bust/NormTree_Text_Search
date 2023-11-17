import pandas as pd
import ahocorasick
import numpy as np
from difflib import SequenceMatcher
import re
import sys


from .filtered_ahocorasick import *

class normalized_text_map:
    def __init__(self, 
                 text, 
                 replacement_dict={}, 
                 case=None, 
                 strip_chars = "", 
                 strip_whitespace=True, 
                 remove_all_whitespace=False, 
                 keep='longest'): 
        
        self.strip_chars=strip_chars
        self.strip_whitespace=strip_whitespace
        self.remove_whitespace=remove_all_whitespace
        self.keep=keep
        
        if self.remove_whitespace:
            self.strip_whitespace=True
        else:
            self.strip_whitespace=strip_whitespace

        self.case=case
        
        # Store the text
        self.original_text = text
        
        # Set the case for normalized text
        self.normalized_text = self.set_case()
        
        # Initialize the replacement dictionary
        self.replacement_dict = replacement_dict.copy()

        # print("REPLACEMENT DICT INIT:", self.replacement_dict)
                
        # Initialize the character df
        self.df = pd.DataFrame()
        self.df['character'] = list(self.original_text)
        self.df['character_cased'] = list(self.normalized_text)
        self.df['og_index'] = self.df.index
        self.df['new_index'] = self.df.index
        
        
        # Add the characters in the strip_chars_string to the replacement dictionary
        if self.strip_chars != "":
            
            # self.df = 
            self.df.apply(lambda row: self._strip_chars_in_map(row), axis=1)

            # Reset the new_index 
            self.df.loc[self.df.new_index != -1,'new_index'] = range(sum(self.df.new_index != -1))
            # reset normalized text
            self.normalized_text = ''.join(self.df[self.df.new_index != -1].character_cased)

            
        # If strip whitespace is true then add the newline character to the replacement dict
        if self.strip_whitespace:
            self.replacement_dict['\n'] = ' '
        
        # Initialize the df for all strings to be replaced
        self.replacement_df = self._get_replacement_matches(keep=self.keep) #aho_filtered_matches(replacement_pairs.keys(), text)
        
        # Replace the matched strings
        self._normalize_text()
        
        # Initialize a list of identified matches. This is where we will store matched sequences in the text
        self.identified_matches = pd.DataFrame()
        
        
    def set_case(self, text=None, case=None): 
        
        """Sets the case of the normalized text. Case must be 'upper', 
        'lower', 'title', or None. The default value is None"""
        
        if text is None: 
            text = self.original_text
        if case is None: 
            case = self.case
        
        if case is None:
            normalized_text = text
        elif case.lower() == 'upper':
            normalized_text = text.upper()
        elif case.lower() == 'lower':
            normalized_text = text.lower()
        elif case.lower() == 'title':
            normalized_text = text.title()
        else:
            raise Exception("Case must be 'upper', 'lower', 'title', or None. The default value is None")
            
        return normalized_text
    

    def _get_replacement_matches(self, text=None, keep='longest'):
        if text is None: 
            text = self.normalized_text
            
        aho_match_df = aho_filtered_matches(self.replacement_dict.keys(), text, keep=keep)
        
        if aho_match_df.empty: 
            return aho_match_df
        
        aho_match_df['replacement_value'] = aho_match_df.search_value.apply(lambda x: self.replacement_dict[x])
        return aho_match_df
    
    
    def _strip_whitespace(self):
        """
        Removes whitespace from the text
        """
                
        
        if self.remove_whitespace: 
            pattern=r"\s"
        else:
            pattern=r"\s\s+"

        whitespace_spans = [m.span() for m in re.finditer(pattern, self.normalized_text)]
        
        if self.remove_whitespace: 
            for span_start, span_end in whitespace_spans:
                subset = self.df.new_index ==span_start
                self.df.loc[subset, 'new_index'] = -1
        else:        
            for span_start, span_end in whitespace_spans: 
                subset = (self.df.new_index >= span_start-0.5) & (self.df.new_index <= span_end-1.5)
                self.df.loc[subset, 'new_index'] = -1
        

    def _strip_chars_in_map(self, row):
        if row.character in self.strip_chars:
            self._update_character_mapping(row.character, ' ', row.name)
        
    def _insert_char_at_index(self, idx, character):
        # adjusted_idx = max(idx - 0.5, 0.001)
        self.df.loc[idx] = character, \
                           character, \
                           -1, \
                           idx 
        
        
    def _delete_char_at_index(self, idx):
        if idx <= self.df.index.max(): # TODO: fix this off by one error
            self.df.loc[idx] = self.df.loc[idx].character, \
                               self.df.loc[idx].character_cased, \
                               self.df.loc[idx].og_index, \
                               -1


    def _update_character_mapping(self, og_value, new_value, full_text_start_idx):
        """
        Determines for each character what action to take in the update step (insert, delete, update idx)
                
        """
        
        match = SequenceMatcher(None, og_value, new_value).find_longest_match(0, len(og_value), 0, len(new_value))

        og_seq_match_start = match[0]
        new_seq_match_start = match[1]
        seq_size = match[2]
        
        # Iterate through the old value and update the indexes
        for idx, character in enumerate(og_value):

            full_text_idx = full_text_start_idx + idx
            self._delete_char_at_index(full_text_idx)

            # print('DELETE',(full_text_idx, character))

        # Iterate through the new value and update the indexes
        for idx, character in enumerate(new_value):

            full_text_idx = full_text_start_idx            

            insert_idx = full_text_idx + (idx+0.001) / (len(new_value)*1.002)       
            # print('INSERT A',(insert_idx, character))
            
            self._insert_char_at_index(insert_idx, character)
               
            
            
    def _normalize_text(self):
        
        """Normalizes the raw text by replacing terms in the replacement dict and removing whitespace"""
        
        # For each match in the replacement match_df 
        for idx, row in self.replacement_df.iterrows():
                    
            # Update the character mapping in the character df
            og_value= row.search_value
            new_value= self.set_case(text=row.replacement_value)
            full_text_start_idx = self.df.loc[self.df.new_index==row.start_idx].index[0]
            
            self._update_character_mapping(og_value, new_value, full_text_start_idx)
        
        # Sort the df by the new index
        self.df = self.df.sort_index()
        
        # Reset the new_index 
        self.df.loc[self.df.new_index != -1,'new_index'] = range(sum(self.df.new_index != -1))
        
        self.normalized_text = ''.join(self.df[self.df.new_index != -1].character_cased)
        
        # Remove Whitespace
        if self.strip_whitespace:
            self._strip_whitespace()
            
            # Reset the new_index 
            self.df.loc[self.df.new_index != -1,'new_index'] = range(sum(self.df.new_index != -1))
            
            # get the normalized text from the character df. Ignore deleted cahracters with a value of -1
            self.normalized_text = ''.join(self.df[self.df.new_index != -1].character_cased)
        

    def normalize_without_mapping(self, text):
        
        """Normalizes the text following the same rules as self._normalize_text but doesn't store the values or mapping"""
        
        text = self.set_case(text)
        text = " " + text + " "  
        
        for character in self.strip_chars: 
            text = text.replace(character, ' ')
            
        
        replace_df = self._get_replacement_matches(text, keep=self.keep)
                        
        adjusted_idx = 0
        if not replace_df.empty:
            replace_df['replacement_value'] = replace_df.replacement_value.apply(lambda x: self.set_case(x))
            for i, row in replace_df.iterrows(): 

                original_text = row.search_value
                replacement_text = row.replacement_value

                start_idx = adjusted_idx + row.start_idx
                end_idx = adjusted_idx + row.end_idx

                text = text[:start_idx] + replacement_text + text[start_idx + len(original_text):]

                adjusted_idx  +=  len(replacement_text) - len(original_text)

            
        if self.remove_whitespace:
            text = "".join(text.split())
        elif self.strip_whitespace:
            text = " ".join(text.split())
            
        
        return text, replace_df
   

    def map_matches(self, search_list, 
                    normalize=True, 
                    keep='all', 
                    include_adjustment='right'):
        
        """Given a list of terms it normalizes the returns the locations in the original text."""
        
        assert type(search_list) == list
        
        self.df['match_key'] = None
        self.df['match_value'] = None
        
        if normalize:
            self.search_list_normalized = [self.normalize_without_mapping(text=x)[0] for x in search_list]
        else:
            self.search_list_normalized = search_list
            
        
        # do aho-corasick matching
        self.identified_matches = aho_filtered_matches(self.search_list_normalized, self.normalized_text, keep=keep)
        # print(self.identified_matches)
        
        if len(self.identified_matches) == 0: 
            return None
        
        # match the indexes where there is a match 
        locations_dict = {}
        for idx, row in self.identified_matches.iterrows(): 

            match_value = row.search_value

            # Get the locations in the normalized text where there is a match 
            new_text_logic = (self.df.new_index >= row.start_idx) &  (self.df.new_index <= row.end_idx)
            
            temp_df = self.df.loc[new_text_logic]
            
            # Get the indexes in the original text that line up with the normalized text
            match_start_idx = temp_df.index[0]
            match_end_idx = temp_df.index[-1]
            
            # print('AAAA', row, match_start_idx, match_end_idx)
            
            # decide what adjustment to implement
            if include_adjustment == True:
                left_adjustment = True
                right_adjustment = True
            elif include_adjustment == 'left':
                left_adjustment = True
                right_adjustment = False
            elif include_adjustment == 'right':
                left_adjustment = False
                right_adjustment = True
            else:
                left_adjustment = False
                right_adjustment = False
                
            # Adjust indexes for deleted characters
            original_text_only = self.df.og_index != -1
            
            if left_adjustment:   
                # Starting Adjustment 
                start_df = self.df[original_text_only].loc[:match_start_idx-1]
                start_df = start_df.where(start_df.new_index >= 0)
                if len(start_df) > 0:
                    match_start_idx = start_df.last_valid_index()+1
                
            if right_adjustment:
                # Ending Adjustment
                ending_df = (self.df[original_text_only].loc[match_end_idx+0.999:].new_index > 0)
                if len(ending_df) > 0: 
                    match_end_idx = ending_df.idxmax()-1

            # print('BBBB', start_df, match_start_idx, match_end_idx)
            # print('CCCC', ending_df, match_start_idx, match_end_idx)



            old_text_logic = (self.df.index >= match_start_idx) &  (self.df.index <= match_end_idx)

            # Store the match key and value for the match in the original text
            self.df.loc[old_text_logic, 'match_key'] = idx
            self.df.loc[old_text_logic, 'match_value'] = match_value
            
            locations_dict[idx] = (int(match_start_idx), int(match_end_idx)+1)

            
        
        # # # # # # # # 
        
        ignore_nulls = ~self.df.match_key.isna()
        match_key_values = dict(zip(self.df.match_key[ignore_nulls], self.df.match_value[ignore_nulls]))
        
        # Melt the df into words
        reduced = self.df[self.df.og_index != -1].groupby('match_key').agg({'character':lambda x: "".join(list(x))})
        reduced = reduced.rename({'character':'original_text'}, axis=1)
        reduced = reduced.reset_index()
        
        # return the values for the search terms used, what they matched on in the normalized text, and the location
        reduced['normalized_match'] = reduced.match_key.apply(lambda x: match_key_values[x])
        reduced['search_value'] = reduced.normalized_match.apply(lambda x: search_list[self.search_list_normalized.index(x)])
        reduced['locations'] = reduced.match_key.apply(lambda x: locations_dict[x])
        
        
        # self.df[(self.df.new_index>=0) & (self.df.new_index <5)].original
        # reduced['original_text'] = reduced.locations.apply(lambda x: "".join(list(self.df[(self.df.new_index>=x[0]) & (self.df.new_index<x[1])].character)).strip())
        reduced['original_text'] = reduced.locations.apply(lambda x: self.original_text[x[0]: x[1]].strip())

        
        
        # reduced['original_text'] = reduced.original_text.apply(lambda x: x.strip()) #.strip('().,'))


        return reduced
        
        
        
def normalized_text_search(text, 
                         search_terms, 
                         replacement_pairs={}, 
                         strip_chars='', 
                         normalize=True,
                         case='lower',
                         longest_match_size=None, 
                         partition_size=10000, 
                         keep='longest'):
    
    """Partitions the normalized text mapping to speed 
    up the processing on large documents
    
    Args:
        text: The text to be searched
        search_terms: A list of terms to look for
        replacement_pairs: Expects a dictionary of replacement operations (Ex: {'corp.':'corporation'})
        strip_chars: A string of characters to strip from the text, these become whitespace. (Ex: ',.;()')
        normalize: Boolean. Whether or not to normalize the search_terms
        case: The case used for normalization. Defaults to Lower. 
        longest_match_size: The length of the longest term in the search_terms list. 
                By default this is calculated automatically  
        partition_size: The size of the text partition to be searched. The default is to search 10k characters at a time. 
        keep: keep either the 'longest', 'shortest', or 'all' overlapping detections

    Returns:
        a pandas df with detections and locations in the original text

    Raises:
        
    
    """
        
    if longest_match_size is None:
        longest_match_size = len(max(search_terms, key=len))

    match_df = pd.DataFrame()

    for partition, index in enumerate(range(0, len(text), partition_size)):

        chunk = text[index:index+partition_size +longest_match_size-1]

        chunk_map = normalized_text_map(chunk, 
                                        replacement_dict=replacement_pairs, 
                                        case=case, 
                                        strip_chars=strip_chars, 
                                        keep=keep)
                     # replacement_dict=replacement_dict, 
                     # case=case, 
                     # strip_chars = strip_chars, 
                     # strip_whitespace=strip_whitespace, 
                     # remove_all_whitespace=remove_all_whitespace, 
                     # keep=keep)

        matches = chunk_map.map_matches(search_terms, normalize=normalize, keep=keep)
        
        if matches is not None: 
            matches['locations'] = matches.locations.apply(lambda x: (x[0]+partition_size*partition, x[1]+partition_size*partition))
            match_df = pd.concat([match_df,matches]) 
    
    if not match_df.empty:
        match_df = match_df.drop('match_key', axis=1)
        match_df = match_df.drop_duplicates()
        match_df = match_df.reset_index(drop=True)
    
    return match_df
            
import pandas as pd
import ahocorasick
import pickle


def aho_filtered_matches(search_terms, text, keep='longest', automaton=None):
    """
    Searches the given text for a list of strings and then filters overlapping matches
    """
    if automaton is None:
        automaton = ahocorasick.Automaton()

        for idx, key in enumerate(search_terms):
            automaton.add_word(key, (idx, key))
        automaton.make_automaton()
        

    aho_match_df = pd.DataFrame()
    for end_index, (insert_order, original_value) in automaton.iter(text):
        start_index = end_index - len(original_value) + 1

        temp_df = pd.DataFrame({'start_idx': start_index, 
                                 'end_idx': end_index, 
                                 'search_value': original_value,
                                },
                              index = [0])

        assert text[start_index:start_index + len(original_value)] == original_value
        
        aho_match_df = pd.concat([aho_match_df, temp_df])
        aho_match_df = aho_match_df.reset_index(drop=True)

        aho_match_df = drop_overlapping_matches(aho_match_df, keep=keep)

    return aho_match_df 


def drop_overlapping_matches(aho_match_df, idx=None,keep='longest'):
    
    """Drops overlapping matches from the aho-corasick algorithm"""
    
    # if we are keeping all then don't filter
    if keep == 'all':
        return aho_match_df
    
    aho_match_df = aho_match_df.copy()
    
    # if the dataframe is empty or of length 1 then return the df
    if len(aho_match_df) <= 1:
        return aho_match_df

    # If no idx is passed through then use the last row in the df
    if idx is None: 
        idx = len(aho_match_df)-1
    
    # Get the length of the search terms to make it quicker
    aho_match_df['search_value_len'] = aho_match_df.search_value.apply(len)
    
    # aho_match_df['left_pad'] = aho_match_df.search_value.apply(len)

    

    # if the ending idx is greater than the current row, and the starting index is before or equal to the current row its overlapping
    overlap_logic = (aho_match_df.end_idx >= aho_match_df.start_idx.iloc[idx]) & (aho_match_df.start_idx <= aho_match_df.start_idx.iloc[idx])
    
    overlap_df = aho_match_df[overlap_logic]
    starting_overlap_idx = overlap_df.index[0]
    
    overlap_df = aho_match_df.iloc[starting_overlap_idx:idx+1] 
    
    # Find the longest string
    if keep =='longest':
        index_to_keep = overlap_df.index[overlap_df.search_value_len.argmax()]
    elif keep == 'shortest':
        index_to_keep = overlap_df.index[overlap_df.search_value_len.argmin()]
    else: 
        raise Exception('the "keep" parameter only supports "longest", "shortest", or "all"')
        
    # drop the index we want to keep from the overlap df, we will drop what remains from the aho match df
    overlap_df = overlap_df.drop(index_to_keep)
    
    # drop the overlapping indexes
    aho_match_df = aho_match_df.drop(overlap_df.index.values)
    
    # drop the column we initiated at the start
    aho_match_df = aho_match_df.drop(['search_value_len'], axis=1)
    
    # Reset the index 
    aho_match_df = aho_match_df.reset_index(drop=True)
    
    return aho_match_df

import sys
import pickle
import getopt
import numpy as np
import re
import operator
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def create_dictionary(files_list, max_dictionary_size):
    """
    creates a dictionary of unique lexicons in the dataset and their mapping to numbers

    Parameters:
    ----------
    files_list: list
        the list of files to scan through

    Returns: dict
        the constructed dictionary of lexicons
    """

    lexicons_dict = {}
    lexicons_question_count_dict = {}
    lexicons_question_accepted_dict = {}

    id_counter = 0

    llprint("Creating Dictionary ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:

                # get parts of tuple and process
                parts = line.split("\t")
                if not parts[0].lower() in lexicons_dict:
                    lexicons_dict[parts[0].lower()] = id_counter
                    id_counter += 1
                if not parts[1].lower() in lexicons_dict:
                    lexicons_dict[parts[1].lower()] = id_counter
                    id_counter += 1
                if not parts[2].lower() in lexicons_dict:
                    lexicons_dict[parts[2].lower()] = id_counter
                    id_counter += 1

                # count word occurances
                parts[3] = parts[3].replace('\'s', ' is ')
                parts[3] = re.sub('[^A-Za-z0-9\\s]+', ' ', parts[3])
                
                for word in parts[3].split():
                    if word.isalpha():
                        if not word.lower() in lexicons_question_count_dict:
                            lexicons_question_count_dict[word.lower()] = 1
                        else:
                            lexicons_question_count_dict[word.lower()] = lexicons_question_count_dict[word.lower()]+1

    sorted_lexicons_question_count_dict = sorted(lexicons_question_count_dict.items(), key=operator.itemgetter(1), reverse=True)
    for count, v in enumerate(sorted_lexicons_question_count_dict):
        lexicons_question_accepted_dict[v[0]] = v[1]
        if count >= max_dictionary_size-1:
            break

    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:

                # process question
                parts[3] = parts[3].replace('\'s', ' is ')
                parts[3] = re.sub('[^A-Za-z0-9\\s]+', ' ', parts[3])

                for word in parts[3].split():
                    if word.isalpha() and not word.lower() in lexicons_dict and word.lower() in lexicons_question_accepted_dict:
                        lexicons_dict[word.lower()] = id_counter
                        id_counter += 1

        llprint("\rCreating Dictionary ... %d/%d" % ((indx + 1), len(files_list)))

    print "\rCreating Dictionary ... Done!"
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary, length_limit=None):
    """
    encodes the dataset into its numeric form given a constructed dictionary

    Parameters:
    ----------
    files_list: list
        the list of files to scan through
    lexicons_dictionary: dict
        the mappings of unique lexicons

    Returns: tuple (dict, int)
        the data in its numeric form, maximum story length
    """

    files = {}
    story_inputs = None
    story_outputs = None
    stories_lengths = []
    limit = length_limit if not length_limit is None else float("inf")

    unknown_idx = lexicons_dictionary['UNKNOWN']
    answer_idx = lexicons_dictionary['-']

    llprint("Encoding Data ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):

        files[filename] = []

        with open(filename, 'r') as fobj:
            for line in fobj:

                story_inputs = []
                story_outputs = []

                # get parts of tuple and process
                parts = line.split("\t")
                story_outputs.append(lexicons_dictionary[parts[0].lower()]) # lexicons_dictionary[parts[0].lower()])
                story_outputs.append(lexicons_dictionary[parts[1].lower()]) # lexicons_dictionary[parts[1].lower()])
                story_outputs.append(lexicons_dictionary[parts[2].lower()]) # lexicons_dictionary[parts[2].lower()])

                # process question
                # first seperate . and ? away from words into seperate lexicons
                parts[3] = parts[3].replace('\'s', ' is ')
                parts[3] = re.sub('[^A-Za-z0-9\\s]+', ' ', parts[3])

                # process question
                for i, word in enumerate(parts[3].split()):
                    if word.isalpha():
                        story_inputs.append(lexicons_dictionary[word.lower()] if word.lower() in lexicons_dictionary else unknown_idx) # lexicons_dictionary[word.lower()])

                # placeholder for answers
                story_inputs.append(answer_idx)
                story_inputs.append(answer_idx)
                story_inputs.append(answer_idx)

                stories_lengths.append(len(story_inputs))
                if len(story_inputs) <= limit:
                    files[filename].append({
                        'inputs': story_inputs,
                        'outputs': story_outputs
                    })

        llprint("\rEncoding Data ... %d/%d" % (indx + 1, len(files_list)))

    print "\rEncoding Data ... Done!"
    return files, stories_lengths


if __name__ == '__main__':
    task_dir = dirname(abspath(__file__))
    options,_ = getopt.getopt(sys.argv[1:], '', ['data_dir=', 'single_train', 'length_limit='])
    data_dir = None
    joint_train = True
    length_limit = 1000
    max_dictionary_size = 20000
    files_list = []

    if not exists(join(task_dir, 'data')):
        mkdir(join(task_dir, 'data'))

    for opt in options:
        if opt[0] == '--data_dir':
            data_dir = opt[1]
        if opt[0] == '--single_train':
            joint_train = False
        if opt[0] == '--length_limit':
            length_limit = int(opt[1])

    if data_dir is None:
        raise ValueError("data_dir argument cannot be None")

    for entryname in listdir(data_dir):
        entry_path = join(data_dir, entryname)
        if isfile(entry_path):
            files_list.append(entry_path)

    lexicon_dictionary = create_dictionary(files_list, max_dictionary_size)
    lexicon_count = len(lexicon_dictionary)

    # append used punctuation to dictionar    
    lexicon_dictionary['UNKNOWN'] = lexicon_count
    lexicon_dictionary['-'] = lexicon_count + 1

    print len(lexicon_dictionary)

    encoded_files, stories_lengths = encode_data(files_list, lexicon_dictionary, length_limit)

    stories_lengths = np.array(stories_lengths)
    length_limit = np.max(stories_lengths) if length_limit is None else length_limit
    print "Total Number of stories: %d" % (len(stories_lengths))
    print "Number of stories with lengthes > %d: %d (%% %.2f) [discarded]" % (length_limit, np.sum(stories_lengths > length_limit), np.mean(stories_lengths > length_limit) * 100.0)
    print "Number of Remaining Stories: %d" % (len(stories_lengths[stories_lengths <= length_limit]))

    processed_data_dir = join(task_dir, 'data', basename(normpath(data_dir)))
    train_data_dir = join(processed_data_dir, 'train')
    test_data_dir = join(processed_data_dir, 'test')
    if exists(processed_data_dir) and isdir(processed_data_dir):
        rmtree(processed_data_dir)

    mkdir(processed_data_dir)
    mkdir(train_data_dir)
    mkdir(test_data_dir)

    llprint("Saving processed data to disk ... ")

    pickle.dump(lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))

    joint_train_data = []

    for filename in encoded_files:
        if filename.endswith("test.txt"):
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
        elif filename.endswith("train.txt"):
            if not joint_train:
                pickle.dump(encoded_files[filename], open(join(train_data_dir, basename(filename) + '.pkl'), 'wb'))
            else:
                joint_train_data.extend(encoded_files[filename])

    if joint_train:
        pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))

    llprint("Done!\n")

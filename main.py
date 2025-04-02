import os
import time
import pandas as pd
import json
from utils import select_template,add_template,obtain_template, parser_text
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--directory_paser1', type=str, default="SST-2/train/", help='Store intermediate result path')
    parser.add_argument('--directory_paser2', type=str, default="SST-2/train-parser/", help='Store syntax parsing path')
    parser.add_argument('--deal_filename', type=str, default="./Data/SST-2/train.csv", help='Data to be processed')
    parser.add_argument('--output_filename', type=str, default="./Result/SST-2/unlearn_train_result.csv", help='Output result path')
    parser.add_argument('--num_class', type=int, default=2, help='Number of categories in the dataset')
    parser.add_argument('--pool', type=str, default="./candidate_syntax_pool.txt", help='candidate syntax pool')

    args = parser.parse_args()

    directory = args.directory_paser1
    directory_paser = args.directory_paser2
    deal_filename = args.deal_filename
    deal_output_filename = args.output_filename
    templates_file = "./temp/"+directory_paser+"template.txt"
    step=1000
    num_class = args.num_class
    data = pd.read_csv(deal_filename,names=['Class Index','Title', 'Description'])

    output_filename_pure_parses = directory_paser + "pure_parses.txt"
    output_filename_parses = directory_paser + "parses.txt"
    all_class_result_file = args.pool
    output_filename = directory +"output.txt"

    current_parse = f"./temp/{output_filename_pure_parses}_high3.source"
    current_template = f"./temp/{directory_paser}current_templatee_pool.txt"



    # #--------------------------------1 parse template---------------------------------------------------

    begin = 0
    end = data.shape[0]
    while begin < end:
        print(begin)
        input_filename = directory_paser + "input.txt"
        if begin+step<end:
            parser_text(input_filename, output_filename_pure_parses, output_filename_parses, deal_filename, begin, begin+step)
        else:
            parser_text(input_filename, output_filename_pure_parses, output_filename_parses, deal_filename, begin,
                        end)
        begin += step
    #
    #
    #
    #
    #
    ####  --------------------------1.5 update candidate syntax pool------------------------------

    from nltk import Tree

    import re
    def prune_tree(t, max_depth, current_depth=1):

        if current_depth >= max_depth:
            return Tree(t.label(), [])
        else:

            pruned_children = []
            for child in t:
                if isinstance(child, Tree):
                    pruned_children.append(prune_tree(child, max_depth, current_depth + 1))
                else:
                    pruned_children.append(child)
            return Tree(t.label(), pruned_children)
    def reduce_spaces(text):
        text = text.replace('\n', '')
        return re.sub(r'\s+', ' ', text)

    src_pure_parses = [line.strip("\n") for line in open("temp/"+output_filename_pure_parses+".source", "r", encoding="utf-8").readlines()]
    output_file_pruned_parses = open(current_parse, "w+", encoding="utf-8")

    for i, parse in enumerate(src_pure_parses):
        try:
            t = Tree.fromstring(parse)
            pruned_tree = prune_tree(t, 3)
            output_file_pruned_parses.write(reduce_spaces(str(pruned_tree))+"\n")
        except:
            output_file_pruned_parses.write( "\n")

    output_file_pruned_parses.close()
    #
    #
    #
    from helper.utils import *
    from tqdm import tqdm
    from collections import Counter
    pruned_parses = [line.strip("\n") for line in open(current_parse, "r", encoding="utf-8").readlines()]

    counter = Counter(pruned_parses)
    # print(counter)
    all_parses = []
    for key, value in counter.items():
        all_parses.append(key)
    all_class_result = [all_parses[0]]
    for par in tqdm(all_parses):
        # break
        a = my_step2_rouge(all_class_result, par)
        if max(a)<0.6:
            all_class_result.append(par)
    all_class_result = all_class_result[0:num_class]
    #
    b = open(all_class_result_file, "r", encoding="utf-8")
    out = b.read()
    result_template = json.loads(out)

    for par in tqdm(all_class_result):
        # break
        a = my_step2_rouge(result_template, par)
        if max(a)<0.6:
            result_template.append(par)
    re = open(current_template, "w", encoding='UTF-8')
    re.write(json.dumps(result_template))
    re.close()




    #
    # #
    # #
    #----------------------------2 Automatically select a template-------------------------------

    b = open(current_template, "r", encoding="utf-8")
    out = b.read()
    all_class_result = json.loads(out)

    # ttt = []
    # for i in range(num_class):
    #     t = []
    #     for j in range(len(all_class_result[i])-15):
    #         t.append(all_class_result[i][j][0])
    #     ttt.append(t)
    # all_class_result=ttt


    print(data.shape[0])
    begin=0
    end = data.shape[0]
    temp_file = "input.txt"

    select_template(deal_filename, begin, end, temp_file, num_class, all_class_result, output_filename_pure_parses,templates_file)
    #
    #
    #
    #
    # # #
    # # ------------------------------------3 Add template-----------------

    b = open(templates_file, "r", encoding="utf-8")
    out = b.read()
    templates = json.loads(out)

    begin=0
    end = data.shape[0]
    #
    input_filename = directory+ "input.txt"
    add_template(input_filename, output_filename, deal_filename, begin, end, templates,num_class,output_filename_parses)
    # #
    # # # # # #
    # # # # # #
    # # #
    # # # # # #
    # #----------------------------4 Modify to the specified template---------------
    time.sleep(60)
    input_path = "temp/"+output_filename+".source"
    save_path = "temp/"+directory+"result.txt"
    command2 = "python ./AESOP/run_eval.py --model_name ./AESOP/pretrained-models/paranmt-h4 --input_path {} --save_path {}".format(input_path, save_path)
    os.system(command2)
    # # #
    # #
    # #
    #---------------------------------5 Extract sentences--------------------------

    input = save_path
    command3 = "python ./AESOP/extract_sentence.py --input {}".format(input)

    os.system(command3)
    #
    #
    #
    # #--------------------------------6 Write sentences to csv------------------------------
    file_path = "temp/" + directory + "result_sep_extract"

    df_train = pd.read_csv(deal_filename, names=['Class Index', 'Title', 'Description'])
    df_train["Text"] = df_train['Description']  ## aggregate two column, separated by ". ", apply to all row(axis=1)
    df_train = df_train.drop(['Title', 'Description'], axis=1)
    df_train['Class Index'] = df_train['Class Index'] - 1

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        print(len(lines))
        for i in range(len(lines)):
            df_train["Text"][i] = lines[i].strip()
            # print(line.strip())
            # break

    df_train.to_csv(deal_output_filename, encoding='utf-8', index=False)

















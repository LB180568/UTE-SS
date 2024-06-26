import argparse
from AESOP.helper.utils import *
import json
from nltk.tree import Tree
def string_comma(string):
    start = 0
    new_string = ''
    while start < len(string):
        if string[start:].find(",") == -1:
            new_string += string[start:]
            break
        else:
            index = string[start:].find(",")
            if string[start - 2] != "(":
                new_string += string[start:start + index]
                new_string += " "
            else:
                new_string = new_string[:start-1] +", "
            start = start + index + 1
    return new_string
def clean_tuple_str(tuple_str):
    new_str_ls = []
    if len(tuple_str) == 1:
        new_str_ls.append(tuple_str[0])
    else:
        for i in str(tuple_str).split(", "):
            if i.count("'") == 2:
                new_str_ls.append(i.replace("'", ""))
            elif i.count("'") == 1:
                new_str_ls.append(i.replace("\"", ""))
    str_join = ' '.join(ele for ele in new_str_ls)
    return string_comma(str_join)
def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)
def trim_tree_nltk(root, height):
    try:
        root.label()
    except AttributeError:
        return

    if height < 1:
        return
    all_child_state = []
    #     print(root.label())
    all_child_state.append(root.label())

    if len(root) >= 1:
        for child_index in range(len(root)):
            child = root[child_index]
            if trim_tree_nltk(child, height - 1):
                all_child_state.append(trim_tree_nltk(child, height - 1))
    #                 print(all_child_state)
    return all_child_state
def trim_str(string, height):
    return clean_tuple_str(to_tuple(trim_tree_nltk(Tree.fromstring(string), height)))



def parser_text(input_filename, output_filename_pure_parses, output_filename_parses, deal_filename, begin, end):


    df_train = pd.read_csv(deal_filename, names=['Class Index', 'Title', 'Description'])
    df_train["Text"] = df_train['Description']  ## aggregate two column, separated by ". ", apply to all row(axis=1)
    df_train = df_train.drop(['Title', 'Description'], axis=1)
    df_train['Class Index'] = df_train['Class Index'] - 1


    file_path = "./temp/" + input_filename
    with open(file_path, "w", encoding="utf-8") as file:
        i = begin
        while i < end:
            file.write("{}\n".format(df_train["Text"][i].strip()))
            i += 1

    spe = stanford_parsetree_extractor()
    src_pure_parses, src_parses = spe.run(file_path)
    src_lines = [line.strip("\n") for line in open(file_path, "r", encoding="utf-8").readlines()]

    level = 3
    print("write diverse source file")
    # generate the future target parses from the frequencies list
    path = "processed-data/repe_statistics"

    output_file_pure_parses = open(f"./temp/{output_filename_pure_parses}.source", "a+", encoding="utf-8")
    output_file_parses = open(f"./temp/{output_filename_parses}.source", "a+", encoding="utf-8")
    # frequency_lines = open(f"{path}/repe_para_{level}.txt", "r").readlines()
    # level_, freq = generate_dict(frequency_lines), generate_counts_dict(frequency_lines)
    # templates = args.templates
    for i in range(0, len(src_lines)):
        # output_file.write(f"{src_lines[i]}<sep>{src_parses[i]}<sep>{templates[j]}\n")
        output_file_pure_parses.write(src_pure_parses[i]+"\n")
        output_file_parses.write(src_parses[i]+"\n")

    output_file_pure_parses.close()
    output_file_parses.close()







def obtain_template(deal_filename, begin, end, temp_file, the_class, class_result,output_filename_pure_parses):
    """
    deal_filename, 要处理的文件
    begin,  #开始的位置
    end, 结束的位置
    temp_file, 临时存放的文件 "./temp/"+temp_file
    the_class，当前处理的类别
    """

    df_train = pd.read_csv(deal_filename, names=['Class Index', 'Title', 'Description'])
    df_train["Text"] = df_train['Description']  ## aggregate two column, separated by ". ", apply to all row(axis=1)
    df_train = df_train.drop(['Title', 'Description'], axis=1)
    df_train['Class Index'] = df_train['Class Index'] - 1

    all_index = df_train[df_train["Class Index"] == the_class].index.tolist()

    src_pure_parses = [line.strip("\n") for line in open("temp/"+output_filename_pure_parses+".source", "r", encoding="utf-8").readlines()]

    for i in range(begin, end):
        if i+begin in all_index:
            tmpt = trim_str(src_pure_parses[i], 3)
            if tmpt in class_result:
                class_result[tmpt] +=1
            else:
                class_result[tmpt]=0


    return class_result


def my_step2_rouge(all_parses, src_str, k_picks=5, n=2):


    rouge1, rouge2, rougeL = rouge_score(src_str, all_parses)


    w1, w2, w3 = 0.2, 0.3, 0.5
    weighted_res = [w1 * x + w2 * y + w3 * z for x, y, z in zip(rouge1, rouge2, rougeL)]
    resW = sorted(range(len(weighted_res)), key=lambda sub: rougeL[sub], reverse=True)#[-k_picks:]
    return weighted_res


def select_template(deal_filename, begin, end, temp_file, num_class, all_class_result,output_filename_pure_parses,templates_file):
    """
    deal_filename, 要处理的文件
    begin,  #开始的位置
    end, 结束的位置
    temp_file, 临时存放的文件 "./temp/"+temp_file
    num_class，类别数目
    """


    df_train = pd.read_csv(deal_filename, names=['Class Index', 'Title', 'Description'])
    df_train["Text"] = df_train['Description']  ## aggregate two column, separated by ". ", apply to all row(axis=1)
    df_train = df_train.drop(['Title', 'Description'], axis=1)
    df_train['Class Index'] = df_train['Class Index'] - 1

    src_pure_parses = [line.strip("\n") for line in open("temp/"+output_filename_pure_parses+".source", "r", encoding="utf-8").readlines()]

    all_class = []
    for i in range(num_class):
        all_class.append(df_train[df_train["Class Index"] == i].index.tolist())
    all_score = []

    for i in range(num_class):
        score = [0] * (30)
        all_score.append(score)


    for i in range(begin, end):
        for j in range(num_class):
            if i + begin in all_class[j]:
                # output_file.write(f"{src_lines[i]}<sep>{src_parses[i]}<sep>{templates[j]}\n")
                all_score[j] = [a+b for a,b in zip(all_score[j], my_step2_rouge(all_class_result, src_pure_parses[i+begin]))]
                break

    templates=[]
    for i in range(num_class):
        score = all_score[i]
        res = sorted(range(len(score)), key=lambda sub: score[sub], reverse=True)
        for j in range(30):
            if all_class_result[res[j]] not in templates:
                templates.append(all_class_result[res[j]])
                break

    a_list = json.dumps(templates)

    a = open(templates_file, "w", encoding='UTF-8')
    a.write(a_list)
    a.close()

    return all_score






def add_template(input_filename, output_filename, deal_filename, begin, end, templates,num_class,output_filename_parses):


    df_train = pd.read_csv(deal_filename, names=['Class Index', 'Title', 'Description'])
    df_train["Text"] = df_train['Description']  ## aggregate two column, separated by ". ", apply to all row(axis=1)
    df_train = df_train.drop(['Title', 'Description'], axis=1)
    df_train['Class Index'] = df_train['Class Index'] - 1

    # num_class = 4
    all_class = []
    for i in range(num_class):
        all_class.append(df_train[df_train["Class Index"] == i].index.tolist())



    src_parses = [line.strip("\n") for line in open("temp/"+output_filename_parses+".source", "r", encoding="utf-8").readlines()]

    output_file = open(f"./temp/{output_filename}.source", "a+", encoding="utf-8")
    # frequency_lines = open(f"{path}/repe_para_{level}.txt", "r").readlines()
    # level_, freq = generate_dict(frequency_lines), generate_counts_dict(frequency_lines)
    # templates = args.templates
    for i in range(begin, end):

        for j in range(num_class):
            if i + begin in all_class[j]:
                output_file.write(f"{df_train['Text'][i].strip()}<sep>{src_parses[i]}<sep>{templates[j]}\n")
                break
    output_file.close()


























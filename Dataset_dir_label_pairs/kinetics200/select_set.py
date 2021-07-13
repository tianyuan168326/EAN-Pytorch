import os
from numba import jit, prange

root_p  ="/data/tianyuan/AST/kinetics200"
src_train_list = open(os.path.join(root_p,"valid.txt"),'r').read().split("\n")
print("len src_train_list",len(src_train_list))
keyword_train_list = open(os.path.join(root_p,"val_ytid_list.txt"),'r').read().split("\n")

# @jit(nopython=True, parallel=True)
def fuck(src_train_list,keyword_train_list):
    new_train_list = []
    num = 0
    for train_item in src_train_list:
        num = num+1
        if num %10000 == 0:
            print(num)
        for key in keyword_train_list:
            if key in train_item:
                new_train_list += [train_item+"\n"]
    return new_train_list
new_train_list = fuck(src_train_list,keyword_train_list)
open(os.path.join(root_p,"test_k200.txt"),"w+").writelines(new_train_list)




# import os
# from numba import jit, prange

# root_p  ="/data/tianyuan/AST/kinetics200"
# src_train_list = open(os.path.join(root_p,"train.txt"),'r').read().split("\n")
# print("len src_train_list",len(src_train_list))
# keyword_train_list = open(os.path.join(root_p,"train_ytid_list.txt"),'r').read().split("\n")

# # @jit(nopython=True, parallel=True)
# def fuck(src_train_list,keyword_train_list):
#     new_train_list = []
#     num = 0
#     for train_item in src_train_list:
#         num = num+1
#         if num %10000 == 0:
#             print(num)
#         for key in keyword_train_list:
#             if key in train_item:
#                 new_train_list += [train_item+"\n"]
#     return new_train_list
# new_train_list = fuck(src_train_list,keyword_train_list)
# open(os.path.join(root_p,"train_k200.txt"),"w+").writelines(new_train_list)
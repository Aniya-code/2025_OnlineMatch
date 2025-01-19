# 前置指纹处理工作fp->fp_mixed->fp_fused
import pickle
from collections import defaultdict
import csv
import time
from datetime import datetime
import os, re
import math

# 流量指纹
class ChunkList:
    def __init__(self, row) -> None:
        self.url = row['url']
        # self.quality = row['quality']
        self.chunk_list = list(map(int, row['body_list'].split('/')))

# 在线匹配
class OnlineMatch:
    def __init__(self) -> None:
        # self.match_pattern = match_pattern
        self.RAW_MATCH_STATE = defaultdict(list) # 存放未排序结果
        self.MATCH_STATE = defaultdict(list) # 存放排序结果

    def chunk_match(self, chunk_idx, chunk, c, win_lower, win_upper):
        start_time = time.time()

        # 确定chunk搜索范围
        self.WIN_LOWER = math.ceil(win_lower / c) # -800 // 600 = -1
        self.WIN_UPPER = win_upper // c # 2700 // 600 =4
        chunk = chunk // c

        # 开始搜索
        for chunk_ in range(chunk - self.WIN_UPPER, chunk - self.WIN_LOWER + 1): # 左闭右开
            if chunk_ in FP_DICT:
                raw_result = FP_DICT[chunk_]
                self.process_raw_result(chunk_idx, raw_result)
        
        end_time = time.time()  # 记录结束时间
        print(f"chunk_match took {end_time - start_time:.4f} seconds")

    def process_raw_result(self, chunk_idx, raw_result):
        start_time = time.time()
        for entry in raw_result:
            key = tuple(entry[:4])  # 前四个值(id, url, v_itag, a_itag)作为键
            value = (chunk_idx, (entry[4], entry[5]))  # chunk_idx及后两个值(chunk_idx, (from_idx, to_idx))作为值
            self.RAW_MATCH_STATE[key].append(value) 

        end_time = time.time()  # 记录结束时间
        print(f"process_raw_result took {end_time - start_time:.4f} seconds")

    def find_continuous_intervals(self):
        '''
        1. 对RAW_MATCH_STATE中的每个key-value对进行处理，找出每个key对应的连续匹配区间
        2. 合并所有能连续匹配的区间，并选出每个key对应的最长连续区间
        '''
        start_time = time.time()
        for fp_key, cidx_intervals in self.RAW_MATCH_STATE.items():
            # 处理一个key:value
            cidx_intervals.sort(key=lambda x: (x[0], x[1][0])) # 按照chunk_idx, from_idx排序
            merged_intervals = [] # 在这里排队

            for cidx_interval in cidx_intervals:
                is_merged  = False
                for merged_list in merged_intervals:
                    # 检查是否与当前子列表中的任何区间连续
                    if cidx_interval[0] == merged_list[-1][0] + 1 and cidx_interval[1][0] == merged_list[-1][1][1] + 1 :
                        merged_list.append(cidx_interval)
                        is_merged  = True
                
                if not is_merged:
                    merged_intervals.append([cidx_interval])

            self.MATCH_STATE[fp_key] = max(merged_intervals, key=len)

        end_time = time.time()  # 记录结束时间
        print(f"find_continuous_intervals took {end_time - start_time:.4f} seconds")

    def find_longest_intervals(self):
        '''
        在所有fp的连续区间中找最最长
        '''
        max_length = max(len(value) for value in self.MATCH_STATE.values())
        longest_result_dict = {key : value for key,value in self.MATCH_STATE.items() if len(value) == max_length}
        return longest_result_dict

# 指纹库读入
def csv_to_kv(file_path, window_max, C):
    """从 csv 文件读取fp数据"""
    start = time.time()
    result_dict = defaultdict(list)
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) # 跳过表头
        for row in reader:
            id = row[0]
            url = row[1]
            v_itag = row[2]
            a_itag = row[5]
            video_quality = row[3]
            if video_quality != "1280x720": # 只保留720
                continue
            fingerprint = row[8].split('/')[:100] #### ！！防止爆内存！！ #####
            fingerprint = [int(x) for x in fingerprint if x.isdigit()]
            if len(fingerprint) == 0:
                print("------skip blank fused_fp------ \n", row[1], row[2], row[5])
                continue
            
            for window_len in range(1, window_max + 1): # 遍历窗口长度
                for start_idx in range(len(fingerprint) - window_len + 1): # 遍历fp每个滑动窗口
                    window = fingerprint[start_idx:start_idx + window_len]
                    window_sum = sum(window)
                    key = window_sum // C  
                    
                    # 记录 (id, url, v_itag, a_itag, from_idx, to_idx)
                    from_idx = start_idx
                    to_idx = start_idx + window_len - 1
                    result_dict[key].append((id, url, v_itag, a_itag, from_idx, to_idx)) 
    load_time = time.time()-start 
    print(f"### CSV converted to KV! Total {load_time:.3f}s ###\n")
    return result_dict, load_time


if __name__=='__main__':
    ######################################################################
    # 在线指纹
    online_chunk_file = '../data/corr copy.csv'
    # 指纹库
    offline_fp_csv_file = '../data/combined_fp_67_fused.csv'
    # 日志
    log_path = f'../data/match_result/log/log_{datetime.now():%Y%m%d%H%M}.txt'

    C = 600 # 缩放因子
    WIN_LOWER = 200 - 1000 # 200指body-seg最小值，-1000指corr-body最小值
    WIN_UPPER = 1200 + 1500  # 1200指body-seg最大值，1500指corr-body最大值
    WINDOW_MAX = 8 # 滑动窗口最大长度

    MAX_INPUT_IDX = 20 # 流量指纹参与匹配块数上限idx
    MIN_INPUT_IDX = 5 # 流量指纹参与匹配块数下限idx
    MIN_MATCH_LEN = 5 # 匹配成功所需的连续命中块数
    ######################################################################

    # 读取指纹：将csv转化为kv
    FP_DICT, load_time = csv_to_kv(offline_fp_csv_file, WINDOW_MAX, C)

    start = time.time()
    with open(online_chunk_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        tf_correct = 0 # 匹配成功的流量指纹数
        tf_num = 0 # 总流量指纹数
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(f"load data: {load_time :.3f}s\n") 

            # 遍历流量指纹文件所有行
            for idx, row in enumerate(reader): 
                tf_num += 1
                chunk_list_obj = ChunkList(row)
                print('#'*20 + f"开始识别chunk序列 标答：{chunk_list_obj.url} " + '#'*20) 
                log.write('#'*20 + f"开始识别chunk序列 标答：{chunk_list_obj.url} " + '#'*20+'\n') 

                online_match = OnlineMatch()
                for chunk_idx, chunk in enumerate(chunk_list_obj.chunk_list): # 遍历指纹块序列
                    if chunk_idx == MAX_INPUT_IDX: # URL不唯一或连续命中块数不够
                        print("=-=【匹配结束】=-=")
                        log.write("=-=【匹配结束】=-=\n")
                        break

                    online_match.chunk_match(chunk_idx, chunk, C, WIN_LOWER, WIN_UPPER)

                    if chunk_idx >= MIN_INPUT_IDX: # 从第5块开始对结果整合、排序
                        online_match.find_continuous_intervals()
                        longest_result_dict = online_match.find_longest_intervals()

                        # url去重
                        url_set = {fp_key[1] for fp_key in longest_result_dict.keys()}
                        
                        if len(url_set) == 1: # url唯一，但itag组合可能不唯一。默认取了第0个itag组合
                            fp_key = list(longest_result_dict.keys())[0]
                            cidx_intervals = list(longest_result_dict.values())[0]
                            if len(cidx_intervals) >= MIN_MATCH_LEN:
                                print(f"【chunk_idx={chunk_idx} 匹配结果】：{len(longest_result_dict)}个fp 1个url : {fp_key[1]} {len(cidx_intervals)}块")
                                log.write(f"【chunk_idx={chunk_idx} 匹配结果】：{len(longest_result_dict)}个fp 1个url : {fp_key[1]} {len(cidx_intervals)}块\n")

                                # url唯一，且连续命中长度达到要求
                                if fp_key[1][-11:] == chunk_list_obj.url[-11:]:
                                    tf_correct += 1
                                    for cidx_interval in cidx_intervals:
                                        print(f"chunk_idx={cidx_interval[0]}:{chunk_list_obj.chunk_list[int(cidx_interval[0])]}  片段索引: {cidx_interval[1][0]}--{cidx_interval[1][1]}")
                                        log.write(f"chunk_idx={cidx_interval[0]}:{chunk_list_obj.chunk_list[int(cidx_interval[0])]}  片段索引: {cidx_interval[1][0]}--{cidx_interval[1][1]}\n")
                                    print(f"=-=【匹配成功 {fp_key[2]} {fp_key[3]}】=-=")
                                    log.write(f"=-=【匹配成功 {fp_key[2]} {fp_key[3]}】=-=\n")
                                    break
                                else:
                                    print(f"=-=【匹配碰撞 {fp_key[2]} {fp_key[3]}】 {len(cidx_intervals)}块=-=")
                                    log.write(f"=-=【匹配碰撞 {fp_key[2]} {fp_key[3]}】 {len(cidx_intervals)}块=-=\n")
                                    break
                            # url唯一，但连续命中长度不够 
                            print(f"【chunk_idx={chunk_idx} 临时结果】：{len(longest_result_dict)}个fp {len(cidx_intervals)}块")
                            log.write(f"【chunk_idx={chunk_idx} 临时结果】：{len(longest_result_dict)}个fp {len(cidx_intervals)}块\n")
                        else: # url不唯一
                            print(f"【chunk_idx={chunk_idx} 临时结果】：{len(longest_result_dict)}个fp {len(list(longest_result_dict.values())[0])}块")
                            log.write(f"【chunk_idx={chunk_idx} 临时结果】：{len(longest_result_dict)}个fp {len(list(longest_result_dict.values())[0])}块\n")
            # 打印结果
            correct = tf_correct/tf_num
            print(f"在线指纹数={tf_num} 准确率={correct:.4f}")
            log.write(f"在线指纹数={tf_num} 准确率={correct:.4f}\n")
            end = time.time()
            print(f"总耗时：{end-start:.4f} 平均耗时={(end-start)/tf_num:.4f}")
            log.write(f"总耗时：{end-start:.4f} 平均耗时={(end-start)/tf_num:.4f}\n")
        










                    

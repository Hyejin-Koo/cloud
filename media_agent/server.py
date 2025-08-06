import numpy as np
# grpc proto libs
import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
import proto.kr.co.lguplus.engine.service.v1.stt_pb2_grpc as pb2_grpc
import proto.kr.co.lguplus.engine.model.v1.stt_pb2 as stt_pb2
import asyncio
# for pytorch inference, to be onnx runtime
import torch
# for grpc concurrency : multi-thread, not multi-process
from concurrent import futures
# logging libs
import logging
import logging.handlers
# for pre-define constants
from configs.paths import *
from configs.params import *
import argparse
#import psutil
from paddlespeech_ctcdecoders import ctc_beam_search_decode_chunk, Scorer, ctc_beam_search_decode_chunk_begin, CtcBeamSearchDecoderStorage, ctc_greedy_decoding, get_decode_result


import onnxruntime

from typing import Callable, List, Tuple
import os

from collections import deque
import time
import os

from modules.combine_module import _ModuleFeatureExtractor, _FunctionalModule, GlobalStatsNormalization, rms_flat, to_numpy, piecewise_linear_log
import math
import torchaudio
# import core of grpc
from utils import blank_collapse

import random
from tlohandler import CustomTimedRotatingFileHandler
from datetime import datetime, timedelta
import subprocess

NUM_TH = 16
torch.set_num_threads(NUM_TH)

DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)

streamlogger = logging.getLogger("stream")
statuslogger = logging.getLogger("status")
tlologger = logging.getLogger("tlo")

formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')

def generate_seq_id():
    current_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # YYYYMMDDHHmmSSsss (밀리초 포함)
    random_part_1 = str(random.randint(1000, 9999))  # 첫 번째 랜덤 4자리
    random_part_2 = str(random.randint(1000, 9999))  # 두 번째 랜덤 4자리
    return f"{current_time}{random_part_1}{random_part_2}"

# 로그 메시지 생성 함수 (TRID는 외부에서 받음)
def create_log_message(trid, result_code="20000000",req_time="",rsp_time="",svc_name="IPTV", stt_result=""):
    log_time = datetime.now().strftime('%Y%m%d%H%M%S')
    seq_id = generate_seq_id()

    # 로그 메시지 형식 맞추기
    log_message = (
        f"SEQ_ID={seq_id}|LOG_TIME={log_time}|LOG_TYPE=SVC|SID=|RESULT_CODE={result_code}|"
        f"REQ_TIME={req_time}|RSP_TIME={rsp_time}|CLIENT_IP=|DEV_INFO=|OS_INFO=|"
        f"NW_INFO=|SVC_NAME={svc_name}|DEV_MODEL=|CARRIER_TYPE=|TRID={trid}|"
        f"STT_RESULT={stt_result}"
    )
    return log_message


def configure_logging(port):
    log_dir = LOG_PATH
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Calculate suffix based on port
    suffix = f'{(port - 50051) // 2 + 1:03d}'

    streamfileHandler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, f'server_stream{suffix}.log'), when='midnight', interval=1, encoding='utf-8', backupCount=14)
    streamfileHandler.suffix = '%Y%m%d'
    streamfileHandler.setFormatter(formatter)

    statusfileHandler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, f'server_status{suffix}.log'), when='midnight', interval=1, encoding='utf-8', backupCount=14)
    statusfileHandler.suffix = '%Y%m%d'
    statusfileHandler.setFormatter(formatter)

    streamlogger.addHandler(streamfileHandler)
    streamlogger.setLevel(level=logging.INFO)
    statuslogger.addHandler(statusfileHandler)
    statuslogger.setLevel(level=logging.INFO)

    current_time = datetime.now()
    folder_name = current_time.strftime('%Y%m%d')
    tlo_dir = '/logs/tlo'

    tlo_path = os.path.join(tlo_dir, folder_name)  # 날짜별 폴더 경로

    if not os.path.exists(tlo_path):
        os.makedirs(tlo_path)

    rounded_minute = (current_time.minute // 5) * 5
    rounded_time = current_time.replace(minute=rounded_minute, second=0, microsecond=0)
    formatted_time = rounded_time.strftime('%Y%m%d%H%M')

    tlohandler = CustomTimedRotatingFileHandler(
        filename=f'{tlo_path}/iptvstt.tlo.{suffix}.{formatted_time}.log',  # 기본 파일 이름
        when='M',  # 'M'은 분 단위로 회전
        interval=5,  # 5분마다 회전
        backupCount=0,  # 백업 파일 개수, 직접 관리하므로 0으로 설정
        base_log_dir=tlo_dir,  # 로그 파일이 저장될 기본 디렉토리 설정
        retention_days=14,  # 로그 보관 기간 14일
        port_number=suffix  # 포트 번호를 받아서 파일 이름에 반영
    )
    tlologger.addHandler(tlohandler)
    tlologger.setLevel(level=logging.INFO)



class gRPC_Core(pb2_grpc.SttServiceServicer):
    def __init__(self, encoder, label_list, scorer,port):
        self.encoder = encoder
        self.scorer = scorer
        self.port = port
        self.cache_size = 64
        self.label_list = label_list
        self.blank_id = len(label_list)-1
        self.stream_extractor = _ModuleFeatureExtractor(
        torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=400, n_mels=80, hop_length=160, center=False
            ),
            _FunctionalModule(lambda x: x.transpose(1, 0)),
            _FunctionalModule(lambda x: piecewise_linear_log(x * GAIN)),
            GlobalStatsNormalization('merge_global_stats.json'),
            )
        )
        self.input_names = ['chunk', 'offset', 'att_cache', 'cnn_cache']
        dummy_features = np.zeros((1,76,80), dtype="float32")
        self.init_att_cache = np.zeros((12,8,self.cache_size,128), dtype="float32")
        self.init_cnn_cache = np.zeros((12,1,512,14), dtype="float32")
        self.init_offset = np.array((64), dtype=np.int64)
        
        
        input_tensors = (dummy_features, self.init_offset, self.init_att_cache, self.init_cnn_cache)
        print("warmup start")
        ort_inputs = {}
        for idx, name in enumerate(self.input_names):
            ort_inputs[name] = input_tensors[idx]
        for _ in range(10):
            self.encoder.run(None, ort_inputs)
        print("warmup end")
    

    def encoding(self, buffer, att_cache, cnn_cache, offset):
        with torch.no_grad():
            feature, length = self.stream_extractor(buffer)
            input_tensors = (feature[0:,:].unsqueeze(0).numpy(), offset, att_cache, cnn_cache)
            ort_inputs = {}
            for idx, name in enumerate(self.input_names):
                ort_inputs[name] = input_tensors[idx]
            encoder_output,att_cache, cnn_cache = self.encoder.run(None, ort_inputs)
            encoder_output = torch.FloatTensor(encoder_output)*0.59
            offset +=64

        return encoder_output.softmax(-1), att_cache, cnn_cache, offset

    def recognize(self, buffer, att_cache, cnn_cache, offset, trie, prefixes, collapse=True):
        encoder_output, att_cache, cnn_cache, offset = self.encoding(buffer, att_cache, cnn_cache, offset)
        if collapse:
            encoder_output, _ = blank_collapse(encoder_output, collapse)
        ctc_beam_search_decode_chunk(trie, prefixes, encoder_output.squeeze(0).tolist(), self.label_list, BEAM_SIZE, PRUNE_TH, PRUNE_RM, self.scorer, len(self.label_list)-1)
        stt_result = get_decode_result(prefixes, self.label_list, BEAM_SIZE,self.scorer)[0][1]

        return stt_result, att_cache, cnn_cache, offset

    def convert_to_string(self, tokens, vocab):
        return "".join([self.label_list[x] for x in tokens])


    def Connecting(self, uuid):
        # register client
        streamlogger.info("%s Connected. | result_code=%d", uuid, 1210)


    def LMUpdate(self, request, context):
        all_ports = list(range(50051, 50051 + 8 * 2, 2))
        endpoints_ports = [port for port in all_ports if port != self.port]
        
        print('make backupfile')
        subprocess.run(['mv','-f', 'checkpoints/lm.bin', 'checkpoints/lm_backup.bin'])
        
        #TODO: Download lm.bin from minio
        import boto3
        from botocore.client import Config
 
        # MinIO 접속 정보
        minio_endpoint = "http://minio-pool.stg-sp.violet.ixi-assist.uplus.co.kr"
        access_key = "RQ1KL2wiFBSgpsTB"
        secret_key = "H9hjocz2tqTsk5augGuzIOU9o8WFcOV5"
         
        # S3 클라이언트 생성
        s3 = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
         
        # 접속 확인 (버킷 목록 조회)
        buckets = s3.list_buckets()
        print(buckets)
        print("lm request name", request.lm_name)

        bucket_name = "s3-ixi-stg-ixiassist-admin" #minio-ixi-stt-lm"
        try:
            s3.download_file(bucket_name, request.lm_name, "checkpoints/lm.bin")
            statuslogger.info("Download LM binary file")
        except:
            statuslogger.error("LM Downloading fails!")
            return pb.LMUpdateReply(success=1)
        
        md5sum = subprocess.run(["md5sum", 'checkpoints/lm.bin'], capture_output=True, text=True)
        md5sum = md5sum.stdout.split()[0]
        if md5sum != request.checksum:
            print('request md5sum', request.checksum)
            print('server md5sum', md5sum)
            statuslogger.error("Downloading LM is not completed")
            return pb.LMUpdateReply(success=1)           
        
        #LMUpdateReply 0: success / 1: download failed / 2: update failed
        

        print('swap engine')
        subprocess.run(['sudo', 'cp', '/svc/web/nginx_lguservice/nginx_grpc.conf', '/svc/web/nginx_lguservice/nginx_grpc.temp'])
        subprocess.run(['sudo', 'cp', f'/svc/web/nginx_lguservice/nginx_grpc.{self.port}', '/svc/web/nginx_lguservice/nginx_grpc.conf'])
        subprocess.run(['sudo', 'touch', '/svc/web/nginx_lguservice/nginx_reload.signal'])
        
        statuslogger.info("lm update start %s", request.lm_name)
        lm_path = 'checkpoints/lm.bin'
        self.scorer = Scorer(LM_ALPHA, LM_BETA, lm_path, self.label_list)
        statuslogger.info('lm update end')
        
        
        subprocess.run(['sudo', 'cp', '/svc/web/nginx_lguservice/nginx_grpc.conf', f'/svc/web/nginx_lguservice/nginx_grpc.{self.port}'])
        subprocess.run(['sudo', 'cp', '/svc/web/nginx_lguservice/nginx_grpc.temp', '/svc/web/nginx_lguservice/nginx_grpc.conf'])
        
        statuslogger.info('swap engine')
        subprocess.run(['sudo', 'touch', '/svc/web/nginx_lguservice/nginx_reload.signal'])
        statuslogger.info('first server updating end')

        update_done_list = [self.port]
        
        for containers in endpoints_ports:
            try:
#               idx = containers - 50054 # ?? 먼솔?
                idx = (containers -  50049) // 2
                statuslogger.info(f'stt_server_{idx}:{containers} is going to be updated')
                lm_channel = grpc.insecure_channel(f'stt_server_{idx}:{containers}')
                lm_stub = pb2_grpc.SttServiceStub(lm_channel)
                lmreq = stt_pb2.InterLMUpdateRequest(port=containers)
                lmres = lm_stub.InterLMUpdate(lmreq)

                if lmres.success: # (0: success, 1,2: fails)
                    print('update fails')
                    statuslogger.error("{lm_path} LM Update failed")
                    subprocess.run(['mv', '-f', 'checkpoints/lm_backup.bin', 'checkpoints/lm.bin'])    
                    for rollbacks in update_done_list:
                        rbreq = stt_pb2.InterLMUpdateRequest(port=rollbacks)
                        #stt_pb2.InterLMUpdate(rbreq)
                        lm_stub.InterLMUpdate(rbreq)
                    return stt_pb2.LMUpdateReply(success=2)

                statuslogger.info(f"[{lm_path}] update lm for stt_server_{idx}")
                update_done_list.append(containers)
                print(update_done_list)
            except:
                # roll-back plans for update process fails..
                statuslogger.error(f"Update {containers} failed. Roll-Back start")
                subprocess.run(['mv','-f', 'checkpoints/lm_backup.bin', 'checkpoints/lm.bin'])
                for rollbacks in update_done_list:
                    rbreq = stt_pb2.InterLMUpdateRequest(port=rollbacks)
                    #stt_pb2.InterLMUpdate(rbreq)
                    lm_stub.InterLMUpdate(rbreq)
                return stt_pb2.LMUpdateReply(success=2)
                break #??



        # For the last step, update lm for tmp container, either
        lm_channel = grpc.insecure_channel(f'stt_server_tmp:50067')
        lm_stub = pb2_grpc.SttServiceStub(lm_channel)

        lmreq = stt_pb2.InterLMUpdateRequest(port=50067)
        lmres = lm_stub.InterLMUpdate(lmreq)
        if lmres.success:
            print('update fails in stt_server_tmp')
            statuslogger.error("{lm_path} LM Update failed")
            subprocess.run(['mv', '-f', 'checkpoints/lm_backup.bin', 'checkpoints/lm.bin'])    
            for rollbacks in update_done_list:
                rbreq = stt_pb2.InterLMUpdateRequest(port=rollbacks)
                #stt_pb2.InterLMUpdate(rbreq)
                lm_stub.InterLMUpdate(rbreq)
            return stt_pb2.LMUpdateReply(success=2)

        print("all conainers updated!")
        statuslogger.info('All containers updated!')

        return stt_pb2.LMUpdateReply(success=0)

    def InterLMUpdate(self, request, context):
        idx = request.port
        idx_ = (idx - 50049) // 2

        try:
            if idx != 50067:
                subprocess.run(['sudo', 'cp', '/svc/web/nginx_lguservice/nginx_grpc.conf', '/svc/web/nginx_lguservice/nginx_grpc.temp'])
                subprocess.run(['sudo', 'cp', f'/svc/web/nginx_lguservice/nginx_grpc.{idx}', '/svc/web/nginx_lguservice/nginx_grpc.conf'])
                subprocess.run(['sudo', 'touch', '/svc/web/nginx_lguservice/nginx_reload.signal'])
                #subprocess.run(['sudo', '/svc/web/nginx_lguservice/sbin/nginx', '-s', 'reload'])

                lm_path = 'checkpoints/lm.bin' # is updated lm
                self.scorer = Scorer(LM_ALPHA, LM_BETA, lm_path, self.label_list)


                subprocess.run(['sudo', 'cp', '/svc/web/nginx_lguservice/nginx_grpc.conf', f'/svc/web/nginx_lguservice/nginx_grpc.{idx}'])
                subprocess.run(['sudo', 'cp', '/svc/web/nginx_lguservice/nginx_grpc.temp', '/svc/web/nginx_lguservice/nginx_grpc.conf'])
                subprocess.run(['sudo', 'touch', '/svc/web/nginx_lguservice/nginx_reload.signal'])
                #subprocess.run(['sudo', '/svc/web/nginx_lguservice/sbin/nginx', '-s', 'reload'])

                # TODO: logger.info('update lm')
                print('LM updated')
                statuslogger.info(f"stt_server_{idx_} is updated")
              

                return stt_pb2.InterLMUpdateReply(success=0)

            else: # for tmp server
                lm_path = 'checkpoints/lm.bin' # is updated lm
                self.scorer = Scorer(LM_ALPHA, LM_BETA, lm_path, self.label_list)
                # TODO: logger.info('update lm')
                print('LM updated')
                statuslogger.info(f"stt_server_{idx_} is updated")
                return stt_pb2.InterLMUpdateReply(success=0)


        except:
            statuslogger.error(f"stt_server_{idx_} is failed")
            return stt_pb2.InterLMUpdateReply(success=1)

    def GetStream(self, request_iterator, context):
        count = 0
        frames = bytearray()
        
        end_flag = False
        epd_count = 0
        offset = np.array((64), dtype=np.int64)
        att_cache = np.zeros((12,8,self.cache_size,128), dtype="float32")
        cnn_cache = np.zeros((12,1,512,14), dtype="float32")
        bef_result = ""
        stt_result = ""
        left_frame = torch.zeros(OVERLAP)

        task_on = True
        stor = CtcBeamSearchDecoderStorage()
        trie = stor.root
        prefixes = stor.prefixes
        ctc_beam_search_decode_chunk_begin(trie, self.scorer)
        try:
            for req in request_iterator:
                req_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함

                if len(req.uuid) != 32:
                    streamlogger.error("%s: not length 32, It has %d length | result_code=%d", req.uuid, len(req.uuid), 2310)   
                    rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                    tlologger.error(create_log_message(req.uuid, result_code="50001001",req_time=req_time, rsp_time=rsp_time))
                    yield stt_pb2.StreamReply(query_header=0, connect_reply=2310)
                    continue

                streamlogger.info("%s: %s buffer | result_code=%d", req.uuid, len(req.audio_data), 1200)

                if req.query_header%4 ==1: #<- common stream input
                    if not task_on:
                        streamlogger.warning("%s Already answer returned, no uuid in memory pool. | result_code=%d", req.uuid, 1400)
                        continue

                    buff = req.audio_data
                    buff_len = len(buff)
                    if buff_len != 8192:
                        streamlogger.error("%s: has invalid buffer length %d | result_code=%d", req.uuid, buff_len, 2320)
                        rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                        tlologger.error(create_log_message(req.uuid, result_code="50001002",req_time=req_time, rsp_time=rsp_time))
                        yield stt_pb2.StreamReply(query_header=1, connect_reply=2320)
                        continue
                    
                    count += 1
                    if count == 1:
                        yield stt_pb2.StreamReply(query_header=1, epd_result=2, connect_reply=2200)
                        continue
                    frames += buff        
                    if len(frames) == 8192*3:
                        if end_flag:
                            streamlogger.info("%s epd checked | result_code=%d", req.uuid, 2200)
                            yield stt_pb2.StreamReply(query_header=1, epd_result=3, stt_result=stt_result, connect_reply=2200)

                        try:
                            
                            buffer = torch.from_numpy(np.frombuffer(frames, dtype=np.int16).astype(np.float32)/32767).squeeze()
                            buffer = torch.cat([left_frame,buffer])
                            left_frame = buffer[-OVERLAP:]
                            frames = bytearray()
                            stt_result, att_cache, cnn_cache, offset = self.recognize(
                                buffer, att_cache, cnn_cache, offset, trie, prefixes
                                )
                        except Exception as ex:
                            streamlogger.error("%s error detected | %s | result_code=%d", req.uuid, ex, 9999)
                            rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                            tlologger.error(create_log_message(req.uuid, result_code="50001004",req_time=req_time, rsp_time=rsp_time))

                        stt_result = stt_result.strip()
                        streamlogger.info("%s partial_result: %s | result_code=%d", req.uuid, stt_result, 2200)

                        if end_flag:
                            streamlogger.info("%s final stt_result: %s | result_code=%d", req.uuid, stt_result+'<eos>', 2200)
                            rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                            tlologger.info(create_log_message(req.uuid, result_code="20000000", stt_result=stt_result,req_time=req_time, rsp_time=rsp_time))
                            yield stt_pb2.StreamReply(query_header=1, epd_result=3, stt_result=stt_result, connect_reply=2200)
                            task_on = False
                            continue
                            
                        if stt_result == "":
                            if count > 3*12:
                                streamlogger.warning("%s speech not detected | result_code=%d", req.uuid, 3000)
                                rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                                tlologger.warning(create_log_message(req.uuid, result_code="20000001",req_time=req_time, rsp_time=rsp_time))
                                yield stt_pb2.StreamReply(query_header=1, epd_result=3, connect_reply=2200)
                                yield stt_pb2.StreamReply(query_header=1, epd_result=3, stt_result="",
                                                            connect_reply=3000)
                                task_on = False
                                continue

                            else:
                                continue
                        if bef_result == stt_result:
                            epd_count += 1
                            if epd_count == 1:
                                end_flag = True

                        else:
                            epd_count = 0
                        bef_result = stt_result
                    else:
                        yield stt_pb2.StreamReply(query_header=1, epd_result=4, stt_result=stt_result, connect_reply=2200)
                                




                # connecting stage
                elif req.query_header%4 == 0:
                    # validate trid
                    if req.uuid == "":
                        streamlogger.error("trid missing. | result_code=%d", 4101)
                        rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                        tlologger.error(create_log_message(req.uuid, result_code="50001003",req_time=req_time, rsp_time=rsp_time))
                        yield stt_pb2.StreamReply(query_header=0, connect_reply=4101)
                        continue
                    # try to allocate user object
                    try:
                        self.Connecting(req.uuid)
                        
                    except Exception as ex:
                        # logging error and response fail allocate
                        streamlogger.error("%s : %s | result_code=%d", ex, req.uuid, 4100)
                        rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                        tlologger.error(create_log_message(req.uuid, result_code="50001006",req_time=req_time, rsp_time=rsp_time))
                        yield stt_pb2.StreamReply(query_header=0, connect_reply=4100)
                        continue
                    
                    yield stt_pb2.StreamReply(query_header=0, connect_reply=2100)
                    continue
                
                # forced stop response
                elif req.query_header%4 == 2:
                    streamlogger.info("%s, request was stopped. | result_code=%d", req.uuid, 1310)
                    rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                    tlologger.warning(create_log_message(req.uuid, result_code="20000002",req_time=req_time, rsp_time=rsp_time))
                    # deallocate user object
                    yield stt_pb2.StreamReply(query_header=1, stt_result=stt_result, epd_result=3, connect_reply=2200)
                    task_on = False
                    continue
                
                # get timeout signal from client
                elif req.query_header%4 == 3:
                    streamlogger.info("%s request time out. | result_code=%d", req.uuid, 1320)
                    rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                    tlologger.warning(create_log_message(req.uuid, result_code="20000003",req_time=req_time, rsp_time=rsp_time))
                    # deallocate
                    yield stt_pb2.StreamReply(query_header=1, epd_result=3, stt_result=stt_result, connect_reply=2200)

                    yield stt_pb2.StreamReply(query_header=1, epd_result=3, stt_result=stt_result, connect_reply=2200)
                    task_on = False
                    continue

                else:
                    streamlogger.info("%s undefined query header(%d). | result_code=%d", req.uuid, req.query_header, 9999)
                    rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                    tlologger.error(create_log_message(req.uuid, result_code="20001006",req_time=req_time, rsp_time=rsp_time))

        except grpc.RpcError as grpce:
            if task_on:
                streamlogger.error("%s : %s | result_code=%d", "RpcError", req.uuid, 8888)
                rsp_time = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # 밀리초 포함
                tlologger.error(create_log_message(req.uuid, result_code="50001006",req_time=req_time, rsp_time=rsp_time))


# define server runner
def _run_server(bind_address, health_bind_address):
    statuslogger.info("Process Started %s", bind_address)
    port = int(bind_address.split('[::]:')[-1])
    label_list = list()

    # Todo : define vocab path
    with open(VOCAB_PATH) as voc:
        for i in voc:
            char, idx = i.strip('\n').split('\t')
            label_list.append(char)

    providers = [
               ('CPUExecutionProvider', {
               }),
            ]

    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = NUM_TH  # 적절한 스레드 수로 조정
    opts.inter_op_num_threads = NUM_TH  # 적절한 스레드 수로 조정
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    encoder = onnxruntime.InferenceSession(ONNX_PATH,
                                             providers=providers, sess_options=opts)
    statuslogger.info("Loaded Encoder %s", bind_address)

    # define server's concurrency
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            ("grpc.so_reuseport", 1),
            ("grpc.use_local_subchannel_pool", 1),
        ]
    )
    scorer = Scorer(LM_ALPHA, LM_BETA, LM_PATH, label_list)
    statuslogger.info("Loaded Decoder %s", bind_address)
    service = gRPC_Core(encoder, label_list, scorer, port)
    pb2_grpc.add_SttServiceServicer_to_server(service, server)
    server.add_insecure_port(bind_address)
    statuslogger.info("Server setted port number %s", bind_address)
    server.start()
    statuslogger.info("Server started on %s", bind_address)
#    health_server = startup_health_server(health_bind_address)
    counter = 0
    tlocounter = 0
    while True:
        counter += 1
        tlocounter += 1
        time.sleep(1)
        now = datetime.now()

        if counter % 60 == 0:
            counter = 0
            statuslogger.info("%s now alive...", bind_address)
        if now.minute % 5 == 0 and now.second == 0:
            tlologger.info("")
            tlocounter = 0



# health check with nginx.
def startup_health_server(bind_address):
    statuslogger.info("Health Server setted port number %s", bind_address)
    health_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, health_server)
    health_servicer.set(health.SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("kr.co.lguplus.engine.service.v1.SttService", health_pb2.HealthCheckResponse.SERVING)
    health_server.add_insecure_port(bind_address)
    health_server.start()
    statuslogger.info("Health Server started on %s", bind_address)
    return health_server

def main():
    parser = argparse.ArgumentParser(description='Run the server with specified port.')
    parser.add_argument('port', type=int, help='The port number to run the server on.')
    
    args = parser.parse_args()
    port = args.port
    configure_logging(port)

    _run_server('[::]:' + str(port), '[::]:' + str(int(port) + 1))

if __name__ == '__main__':
    main()

    

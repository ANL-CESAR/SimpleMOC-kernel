#!/bin/sh
nvprof --devices 0 --metrics stall_inst_fetch --metrics stall_exec_dependency --metrics stall_data_request --metrics stall_texture --metrics stall_sync --metrics stall_other ../SimpleMOC-kernel

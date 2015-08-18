#!/bin/sh
#nvprof --devices 0 --metrics stall_inst_fetch --metrics stall_exec_dependency --metrics stall_data_request --metrics stall_texture --metrics stall_sync --metrics stall_other ../SimpleMOC-kernel
nvprof --devices 2 --metrics stall_memory_dependency --metrics stall_constant_memory_dependency --metrics stall_pipe_busy --metrics stall_memory_throttle --metrics stall_not_selected --metrics stall_inst_fetch --metrics stall_exec_dependency --metrics stall_data_request --metrics stall_texture --metrics stall_sync --metrics stall_other ../SimpleMOC-kernel -d 2 -s 50000000

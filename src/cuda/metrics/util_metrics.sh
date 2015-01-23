#!/bin/sh
nvprof --devices 0 --metrics issue_slot_utilization --metrics l1_shared_utilization --metrics l2_utilization --metrics tex_utilization --metrics dram_utilization --metrics sysmem_utilization --metrics ldst_fu_utilization --metrics alu_fu_utilization --metrics cf_fu_utilization --metrics tex_fu_utilization ../SimpleMOC-kernel

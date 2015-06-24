#!/bin/sh
nvprof --metrics flop_sp_efficiency --metrics flop_count_sp ../SimpleMOC-kernel -d 0
nvprof --metrics flop_sp_efficiency --metrics flop_count_sp ../SimpleMOC-kernel -d 2

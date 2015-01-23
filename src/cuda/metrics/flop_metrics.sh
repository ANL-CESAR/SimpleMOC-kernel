#!/bin/sh
nvprof --devices 0 --metrics flops_sp --metrics ldst_issued --metrics ldst_executed ../SimpleMOC-kernel

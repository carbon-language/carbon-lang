// RUN: clang-cc -E %s 2>&1 | grep 'file not found'

#define DO_PRAGMA _Pragma 
DO_PRAGMA ("GCC dependency \"blahblabh\"")


// RUN: clang-cc %s -E 2>&1 | grep 'system_header ignored in main file'

_Pragma ("GCC system_header")


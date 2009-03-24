// RUN: clang-cc -x c-header %s -o %t && 
// RUN: clang-cc -include-pth %t %s -E | grep 'file_to_include' | count 2
#include "file_to_include.h"

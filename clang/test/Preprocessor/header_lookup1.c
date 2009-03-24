// RUN: clang-cc -I /usr/include %s -E | grep 'stdio.h.*3.*4'
#include <stdio.h>

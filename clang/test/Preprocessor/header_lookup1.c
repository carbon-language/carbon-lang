// RUN: %clang -fno-ms-extensions -I /usr/include %s -E | grep 'stdio.h.*3.*4'
#include <stdio.h>

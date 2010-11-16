// RUN: %clang -fno-ms-extensions %s -E | grep 'stddef.h.*3.*4'
#include <stddef.h>

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I%S/Inputs/merge-name-for-linkage -verify %s
// expected-no-diagnostics
typedef union {} pthread_mutex_t;
#include "a.h"
pthread_mutex_t x;
#include "b.h"
pthread_mutex_t y;

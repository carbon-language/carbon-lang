// RUN: %clang -E -frewrite-includes %s -I%S/Inputs/ | %clang -Wall -Wunused-macros -x c -c - 2> %t.1
// RUN: %clang -I%S/Inputs/ -Wall -Wunused-macros -c %s 2> %t.2
// RUN: cmp -s %t.1 %t.2
// expected-no-diagnostics
// REQUIRES: shell

#include "rewrite-includes-messages.h"
#define UNUSED_MACRO

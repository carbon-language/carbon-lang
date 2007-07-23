// RUN: clang -Eonly %s -I.
#  define HEADER <stdio.h>

#  include HEADER

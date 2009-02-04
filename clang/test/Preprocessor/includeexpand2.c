// RUN: clang -Eonly %s -I.
#  define HEADER <file_to_include.h>

#  include HEADER

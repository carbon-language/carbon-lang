// RUN: clang-cc %s -E | grep '! ,'

#define XX
! XX,


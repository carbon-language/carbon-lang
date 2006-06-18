// RUN: clang %s -E | grep '! ,'

#define XX
! XX,


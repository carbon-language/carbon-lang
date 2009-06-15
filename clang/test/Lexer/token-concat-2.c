// RUN: clang-cc -E -x c -o - %s | grep '[.][*]'
// PR4395
#define X .*
X

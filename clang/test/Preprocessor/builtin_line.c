// RUN: clang %s -E | grep "^  4"
#define FOO __LINE__

  FOO

// RUN: %clang_cc1 %s -E | not grep 'scratch space'

#define push _Pragma ("pack(push)")
push

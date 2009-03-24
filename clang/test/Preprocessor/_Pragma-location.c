// RUN: clang-cc %s -E | not grep 'scratch space'

#define push _Pragma ("pack(push)")
push

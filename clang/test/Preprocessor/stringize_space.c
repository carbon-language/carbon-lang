// RUN: clang-cc -E %s | grep -- '-"" , - "" , -"" , - ""'

#define A(b) -#b  ,  - #b  ,  -# b  ,  - # b
A()

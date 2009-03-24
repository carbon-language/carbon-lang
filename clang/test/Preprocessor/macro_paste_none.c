// RUN: clang-cc -E %s | grep '!!'

#define A(B,C) B ## C

!A(,)!


// RUN: clang -E %s | grep '^Y$'

#define X() Y
#define Y() X

X()()()


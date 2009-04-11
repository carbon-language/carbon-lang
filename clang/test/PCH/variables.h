// RUN: clang-cc -emit-pch -o variables.h.pch variables.h
// Do not mess with the whitespace in this file. It's important.




extern float y;
extern int *ip, x;

float z;



#define MAKE_HAPPY(X) X##Happy
int MAKE_HAPPY(Very);

#define A_MACRO_IN_THE_PCH 492
#define FUNCLIKE_MACRO(X, Y) X ## Y

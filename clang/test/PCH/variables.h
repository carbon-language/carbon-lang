// RUN: clang-cc -emit-pch -o variables.h.pch variables.h
extern int x;
extern float y;
extern int *ip;
float z;

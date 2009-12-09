// RUN: clang-cc -fsyntax-only -verify %s

extern "C" int myarray[];
int myarray[12] = {0};

extern "C" int anotherarray[][3];
int anotherarray[2][3] = {1,2,3,4,5,6};

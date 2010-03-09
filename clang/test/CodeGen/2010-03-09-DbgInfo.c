// RUN: %clang -dA -S -O0 -g %s -o - | grep DW_TAG_variable
unsigned char ctable1[1] = { 0001 };

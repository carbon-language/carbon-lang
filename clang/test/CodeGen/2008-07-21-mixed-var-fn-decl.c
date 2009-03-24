// RUN: clang-cc -emit-llvm -o - %s | grep -e "@g[0-9] " | count 2

int g0, f0();
int f1(), g1;


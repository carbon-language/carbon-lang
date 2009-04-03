// RUN: clang-cc -fvisibility=hidden -emit-llvm -o - %s | grep -e "hidden" | count 2

int Global = 10; 

void Func() {}


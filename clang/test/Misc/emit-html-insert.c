// RUN: clang-cc %s -emit-html -o - | grep ">&lt; 10; }"

int a(int x) { return x
< 10; }

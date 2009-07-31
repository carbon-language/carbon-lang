// RUN: clang-cc -S -g -o %t.s %s
void foo() {
     int i = 0;
     i = 42;
}

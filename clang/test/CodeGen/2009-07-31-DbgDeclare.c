// RUN: %clang_cc1 -S -debug-info-kind=limited -o %t.s %s
void foo() {
     int i = 0;
     i = 42;
}

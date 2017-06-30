void f1();
__attribute__((always_inline)) void f2() {
  f1();
}
void f3() {
  f2();
}

void f1();
inline __attribute__((always_inline)) void f2() {
  f1();
}
void f3() {
  f2();
}

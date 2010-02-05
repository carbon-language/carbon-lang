// RUN: %clang_cc1 -emit-llvm < %s | grep 'fastcallcc' | count 6
// RUN: %clang_cc1 -emit-llvm < %s | grep 'stdcallcc' | count 6

void __attribute__((fastcall)) f1(void);
void __attribute__((stdcall)) f2(void);
void __attribute__((fastcall)) f3(void) {
  f1();
}
void __attribute__((stdcall)) f4(void) {
  f2();
}

// PR5280
void (__attribute__((fastcall)) *pf1)(void) = f1;
void (__attribute__((stdcall)) *pf2)(void) = f2;
void (__attribute__((fastcall)) *pf3)(void) = f3;
void (__attribute__((stdcall)) *pf4)(void) = f4;

int main(void) {
    f3(); f4();
    pf1(); pf2(); pf3(); pf4();
    return 0;
}


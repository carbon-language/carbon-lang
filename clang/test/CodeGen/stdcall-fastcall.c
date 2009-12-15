// RUN: %clang_cc1 -emit-llvm < %s | grep 'fastcallcc' | count 4
// RUN: %clang_cc1 -emit-llvm < %s | grep 'stdcallcc' | count 4

void __attribute__((fastcall)) f1(void);
void __attribute__((stdcall)) f2(void);
void __attribute__((fastcall)) f3(void) {
  f1();
}
void __attribute__((stdcall)) f4(void) {
  f2();
}

int main(void) {
    f3(); f4();
    return 0;
}


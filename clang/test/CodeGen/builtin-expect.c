// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

int x;
int y(void);
void foo();
void FUNC() {
// CHECK: [[call:%.*]] = call i32 @y
  if (__builtin_expect (x, y()))
    foo ();
}

// rdar://9330105
void isigprocmask(void);
long bar();

int main() {
    (void) __builtin_expect((isigprocmask(), 0), bar());
}

// CHECK: call void @isigprocmask()
// CHECK: [[C:%.*]] = call i64 (...)* @bar()

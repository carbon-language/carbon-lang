// RUN: %clang -target x86_64-linux-gnu -fsplit-stack -S %s -emit-llvm -o - | FileCheck -check-prefix=CHECK-SEGSTK %s
// RUN: %clang -target x86_64-linux-gnu -S %s -emit-llvm -o - | FileCheck -check-prefix=CHECK-NOSEGSTK %s

int foo() {
  return 0;
}

int main() {
  return foo();
}

// CHECK-SEGSTK: define i32 @foo() #0 {
// CHECK-SEGSTK: define i32 @main() #0 {
// CHECK-SEGSTK: #0 = { {{.*}} "split-stack" {{.*}} }

// CHECK-NOSEGSTK: define i32 @foo() #0 {
// CHECK-NOSEGSTK: define i32 @main() #0 {
// CHECK-NOSEGSTK-NOT: #0 = { {{.*}} "split-stack" {{.*}} }

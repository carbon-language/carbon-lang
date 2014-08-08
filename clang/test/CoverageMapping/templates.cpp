// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name templates.cpp %s | FileCheck %s

template<typename T>
void unused(T x) {
  return;
}

template<typename T>
int func(T x) {
  if(x)
    return 0;
  else
    return 1;
  int j = 1;
}

// CHECK: func
// CHECK-NEXT: File 0, 9:15 -> 15:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 11:5 -> 11:13 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 13:5 -> 13:13 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 14:3 -> 14:12 = 0 (HasCodeBefore = 0)

// CHECK-NEXT: func
// CHECK-NEXT: File 0, 9:15 -> 15:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 11:5 -> 11:13 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 13:5 -> 13:13 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 14:3 -> 14:12 = 0 (HasCodeBefore = 0)

int main() {
  func<int>(0);
  func<bool>(true);
  return 0;
}

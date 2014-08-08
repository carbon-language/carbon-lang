// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name return.c %s | FileCheck %s

void func() {
  return;
  int i = 0;
}

// CHECK: func
// CHECK: File 0, 3:13 -> 6:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:3 -> 5:12 = 0 (HasCodeBefore = 0)

void func2() {
  for(int i = 0; i < 10; ++i) {
    if(i > 2) {
      return;
    } else {
      int j = 0;
    }
    if(i == 3) {
      int j = 1;
    } else {
      int j = 2;
    }
  }
}

// CHECK-NEXT: func2
// CHECK-NEXT: File 0, 12:14 -> 25:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 13:18 -> 13:24 = ((#0 + #1) - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 13:26 -> 13:29 = (#1 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 13:31 -> 24:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 14:15 -> 16:6 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 16:12 -> 21:11 = (#1 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 19:16 -> 21:6 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 21:12 -> 23:6 = ((#1 - #2) - #3) (HasCodeBefore = 0)

void func3(int x) {
  if(x > 5) {
    while(x >= 9) {
      return;
      --x;
    }
    int i = 0;
  }
  int j = 0;
}

// CHECK-NEXT: func3
// CHECK-NEXT: File 0, 37:19 -> 46:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 38:13 -> 44:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 39:11 -> 39:17 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 39:19 -> 42:6 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 41:7 -> 41:10 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 43:5 -> 43:14 = (#1 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 45:3 -> 45:12 = (#0 - #2) (HasCodeBefore = 0)

int main() {
  func();
  func2();
  for(int i = 0; i < 10; ++i)
    func3(i);
  return 0;
}

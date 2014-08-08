// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name switch.c %s | FileCheck %s

void foo(int i) {
  switch(i) {
  case 1:
    return;
  case 2:
    break;
  }
  int x = 0;
}

// CHECK: foo
// CHECK-NEXT: File 0, 3:17 -> 11:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:3 -> 6:11 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 7:3 -> 8:10 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 9:3 -> 10:12 = #1 (HasCodeBefore = 0)

int main() {
  int i = 0;
  switch(i) {
  case 0:
    i = 1;
    break;
  case 1:
    i = 2;
    break;
  default:
    break;
  }
  switch(i) {
  case 0:
    i = 1;
    break;
  case 1:
    i = 2;
  default:
    break;
  }


  switch(i) {
  case 1:
  case 2:
    i = 11;
  case 3:
  case 4:
    i = 99;
  }
  switch(i) {
  case 1:
    return 1;
    break;
  case 2:
    break;
  }

  foo(1);
  return 0;
}

// CHECK-NEXT: main
// CHECK-NEXT: File 0, 19:12 -> 60:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 22:3 -> 24:10 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 25:3 -> 27:10 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 28:3 -> 29:10 = #4 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 30:3 -> 31:14 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 32:3 -> 34:10 = #6 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 35:3 -> 36:10 = #7 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 37:3 -> 38:10 = (#7 + #8) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 39:3 -> 42:14 = #5 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 43:3 -> 43:10 = #10 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 44:3 -> 45:11 = (#10 + #11) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 46:3 -> 46:10 = ((#10 + #11) + #12) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 47:3 -> 48:11 = (((#10 + #11) + #12) + #13) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 49:3 -> 50:14 = #9 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 51:3 -> 52:13 = #15 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 53:5 -> 53:10 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 54:3 -> 55:10 = #16 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 56:3 -> 59:11 = #14 (HasCodeBefore = 0)

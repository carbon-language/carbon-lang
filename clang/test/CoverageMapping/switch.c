// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name switch.c %s | FileCheck %s
                    // CHECK: foo
void foo(int i) {   // CHECK-NEXT: File 0, [[@LINE]]:17 -> [[@LINE+8]]:2 = #0 (HasCodeBefore = 0)
  switch(i) {
  case 1:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:11 = #2 (HasCodeBefore = 0)
    return;
  case 2:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:10 = #3 (HasCodeBefore = 0)
    break;
  }                 // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:12 = #1 (HasCodeBefore = 0)
  int x = 0;
}

                    // CHECK-NEXT: main
int main() {        // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+34]]:2 = #0 (HasCodeBefore = 0)
  int i = 0;
  switch(i) {
  case 0:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+2]]:10 = #2 (HasCodeBefore = 0)
    i = 1;
    break;
  case 1:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+2]]:10 = #3 (HasCodeBefore = 0)
    i = 2;
    break;
  default:          // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:10 = #4 (HasCodeBefore = 0)
    break;
  }                 // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:14 = #1 (HasCodeBefore = 0)
  switch(i) {
  case 0:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+2]]:10 = #6 (HasCodeBefore = 0)
    i = 1;
    break;
  case 1:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:10 = #7 (HasCodeBefore = 0)
    i = 2;
  default:          // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:10 = (#7 + #8) (HasCodeBefore = 0)
    break;
  }                 // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+3]]:14 = #5 (HasCodeBefore = 0)


  switch(i) {
  case 1:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:10 = #10 (HasCodeBefore = 0)
  case 2:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:11 = (#10 + #11) (HasCodeBefore = 0)
    i = 11;
  case 3:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:10 = ((#10 + #11) + #12) (HasCodeBefore = 0)
  case 4:           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:11 = (((#10 + #11) + #12) + #13) (HasCodeBefore = 0)
    i = 99;
  }                 // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+3]]:11 = #9 (HasCodeBefore = 0)

  foo(1);
  return 0;
}

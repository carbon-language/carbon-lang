// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name label.cpp %s | FileCheck %s

                             // CHECK: func
void func() {                // CHECK-NEXT: File 0, [[@LINE]]:13 -> [[@LINE+18]]:2 = #0 (HasCodeBefore = 0)
  int i = 0;                 // CHECK-NEXT: File 0, [[@LINE+2]]:14 -> [[@LINE+2]]:20 = (#0 + #3) (HasCodeBefore = 0)
                             // CHECK-NEXT: File 0, [[@LINE+1]]:22 -> [[@LINE+1]]:25 = #3 (HasCodeBefore = 0)
  for(i = 0; i < 10; ++i) {  // CHECK-NEXT: File 0, [[@LINE]]:27 -> [[@LINE+10]]:4 = #1 (HasCodeBefore = 0)
    if(i < 5) {              // CHECK-NEXT: File 0, [[@LINE]]:15 -> [[@LINE+6]]:6 = #2 (HasCodeBefore = 0)
      {
        x:                   // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE+6]]:14 = #3 (HasCodeBefore = 0)
          int j = 1;
      }
      int m = 2;
    } else
      goto x;                // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:13 = (#1 - #2) (HasCodeBefore = 0)
    int k = 3;
  }
  static int j = 0;          // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+2]]:12 = ((#0 + #3) - #1) (HasCodeBefore = 0)
  ++j;
  if(j == 1)
    goto x;                  // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:11 = #4 (HasCodeBefore = 0)
}

                             // CHECK-NEXT: test1
void test1(int x) {          // CHECK-NEXT: File 0, [[@LINE]]:19 -> [[@LINE+7]]:2 = #0 (HasCodeBefore = 0)
  if(x == 0)
    goto a;                  // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:11 = #1 (HasCodeBefore = 0)
  goto b;                    // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:9 = (#0 - #1) (HasCodeBefore = 0)
a:                           // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE]]:2 = #2 (HasCodeBefore = 0)
b:                           // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE+1]]:12 = #3 (HasCodeBefore = 0)
  x = x + 1;
}

                             // CHECK-NEXT: test2
void test2(int x) {          // CHECK-NEXT: File 0, [[@LINE]]:19 -> [[@LINE+8]]:2 = #0 (HasCodeBefore = 0)
  if(x == 0)
    goto a;                  // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:11 = #1 (HasCodeBefore = 0)
                             // CHECK-NEXT: File 0, [[@LINE+1]]:8 -> [[@LINE+1]]:17 = (#0 - #1) (HasCodeBefore = 0)
  else if(x == 1) goto b;    // CHECK-NEXT: File 0, [[@LINE]]:19 -> [[@LINE]]:25 = #2 (HasCodeBefore = 0)
a:                           // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE]]:2 = #3 (HasCodeBefore = 0)
b:                           // CHECK-NEXT: File 0, [[@LINE]]:1 -> [[@LINE+1]]:12 = #4 (HasCodeBefore = 0)
  x = x + 1;
}

                             // CHECK-NEXT: main
int main() {                 // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+17]]:2 = #0 (HasCodeBefore = 0)
  int j = 0;
  for(int i = 0; i < 10; ++i) { // CHECK: File 0, [[@LINE]]:31 -> [[@LINE+11]]:4 = #1 (HasCodeBefore = 0)
  a:                         // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:13 = #2 (HasCodeBefore = 0)
    if(i < 3)
      goto e;                // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:13 = #3 (HasCodeBefore = 0)
    goto c;                  // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:11 = (#2 - #3) (HasCodeBefore = 0)
  b:                         // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:10 = #4 (HasCodeBefore = 0)
    j = 2;
  c:                         // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+1]]:10 = #5 (HasCodeBefore = 0)
    j = 1;
                             // CHECK-NEXT: File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:4 = #6 (HasCodeBefore = 0)
  e: f: ;                    // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE]]:10 = #7 (HasCodeBefore = 0)
  }
  func();                    // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+2]]:11 = ((#0 + #7) - #1) (HasCodeBefore = 0)
  test1(0);
  test2(2);
}

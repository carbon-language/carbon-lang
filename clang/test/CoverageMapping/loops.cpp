// RUN: %clang_cc1 -std=c++11 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name loops.cpp %s | FileCheck %s

                                    // CHECK: rangedFor
void rangedFor() {                  // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE+6]]:2 = #0 (HasCodeBefore = 0)
  int arr[] = { 1, 2, 3, 4, 5 };
  int sum = 0;
  for(auto i : arr) {               // CHECK-NEXT: File 0, [[@LINE]]:21 -> [[@LINE+2]]:4 = #1 (HasCodeBefore = 0)
    sum += i;
  }
}

                                    // CHECK-NEXT: main
int main() {                        // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+24]]:2 = #0 (HasCodeBefore = 0)
                                    // CHECK-NEXT: File 0, [[@LINE+1]]:18 -> [[@LINE+1]]:24 = (#0 + #1) (HasCodeBefore = 0)
  for(int i = 0; i < 10; ++i)       // CHECK-NEXT: File 0, [[@LINE]]:26 -> [[@LINE]]:29 = #1 (HasCodeBefore = 0)
     ;                              // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE]]:7 = #1 (HasCodeBefore = 0)
  for(int i = 0;
      i < 10;                       // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:13 = (#0 + #2) (HasCodeBefore = 0)
      ++i)                          // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:10 = #2 (HasCodeBefore = 0)
  {                                 // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE+2]]:4 = #2 (HasCodeBefore = 0)
    int x = 0;
  }
  int j = 0;                        // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:14 = (#0 + #3) (HasCodeBefore = 0)
  while(j < 5) ++j;                 // CHECK-NEXT: File 0, [[@LINE]]:16 -> [[@LINE]]:19 = #3 (HasCodeBefore = 0)
  do {                              // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE+2]]:17 = (#0 + #4) (HasCodeBefore = 0)
    ++j;
  } while(j < 10);
  j = 0;
  while
   (j < 5)                          // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:10 = (#0 + #5) (HasCodeBefore = 0)
     ++j;                           // CHECK-NEXT: File 0, [[@LINE]]:6 -> [[@LINE]]:9 = #5 (HasCodeBefore = 0)
  do
    ++j;                            // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE+1]]:15 = (#0 + #6) (HasCodeBefore = 0)
  while(j < 10);
  rangedFor();
  return 0;
}

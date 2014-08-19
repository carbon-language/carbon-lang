// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name break.c %s | FileCheck %s

int main() {         // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+28]]:2 = #0 (HasCodeBefore = 0)
  int cnt = 0;       // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = #0 (HasCodeBefore = 0)
  while(cnt < 100) { // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+3]]:4 = #1 (HasCodeBefore = 0)
    break;
    ++cnt;           // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:10 = 0 (HasCodeBefore = 0)
  }                  // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = #0 (HasCodeBefore = 0)
  while(cnt < 100) { // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+6]]:4 = #2 (HasCodeBefore = 0)
    {
      break;
      ++cnt;         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE+2]]:10 = 0 (HasCodeBefore = 0)
    }
    ++cnt;
  }                  // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = ((#0 + #3) - #4) (HasCodeBefore = 0)
  while(cnt < 100) { // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+6]]:4 = #3 (HasCodeBefore = 0)
    if(cnt == 0) {   // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE+3]]:6 = #4 (HasCodeBefore = 0)
      break;
      ++cnt;         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:12 = 0 (HasCodeBefore = 0)
    }
    ++cnt;           // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:10 = (#3 - #4) (HasCodeBefore = 0)
  }                  // CHECK-NEXT: File 0, [[@LINE+1]]:9 -> [[@LINE+1]]:18 = (#0 + #6) (HasCodeBefore = 0)
  while(cnt < 100) { // CHECK-NEXT: File 0, [[@LINE]]:20 -> [[@LINE+7]]:4 = #5 (HasCodeBefore = 0)
    if(cnt == 0) {   // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE+5]]:10 = #6 (HasCodeBefore = 0)
      ++cnt;
    } else {         // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+2]]:6 = (#5 - #6) (HasCodeBefore = 0)
      break;
    }
    ++cnt;
  }
}

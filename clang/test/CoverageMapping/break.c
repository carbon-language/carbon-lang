// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name break.c %s | FileCheck %s

int main() {
  int cnt = 0;
  while(cnt < 100) {
    break;
    ++cnt;
  }
  while(cnt < 100) {
    {
      break;
      ++cnt;
    }
    ++cnt;
  }
  while(cnt < 100) {
    if(cnt == 0) {
      break;
      ++cnt;
    }
    ++cnt;
  }
  while(cnt < 100) {
    if(cnt == 0) {
      ++cnt;
    } else {
      break;
    }
    ++cnt;
  }
}

// CHECK: File 0, 3:12 -> 31:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:9 -> 5:18 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:20 -> 8:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 7:5 -> 7:10 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 9:9 -> 9:18 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 9:20 -> 15:4 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 12:7 -> 14:10 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 16:9 -> 16:18 = ((#0 + #3) - #4) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 16:20 -> 22:4 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 17:18 -> 20:6 = #4 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 19:7 -> 19:12 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 21:5 -> 21:10 = (#3 - #4) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 23:9 -> 23:18 = (#0 + #6) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 23:20 -> 30:4 = #5 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 24:18 -> 29:10 = #6 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 26:12 -> 28:6 = (#5 - #6) (HasCodeBefore = 0)

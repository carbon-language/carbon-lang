// RUN: %clang_cc1 -std=c++11 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name loops.cpp %s | FileCheck %s

void rangedFor() {
  int arr[] = { 1, 2, 3, 4, 5 };
  int sum = 0;
  for(auto i : arr) {
    sum += i;
    if(i == 3)
      break;
  }
}

// CHECK: rangedFor
// CHECK-NEXT: File 0, 3:18 -> 11:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 6:21 -> 10:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 9:7 -> 9:12 = #2 (HasCodeBefore = 0)

int main() {
  for(int i = 0; i < 10; ++i)
     ;
  for(int i = 0; i < 0; ++i)
     ;
  for(int i = 0;
      i < 10;
      ++i)
  {
    int x = 0;
  }
  int j = 0;
  while(j < 5) ++j;
  do {
    ++j;
  } while(j < 10);
  j = 0;
  while
   (j < 5)
     ++j;
  do
    ++j;
  while(j < 10);
  rangedFor();
  return 0;
}

// CHECK-NEXT: main
// CHECK-NEXT: File 0, 18:12 -> 43:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 19:18 -> 19:24 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 19:26 -> 19:29 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 20:6 -> 20:7 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 21:18 -> 21:23 = (#0 + #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 21:25 -> 21:28 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 22:6 -> 22:7 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 24:7 -> 24:13 = (#0 + #3) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 25:7 -> 25:10 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 26:3 -> 28:4 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 30:9 -> 30:14 = (#0 + #4) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 30:16 -> 30:19 = #4 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 31:6 -> 33:17 = (#0 + #5) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 36:5 -> 36:10 = (#0 + #6) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 37:6 -> 37:9 = #6 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 39:5 -> 40:15 = (#0 + #7) (HasCodeBefore = 0)

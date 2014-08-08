// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name trycatch.cpp %s | FileCheck %s

class Error {
};

class ImportantError {
};

class Warning {
};

void func(int i) {
  if(i % 2)
    throw Error();
  else if(i == 8)
    throw ImportantError();
}

// CHECK: func
// CHECK-NEXT: File 0, 12:18 -> 17:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 14:5 -> 14:16 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 15:8 -> 15:17 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 16:5 -> 16:25 = #2 (HasCodeBefore = 0)

int main() {
  int j = 0;
  for(int i = 0; i < 9; ++i) {
    try {
      func(i);
    } catch(const Error &e) {
      j = 1;
    } catch(const ImportantError &e) {
      j = 11;
    }
    catch(const Warning &w) {
      j = 0;
    }
  }
  return 0;
}

// CHECK-NEXT: main
// CHECK-NEXT: File 0, 25:12 -> 40:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 27:18 -> 27:23 = (#0 + #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 27:25 -> 27:28 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 27:30 -> 38:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 30:29 -> 32:12 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 32:38 -> 35:10 = #4 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 35:29 -> 37:6 = #5 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 39:3 -> 39:11 = ((#0 + #2) - #1) (HasCodeBefore = 0)

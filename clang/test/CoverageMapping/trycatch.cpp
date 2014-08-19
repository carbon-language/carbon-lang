// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fexceptions -fcxx-exceptions -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name trycatch.cpp %s | FileCheck %s

class Error {
};

class ImportantError {
};

class Warning {
};

                                      // CHECK: func
void func(int i) {                    // CHECK-NEXT: File 0, [[@LINE]]:18 -> [[@LINE+5]]:2 = #0 (HasCodeBefore = 0)
  if(i % 2)
    throw Error();                    // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:16 = #1 (HasCodeBefore = 0)
  else if(i == 8)                     // CHECK-NEXT: File 0, [[@LINE]]:8 -> [[@LINE]]:17 = (#0 - #1) (HasCodeBefore = 0)
    throw ImportantError();           // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:25 = #2 (HasCodeBefore = 0)
}

                                      // CHECK-NEXT: main
int main() {                          // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+13]]:2 = #0 (HasCodeBefore = 0)
  int j = 0;
  try {
    func(j);
  } catch(const Error &e) {           // CHECK-NEXT: File 0, [[@LINE]]:27 -> [[@LINE+2]]:10 = #2 (HasCodeBefore = 0)
    j = 1;
  } catch(const ImportantError &e) {  // CHECK-NEXT: File 0, [[@LINE]]:36 -> [[@LINE+3]]:8 = #3 (HasCodeBefore = 0)
    j = 11;
  }
  catch(const Warning &w) {           // CHECK-NEXT: File 0, [[@LINE]]:27 -> [[@LINE+2]]:4 = #4 (HasCodeBefore = 0)
    j = 0;
  }
  return 0;                           // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:11 = #1 (HasCodeBefore = 0)
}

// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %S/Inputs/lineExecutionCounts.profdata -no-colors -filename-equivalence %s | FileCheck -check-prefix=CHECK -check-prefix=WHOLE-FILE %s
// RUN: llvm-cov show %S/Inputs/lineExecutionCounts.covmapping -instr-profile %S/Inputs/lineExecutionCounts.profdata -no-colors -filename-equivalence -name=main %s | FileCheck -check-prefix=CHECK -check-prefix=FILTER %s

// before any coverage              // WHOLE-FILE:    | [[@LINE]]|// before
                                    // FILTER-NOT:    | [[@LINE-1]]|// before
int main() {                             // CHECK:   1| [[@LINE]]|int main(
  int x = 0;                             // CHECK:   1| [[@LINE]]|  int x
                                         // CHECK:   1| [[@LINE]]|
  if (x) {                               // CHECK:   0| [[@LINE]]|  if (x)
    x = 0;                               // CHECK:   0| [[@LINE]]|    x = 0
  } else {                               // CHECK:   1| [[@LINE]]|  } else
    x = 1;                               // CHECK:   1| [[@LINE]]|    x = 1
  }                                      // CHECK:   1| [[@LINE]]|  }
                                         // CHECK:   1| [[@LINE]]|
  for (int i = 0; i < 100; ++i) {        // CHECK: 101| [[@LINE]]|  for (
    x = 1;                               // CHECK: 100| [[@LINE]]|    x = 1
  }                                      // CHECK: 100| [[@LINE]]|  }
                                         // CHECK:   1| [[@LINE]]|
  x = x < 10 ? x + 1 : x - 1;            // CHECK:   1| [[@LINE]]|  x =
  x = x > 10 ?                           // CHECK:   1| [[@LINE]]|  x =
        x - 1:                           // CHECK:   0| [[@LINE]]|        x
        x + 1;                           // CHECK:   1| [[@LINE]]|        x
                                         // CHECK:   1| [[@LINE]]|
  return 0;                              // CHECK:   1| [[@LINE]]|  return
}                                        // CHECK:   1| [[@LINE]]|}
// after coverage                   // WHOLE-FILE:    | [[@LINE]]|// after
                                    // FILTER-NOT:    | [[@LINE-1]]|// after

// llvm-cov doesn't work on big endian yet
// XFAIL: powerpc64-, s390x, mips-, mips64-, sparc

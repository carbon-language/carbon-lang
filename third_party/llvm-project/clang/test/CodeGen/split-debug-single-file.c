// REQUIRES: x86-registered-target

// Testing to ensure that setting only -split-dwarf-file allows to place .dwo sections into regular output object.
//  RUN: %clang_cc1 -debug-info-kind=limited -triple x86_64-unknown-linux \
//  RUN:   -split-dwarf-file %t.o -emit-obj -o %t.o %s
//  RUN: llvm-readobj -S %t.o | FileCheck --check-prefix=MODE-SINGLE %s
//  MODE-SINGLE: .dwo

// Testing to ensure that setting both -split-dwarf-file and -split-dwarf-output
// does not place .dwo sections into regular output object.
//  RUN: %clang_cc1 -debug-info-kind=limited -triple x86_64-unknown-linux \
//  RUN:   -split-dwarf-file %t.dwo -split-dwarf-output %t.dwo -emit-obj -o %t.o %s
//  RUN: llvm-readobj -S %t.o | FileCheck --check-prefix=MODE-SPLIT %s
//  MODE-SPLIT-NOT: .dwo

int main (void) {
  return 0;
}

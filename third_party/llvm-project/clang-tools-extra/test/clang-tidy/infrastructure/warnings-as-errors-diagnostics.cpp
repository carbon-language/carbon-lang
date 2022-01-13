// RUN: clang-tidy %s -checks='-*,llvm-namespace-comment,clang-diagnostic*' \
// RUN:   -- -Wunused-variable 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WARN -implicit-check-not='{{warning|error}}:'
// RUN: not clang-tidy %s -checks='-*,llvm-namespace-comment,clang-diagnostic*' \
// RUN:   -warnings-as-errors='clang-diagnostic*' -- -Wunused-variable 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WERR -implicit-check-not='{{warning|error}}:'
// RUN: not clang-tidy %s -checks='-*,llvm-namespace-comment,clang-diagnostic*' \
// RUN:   -warnings-as-errors='clang-diagnostic*' -quiet -- -Wunused-variable 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WERR-QUIET -implicit-check-not='{{warning|error}}:'

void f() { int i; }
// CHECK-WARN: warning: unused variable 'i' [clang-diagnostic-unused-variable]
// CHECK-WERR: error: unused variable 'i' [clang-diagnostic-unused-variable,-warnings-as-errors]
// CHECK-WERR-QUIET: error: unused variable 'i' [clang-diagnostic-unused-variable,-warnings-as-errors]

// CHECK-WARN-NOT: treated as
// CHECK-WERR: 1 warning treated as error
// CHECK-WERR-QUIET-NOT: treated as

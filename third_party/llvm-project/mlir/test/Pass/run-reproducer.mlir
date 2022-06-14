// configuration: -mlir-disable-threading=true -pass-pipeline='func.func(cse,canonicalize)' -mlir-print-ir-before=cse

// Test of the reproducer run option. The first line has to be the
// configuration (matching what is produced by reproducer).

// RUN: mlir-opt %s -run-reproducer 2>&1 | FileCheck -check-prefix=BEFORE %s

func.func @foo() {
  %0 = arith.constant 0 : i32
  return
}

func.func @bar() {
  return
}

// BEFORE: // -----// IR Dump Before{{.*}}CSE //----- //
// BEFORE-NEXT: func @foo()
// BEFORE: // -----// IR Dump Before{{.*}}CSE //----- //
// BEFORE-NEXT: func @bar()
// BEFORE-NOT: // -----// IR Dump Before{{.*}}Canonicalizer //----- //
// BEFORE-NOT: // -----// IR Dump After

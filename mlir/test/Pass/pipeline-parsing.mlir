// RUN: mlir-opt %s -mlir-disable-threading -pass-pipeline='builtin.module(test-module-pass,builtin.func(test-function-pass)),builtin.func(test-function-pass)' -pass-pipeline="builtin.func(cse,canonicalize)" -verify-each=false -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck %s
// RUN: mlir-opt %s -mlir-disable-threading -test-textual-pm-nested-pipeline -verify-each=false -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck %s --check-prefix=TEXTUAL_CHECK
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(test-module-pass' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_1 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(test-module-pass))' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_2 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module()(' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_3 %s
// RUN: not mlir-opt %s -pass-pipeline=',' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_4 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.func(test-module-pass)' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_5 %s

// CHECK_ERROR_1: encountered unbalanced parentheses while parsing pipeline
// CHECK_ERROR_2: encountered extra closing ')' creating unbalanced parentheses while parsing pipeline
// CHECK_ERROR_3: expected ',' after parsing pipeline
// CHECK_ERROR_4: does not refer to a registered pass or pass pipeline
// CHECK_ERROR_5:  Can't add pass '{{.*}}TestModulePass' restricted to 'builtin.module' on a PassManager intended to run on 'builtin.func', did you intend to nest?
func @foo() {
  return
}

module {
  func @foo() {
    return
  }
}

// CHECK: Pipeline Collection : ['builtin.func', 'builtin.module']
// CHECK-NEXT:   'builtin.func' Pipeline
// CHECK-NEXT:     TestFunctionPass
// CHECK-NEXT:     CSE
// CHECK-NEXT:       DominanceInfo
// CHECK-NEXT:     Canonicalizer
// CHECK-NEXT:   'builtin.module' Pipeline
// CHECK-NEXT:     TestModulePass
// CHECK-NEXT:     'builtin.func' Pipeline
// CHECK-NEXT:       TestFunctionPass

// TEXTUAL_CHECK: Pipeline Collection : ['builtin.func', 'builtin.module']
// TEXTUAL_CHECK-NEXT:   'builtin.func' Pipeline
// TEXTUAL_CHECK-NEXT:     TestFunctionPass
// TEXTUAL_CHECK-NEXT:   'builtin.module' Pipeline
// TEXTUAL_CHECK-NEXT:     TestModulePass
// TEXTUAL_CHECK-NEXT:     'builtin.func' Pipeline
// TEXTUAL_CHECK-NEXT:       TestFunctionPass

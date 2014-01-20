; This test is essentially doing very basic things with the opt tool and the
; new pass manager pipeline. It will be used to flesh out the feature
; completeness of the opt tool when the new pass manager is engaged. The tests
; may not be useful once it becomes the default or may get spread out into other
; files, but for now this is just going to step the new process through its
; paces.

; RUN: opt -disable-output -debug-pass-manager -passes=print %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-PRINT
; CHECK-MODULE-PRINT: Starting module pass manager
; CHECK-MODULE-PRINT: Running module pass: VerifierPass
; CHECK-MODULE-PRINT: Running module pass: PrintModulePass
; CHECK-MODULE-PRINT: ModuleID
; CHECK-MODULE-PRINT: define void @foo()
; CHECK-MODULE-PRINT: Running module pass: VerifierPass
; CHECK-MODULE-PRINT: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager -passes='function(print)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PRINT
; CHECK-FUNCTION-PRINT: Starting module pass manager
; CHECK-FUNCTION-PRINT: Running module pass: VerifierPass
; CHECK-FUNCTION-PRINT: Starting function pass manager
; CHECK-FUNCTION-PRINT: Running function pass: PrintFunctionPass
; CHECK-FUNCTION-PRINT-NOT: ModuleID
; CHECK-FUNCTION-PRINT: define void @foo()
; CHECK-FUNCTION-PRINT: Finished function pass manager
; CHECK-FUNCTION-PRINT: Running module pass: VerifierPass
; CHECK-FUNCTION-PRINT: Finished module pass manager

; RUN: opt -S -o - -passes='no-op-module,no-op-module' %s \
; RUN:     | FileCheck %s --check-prefix=CHECK-NOOP
; CHECK-NOOP: define void @foo() {
; CHECK-NOOP:   ret void
; CHECK-NOOP: }

; Round trip through bitcode.
; RUN: opt -f -o - -passes='no-op-module,no-op-module' %s \
; RUN:     | llvm-dis \
; RUN:     | FileCheck %s --check-prefix=CHECK-NOOP

; RUN: opt -disable-output -debug-pass-manager -verify-each -passes='no-op-module,function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-VERIFY-EACH
; CHECK-VERIFY-EACH: Starting module pass manager
; CHECK-VERIFY-EACH: Running module pass: VerifierPass
; CHECK-VERIFY-EACH: Running module pass: NoOpModulePass
; CHECK-VERIFY-EACH: Running module pass: VerifierPass
; CHECK-VERIFY-EACH: Starting function pass manager
; CHECK-VERIFY-EACH: Running function pass: NoOpFunctionPass
; CHECK-VERIFY-EACH: Running function pass: VerifierPass
; CHECK-VERIFY-EACH: Finished function pass manager
; CHECK-VERIFY-EACH: Running module pass: VerifierPass
; CHECK-VERIFY-EACH: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager -disable-verify -passes='no-op-module,function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NO-VERIFY
; CHECK-NO-VERIFY: Starting module pass manager
; CHECK-NO-VERIFY-NOT: VerifierPass
; CHECK-NO-VERIFY: Running module pass: NoOpModulePass
; CHECK-NO-VERIFY-NOT: VerifierPass
; CHECK-NO-VERIFY: Starting function pass manager
; CHECK-NO-VERIFY: Running function pass: NoOpFunctionPass
; CHECK-NO-VERIFY-NOT: VerifierPass
; CHECK-NO-VERIFY: Finished function pass manager
; CHECK-NO-VERIFY-NOT: VerifierPass
; CHECK-NO-VERIFY: Finished module pass manager

define void @foo() {
  ret void
}

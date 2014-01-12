; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes=no-op-module,no-op-module %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-MP
; CHECK-TWO-NOOP-MP: Starting module pass manager
; CHECK-TWO-NOOP-MP: Running module pass: NoOpModulePass
; CHECK-TWO-NOOP-MP: Running module pass: NoOpModulePass
; CHECK-TWO-NOOP-MP: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(no-op-module,no-op-module)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-TWO-NOOP-MP
; CHECK-NESTED-TWO-NOOP-MP: Starting module pass manager
; CHECK-NESTED-TWO-NOOP-MP: Running module pass: ModulePassManager
; CHECK-NESTED-TWO-NOOP-MP: Starting module pass manager
; CHECK-NESTED-TWO-NOOP-MP: Running module pass: NoOpModulePass
; CHECK-NESTED-TWO-NOOP-MP: Running module pass: NoOpModulePass
; CHECK-NESTED-TWO-NOOP-MP: Finished module pass manager
; CHECK-NESTED-TWO-NOOP-MP: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes=no-op-function,no-op-function %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-FP
; CHECK-TWO-NOOP-FP: Starting module pass manager
; CHECK-TWO-NOOP-FP: Running module pass: ModuleToFunctionPassAdaptor
; CHECK-TWO-NOOP-FP: Starting function pass manager
; CHECK-TWO-NOOP-FP: Running function pass: NoOpFunctionPass
; CHECK-TWO-NOOP-FP: Running function pass: NoOpFunctionPass
; CHECK-TWO-NOOP-FP: Finished function pass manager
; CHECK-TWO-NOOP-FP: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function,no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-TWO-NOOP-FP
; CHECK-NESTED-TWO-NOOP-FP: Starting module pass manager
; CHECK-NESTED-TWO-NOOP-FP: Running module pass: ModuleToFunctionPassAdaptor
; CHECK-NESTED-TWO-NOOP-FP: Starting function pass manager
; CHECK-NESTED-TWO-NOOP-FP: Running function pass: FunctionPassManager
; CHECK-NESTED-TWO-NOOP-FP: Starting function pass manager
; CHECK-NESTED-TWO-NOOP-FP: Running function pass: NoOpFunctionPass
; CHECK-NESTED-TWO-NOOP-FP: Running function pass: NoOpFunctionPass
; CHECK-NESTED-TWO-NOOP-FP: Finished function pass manager
; CHECK-NESTED-TWO-NOOP-FP: Finished function pass manager
; CHECK-NESTED-TWO-NOOP-FP: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module,function(no-op-function,no-op-function),no-op-module' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MIXED-FP-AND-MP
; CHECK-MIXED-FP-AND-MP: Starting module pass manager
; CHECK-MIXED-FP-AND-MP: Running module pass: NoOpModulePass
; CHECK-MIXED-FP-AND-MP: Running module pass: ModuleToFunctionPassAdaptor
; CHECK-MIXED-FP-AND-MP: Starting function pass manager
; CHECK-MIXED-FP-AND-MP: Running function pass: NoOpFunctionPass
; CHECK-MIXED-FP-AND-MP: Running function pass: NoOpFunctionPass
; CHECK-MIXED-FP-AND-MP: Finished function pass manager
; CHECK-MIXED-FP-AND-MP: Running module pass: NoOpModulePass
; CHECK-MIXED-FP-AND-MP: Finished module pass manager

define void @f() {
 ret void
}

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

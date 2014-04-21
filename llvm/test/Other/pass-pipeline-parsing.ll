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

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED1
; CHECK-UNBALANCED1: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(no-op-module))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED2
; CHECK-UNBALANCED2: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(no-op-module' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED3
; CHECK-UNBALANCED3: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED4
; CHECK-UNBALANCED4: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED5
; CHECK-UNBALANCED5: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(function(no-op-function)))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED6
; CHECK-UNBALANCED6: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED7
; CHECK-UNBALANCED7: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED8
; CHECK-UNBALANCED8: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module,)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED9
; CHECK-UNBALANCED9: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function,)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED10
; CHECK-UNBALANCED10: unable to parse pass pipeline description

; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes=no-op-cgscc,no-op-cgscc %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-CG
; CHECK-TWO-NOOP-CG: Starting module pass manager
; CHECK-TWO-NOOP-CG: Running module pass: ModuleToPostOrderCGSCCPassAdaptor
; CHECK-TWO-NOOP-CG: Starting CGSCC pass manager
; CHECK-TWO-NOOP-CG: Running CGSCC pass: NoOpCGSCCPass
; CHECK-TWO-NOOP-CG: Running CGSCC pass: NoOpCGSCCPass
; CHECK-TWO-NOOP-CG: Finished CGSCC pass manager
; CHECK-TWO-NOOP-CG: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='module(function(no-op-function),cgscc(no-op-cgscc,function(no-op-function),no-op-cgscc),function(no-op-function))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-MP-CG-FP
; CHECK-NESTED-MP-CG-FP: Starting module pass manager
; CHECK-NESTED-MP-CG-FP: Starting module pass manager
; CHECK-NESTED-MP-CG-FP: Running module pass: ModuleToFunctionPassAdaptor
; CHECK-NESTED-MP-CG-FP: Starting function pass manager
; CHECK-NESTED-MP-CG-FP: Running function pass: NoOpFunctionPass
; CHECK-NESTED-MP-CG-FP: Finished function pass manager
; CHECK-NESTED-MP-CG-FP: Running module pass: ModuleToPostOrderCGSCCPassAdaptor
; CHECK-NESTED-MP-CG-FP: Starting CGSCC pass manager
; CHECK-NESTED-MP-CG-FP: Running CGSCC pass: NoOpCGSCCPass
; CHECK-NESTED-MP-CG-FP: Running CGSCC pass: CGSCCToFunctionPassAdaptor
; CHECK-NESTED-MP-CG-FP: Starting function pass manager
; CHECK-NESTED-MP-CG-FP: Running function pass: NoOpFunctionPass
; CHECK-NESTED-MP-CG-FP: Finished function pass manager
; CHECK-NESTED-MP-CG-FP: Running CGSCC pass: NoOpCGSCCPass
; CHECK-NESTED-MP-CG-FP: Finished CGSCC pass manager
; CHECK-NESTED-MP-CG-FP: Running module pass: ModuleToFunctionPassAdaptor
; CHECK-NESTED-MP-CG-FP: Starting function pass manager
; CHECK-NESTED-MP-CG-FP: Running function pass: NoOpFunctionPass
; CHECK-NESTED-MP-CG-FP: Finished function pass manager
; CHECK-NESTED-MP-CG-FP: Finished module pass manager
; CHECK-NESTED-MP-CG-FP: Finished module pass manager

define void @f() {
 ret void
}

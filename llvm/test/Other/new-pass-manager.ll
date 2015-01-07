; This test is essentially doing very basic things with the opt tool and the
; new pass manager pipeline. It will be used to flesh out the feature
; completeness of the opt tool when the new pass manager is engaged. The tests
; may not be useful once it becomes the default or may get spread out into other
; files, but for now this is just going to step the new process through its
; paces.

; RUN: opt -disable-output -disable-verify -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes=no-op-module %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-PASS
; CHECK-MODULE-PASS: Starting module pass manager
; CHECK-MODULE-PASS-NEXT: Running module pass: NoOpModulePass
; CHECK-MODULE-PASS-NEXT: Finished module pass manager run.

; RUN: opt -disable-output -disable-verify -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes=no-op-cgscc %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-CGSCC-PASS
; RUN: opt -disable-output -disable-verify -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='cgscc(no-op-cgscc)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-CGSCC-PASS
; CHECK-CGSCC-PASS: Starting module pass manager
; CHECK-CGSCC-PASS-NEXT: Running module pass: ModuleToPostOrderCGSCCPassAdaptor
; CHECK-CGSCC-PASS-NEXT: Running module analysis: CGSCCAnalysisManagerModuleProxy
; CHECK-CGSCC-PASS-NEXT: Running module analysis: Lazy CallGraph Analysis
; CHECK-CGSCC-PASS-NEXT: Starting CGSCC pass manager run.
; CHECK-CGSCC-PASS-NEXT: Running CGSCC pass: NoOpCGSCCPass
; CHECK-CGSCC-PASS-NEXT: Finished CGSCC pass manager run.
; CHECK-CGSCC-PASS-NEXT: Finished module pass manager run.

; RUN: opt -disable-output -disable-verify -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes=no-op-function %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PASS
; RUN: opt -disable-output -disable-verify -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PASS
; CHECK-FUNCTION-PASS: Starting module pass manager
; CHECK-FUNCTION-PASS-NEXT: Running module pass: ModuleToFunctionPassAdaptor
; CHECK-FUNCTION-PASS-NEXT: Running module analysis: FunctionAnalysisManagerModuleProxy
; CHECK-FUNCTION-PASS-NEXT: Starting function pass manager run.
; CHECK-FUNCTION-PASS-NEXT: Running function pass: NoOpFunctionPass
; CHECK-FUNCTION-PASS-NEXT: Finished function pass manager run.
; CHECK-FUNCTION-PASS-NEXT: Finished module pass manager run.

; RUN: opt -disable-output -debug-pass-manager -passes=print %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-PRINT
; CHECK-MODULE-PRINT: Starting module pass manager
; CHECK-MODULE-PRINT: Running module pass: VerifierPass
; CHECK-MODULE-PRINT: Running module pass: PrintModulePass
; CHECK-MODULE-PRINT: ModuleID
; CHECK-MODULE-PRINT: define void @foo()
; CHECK-MODULE-PRINT: Running module pass: VerifierPass
; CHECK-MODULE-PRINT: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager -disable-verify -passes='print,verify' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-VERIFY
; CHECK-MODULE-VERIFY: Starting module pass manager
; CHECK-MODULE-VERIFY: Running module pass: PrintModulePass
; CHECK-MODULE-VERIFY: ModuleID
; CHECK-MODULE-VERIFY: define void @foo()
; CHECK-MODULE-VERIFY: Running module pass: VerifierPass
; CHECK-MODULE-VERIFY: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager -passes='function(print)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PRINT
; CHECK-FUNCTION-PRINT: Starting module pass manager
; CHECK-FUNCTION-PRINT: Running module pass: VerifierPass
; CHECK-FUNCTION-PRINT: Running module pass: ModuleToFunctionPassAdaptor
; CHECK-FUNCTION-PRINT: Running module analysis: FunctionAnalysisManagerModuleProxy
; CHECK-FUNCTION-PRINT: Starting function pass manager
; CHECK-FUNCTION-PRINT: Running function pass: PrintFunctionPass
; CHECK-FUNCTION-PRINT-NOT: ModuleID
; CHECK-FUNCTION-PRINT: define void @foo()
; CHECK-FUNCTION-PRINT: Finished function pass manager
; CHECK-FUNCTION-PRINT: Running module pass: VerifierPass
; CHECK-FUNCTION-PRINT: Finished module pass manager

; RUN: opt -disable-output -debug-pass-manager -disable-verify -passes='function(print,verify)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-VERIFY
; CHECK-FUNCTION-VERIFY: Starting module pass manager
; CHECK-FUNCTION-VERIFY: Starting function pass manager
; CHECK-FUNCTION-VERIFY: Running function pass: PrintFunctionPass
; CHECK-FUNCTION-VERIFY-NOT: ModuleID
; CHECK-FUNCTION-VERIFY: define void @foo()
; CHECK-FUNCTION-VERIFY: Running function pass: VerifierPass
; CHECK-FUNCTION-VERIFY: Finished function pass manager
; CHECK-FUNCTION-VERIFY: Finished module pass manager

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

; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='require<no-op-module>,cgscc(require<no-op-cgscc>,function(require<no-op-function>))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-ANALYSES
; CHECK-ANALYSES: Starting module pass manager
; CHECK-ANALYSES: Running module pass: RequireAnalysisPass
; CHECK-ANALYSES: Running module analysis: NoOpModuleAnalysis
; CHECK-ANALYSES: Starting CGSCC pass manager
; CHECK-ANALYSES: Running CGSCC pass: RequireAnalysisPass
; CHECK-ANALYSES: Running CGSCC analysis: NoOpCGSCCAnalysis
; CHECK-ANALYSES: Starting function pass manager
; CHECK-ANALYSES: Running function pass: RequireAnalysisPass
; CHECK-ANALYSES: Running function analysis: NoOpFunctionAnalysis

; Make sure no-op passes that preserve all analyses don't even try to do any
; analysis invalidation.
; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='require<no-op-module>,cgscc(require<no-op-cgscc>,function(require<no-op-function>))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NO-OP-INVALIDATION
; CHECK-NO-OP-INVALIDATION: Starting module pass manager
; CHECK-NO-OP-INVALIDATION-NOT: Invalidating all non-preserved analyses

; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='require<no-op-module>,require<no-op-module>,require<no-op-module>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS
; CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS: Starting module pass manager
; CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS: Running module pass: RequireAnalysisPass
; CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS: Running module analysis: NoOpModuleAnalysis
; CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS-NOT: Running module analysis: NoOpModuleAnalysis

; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='require<no-op-module>,invalidate<no-op-module>,require<no-op-module>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Starting module pass manager
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Running module pass: RequireAnalysisPass
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Running module analysis: NoOpModuleAnalysis
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Invalidating module analysis: NoOpModuleAnalysis
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Running module analysis: NoOpModuleAnalysis

; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='cgscc(require<no-op-cgscc>,require<no-op-cgscc>,require<no-op-cgscc>)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS
; CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS: Starting CGSCC pass manager
; CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS: Running CGSCC pass: RequireAnalysisPass
; CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS: Running CGSCC analysis: NoOpCGSCCAnalysis
; CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS-NOT: Running CGSCC analysis: NoOpCGSCCAnalysis

; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='cgscc(require<no-op-cgscc>,invalidate<no-op-cgscc>,require<no-op-cgscc>)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Starting CGSCC pass manager
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Running CGSCC pass: RequireAnalysisPass
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Running CGSCC analysis: NoOpCGSCCAnalysis
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Invalidating CGSCC analysis: NoOpCGSCCAnalysis
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Running CGSCC analysis: NoOpCGSCCAnalysis

; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='function(require<no-op-function>,require<no-op-function>,require<no-op-function>)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS
; CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS: Starting function pass manager
; CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS: Running function pass: RequireAnalysisPass
; CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS: Running function analysis: NoOpFunctionAnalysis
; CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS-NOT: Running function analysis: NoOpFunctionAnalysis

; RUN: opt -disable-output -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='function(require<no-op-function>,invalidate<no-op-function>,require<no-op-function>)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Starting function pass manager
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Running function pass: RequireAnalysisPass
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Running function analysis: NoOpFunctionAnalysis
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Invalidating function analysis: NoOpFunctionAnalysis
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Running function analysis: NoOpFunctionAnalysis

; RUN: opt -disable-output -disable-verify -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='require<no-op-module>,module(require<no-op-module>,function(require<no-op-function>,invalidate<all>,require<no-op-function>),require<no-op-module>),require<no-op-module>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-INVALIDATE-ALL
; CHECK-INVALIDATE-ALL: Starting module pass manager run.
; CHECK-INVALIDATE-ALL: Running module pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL: Running module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Starting module pass manager run.
; CHECK-INVALIDATE-ALL: Running module pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-NOT: Running module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Starting function pass manager run.
; CHECK-INVALIDATE-ALL: Running function pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL: Running function analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL: Running function pass: InvalidateAllAnalysesPass
; CHECK-INVALIDATE-ALL: Invalidating all non-preserved analyses for function
; CHECK-INVALIDATE-ALL: Invalidating function analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL: Running function pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL: Running function analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL: Finished function pass manager run.
; CHECK-INVALIDATE-ALL: Invalidating all non-preserved analyses for function
; CHECK-INVALIDATE-ALL-NOT: Running function analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL: Invalidating all non-preserved analyses for module
; CHECK-INVALIDATE-ALL: Invalidating module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Running module pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL: Running module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Finished module pass manager run.
; CHECK-INVALIDATE-ALL: Invalidating all non-preserved analyses for module
; CHECK-INVALIDATE-ALL-NOT: Invalidating module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Running module pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-NOT: Running module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Finished module pass manager run.

; RUN: opt -disable-output -disable-verify -debug-pass-manager -debug-cgscc-pass-manager \
; RUN:     -passes='require<no-op-module>,module(require<no-op-module>,cgscc(require<no-op-cgscc>,function(require<no-op-function>,invalidate<all>,require<no-op-function>),require<no-op-cgscc>),require<no-op-module>),require<no-op-module>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-INVALIDATE-ALL-CG
; CHECK-INVALIDATE-ALL-CG: Starting module pass manager run.
; CHECK-INVALIDATE-ALL-CG: Running module pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Starting module pass manager run.
; CHECK-INVALIDATE-ALL-CG: Running module pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG-NOT: Running module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Starting CGSCC pass manager run.
; CHECK-INVALIDATE-ALL-CG: Running CGSCC pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running CGSCC analysis: NoOpCGSCCAnalysis
; CHECK-INVALIDATE-ALL-CG: Starting function pass manager run.
; CHECK-INVALIDATE-ALL-CG: Running function pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running function analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL-CG: Running function pass: InvalidateAllAnalysesPass
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses for function
; CHECK-INVALIDATE-ALL-CG: Invalidating function analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL-CG: Running function pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running function analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL-CG: Finished function pass manager run.
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses for function
; CHECK-INVALIDATE-ALL-CG-NOT: Running function analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses for SCC
; CHECK-INVALIDATE-ALL-CG: Invalidating CGSCC analysis: NoOpCGSCCAnalysis
; CHECK-INVALIDATE-ALL-CG: Running CGSCC pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running CGSCC analysis: NoOpCGSCCAnalysis
; CHECK-INVALIDATE-ALL-CG: Finished CGSCC pass manager run.
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses for SCC
; CHECK-INVALIDATE-ALL-CG-NOT: Invalidating CGSCC analysis: NoOpCGSCCAnalysis
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses for module
; CHECK-INVALIDATE-ALL-CG: Invalidating module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Running module pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Finished module pass manager run.
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses for module
; CHECK-INVALIDATE-ALL-CG-NOT: Invalidating module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Running module pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG-NOT: Running module analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Finished module pass manager run.

define void @foo() {
  ret void
}

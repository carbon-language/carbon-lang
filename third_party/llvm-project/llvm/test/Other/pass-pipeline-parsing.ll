; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes=no-op-module,no-op-module %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-MP
; CHECK-TWO-NOOP-MP: Running pass: NoOpModulePass
; CHECK-TWO-NOOP-MP: Running pass: NoOpModulePass

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(no-op-module,no-op-module)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-TWO-NOOP-MP
; CHECK-NESTED-TWO-NOOP-MP: Running pass: NoOpModulePass
; CHECK-NESTED-TWO-NOOP-MP: Running pass: NoOpModulePass

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes=no-op-function,no-op-function %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-FP
; CHECK-TWO-NOOP-FP: Running pass: NoOpFunctionPass
; CHECK-TWO-NOOP-FP: Running pass: NoOpFunctionPass

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function,no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-TWO-NOOP-FP
; CHECK-NESTED-TWO-NOOP-FP: Running pass: NoOpFunctionPass
; CHECK-NESTED-TWO-NOOP-FP: Running pass: NoOpFunctionPass

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module,function(no-op-function,no-op-function),no-op-module' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MIXED-FP-AND-MP
; CHECK-MIXED-FP-AND-MP: Running pass: NoOpModulePass
; CHECK-MIXED-FP-AND-MP: Running pass: NoOpFunctionPass
; CHECK-MIXED-FP-AND-MP: Running pass: NoOpFunctionPass
; CHECK-MIXED-FP-AND-MP: Running pass: NoOpModulePass

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -aa-pipeline= -passes='require<aa>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-EMPTY-AA
; CHECK-EMPTY-AA: Running analysis: AAManager
; CHECK-EMPTY-AA-NOT: Running analysis: BasicAA

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -aa-pipeline=basic-aa -passes='require<aa>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-BASIC-AA
; CHECK-BASIC-AA: Running analysis: AAManager
; CHECK-BASIC-AA: Running analysis: BasicAA
; CHECK-BASIC-AA-NOT: Running analysis: TypeBasedAA

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -aa-pipeline=basic-aa,tbaa -passes='require<aa>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-AA
; CHECK-TWO-AA: Running analysis: AAManager
; CHECK-TWO-AA: Running analysis: BasicAA
; CHECK-TWO-AA: Running analysis: TypeBasedAA

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -aa-pipeline=default -passes='require<aa>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DEFAULT-AA
; CHECK-DEFAULT-AA: Running analysis: AAManager
; CHECK-DEFAULT-AA-DAG: Running analysis: BasicAA
; CHECK-DEFAULT-AA-DAG: Running analysis: TypeBasedAA

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED1
; CHECK-UNBALANCED1: invalid pipeline 'no-op-module)'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(no-op-module))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED2
; CHECK-UNBALANCED2: invalid pipeline 'module(no-op-module))'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(no-op-module' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED3
; CHECK-UNBALANCED3: invalid pipeline 'module(no-op-module'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED4
; CHECK-UNBALANCED4: invalid pipeline 'no-op-function)'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED5
; CHECK-UNBALANCED5: invalid pipeline 'function(no-op-function))'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(function(no-op-function)))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED6
; CHECK-UNBALANCED6: invalid pipeline 'function(function(no-op-function)))'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED7
; CHECK-UNBALANCED7: invalid pipeline 'function(no-op-function'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED8
; CHECK-UNBALANCED8: invalid pipeline 'function(function(no-op-function)'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module,)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED9
; CHECK-UNBALANCED9: invalid pipeline 'no-op-module,)'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function,)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED10
; CHECK-UNBALANCED10: invalid pipeline 'no-op-function,)'

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes=no-op-cgscc,no-op-cgscc %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-CG
; CHECK-TWO-NOOP-CG: Running pass: NoOpCGSCCPass
; CHECK-TWO-NOOP-CG: Running pass: NoOpCGSCCPass

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(function(no-op-function),cgscc(no-op-cgscc,function(no-op-function),no-op-cgscc),function(no-op-function))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-MP-CG-FP
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpFunctionPass
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpCGSCCPass
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpFunctionPass
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpCGSCCPass
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpFunctionPass

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-loop,no-op-loop' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-LOOP
; CHECK-TWO-NOOP-LOOP: Running pass: NoOpLoopPass
; CHECK-TWO-NOOP-LOOP: Running pass: NoOpLoopPass

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(function(loop(no-op-loop)))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-FP-LP
; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(loop(no-op-loop))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-FP-LP
; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='loop(no-op-loop)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-FP-LP
; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-loop' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-FP-LP
; CHECK-NESTED-FP-LP: Running pass: NoOpLoopPass

; RUN: opt -disable-output -debug-pass-manager=verbose \
; RUN:     -passes='module(no-op-function,no-op-loop,no-op-cgscc,cgscc(no-op-function,no-op-loop),function(no-op-loop))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-ADAPTORS
; CHECK-ADAPTORS: Running pass: ModuleToFunctionPassAdaptor
; CHECK-ADAPTORS: Running pass: NoOpFunctionPass
; CHECK-ADAPTORS: Running pass: ModuleToFunctionPassAdaptor
; CHECK-ADAPTORS: Running pass: FunctionToLoopPassAdaptor
; CHECK-ADAPTORS: Running pass: NoOpLoopPass on Loop at depth 1 containing: %loop
; CHECK-ADAPTORS: Running pass: ModuleToPostOrderCGSCCPassAdaptor
; CHECK-ADAPTORS: Running pass: NoOpCGSCCPass
; CHECK-ADAPTORS: Running pass: ModuleToPostOrderCGSCCPassAdaptor
; CHECK-ADAPTORS: Running pass: PassManager{{.*}}SCC
; CHECK-ADAPTORS: Running pass: CGSCCToFunctionPassAdaptor
; CHECK-ADAPTORS: Running pass: NoOpFunctionPass
; CHECK-ADAPTORS: Running pass: CGSCCToFunctionPassAdaptor
; CHECK-ADAPTORS: Running pass: FunctionToLoopPassAdaptor
; CHECK-ADAPTORS: Running pass: NoOpLoopPass on Loop at depth 1 containing: %loop
; CHECK-ADAPTORS: Running pass: ModuleToFunctionPassAdaptor
; CHECK-ADAPTORS: Running pass: PassManager{{.*}}Function
; CHECK-ADAPTORS: Running pass: FunctionToLoopPassAdaptor
; CHECK-ADAPTORS: Running pass: NoOpLoopPass on Loop at depth 1 containing: %loop

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(function(no-op-function,loop(no-op-loop,no-op-loop)))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MANAGERS-NO-VERBOSE
; RUN: opt -disable-output -debug-pass-manager=verbose \
; RUN:     -passes='module(function(no-op-function,loop(no-op-loop,no-op-loop)))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MANAGERS
; CHECK-MANAGERS: Running pass: PassManager{{.*}}Function
; CHECK-MANAGERS: Running pass: PassManager{{.*}}Loop
; CHECK-MANAGERS-NO-VERBOSE-NOT: PassManager

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='cgscc(print)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-PRINT-IN-CGSCC
; CHECK-PRINT-IN-CGSCC: Running pass: PrintFunctionPass
; CHECK-PRINT-IN-CGSCC: Running pass: VerifierPass

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function)function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MISSING-COMMA1
; CHECK-MISSING-COMMA1: invalid pipeline 'function(no-op-function)function(no-op-function)'

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function()' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-EMPTY-INNER-PIPELINE
; CHECK-EMPTY-INNER-PIPELINE: unknown function pass ''

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module(no-op-module,whatever)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-PIPELINE-ON-MODULE-PASS
; CHECK-PIPELINE-ON-MODULE-PASS: invalid use of 'no-op-module' pass as module pipeline

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-cgscc(no-op-cgscc,whatever)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-PIPELINE-ON-CGSCC-PASS
; CHECK-PIPELINE-ON-CGSCC-PASS: invalid use of 'no-op-cgscc' pass as cgscc pipeline

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function(no-op-function,whatever)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-PIPELINE-ON-FUNCTION-PASS
; CHECK-PIPELINE-ON-FUNCTION-PASS: invalid use of 'no-op-function' pass as function pipeline

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-loop(no-op-loop,whatever)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-PIPELINE-ON-LOOP-PASS
; CHECK-PIPELINE-ON-LOOP-PASS: invalid use of 'no-op-loop' pass as loop pipeline

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function()' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-EMPTY-PIPELINE-ON-PASS
; CHECK-EMPTY-PIPELINE-ON-PASS: invalid use of 'no-op-function' pass as function pipeline

; RUN: not opt -passes='no-op-module,bad' \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=CHECK-UNKNOWN-MODULE
; CHECK-UNKNOWN-MODULE: unknown module pass 'bad'

; RUN: not opt -passes='no-op-loop,bad' \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=CHECK-UNKNOWN-LOOP
; CHECK-UNKNOWN-LOOP: unknown loop pass 'bad'

; RUN: not opt -passes='no-op-cgscc,bad' \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=CHECK-UNKNOWN-CGSCC
; CHECK-UNKNOWN-CGSCC: unknown cgscc pass 'bad'

; RUN: not opt -passes='no-op-function,bad' \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=CHECK-UNKNOWN-FUNCTION
; RUN: not opt -passes='function(bad,pipeline,text)' \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=CHECK-UNKNOWN-FUNCTION
; RUN: not opt -passes='module(no-op-module,function(bad,pipeline,text))' \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=CHECK-UNKNOWN-FUNCTION
; RUN: not opt -passes='no-op-module,function(bad,pipeline,text)' \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=CHECK-UNKNOWN-FUNCTION
; RUN: not opt -passes='module(cgscc(function(bad,pipeline,text)))' \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=CHECK-UNKNOWN-FUNCTION
; CHECK-UNKNOWN-FUNCTION: unknown function pass 'bad'

; RUN: not opt -aa-pipeline=bad -passes=no-op-function \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=AA-PIPELINE-ERR
; AA-PIPELINE-ERR: unknown alias analysis name 'bad'
; RUN: opt -passes-ep-peephole=bad -passes=no-op-function \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=PASSES-EP-PEEPHOLE-ERR
; PASSES-EP-PEEPHOLE-ERR: Could not parse -passes-ep-peephole pipeline: unknown function pass 'bad'
; RUN: opt -passes-ep-late-loop-optimizations=bad -passes=no-op-function \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=PASSES-EP-LATELOOPOPT-ERR
; PASSES-EP-LATELOOPOPT-ERR: Could not parse -passes-ep-late-loop-optimizations pipeline: unknown loop pass 'bad'
; RUN: opt -passes-ep-loop-optimizer-end=bad -passes=no-op-function \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=PASSES-EP-LOOPOPTEND-ERR
; PASSES-EP-LOOPOPTEND-ERR: Could not parse -passes-ep-loop-optimizer-end pipeline: unknown loop pass 'bad'
; RUN: opt -passes-ep-scalar-optimizer-late=bad -passes=no-op-function \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=PASSES-EP-SCALAROPTLATE-ERR
; PASSES-EP-SCALAROPTLATE-ERR: Could not parse -passes-ep-scalar-optimizer-late pipeline: unknown function pass 'bad'
; RUN: opt -passes-ep-cgscc-optimizer-late=bad -passes=no-op-function \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=PASSES-EP-CGSCCOPTLATE-ERR
; PASSES-EP-CGSCCOPTLATE-ERR: Could not parse -passes-ep-cgscc-optimizer-late pipeline: unknown cgscc pass 'bad'
; RUN: opt -passes-ep-vectorizer-start=bad -passes=no-op-function \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=PASSES-EP-VECTORIZERSTART-ERR
; PASSES-EP-VECTORIZERSTART-ERR: Could not parse -passes-ep-vectorizer-start pipeline: unknown function pass 'bad'
; RUN: opt -passes-ep-pipeline-start=bad -passes=no-op-function \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=PASSES-EP-PIPELINESTART-ERR
; PASSES-EP-PIPELINESTART-ERR: Could not parse -passes-ep-pipeline-start pipeline: unknown pass name 'bad'
; RUN: opt -passes-ep-pipeline-early-simplification=bad -passes=no-op-function \
; RUN:       /dev/null -disable-output 2>&1 | FileCheck %s -check-prefix=PASSES-EP-PIPELINEEARLYSIMPLIFICATION-ERR
; PASSES-EP-PIPELINEEARLYSIMPLIFICATION-ERR: Could not parse -passes-ep-pipeline-early-simplification pipeline: unknown pass name 'bad'

define void @f() {
entry:
 br label %loop
loop:
 br label %loop
}

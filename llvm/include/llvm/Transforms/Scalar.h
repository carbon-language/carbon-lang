//===-- Scalar.h - Scalar Transformations -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Scalar transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_H
#define LLVM_TRANSFORMS_SCALAR_H

#include <cstdlib>

namespace llvm {

class ModulePass;
class FunctionPass;
class GetElementPtrInst;
class PassInfo;
class TerminatorInst;
class TargetLowering;

//===----------------------------------------------------------------------===//
//
// RaisePointerReferences - Try to eliminate as many pointer arithmetic
// expressions as possible, by converting expressions to use getelementptr and
// friends.
//
FunctionPass *createRaisePointerReferencesPass();

//===----------------------------------------------------------------------===//
//
// Constant Propagation Pass - A worklist driven constant propagation pass
//
FunctionPass *createConstantPropagationPass();


//===----------------------------------------------------------------------===//
//
// Sparse Conditional Constant Propagation Pass
//
FunctionPass *createSCCPPass();


//===----------------------------------------------------------------------===//
//
// DeadInstElimination - This pass quickly removes trivially dead instructions
// without modifying the CFG of the function.  It is a BasicBlockPass, so it
// runs efficiently when queued next to other BasicBlockPass's.
//
FunctionPass *createDeadInstEliminationPass();


//===----------------------------------------------------------------------===//
//
// DeadCodeElimination - This pass is more powerful than DeadInstElimination,
// because it is worklist driven that can potentially revisit instructions when
// their other instructions become dead, to eliminate chains of dead
// computations.
//
FunctionPass *createDeadCodeEliminationPass();

//===----------------------------------------------------------------------===//
//
// DeadStoreElimination - This pass deletes stores that are post-dominated by
// must-aliased stores and are not loaded used between the stores.
//
FunctionPass *createDeadStoreEliminationPass();

//===----------------------------------------------------------------------===//
//
// AggressiveDCE - This pass uses the SSA based Aggressive DCE algorithm.  This
// algorithm assumes instructions are dead until proven otherwise, which makes
// it more successful are removing non-obviously dead instructions.
//
FunctionPass *createAggressiveDCEPass();


//===----------------------------------------------------------------------===//
//
// Scalar Replacement of Aggregates - Break up alloca's of aggregates into
// multiple allocas if possible.
//
FunctionPass *createScalarReplAggregatesPass();


//===----------------------------------------------------------------------===//
//
// GCSE - This pass is designed to be a very quick global transformation that
// eliminates global common subexpressions from a function.  It does this by
// examining the SSA value graph of the function, instead of doing slow
// bit-vector computations.
//
FunctionPass *createGCSEPass();


//===----------------------------------------------------------------------===//
//
// InductionVariableSimplify - Transform induction variables in a program to all
// use a single canonical induction variable per loop.
//
FunctionPass *createIndVarSimplifyPass();


//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
//   instructions.  This pass does not modify the CFG, and has a tendency to
//   make instructions dead, so a subsequent DCE pass is useful.
//
// This pass combines things like:
//    %Y = add int 1, %X
//    %Z = add int 1, %Y
// into:
//    %Z = add int 2, %X
//
FunctionPass *createInstructionCombiningPass();


//===----------------------------------------------------------------------===//
//
// LICM - This pass is a loop invariant code motion and memory promotion pass.
//
FunctionPass *createLICMPass();

//===----------------------------------------------------------------------===//
//
// LoopStrengthReduce - This pass is strength reduces GEP instructions that use
// a loop's canonical induction variable as one of their indices.  It takes an
// optional parameter used to consult the target machine whether certain
// transformations are profitable.
//
FunctionPass *createLoopStrengthReducePass(const TargetLowering *TLI = NULL);

//===----------------------------------------------------------------------===//
//
// LoopUnswitch - This pass is a simple loop unswitching pass.
//
FunctionPass *createLoopUnswitchPass();


//===----------------------------------------------------------------------===//
//
// LoopUnroll - This pass is a simple loop unrolling pass.
//
FunctionPass *createLoopUnrollPass();

//===----------------------------------------------------------------------===//
//
// This pass is used to promote memory references to be register references.  A
// simple example of the transformation performed by this pass is:
//
//        FROM CODE                           TO CODE
//   %X = alloca int, uint 1                 ret int 42
//   store int 42, int *%X
//   %Y = load int* %X
//   ret int %Y
//
FunctionPass *createPromoteMemoryToRegisterPass();
extern const PassInfo *PromoteMemoryToRegisterID;

//===----------------------------------------------------------------------===//
//
// This pass is used to demote registers to memory references .
// In basically undoes the PromoteMemoryToRegister pass to
// make cfg hacking easier.
FunctionPass *createDemoteRegisterToMemoryPass();
extern const PassInfo *DemoteRegisterToMemoryID;

//===----------------------------------------------------------------------===//
//
// This pass reassociates commutative expressions in an order that is designed
// to promote better constant propagation, GCSE, LICM, PRE...
//
// For example:  4 + (x + 5)  ->  x + (4 + 5)
//
FunctionPass *createReassociatePass();

//===----------------------------------------------------------------------===//
//
// This pass eliminates correlated conditions, such as these:
//  if (X == 0)
//    if (X > 2) ;   // Known false
//    else
//      Y = X * Z;   // = 0
//
FunctionPass *createCorrelatedExpressionEliminationPass();


// createCondPropagationPass - This pass propagates information about
// conditional expressions through the program, allowing it to eliminate
// conditional branches in some cases.
//
FunctionPass *createCondPropagationPass();

//===----------------------------------------------------------------------===//
//
// TailDuplication - Eliminate unconditional branches through controlled code
// duplication, creating simpler CFG structures.
//
FunctionPass *createTailDuplicationPass();


//===----------------------------------------------------------------------===//
//
// CFG Simplification - Merge basic blocks, eliminate unreachable blocks,
// simplify terminator instructions, etc...
//
FunctionPass *createCFGSimplificationPass();


//===----------------------------------------------------------------------===//
//
// BreakCriticalEdges pass - Break all of the critical edges in the CFG by
// inserting a dummy basic block.  This pass may be "required" by passes that
// cannot deal with critical edges.  For this usage, a pass must call:
//
//   AU.addRequiredID(BreakCriticalEdgesID);
//
// This pass obviously invalidates the CFG, but can update forward dominator
// (set, immediate dominators, tree, and frontier) information.
//
FunctionPass *createBreakCriticalEdgesPass();
extern const PassInfo *BreakCriticalEdgesID;

//===----------------------------------------------------------------------===//
//
// LoopSimplify pass - Insert Pre-header blocks into the CFG for every function
// in the module.  This pass updates dominator information, loop information,
// and does not add critical edges to the CFG.
//
//   AU.addRequiredID(LoopSimplifyID);
//
FunctionPass *createLoopSimplifyPass();
extern const PassInfo *LoopSimplifyID;

//===----------------------------------------------------------------------===//
// This pass converts SelectInst instructions into conditional branch and PHI
// instructions.  If the OnlyFP flag is set to true, then only floating point
// select instructions are lowered.
//
FunctionPass *createLowerSelectPass(bool OnlyFP = false);
extern const PassInfo *LowerSelectID;

//===----------------------------------------------------------------------===//
//
// LowerAllocations Pass - Turn malloc and free instructions into %malloc and
// %free calls.
//
//   AU.addRequiredID(LowerAllocationsID);
//
FunctionPass *createLowerAllocationsPass(bool LowerMallocArgToInteger = false);
extern const PassInfo *LowerAllocationsID;

//===----------------------------------------------------------------------===//
//
// This pass eliminates call instructions to the current function which occur
// immediately before return instructions.
//
FunctionPass *createTailCallEliminationPass();

//===----------------------------------------------------------------------===//
// This pass converts SwitchInst instructions into a sequence of chained binary
// branch instructions.
//
FunctionPass *createLowerSwitchPass();
extern const PassInfo *LowerSwitchID;

//===----------------------------------------------------------------------===//
// This pass converts PackedType operations into low-level scalar operations.
//
FunctionPass *createLowerPackedPass();

//===----------------------------------------------------------------------===//
// This pass converts invoke and unwind instructions to use sjlj exception
// handling mechanisms.  Note that after this pass runs the CFG is not entirely
// accurate (exceptional control flow edges are not correct anymore) so only
// very simple things should be done after the lowerinvoke pass has run (like
// generation of native code).  This should *NOT* be used as a general purpose
// "my LLVM-to-LLVM pass doesn't support the invoke instruction yet" lowering
// pass.
//
FunctionPass *createLowerInvokePass(unsigned JumBufSize = 200, 
                                    unsigned JumpBufAlign = 0);
extern const PassInfo *LowerInvokePassID;


//===----------------------------------------------------------------------===//
/// createLowerGCPass - This function returns an instance of the "lowergc"
/// pass, which lowers garbage collection intrinsics to normal LLVM code.
///
FunctionPass *createLowerGCPass();

//===----------------------------------------------------------------------===//
// This pass reorders basic blocks in order to increase the number of fall-
// through conditional branches.
FunctionPass *createBlockPlacementPass();

//===----------------------------------------------------------------------===//
// This pass inserts phi nodes at loop boundaries to simplify other loop 
// optimizations.
FunctionPass *createLCSSAPass();
extern const PassInfo *LCSSAID;

//===----------------------------------------------------------------------===//
// This pass collapses duplicate variables into one canonical form,
// and tries to simplify expressions along the way.
FunctionPass *createPredicateSimplifierPass();

} // End llvm namespace

#endif

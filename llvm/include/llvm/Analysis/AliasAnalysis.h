//===- llvm/Analysis/AliasAnalysis.h - Alias Analysis Interface -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the generic AliasAnalysis interface, which is used as the
// common interface used by all clients of alias analysis information, and
// implemented by all alias analysis implementations.  Mod/Ref information is
// also captured by this interface.
//
// Implementations of this interface must implement the various virtual methods,
// which automatically provides functionality for the entire suite of client
// APIs.
//
// This API represents memory as a (Pointer, Size) pair.  The Pointer component
// specifies the base memory address of the region, the Size specifies how large
// of an area is being queried.  If Size is 0, two pointers only alias if they
// are exactly equal.  If size is greater than zero, but small, the two pointers
// alias if the areas pointed to overlap.  If the size is very large (ie, ~0U),
// then the two pointers alias if they may be pointing to components of the same
// memory object.  Pointers that point to two completely different objects in
// memory never alias, regardless of the value of the Size component.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIAS_ANALYSIS_H
#define LLVM_ANALYSIS_ALIAS_ANALYSIS_H

#include "llvm/Support/CallSite.h"
#include "llvm/Pass.h"    // Need this for IncludeFile

namespace llvm {

class LoadInst;
class StoreInst;
class TargetData;

class AliasAnalysis {
  const TargetData *TD;
protected:
  /// InitializeAliasAnalysis - Subclasses must call this method to initialize
  /// the AliasAnalysis interface before any other methods are called.  This is
  /// typically called by the run* methods of these subclasses.  This may be
  /// called multiple times.
  ///
  void InitializeAliasAnalysis(Pass *P);
  
  // getAnalysisUsage - All alias analysis implementations should invoke this
  // directly (using AliasAnalysis::getAnalysisUsage(AU)) to make sure that
  // TargetData is required by the pass.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

public:
  AliasAnalysis() : TD(0) {}
  virtual ~AliasAnalysis();  // We want to be subclassed

  /// getTargetData - Every alias analysis implementation depends on the size of
  /// data items in the current Target.  This provides a uniform way to handle
  /// it.
  ///
  const TargetData &getTargetData() const { return *TD; }

  //===--------------------------------------------------------------------===//
  /// Alias Queries...
  ///

  /// Alias analysis result - Either we know for sure that it does not alias, we
  /// know for sure it must alias, or we don't know anything: The two pointers
  /// _might_ alias.  This enum is designed so you can do things like:
  ///     if (AA.alias(P1, P2)) { ... }
  /// to check to see if two pointers might alias.
  ///
  enum AliasResult { NoAlias = 0, MayAlias = 1, MustAlias = 2 };

  /// alias - The main low level interface to the alias analysis implementation.
  /// Returns a Result indicating whether the two pointers are aliased to each
  /// other.  This is the interface that must be implemented by specific alias
  /// analysis implementations.
  ///
  virtual AliasResult alias(const Value *V1, unsigned V1Size,
                            const Value *V2, unsigned V2Size) {
    return MayAlias;
  }

  /// getMustAliases - If there are any pointers known that must alias this
  /// pointer, return them now.  This allows alias-set based alias analyses to
  /// perform a form a value numbering (which is exposed by load-vn).  If an
  /// alias analysis supports this, it should ADD any must aliased pointers to
  /// the specified vector.
  ///
  virtual void getMustAliases(Value *P, std::vector<Value*> &RetVals) {}

  /// pointsToConstantMemory - If the specified pointer is known to point into
  /// constant global memory, return true.  This allows disambiguation of store
  /// instructions from constant pointers.
  ///
  virtual bool pointsToConstantMemory(const Value *P) { return false; }

  /// doesNotAccessMemory - If the specified function is known to never read or
  /// write memory, return true.  If the function only reads from known-constant
  /// memory, it is also legal to return true.  Functions that unwind the stack
  /// are not legal for this predicate.
  ///
  /// Many optimizations (such as CSE and LICM) can be performed on calls to it,
  /// without worrying about aliasing properties, and many functions have this
  /// property (e.g. 'sin' and 'cos').
  ///
  /// This property corresponds to the GCC 'const' attribute.
  ///
  virtual bool doesNotAccessMemory(Function *F) { return false; }

  /// onlyReadsMemory - If the specified function is known to only read from
  /// non-volatile memory (or not access memory at all), return true.  Functions
  /// that unwind the stack are not legal for this predicate.
  ///
  /// This property allows many common optimizations to be performed in the
  /// absence of interfering store instructions, such as CSE of strlen calls.
  ///
  /// This property corresponds to the GCC 'pure' attribute.
  ///
  virtual bool onlyReadsMemory(Function *F) { return doesNotAccessMemory(F); }


  //===--------------------------------------------------------------------===//
  /// Simple mod/ref information...
  ///

  /// ModRefResult - Represent the result of a mod/ref query.  Mod and Ref are
  /// bits which may be or'd together.
  ///
  enum ModRefResult { NoModRef = 0, Ref = 1, Mod = 2, ModRef = 3 };

  /// getModRefInfo - Return information about whether or not an instruction may
  /// read or write memory specified by the pointer operand.  An instruction
  /// that doesn't read or write memory may be trivially LICM'd for example.

  /// getModRefInfo (for call sites) - Return whether information about whether
  /// a particular call site modifies or reads the memory specified by the
  /// pointer.
  ///
  virtual ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);

  /// getModRefInfo - Return information about whether two call sites may refer
  /// to the same set of memory locations.  This function returns NoModRef if
  /// the two calls refer to disjoint memory locations, Ref if they both read
  /// some of the same memory, Mod if they both write to some of the same
  /// memory, and ModRef if they read and write to the same memory.
  ///
  virtual ModRefResult getModRefInfo(CallSite CS1, CallSite CS2);

  /// Convenience functions...
  ModRefResult getModRefInfo(LoadInst *L, Value *P, unsigned Size);
  ModRefResult getModRefInfo(StoreInst *S, Value *P, unsigned Size);
  ModRefResult getModRefInfo(CallInst *C, Value *P, unsigned Size) {
    return getModRefInfo(CallSite(C), P, Size);
  }
  ModRefResult getModRefInfo(InvokeInst *I, Value *P, unsigned Size) {
    return getModRefInfo(CallSite(I), P, Size);
  }
  ModRefResult getModRefInfo(Instruction *I, Value *P, unsigned Size) {
    switch (I->getOpcode()) {
    case Instruction::Load:   return getModRefInfo((LoadInst*)I, P, Size);
    case Instruction::Store:  return getModRefInfo((StoreInst*)I, P, Size);
    case Instruction::Call:   return getModRefInfo((CallInst*)I, P, Size);
    case Instruction::Invoke: return getModRefInfo((InvokeInst*)I, P, Size);
    default:                  return NoModRef;
    }
  }

  /// canBasicBlockModify - Return true if it is possible for execution of the
  /// specified basic block to modify the value pointed to by Ptr.
  ///
  bool canBasicBlockModify(const BasicBlock &BB, const Value *P, unsigned Size);

  /// canInstructionRangeModify - Return true if it is possible for the
  /// execution of the specified instructions to modify the value pointed to by
  /// Ptr.  The instructions to consider are all of the instructions in the
  /// range of [I1,I2] INCLUSIVE.  I1 and I2 must be in the same basic block.
  ///
  bool canInstructionRangeModify(const Instruction &I1, const Instruction &I2,
                                 const Value *Ptr, unsigned Size);
};

// Because of the way .a files work, we must force the BasicAA implementation to
// be pulled in if the AliasAnalysis header is included.  Otherwise we run
// the risk of AliasAnalysis being used, but the default implementation not
// being linked into the tool that uses it.
//
extern void BasicAAStub();
static IncludeFile HDR_INCLUDE_BASICAA_CPP((void*)&BasicAAStub);

} // End llvm namespace

#endif

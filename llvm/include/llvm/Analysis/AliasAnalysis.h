//===- llvm/Analysis/AliasAnalysis.h - Alias Analysis Interface -*- C++ -*-===//
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
class LoadInst;
class StoreInst;
class TargetData;
class AnalysisUsage;
class Pass;

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
  virtual ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size) {
    return ModRef;
  }

  /// getModRefInfo - Return information about whether two call sites may refer
  /// to the same set of memory locations.  This function returns NoModRef if
  /// the two calls refer to disjoint memory locations, Ref if they both read
  /// some of the same memory, Mod if they both write to some of the same
  /// memory, and ModRef if they read and write to the same memory.
  ///
  virtual ModRefResult getModRefInfo(CallSite CS1, CallSite CS2) {
    return ModRef;
  }

  /// Convenience functions...
  ModRefResult getModRefInfo(LoadInst *L, Value *P, unsigned Size);
  ModRefResult getModRefInfo(StoreInst*S, Value *P, unsigned Size);
  ModRefResult getModRefInfo(CallInst  *C, Value *P, unsigned Size) {
    return getModRefInfo(CallSite(C), P, Size);
  }
  ModRefResult getModRefInfo(InvokeInst*I, Value *P, unsigned Size) {
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

#endif

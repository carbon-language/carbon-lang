//===- llvm/Analysis/AliasAnalysis.h - Alias Analysis Interface -*- C++ -*-===//
//
// This file defines the generic AliasAnalysis interface, which is used as the
// common interface used by all clients of alias analysis information, and
// implemented by all alias analysis implementations.
//
// Implementations of this interface must implement the various virtual methods,
// which automatically provides functionality for the entire suite of client
// APIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIAS_ANALYSIS_H
#define LLVM_ANALYSIS_ALIAS_ANALYSIS_H

class Value;
class CallInst;
class InvokeInst;
class BasicBlock;
class Instruction;

struct AliasAnalysis {

  // Alias analysis result - Either we know for sure that it does not alias, we
  // know for sure it must alias, or we don't know anything: The two pointers
  // _might_ alias.  This enum is designed so you can do things like:
  //     if (AA.alias(P1, P2)) { ... }
  // to check to see if two pointers might alias.
  //
  enum Result { NoAlias = 0, MayAlias = 1, MustAlias = 2 };

  // alias - The main low level interface to the alias analysis implementation.
  // Returns a Result indicating whether the two pointers are aliased to each
  // other.  This is the interface that must be implemented by specific alias
  // analysis implementations.
  //
  virtual Result alias(const Value *V1, const Value *V2) const = 0;

  // canCallModify - Return a Result that indicates whether the specified
  // function call can modify the memory location pointed to by Ptr.
  //
  virtual Result canCallModify(const CallInst &CI, const Value *Ptr) const = 0;

  // canInvokeModify - Return a Result that indicates whether the specified
  // function invoke can modify the memory location pointed to by Ptr.
  //
  virtual Result canInvokeModify(const InvokeInst &I, const Value *Ptr) const=0;

  // canBasicBlockModify - Return true if it is possible for execution of the
  // specified basic block to modify the value pointed to by Ptr.
  //
  bool canBasicBlockModify(const BasicBlock &BB, const Value *Ptr) const;

  // canInstructionRangeModify - Return true if it is possible for the execution
  // of the specified instructions to modify the value pointed to by Ptr.  The
  // instructions to consider are all of the instructions in the range of
  // [I1,I2] INCLUSIVE.  I1 and I2 must be in the same basic block.
  //
  bool canInstructionRangeModify(const Instruction &I1, const Instruction &I2,
                                 const Value *Ptr) const;

  virtual ~AliasAnalysis();  // We want to be subclassed
};

#endif

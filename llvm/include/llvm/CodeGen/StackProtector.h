//===-- StackProtector.h - Stack Protector Insertion ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass inserts stack protectors into functions which need them. A variable
// with a random value in it is stored onto the stack before the local variables
// are allocated. Upon exiting the block, the stored value is checked. If it's
// changed, then there was some sort of violation and the program aborts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_STACKPROTECTOR_H
#define LLVM_CODEGEN_STACKPROTECTOR_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
class DominatorTree;
class Function;
class Module;
class PHINode;

class StackProtector : public FunctionPass {
  const TargetMachine *TM;

  /// TLI - Keep a pointer of a TargetLowering to consult for determining
  /// target type sizes.
  const TargetLoweringBase *TLI;
  const Triple Trip;

  Function *F;
  Module *M;

  DominatorTree *DT;

  /// \brief The minimum size of buffers that will receive stack smashing
  /// protection when -fstack-protection is used.
  unsigned SSPBufferSize;

  /// VisitedPHIs - The set of PHI nodes visited when determining
  /// if a variable's reference has been taken.  This set
  /// is maintained to ensure we don't visit the same PHI node multiple
  /// times.
  SmallPtrSet<const PHINode*, 16> VisitedPHIs;

  /// InsertStackProtectors - Insert code into the prologue and epilogue of
  /// the function.
  ///
  ///  - The prologue code loads and stores the stack guard onto the stack.
  ///  - The epilogue checks the value stored in the prologue against the
  ///    original value. It calls __stack_chk_fail if they differ.
  bool InsertStackProtectors();

  /// CreateFailBB - Create a basic block to jump to when the stack protector
  /// check fails.
  BasicBlock *CreateFailBB();

  /// ContainsProtectableArray - Check whether the type either is an array or
  /// contains an array of sufficient size so that we need stack protectors
  /// for it.
  bool ContainsProtectableArray(Type *Ty, bool Strong = false,
                                bool InStruct = false) const;

  /// \brief Check whether a stack allocation has its address taken.
  bool HasAddressTaken(const Instruction *AI);

  /// RequiresStackProtector - Check whether or not this function needs a
  /// stack protector based upon the stack protector level.
  bool RequiresStackProtector();
public:
  static char ID;             // Pass identification, replacement for typeid.
  StackProtector() : FunctionPass(ID), TM(0), TLI(0), SSPBufferSize(0) {
    initializeStackProtectorPass(*PassRegistry::getPassRegistry());
  }
  StackProtector(const TargetMachine *TM)
    : FunctionPass(ID), TM(TM), TLI(0), Trip(TM->getTargetTriple()),
      SSPBufferSize(8) {
    initializeStackProtectorPass(*PassRegistry::getPassRegistry());
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addPreserved<DominatorTree>();
  }

  virtual bool runOnFunction(Function &Fn);
};
} // end namespace llvm

#endif // LLVM_CODEGEN_STACKPROTECTOR_H

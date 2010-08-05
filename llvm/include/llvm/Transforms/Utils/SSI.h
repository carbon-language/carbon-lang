//===------------------- SSI.h - Creates SSI Representation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass converts a list of variables to the Static Single Information
// form. This is a program representation described by Scott Ananian in his
// Master Thesis: "The Static Single Information Form (1999)".
// We are building an on-demand representation, that is, we do not convert
// every single variable in the target function to SSI form. Rather, we receive
// a list of target variables that must be converted. We also do not
// completely convert a target variable to the SSI format. Instead, we only
// change the variable in the points where new information can be attached
// to its live range, that is, at branch points.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SSI_H
#define LLVM_TRANSFORMS_UTILS_SSI_H

#include "llvm/InstrTypes.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

  class DominatorTree;
  class PHINode;
  class Instruction;
  class CmpInst;

  class SSI : public FunctionPass {
    public:
      static char ID; // Pass identification, replacement for typeid.
      SSI() :
        FunctionPass(ID) {
      }

      void getAnalysisUsage(AnalysisUsage &AU) const;

      bool runOnFunction(Function&);

      void createSSI(SmallVectorImpl<Instruction *> &value);

    private:
      // Variables always live
      DominatorTree *DT_;

      // Stores variables created by SSI
      SmallPtrSet<Instruction *, 16> created;

      // Phis created by SSI
      DenseMap<PHINode *, Instruction*> phis;

      // Sigmas created by SSI
      DenseMap<PHINode *, Instruction*> sigmas;

      // Phi nodes that have a phi as operand and has to be fixed
      SmallPtrSet<PHINode *, 1> phisToFix;

      // List of definition points for every variable
      DenseMap<Instruction*, SmallVector<BasicBlock*, 4> > defsites;

      // Basic Block of the original definition of each variable
      DenseMap<Instruction*, BasicBlock*> value_original;

      // Stack of last seen definition of a variable
      DenseMap<Instruction*, SmallVector<Instruction *, 1> > value_stack;

      void insertSigmaFunctions(SmallPtrSet<Instruction*, 4> &value);
      void insertSigma(TerminatorInst *TI, Instruction *I);
      void insertPhiFunctions(SmallPtrSet<Instruction*, 4> &value);
      void renameInit(SmallPtrSet<Instruction*, 4> &value);
      void rename(BasicBlock *BB);

      void substituteUse(Instruction *I);
      bool dominateAny(BasicBlock *BB, Instruction *value);
      void fixPhis();

      Instruction* getPositionPhi(PHINode *PN);
      Instruction* getPositionSigma(PHINode *PN);

      void init(SmallVectorImpl<Instruction *> &value);
      void clean();
  };
} // end namespace
#endif

//===- StableBasicBlockNumbering.h - Provide BB identifiers -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This class provides a *stable* numbering of basic blocks that does not depend
// on their address in memory (which is nondeterministic).  When requested, this
// class simply provides a unique ID for each basic block in the function
// specified and the inverse mapping.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STABLEBASICBLOCKNUMBERING_H
#define LLVM_SUPPORT_STABLEBASICBLOCKNUMBERING_H

#include "llvm/Function.h"
#include <map>

namespace llvm {
  class StableBasicBlockNumbering {
    // BasicBlockNumbering - Holds a numbering of the basic blocks in the
    // function in a stable order that does not depend on their address.
    std::map<BasicBlock*, unsigned> BasicBlockNumbering;

    // NumberedBasicBlock - Holds the inverse mapping of BasicBlockNumbering.
    std::vector<BasicBlock*> NumberedBasicBlock;
  public:

    StableBasicBlockNumbering(Function *F = 0) {
      if (F) compute(*F);
    }

    /// compute - If we have not computed a numbering for the function yet, do
    /// so.
    void compute(Function &F) {
      if (NumberedBasicBlock.empty()) {
        unsigned n = 0;
        for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I, ++n) {
          NumberedBasicBlock.push_back(I);
          BasicBlockNumbering[I] = n;
        }
      }
    }

    /// getNumber - Return the ID number for the specified BasicBlock.
    ///
    unsigned getNumber(BasicBlock *BB) const {
      std::map<BasicBlock*, unsigned>::const_iterator I =
        BasicBlockNumbering.find(BB);
      assert(I != BasicBlockNumbering.end() &&
             "Invalid basic block or numbering not computed!");
      return I->second;
    }

    /// getBlock - Return the BasicBlock corresponding to a particular ID.
    ///
    BasicBlock *getBlock(unsigned N) const {
      assert(N < NumberedBasicBlock.size() &&
             "Block ID out of range or numbering not computed!");
      return NumberedBasicBlock[N];
    }

  };
}

#endif

//===- Dominators.cpp - Dominator Calculation -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// forward dominators.  Postdominators are available in libanalysis, but are not
// included in libvmcore, because it's not needed.  Forward dominators are
// needed to support the Verifier pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DominatorInternals.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
using namespace llvm;

// Always verify dominfo if expensive checking is enabled.
#ifdef XDEBUG
static bool VerifyDomInfo = true;
#else
static bool VerifyDomInfo = false;
#endif
static cl::opt<bool,true>
VerifyDomInfoX("verify-dom-info", cl::location(VerifyDomInfo),
               cl::desc("Verify dominator info (time consuming)"));

//===----------------------------------------------------------------------===//
//  DominatorTree Implementation
//===----------------------------------------------------------------------===//
//
// Provide public access to DominatorTree information.  Implementation details
// can be found in DominatorCalculation.h.
//
//===----------------------------------------------------------------------===//

TEMPLATE_INSTANTIATION(class llvm::DomTreeNodeBase<BasicBlock>);
TEMPLATE_INSTANTIATION(class llvm::DominatorTreeBase<BasicBlock>);

char DominatorTree::ID = 0;
INITIALIZE_PASS(DominatorTree, "domtree",
                "Dominator Tree Construction", true, true)

bool DominatorTree::runOnFunction(Function &F) {
  DT->recalculate(F);
  return false;
}

void DominatorTree::verifyAnalysis() const {
  if (!VerifyDomInfo) return;

  Function &F = *getRoot()->getParent();

  DominatorTree OtherDT;
  OtherDT.getBase().recalculate(F);
  if (compare(OtherDT)) {
    errs() << "DominatorTree is not up to date!  Computed:\n";
    print(errs());
    
    errs() << "\nActual:\n";
    OtherDT.print(errs());
    abort();
  }
}

void DominatorTree::print(raw_ostream &OS, const Module *) const {
  DT->print(OS);
}

// dominates - Return true if A dominates a use in B. This performs the
// special checks necessary if A and B are in the same basic block.
bool DominatorTree::dominates(const Instruction *A, const Instruction *B) const{
  const BasicBlock *BBA = A->getParent(), *BBB = B->getParent();
  
  // If A is an invoke instruction, its value is only available in this normal
  // successor block.
  if (const InvokeInst *II = dyn_cast<InvokeInst>(A))
    BBA = II->getNormalDest();
  
  if (BBA != BBB) return dominates(BBA, BBB);
  
  // It is not possible to determine dominance between two PHI nodes 
  // based on their ordering.
  if (isa<PHINode>(A) && isa<PHINode>(B)) 
    return false;
  
  // Loop through the basic block until we find A or B.
  BasicBlock::const_iterator I = BBA->begin();
  for (; &*I != A && &*I != B; ++I)
    /*empty*/;
  
  return &*I == A;
}

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

#include "llvm/Analysis/DominanceFrontier.h"
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



//===----------------------------------------------------------------------===//
//  DominanceFrontier Implementation
//===----------------------------------------------------------------------===//

char DominanceFrontier::ID = 0;
INITIALIZE_PASS_BEGIN(DominanceFrontier, "domfrontier",
                "Dominance Frontier Construction", true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(DominanceFrontier, "domfrontier",
                "Dominance Frontier Construction", true, true)

void DominanceFrontier::verifyAnalysis() const {
  if (!VerifyDomInfo) return;

  DominatorTree &DT = getAnalysis<DominatorTree>();

  DominanceFrontier OtherDF;
  const std::vector<BasicBlock*> &DTRoots = DT.getRoots();
  OtherDF.calculate(DT, DT.getNode(DTRoots[0]));
  assert(!compare(OtherDF) && "Invalid DominanceFrontier info!");
}

namespace {
  class DFCalculateWorkObject {
  public:
    DFCalculateWorkObject(BasicBlock *B, BasicBlock *P, 
                          const DomTreeNode *N,
                          const DomTreeNode *PN)
    : currentBB(B), parentBB(P), Node(N), parentNode(PN) {}
    BasicBlock *currentBB;
    BasicBlock *parentBB;
    const DomTreeNode *Node;
    const DomTreeNode *parentNode;
  };
}

const DominanceFrontier::DomSetType &
DominanceFrontier::calculate(const DominatorTree &DT,
                             const DomTreeNode *Node) {
  BasicBlock *BB = Node->getBlock();
  DomSetType *Result = NULL;

  std::vector<DFCalculateWorkObject> workList;
  SmallPtrSet<BasicBlock *, 32> visited;

  workList.push_back(DFCalculateWorkObject(BB, NULL, Node, NULL));
  do {
    DFCalculateWorkObject *currentW = &workList.back();
    assert (currentW && "Missing work object.");

    BasicBlock *currentBB = currentW->currentBB;
    BasicBlock *parentBB = currentW->parentBB;
    const DomTreeNode *currentNode = currentW->Node;
    const DomTreeNode *parentNode = currentW->parentNode;
    assert (currentBB && "Invalid work object. Missing current Basic Block");
    assert (currentNode && "Invalid work object. Missing current Node");
    DomSetType &S = Frontiers[currentBB];

    // Visit each block only once.
    if (visited.count(currentBB) == 0) {
      visited.insert(currentBB);

      // Loop over CFG successors to calculate DFlocal[currentNode]
      for (succ_iterator SI = succ_begin(currentBB), SE = succ_end(currentBB);
           SI != SE; ++SI) {
        // Does Node immediately dominate this successor?
        if (DT[*SI]->getIDom() != currentNode)
          S.insert(*SI);
      }
    }

    // At this point, S is DFlocal.  Now we union in DFup's of our children...
    // Loop through and visit the nodes that Node immediately dominates (Node's
    // children in the IDomTree)
    bool visitChild = false;
    for (DomTreeNode::const_iterator NI = currentNode->begin(), 
           NE = currentNode->end(); NI != NE; ++NI) {
      DomTreeNode *IDominee = *NI;
      BasicBlock *childBB = IDominee->getBlock();
      if (visited.count(childBB) == 0) {
        workList.push_back(DFCalculateWorkObject(childBB, currentBB,
                                                 IDominee, currentNode));
        visitChild = true;
      }
    }

    // If all children are visited or there is any child then pop this block
    // from the workList.
    if (!visitChild) {

      if (!parentBB) {
        Result = &S;
        break;
      }

      DomSetType::const_iterator CDFI = S.begin(), CDFE = S.end();
      DomSetType &parentSet = Frontiers[parentBB];
      for (; CDFI != CDFE; ++CDFI) {
        if (!DT.properlyDominates(parentNode, DT[*CDFI]))
          parentSet.insert(*CDFI);
      }
      workList.pop_back();
    }

  } while (!workList.empty());

  return *Result;
}

void DominanceFrontierBase::print(raw_ostream &OS, const Module* ) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    OS << "  DomFrontier for BB ";
    if (I->first)
      WriteAsOperand(OS, I->first, false);
    else
      OS << " <<exit node>>";
    OS << " is:\t";
    
    const std::set<BasicBlock*> &BBs = I->second;
    
    for (std::set<BasicBlock*>::const_iterator I = BBs.begin(), E = BBs.end();
         I != E; ++I) {
      OS << ' ';
      if (*I)
        WriteAsOperand(OS, *I, false);
      else
        OS << "<<exit node>>";
    }
    OS << "\n";
  }
}

void DominanceFrontierBase::dump() const {
  print(dbgs());
}


//===- llvm/Transforms/HoistPHIConstants.h - Normalize PHI nodes ------------=//
//
// HoistPHIConstants - Remove literal constants that are arguments of PHI nodes
// by inserting cast instructions in the preceeding basic blocks, and changing
// constant references into references of the casted value.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/HoistPHIConstants.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include <map>

typedef pair<BasicBlock *, Value*> BBConstTy;
typedef map<BBConstTy, CastInst *> CachedCopyMap;

static Value *NormalizePhiOperand(PHINode *PN, Value *CPV,
                                  BasicBlock *Pred, CachedCopyMap &CopyCache) {
  
  // Check if we've already inserted a copy for this constant in Pred
  // Note that `copyCache[Pred]' will create an empty vector the first time
  //
  CachedCopyMap::iterator CCI = CopyCache.find(BBConstTy(Pred, CPV));
  if (CCI != CopyCache.end()) return CCI->second;
  
  // Create a copy instruction and add it to the cache...
  CastInst *Inst = new CastInst(CPV, CPV->getType());
  CopyCache.insert(make_pair(BBConstTy(Pred, CPV), Inst));
    
  // Insert the copy just before the terminator inst of the predecessor BB
  assert(Pred->getTerminator() && "Degenerate BB encountered!");
  Pred->getInstList().insert(Pred->getInstList().end()-1, Inst);
  
  return Inst;
}


//---------------------------------------------------------------------------
// Entry point for normalizing constant args in PHIs
//---------------------------------------------------------------------------

bool HoistPHIConstants::doPerMethodWork(Method *M) {
  CachedCopyMap Cache;
  
  for (Method::iterator BI = M->begin(), BE = M->end(); BI != BE; ++BI)
    for (BasicBlock::iterator II = (*BI)->begin(); II != (*BI)->end(); ++II) {
      Instruction *Inst = *II;
      if (!isa<PHINode>(Inst)) break;   // All PHIs occur at top of BB!
      
      PHINode *PN = cast<PHINode>(Inst);
      for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i) {
        Value *Op = PN->getIncomingValue(i);
        if (isa<ConstPoolVal>(Op))
          PN->setIncomingValue(i,
                NormalizePhiOperand(PN, Op, PN->getIncomingBlock(i), Cache));
      }
    }

  return false;
}

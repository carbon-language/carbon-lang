//===- llvm/Transforms/HoistPHIConstants.h - Normalize PHI nodes ------------=//
//
// HoistPHIConstants - Remove literal constants that are arguments of PHI nodes
// by inserting cast instructions in the preceeding basic blocks, and changing
// constant references into references of the casted value.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/HoistPHIConstants.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"

typedef std::pair<BasicBlock *, Value*> BBConstTy;
typedef std::map<BBConstTy, CastInst *> CachedCopyMap;

static Value *NormalizePhiOperand(PHINode *PN, Value *CPV,
                                  BasicBlock *Pred, CachedCopyMap &CopyCache) {
  // Check if we've already inserted a copy for this constant in Pred
  // Note that `copyCache[Pred]' will create an empty vector the first time
  //
  CachedCopyMap::iterator CCI = CopyCache.find(BBConstTy(Pred, CPV));
  if (CCI != CopyCache.end()) return CCI->second;
  
  // Create a copy instruction and add it to the cache...
  CastInst *Inst = new CastInst(CPV, CPV->getType());
  CopyCache.insert(std::make_pair(BBConstTy(Pred, CPV), Inst));
    
  // Insert the copy just before the terminator inst of the predecessor BB
  assert(Pred->getTerminator() && "Degenerate BB encountered!");
  Pred->getInstList().insert(Pred->getInstList().end()-1, Inst);
  
  return Inst;
}


//---------------------------------------------------------------------------
// Entry point for normalizing constant args in PHIs
//---------------------------------------------------------------------------

static bool doHoistPHIConstants(Function *M) {
  CachedCopyMap Cache;
  bool Changed = false;
  
  for (Function::iterator BI = M->begin(), BE = M->end(); BI != BE; ++BI) {
    std::vector<PHINode*> phis;          // normalizing invalidates BB iterator
      
    for (BasicBlock::iterator II = (*BI)->begin(); II != (*BI)->end(); ++II) {
      if (PHINode *PN = dyn_cast<PHINode>(*II))
        phis.push_back(PN);
      else
        break;                      // All PHIs occur at top of BB!
    }
      
    for (std::vector<PHINode*>::iterator PI=phis.begin(); PI != phis.end();++PI)
      for (unsigned i = 0; i < (*PI)->getNumIncomingValues(); ++i) {
        Value *Op = (*PI)->getIncomingValue(i);
        
        if (isa<Constant>(Op)) {
          (*PI)->setIncomingValue(i,
                    NormalizePhiOperand((*PI),
                                        (*PI)->getIncomingValue(i),
                                        (*PI)->getIncomingBlock(i), Cache));
          Changed = true;
        }
      }
  }
  
  return Changed;
}

namespace {
  struct HoistPHIConstants : public FunctionPass {
    const char *getPassName() const { return "Hoist Constants from PHI Nodes"; }

    virtual bool runOnFunction(Function *F) { return doHoistPHIConstants(F); }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
    }
  };
}

Pass *createHoistPHIConstantsPass() { return new HoistPHIConstants(); }

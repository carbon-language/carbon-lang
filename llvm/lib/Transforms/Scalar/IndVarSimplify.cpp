//===- IndVarSimplify.cpp - Induction Variable Elimination ----------------===//
//
// InductionVariableSimplify - Transform induction variables in a program
//   to all use a single cannonical induction variable per loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Analysis/InductionVariable.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/iPHINode.h"
#include "Support/STLExtras.h"

static bool TransformLoop(cfg::LoopInfo *Loops, cfg::Loop *Loop) {
  // Transform all subloops before this loop...
  bool Changed = reduce_apply_bool(Loop->getSubLoops().begin(),
                                   Loop->getSubLoops().end(),
                                   std::bind1st(ptr_fun(TransformLoop), Loops));
  // Get the header node for this loop.  All of the phi nodes that could be
  // induction variables must live in this basic block.
  BasicBlock *Header = (BasicBlock*)Loop->getBlocks().front();
  
  // Loop over all of the PHI nodes in the basic block, calculating the
  // induction variables that they represent... stuffing the induction variable
  // info into a vector...
  //
  vector<InductionVariable> IndVars;    // Induction variables for block
  for (BasicBlock::iterator I = Header->begin(); 
       PHINode *PN = dyn_cast<PHINode>(*I); ++I)
    IndVars.push_back(InductionVariable(PN, Loops));

  // If there are phi nodes in this basic block, there can't be indvars...
  if (IndVars.empty()) return Changed;
  
  // Loop over the induction variables, looking for a cannonical induction
  // variable, and checking to make sure they are not all unknown induction
  // variables.
  //
  bool FoundIndVars = false;
  InductionVariable *Cannonical = 0;
  for (unsigned i = 0; i < IndVars.size(); ++i) {
    if (IndVars[i].InductionType == InductionVariable::Cannonical)
      Cannonical = &IndVars[i];
    if (IndVars[i].InductionType != InductionVariable::Unknown)
      FoundIndVars = true;
  }

  // No induction variables, bail early... don't add a cannonnical indvar
  if (!FoundIndVars) return Changed;

  // Okay, we want to convert other induction variables to use a cannonical
  // indvar.  If we don't have one, add one now...
  if (!Cannonical) {

  }

  return Changed;
}

bool InductionVariableSimplify::doit(Method *M) {
  // Figure out the loop structure of the method...
  cfg::LoopInfo Loops(M);

  // Induction Variables live in the header nodes of the loops of the method...
  return reduce_apply_bool(Loops.getTopLevelLoops().begin(),
                           Loops.getTopLevelLoops().end(),
                           std::bind1st(std::ptr_fun(TransformLoop), &Loops));
}

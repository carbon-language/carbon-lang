//===-- DCE.h - Functions that perform Dead Code Elimination -----*- C++ -*--=//
//
// This family of functions is useful for performing dead code elimination of 
// various sorts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_DCE_H
#define LLVM_TRANSFORMS_SCALAR_DCE_H

#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
class Pass;

//===----------------------------------------------------------------------===//
// DeadInstElimination - This pass quickly removes trivially dead instructions
// without modifying the CFG of the function.  It is a BasicBlockPass, so it
// runs efficiently when queued next to other BasicBlockPass's.
//
Pass *createDeadInstEliminationPass();


//===----------------------------------------------------------------------===//
// DeadCodeElimination - This pass is more powerful than DeadInstElimination,
// because it will remove dead basic blocks as well as all of the instructions
// contained within them.  This pass is useful to run after another pass has
// reorganized the CFG and possibly modified control flow.
//
// TODO: In addition to DCE stuff, this also merges basic blocks together and
// otherwise simplifies control flow.  This should be factored out of this pass
// eventually into it's own pass.
//

Pass *createDeadCodeEliminationPass();

// dceInstruction - Inspect the instruction at *BBI and figure out if it's
// [trivially] dead.  If so, remove the instruction and update the iterator
// to point to the instruction that immediately succeeded the original
// instruction.
//
bool dceInstruction(BasicBlock::InstListType &BBIL,
                    BasicBlock::iterator &BBI);



//===----------------------------------------------------------------------===//
// AgressiveDCE - This pass uses the SSA based Agressive DCE algorithm.  This
// algorithm assumes instructions are dead until proven otherwise, which makes
// it more successful are removing non-obviously dead instructions.
//
Pass *createAgressiveDCEPass();


// SimplifyCFG - This function is used to do simplification of a CFG.  For
// example, it adjusts branches to branches to eliminate the extra hop, it
// eliminates unreachable basic blocks, and does other "peephole" optimization
// of the CFG.  It returns true if a modification was made, and returns an 
// iterator that designates the first element remaining after the block that
// was deleted.
//
// WARNING:  The entry node of a method may not be simplified.
//
bool SimplifyCFG(Function::iterator &BBIt);

#endif

//===-- DCE.h - Passes that perform Dead Code Elimination --------*- C++ -*--=//
//
// This family of passes is useful for performing dead code elimination of
// various strengths.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_DCE_H
#define LLVM_TRANSFORMS_SCALAR_DCE_H

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


//===----------------------------------------------------------------------===//
// AgressiveDCE - This pass uses the SSA based Agressive DCE algorithm.  This
// algorithm assumes instructions are dead until proven otherwise, which makes
// it more successful are removing non-obviously dead instructions.
//
Pass *createAgressiveDCEPass();

#endif

//===- llvm/Transforms/DecomposeArrayRefs.h - Lower array refs --*- C++ -*--=//
// 
// DecomposeArrayRefs - 
// Convert multi-dimensional array references into a sequence of
// instructions (using getelementpr and cast) so that each instruction
// has at most one array offset.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_DECOMPOSEARRAYREFS_H
#define LLVM_TRANSFORMS_DECOMPOSEARRAYREFS_H

class Pass;
Pass *createDecomposeArrayRefsPass();

#endif

//===- llvm/Transforms/DecomposeMultiDimRefs.h - Lower multi-dim refs --*- C++ -*--=//
// 
// DecomposeMultiDimRefs - 
// Convert multi-dimensional references consisting of any combination
// of 2 or more array and structure indices into a sequence of
// instructions (using getelementpr and cast) so that each instruction
// has at most one index (except structure references,
// which need an extra leading index of [0]).
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_DECOMPOSEMULTIDIMREFS_H
#define LLVM_TRANSFORMS_SCALAR_DECOMPOSEMULTIDIMREFS_H

class Pass;
Pass *createDecomposeMultiDimRefsPass();

#endif

//===- llvm/Transforms/CleanupGCCOutput.h - Cleanup GCC Output ---*- C++ -*--=//
//
// These passes are used to cleanup the output of GCC.  GCC's output is
// unneccessarily gross for a couple of reasons. This pass does the following
// things to try to clean it up:
//
// * Eliminate names for GCC types that we know can't be needed by the user.
// * Eliminate names for types that are unused in the entire translation unit
// * Fix various problems that we might have in PHI nodes and casts
// * Link uses of 'void %foo(...)' to 'void %foo(sometypes)'
//
// Note:  This code produces dead declarations, it is a good idea to run DCE
//        after this pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H
#define LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H

class Pass;

// CleanupGCCOutputPass - Perform all of the function body transformations.
//
Pass *createCleanupGCCOutputPass();


// FunctionResolvingPass - Go over the functions that are in the module and
// look for functions that have the same name.  More often than not, there will
// be things like:
//    void "foo"(...)
//    void "foo"(int, int)
// because of the way things are declared in C.  If this is the case, patch
// things up.
//
// This is an interprocedural pass.
//
Pass *createFunctionResolvingPass();

#endif

//===- Cilkifier.h - Support routines for Cilk code generation --*- C++ -*-===//
//
// This is located here so that the code generator (dis) does not have to
// include and link with the libtipo.a archive containing class Cilkifier
// and the rest of the automatic parallelization code.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CILKIFIER_H
#define LLVM_SUPPORT_CILKIFIER_H

#include <string>
class Function;
class CallInst;


//---------------------------------------------------------------------------- 
// Global constants used in marking Cilk functions and function calls.
// These should be used only by the auto-parallelization pass.
//---------------------------------------------------------------------------- 

extern const std::string  CilkSuffix;
extern const std::string  DummySyncFuncName;

//---------------------------------------------------------------------------- 
// Routines to identify Cilk functions, calls to Cilk functions, and syncs.
//---------------------------------------------------------------------------- 

extern bool  isCilk     (const Function& F);
extern bool  isCilkMain (const Function& F);
extern bool  isCilk     (const CallInst& CI);
extern bool  isSync     (const CallInst& CI);

//===----------------------------------------------------------------------===//

#endif

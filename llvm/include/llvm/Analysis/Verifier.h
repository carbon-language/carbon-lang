//===-- llvm/Analysis/Verifier.h - Module Verifier ---------------*- C++ -*-==//
//
// This file defines the method verifier interface, that can be used for some
// sanity checking of input to the system.
//
// Note that this does not provide full 'java style' security and verifications,
// instead it just tries to ensure that code is well formed.
//
// To see what specifically is checked, look at the top of Verifier.cpp
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_VERIFIER_H
#define LLVM_ANALYSIS_VERIFIER_H

#include <vector>
#include <string>
class Module;
class Method;

// verify - Check a module or method for validity.  If errors are detected, 
// error messages corresponding to the problem are added to the errorMsgs 
// vectors, and a value of true is returned. 
//
bool verify(const Module *M, std::vector<std::string> &ErrorMsgs);
bool verify(const Method *M, std::vector<std::string> &ErrorMsgs);

#endif

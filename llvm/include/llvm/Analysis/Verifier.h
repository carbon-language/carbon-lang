//===-- llvm/Analysis/Verifier.h - Module Verifier ---------------*- C++ -*-==//
//
// This file defines the function verifier interface, that can be used for some
// sanity checking of input to the system, and for checking that transformations
// haven't done something bad.
//
// Note that this does not provide full 'java style' security and verifications,
// instead it just tries to ensure that code is well formed.
//
// To see what specifically is checked, look at the top of Verifier.cpp
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_VERIFIER_H
#define LLVM_ANALYSIS_VERIFIER_H

class FunctionPass;
class Module;
class Function;

// createVerifierPass - Check a module or function for validity.  If errors are
// detected, error messages corresponding to the problem are printed to stderr.
//
FunctionPass *createVerifierPass();

// verifyModule - Check a module for errors, printing messages on stderr.
// Return true if the module is corrupt.  This should only be used for
// debugging, because it plays games with PassManagers and stuff.
//
bool verifyModule(const Module &M);

// verifyFunction - Check a function for errors, useful for use when debugging a
// pass.
bool verifyFunction(const Function &F);

#endif

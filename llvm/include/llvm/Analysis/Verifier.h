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

class Pass;
class Module;
class Method;

// createVerifierPass - Check a module or method for validity.  If errors are
// detected, error messages corresponding to the problem are printed to stderr.
//
Pass *createVerifierPass();

// verifyModule - Check a module for errors, printing messages on stderr.
// Return true if the module is corrupt.
//
bool verifyModule(const Module *M);

// verifyMethod - Check a method for errors, useful for use when debugging a
// pass.
bool verifyMethod(const Method *M);

#endif

//===- llvm/Analysis/SafePointerAccess.h - Check pointer safety ---*- C++ -*-=//
//
// This file defines a pass that can be used to determine, interprocedurally, 
// which pointer types are accessed unsafely in a program.  If there is an
// "unsafe" access to a specific pointer type, transformations that depend on
// type safety cannot be permitted.
//
// The result of running this analysis over a program is a set of unsafe pointer
// types that cannot be transformed.  Safe pointer types are not tracked.
//
// Additionally, this analysis exports a hidden command line argument that (when
// enabled) prints out the reasons a type was determined to be unsafe.  Just add
// -unsafeptrinst to the command line of the tool you want to get it.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SAFEPOINTERACCESS_H
#define LLVM_ANALYSIS_SAFEPOINTERACCESS_H

#include "llvm/Pass.h"
#include <set>

class PointerType;

struct FindUnsafePointerTypes : public MethodPass {
  // UnsafeTypes - Set of types that are not safe to transform.
  std::set<PointerType*> UnsafeTypes;
public:

  // Accessor for underlying type set...
  inline const std::set<PointerType*> &getUnsafeTypes() const {
    return UnsafeTypes;
  }

  // runOnMethod - Inspect the operations that the specified method does on
  // values of various types.  If they are deemed to be 'unsafe' note that the
  // type is not safe to transform.
  //
  virtual bool runOnMethod(Method *M);

  // printResults - Loop over the results of the analysis, printing out unsafe
  // types.
  //
  void printResults(const Module *Mod, std::ostream &o);
};

#endif

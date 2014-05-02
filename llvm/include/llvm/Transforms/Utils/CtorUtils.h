//===- CtorUtils.h - Helpers for working with global_ctors ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines functions that are used to process llvm.global_ctors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CTOR_UTILS_H
#define LLVM_TRANSFORMS_UTILS_CTOR_UTILS_H

#include <functional>
#include <vector>

namespace llvm {

class GlobalVariable;
class Function;
class Module;

typedef bool (*ShouldRemoveCtor)(void *, Function *);

/// Call "ShouldRemove" for every entry in M's global_ctor list and remove the
/// entries for which it returns true.  Return true if anything changed.
bool optimizeGlobalCtorsList(Module &M, ShouldRemoveCtor ShouldRemove,
                             void *Context);

} // End llvm namespace

#endif

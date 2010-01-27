//===- CocoaConventions.h - Special handling of Cocoa conventions -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CHECKER_DS_COCOA
#define LLVM_CLANG_CHECKER_DS_COCOA

#include "clang/Basic/IdentifierTable.h"

namespace clang {
namespace cocoa {
 
enum NamingConvention { NoConvention, CreateRule, InitRule };

NamingConvention deriveNamingConvention(Selector S);

static inline bool followsFundamentalRule(Selector S) {
  return deriveNamingConvention(S) == CreateRule;
}

}}

#endif

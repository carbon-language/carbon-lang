//===--- Rewriters.h - Rewriter implementations     -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header contains miscellaneous utilities for various front-end actions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_REWRITE_REWRITERS_H
#define LLVM_CLANG_REWRITE_REWRITERS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
class Preprocessor;

/// RewriteMacrosInInput - Implement -rewrite-macros mode.
void RewriteMacrosInInput(Preprocessor &PP, llvm::raw_ostream* OS);

/// DoRewriteTest - A simple test for the TokenRewriter class.
void DoRewriteTest(Preprocessor &PP, llvm::raw_ostream* OS);

}  // end namespace clang

#endif

//==- HTMLRewrite.h - Translate source code into prettified HTML ---*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of functions used for translating source code
//  into beautified HTML.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_HTMLREWRITER_H
#define LLVM_CLANG_HTMLREWRITER_H

#include "clang/Basic/SourceLocation.h"

namespace clang {
  
class Rewriter;
  
namespace html {

  void EscapeText(Rewriter& R, unsigned FileID, bool EscapeSpaces = false);
  
  void AddLineNumbers(Rewriter& R, unsigned FileID);  

} // end html namespace
} // end clang namespace

#endif

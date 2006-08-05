//===--- ParserActions.h - Parser Actions Interface -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ParserActions interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSERACTIONS_H
#define LLVM_CLANG_PARSE_PARSERACTIONS_H

namespace llvm {
namespace clang {

/// ParserActions - 
class ParserActions {
  // SYMBOL TABLE
public:
  // Types - Though these don't actually enforce strong typing, they document
  // what types are required to be identical for the actions.
  typedef void* ExprTy;
  
  
};
  
}  // end namespace clang
}  // end namespace llvm

#endif

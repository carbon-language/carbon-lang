//===--- ParentMap.h - Mappings from Stmts to their Parents -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ParentMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARENTMAP_H
#define LLVM_CLANG_PARENTMAP_H

namespace clang {
class Stmt;
  
class ParentMap {
  void* Impl;
public:
  ParentMap(Stmt* ASTRoot);
  ~ParentMap();

  Stmt* getParent(Stmt*) const;  

  bool hasParent(Stmt* S) const {
    return !getParent(S);
  }
  
  bool isSubExpr(Stmt *S) const;
};
  
} // end clang namespace
#endif

//===--- QuerySession.h - clang-query ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_SESSION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_SESSION_H

#include "llvm/ADT/ArrayRef.h"
#include "Query.h"

namespace clang {

class ASTUnit;

namespace query {

/// Represents the state for a particular clang-query session.
class QuerySession {
public:
  QuerySession(llvm::ArrayRef<ASTUnit *> ASTs)
      : ASTs(ASTs), OutKind(OK_Diag), BindRoot(true) {}

  llvm::ArrayRef<ASTUnit *> ASTs;
  OutputKind OutKind;
  bool BindRoot;
};

} // namespace query
} // namespace clang

#endif

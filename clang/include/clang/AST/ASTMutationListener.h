//===--- ASTMutationListener.h - AST Mutation Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTMutationListener interface.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_ASTMUTATIONLISTENER_H
#define LLVM_CLANG_AST_ASTMUTATIONLISTENER_H

namespace clang {
  class CXXRecordDecl;
  class CXXMethodDecl;

/// \brief An abstract interface that should be implemented by listeners
/// that want to be notified when an AST entity gets modified after its
/// initial creation.
class ASTMutationListener {
public:
  virtual ~ASTMutationListener();
};

} // end namespace clang

#endif

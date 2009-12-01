//===--- ExternalSemaSource.h - External Sema Interface ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ExternalSemaSource interface.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_EXTERNAL_SEMA_SOURCE_H
#define LLVM_CLANG_SEMA_EXTERNAL_SEMA_SOURCE_H

#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExternalASTSource.h"

namespace clang {

class Sema;

/// \brief An abstract interface that should be implemented by
/// external AST sources that also provide information for semantic
/// analysis.
class ExternalSemaSource : public ExternalASTSource {
public:
  ExternalSemaSource() {
    ExternalASTSource::SemaSource = true;
  }

  /// \brief Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  virtual void InitializeSema(Sema &S) {}

  /// \brief Inform the semantic consumer that Sema is no longer available.
  virtual void ForgetSema() {}

  /// \brief Load the contents of the global method pool for a given
  /// selector.
  ///
  /// \returns a pair of Objective-C methods lists containing the
  /// instance and factory methods, respectively, with this selector.
  virtual std::pair<ObjCMethodList, ObjCMethodList>
  ReadMethodPool(Selector Sel) {
    return std::pair<ObjCMethodList, ObjCMethodList>();
  }

  // isa/cast/dyn_cast support
  static bool classof(const ExternalASTSource *Source) {
    return Source->SemaSource;
  }
  static bool classof(const ExternalSemaSource *) { return true; }
};

} // end namespace clang

#endif

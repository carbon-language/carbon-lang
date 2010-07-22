//===--- TranslationUnit.h - Interface for a translation unit ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Abstract interface for a translation unit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_TRANSLATIONUNIT_H
#define LLVM_CLANG_INDEX_TRANSLATIONUNIT_H

namespace clang {
  class ASTContext;
  class Diagnostic;
  class Preprocessor;

namespace idx {
  class DeclReferenceMap;
  class SelectorMap;

/// \brief Abstract interface for a translation unit.
class TranslationUnit {
public:
  virtual ~TranslationUnit();
  virtual ASTContext &getASTContext() = 0;
  virtual Preprocessor &getPreprocessor() = 0;
  virtual Diagnostic &getDiagnostic() = 0;
  virtual DeclReferenceMap &getDeclReferenceMap() = 0;
  virtual SelectorMap &getSelectorMap() = 0;
};

} // namespace idx

} // namespace clang

#endif

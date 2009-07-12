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

namespace idx {

/// \brief Abstract interface for a translation unit.
class TranslationUnit {
public:
  virtual ~TranslationUnit();
  virtual ASTContext &getASTContext() = 0;
};

} // namespace idx

} // namespace clang

#endif

//===--- TypoCorrection.h - Class for typo correction results ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TypoCorrection class, which stores the results of
// Sema's typo correction (Sema::CorrectTypo).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_TYPOCORRECTION_H
#define LLVM_CLANG_SEMA_TYPOCORRECTION_H

#include "clang/AST/DeclCXX.h"

namespace clang {

/// @brief Simple class containing the result of Sema::CorrectTypo
class TypoCorrection {
public:
  TypoCorrection(const DeclarationName &Name, NamedDecl *NameDecl,
                 NestedNameSpecifier *NNS=NULL, unsigned distance=0)
      : CorrectionName(Name),
        CorrectionNameSpec(NNS),
        CorrectionDecl(NameDecl),
        EditDistance(distance) {}

  TypoCorrection(NamedDecl *Name, NestedNameSpecifier *NNS=NULL,
                 unsigned distance=0)
      : CorrectionName(Name->getDeclName()),
        CorrectionNameSpec(NNS),
        CorrectionDecl(Name),
        EditDistance(distance)  {}

  TypoCorrection(DeclarationName Name, NestedNameSpecifier *NNS=NULL,
                 unsigned distance=0)
      : CorrectionName(Name),
        CorrectionNameSpec(NNS),
        CorrectionDecl(NULL),
        EditDistance(distance)  {}

  TypoCorrection()
      : CorrectionName(), CorrectionNameSpec(NULL), CorrectionDecl(NULL),
        EditDistance(0) {}

  /// \brief Gets the DeclarationName of the typo correction
  DeclarationName getCorrection() const { return CorrectionName; }
  IdentifierInfo* getCorrectionAsIdentifierInfo() const {
    return CorrectionName.getAsIdentifierInfo();
  }

  /// \brief Gets the NestedNameSpecifier needed to use the typo correction
  NestedNameSpecifier* getCorrectionSpecifier() const {
    return CorrectionNameSpec;
  }
  void setCorrectionSpecifier(NestedNameSpecifier* NNS) {
    CorrectionNameSpec = NNS;
  }

  /// \brief Gets the "edit distance" of the typo correction from the typo
  unsigned getEditDistance() const { return EditDistance; }

  /// \brief Gets the pointer to the declaration of the typo correction
  NamedDecl* getCorrectionDecl() const {
    return isKeyword() ? NULL : CorrectionDecl;
  }
  template <class DeclClass>
  DeclClass *getCorrectionDeclAs() const {
    return dyn_cast_or_null<DeclClass>(getCorrectionDecl());
  }
  
  void setCorrectionDecl(NamedDecl *CDecl) {
    CorrectionDecl = CDecl;
    if (!CorrectionName)
      CorrectionName = CDecl->getDeclName();
  }

  std::string getAsString(const LangOptions &LO) const;
  std::string getQuoted(const LangOptions &LO) const {
    return "'" + getAsString(LO) + "'";
  }

  operator bool() const { return bool(CorrectionName); }

  static inline NamedDecl *KeywordDecl() { return (NamedDecl*)-1; }
  bool isKeyword() const { return CorrectionDecl == KeywordDecl(); }

  // Returns true if the correction either is a keyword or has a known decl.
  bool isResolved() const { return CorrectionDecl != NULL; }

private:
  // Results.
  DeclarationName CorrectionName;
  NestedNameSpecifier *CorrectionNameSpec;
  NamedDecl *CorrectionDecl;
  unsigned EditDistance;
};

}

#endif

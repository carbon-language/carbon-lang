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
#include "llvm/ADT/SmallVector.h"

namespace clang {

/// @brief Simple class containing the result of Sema::CorrectTypo
class TypoCorrection {
public:
  TypoCorrection(const DeclarationName &Name, NamedDecl *NameDecl,
                 NestedNameSpecifier *NNS=0, unsigned distance=0)
      : CorrectionName(Name),
        CorrectionNameSpec(NNS),
        EditDistance(distance) {
    if (NameDecl)
      CorrectionDecls.push_back(NameDecl);
  }

  TypoCorrection(NamedDecl *Name, NestedNameSpecifier *NNS=0,
                 unsigned distance=0)
      : CorrectionName(Name->getDeclName()),
        CorrectionNameSpec(NNS),
        EditDistance(distance) {
    if (Name)
      CorrectionDecls.push_back(Name);
  }

  TypoCorrection(DeclarationName Name, NestedNameSpecifier *NNS=0,
                 unsigned distance=0)
      : CorrectionName(Name),
        CorrectionNameSpec(NNS),
        EditDistance(distance) {}

  TypoCorrection()
      : CorrectionNameSpec(0), EditDistance(0) {}

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
    return hasCorrectionDecl() ? *(CorrectionDecls.begin()) : 0;
  }
  template <class DeclClass>
  DeclClass *getCorrectionDeclAs() const {
    return dyn_cast_or_null<DeclClass>(getCorrectionDecl());
  }
  
  /// \brief Clears the list of NamedDecls before adding the new one.
  void setCorrectionDecl(NamedDecl *CDecl) {
    CorrectionDecls.clear();
    addCorrectionDecl(CDecl);
  }

  /// \brief Add the given NamedDecl to the list of NamedDecls that are the
  /// declarations associated with the DeclarationName of this TypoCorrection
  void addCorrectionDecl(NamedDecl *CDecl);

  std::string getAsString(const LangOptions &LO) const;
  std::string getQuoted(const LangOptions &LO) const {
    return "'" + getAsString(LO) + "'";
  }

  /// \brief Returns whether this TypoCorrection has a non-empty DeclarationName
  operator bool() const { return bool(CorrectionName); }

  /// \brief Mark this TypoCorrection as being a keyword.
  /// Since addCorrectionDeclsand setCorrectionDecl don't allow NULL to be
  /// added to the list of the correction's NamedDecl pointers, NULL is added
  /// as the only element in the list to mark this TypoCorrection as a keyword.
  void makeKeyword() {
    CorrectionDecls.clear();
    CorrectionDecls.push_back(0);
  }

  // Check if this TypoCorrection is a keyword by checking if the first
  // item in CorrectionDecls is NULL.
  bool isKeyword() const {
    return !CorrectionDecls.empty() &&
        CorrectionDecls.front() == 0;
  }

  // Returns true if the correction either is a keyword or has a known decl.
  bool isResolved() const { return !CorrectionDecls.empty(); }

  bool isOverloaded() const {
    return CorrectionDecls.size() > 1;
  }

  typedef llvm::SmallVector<NamedDecl*, 1>::iterator decl_iterator;
  decl_iterator begin() {
    return isKeyword() ? CorrectionDecls.end() : CorrectionDecls.begin();
  }
  decl_iterator end() { return CorrectionDecls.end(); }

private:
  bool hasCorrectionDecl() const {
    return (!isKeyword() && !CorrectionDecls.empty());
  }

  // Results.
  DeclarationName CorrectionName;
  NestedNameSpecifier *CorrectionNameSpec;
  llvm::SmallVector<NamedDecl*, 1> CorrectionDecls;
  unsigned EditDistance;
};

}

#endif

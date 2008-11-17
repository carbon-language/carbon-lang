//===-- DeclarationName.cpp - Declaration names implementation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DeclarationName and DeclarationNameTable
// classes.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/DeclarationName.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"
using namespace clang;

namespace clang {
/// CXXSpecialName - Records the type associated with one of the
/// "special" kinds of declaration names in C++, e.g., constructors,
/// destructors, and conversion functions.
class CXXSpecialName 
  : public DeclarationNameExtra, public llvm::FoldingSetNode {
public:
  QualType Type;

  void Profile(llvm::FoldingSetNodeID &ID) {
    ID.AddInteger(ExtraKindOrNumArgs);
    ID.AddPointer(Type.getAsOpaquePtr());
  }
};

bool operator<(DeclarationName LHS, DeclarationName RHS) {
  if (IdentifierInfo *LhsId = LHS.getAsIdentifierInfo())
    if (IdentifierInfo *RhsId = RHS.getAsIdentifierInfo())
      return strcmp(LhsId->getName(), RhsId->getName()) < 0;

  return LHS.getAsOpaqueInteger() < RHS.getAsOpaqueInteger();
}

} // end namespace clang

DeclarationName::DeclarationName(Selector Sel) {
  switch (Sel.getNumArgs()) {
  case 0:
    Ptr = reinterpret_cast<uintptr_t>(Sel.getAsIdentifierInfo());
    Ptr |= StoredObjCZeroArgSelector;
    break;

  case 1:
    Ptr = reinterpret_cast<uintptr_t>(Sel.getAsIdentifierInfo());
    Ptr |= StoredObjCOneArgSelector;
    break;

  default:
    Ptr = Sel.InfoPtr & ~Selector::ArgFlags;
    Ptr |= StoredObjCMultiArgSelectorOrCXXName;
    break;
  }
}

DeclarationName::NameKind DeclarationName::getNameKind() const {
  switch (getStoredNameKind()) {
  case StoredIdentifier:          return Identifier;
  case StoredObjCZeroArgSelector: return ObjCZeroArgSelector;
  case StoredObjCOneArgSelector:  return ObjCOneArgSelector;

  case StoredObjCMultiArgSelectorOrCXXName:
    switch (getExtra()->ExtraKindOrNumArgs) {
    case DeclarationNameExtra::CXXConstructor: 
      return CXXConstructorName;

    case DeclarationNameExtra::CXXDestructor: 
      return CXXDestructorName;

    case DeclarationNameExtra::CXXConversionFunction: 
      return CXXConversionFunctionName;

    default:
      return ObjCMultiArgSelector;
    }
    break;
  }

  // Can't actually get here.
  return Identifier;
}

QualType DeclarationName::getCXXNameType() const {
  if (CXXSpecialName *CXXName = getAsCXXSpecialName())
    return CXXName->Type;
  else
    return QualType();
}

Selector DeclarationName::getObjCSelector() const {
  switch (getNameKind()) {
  case ObjCZeroArgSelector:
    return Selector(reinterpret_cast<IdentifierInfo *>(Ptr & ~PtrMask), 0);

  case ObjCOneArgSelector:
    return Selector(reinterpret_cast<IdentifierInfo *>(Ptr & ~PtrMask), 1);

  case ObjCMultiArgSelector:
    return Selector(reinterpret_cast<MultiKeywordSelector *>(Ptr & ~PtrMask));

  default:
    break;
  }

  return Selector();
}

DeclarationNameTable::DeclarationNameTable() {
  CXXSpecialNamesImpl = new llvm::FoldingSet<CXXSpecialName>;
}

DeclarationNameTable::~DeclarationNameTable() {
  delete static_cast<llvm::FoldingSet<CXXSpecialName>*>(CXXSpecialNamesImpl);
}

DeclarationName 
DeclarationNameTable::getCXXSpecialName(DeclarationName::NameKind Kind, 
                                        QualType Ty) {
  assert(Kind >= DeclarationName::CXXConstructorName &&
         Kind <= DeclarationName::CXXConversionFunctionName &&
         "Kind must be a C++ special name kind");
  
  llvm::FoldingSet<CXXSpecialName> *SpecialNames 
    = static_cast<llvm::FoldingSet<CXXSpecialName>*>(CXXSpecialNamesImpl);

  DeclarationNameExtra::ExtraKind EKind;
  switch (Kind) {
  case DeclarationName::CXXConstructorName: 
    EKind = DeclarationNameExtra::CXXConstructor;
    break;
  case DeclarationName::CXXDestructorName:
    EKind = DeclarationNameExtra::CXXDestructor;
    break;
  case DeclarationName::CXXConversionFunctionName:
    EKind = DeclarationNameExtra::CXXConversionFunction;
    break;
  default:
    return DeclarationName();
  }

  // Unique selector, to guarantee there is one per name.
  llvm::FoldingSetNodeID ID;
  ID.AddInteger(EKind);
  ID.AddPointer(Ty.getAsOpaquePtr());

  void *InsertPos = 0;
  if (CXXSpecialName *Name = SpecialNames->FindNodeOrInsertPos(ID, InsertPos))
    return DeclarationName(Name);

  CXXSpecialName *SpecialName = new CXXSpecialName;
  SpecialName->ExtraKindOrNumArgs = EKind;
  SpecialName->Type = Ty;

  SpecialNames->InsertNode(SpecialName, InsertPos);
  return DeclarationName(SpecialName);
}


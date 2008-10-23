//===---- SemaInherit.cpp - C++ Inheritance ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sema routines for C++ inheritance semantics,
// including searching the inheritance hierarchy and (eventually)
// access checking.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"

namespace clang {

/// IsDerivedFrom - Determine whether the class type Derived is
/// derived from the class type Base, ignoring qualifiers on Base and
/// Derived. This routine does not assess whether an actual conversion
/// from a Derived* to a Base* is legal, because it does not account
/// for ambiguous conversions or conversions to private/protected
/// bases.
bool Sema::IsDerivedFrom(QualType Derived, QualType Base)
{
  Derived = Context.getCanonicalType(Derived).getUnqualifiedType();
  Base = Context.getCanonicalType(Base).getUnqualifiedType();
  
  assert(Derived->isRecordType() && "IsDerivedFrom requires a class type");
  assert(Base->isRecordType() && "IsDerivedFrom requires a class type");

  if (Derived == Base)
    return false;

  if (const RecordType *DerivedType = Derived->getAsRecordType()) {
    const CXXRecordDecl *Decl 
      = static_cast<const CXXRecordDecl *>(DerivedType->getDecl());
    for (CXXRecordDecl::base_class_const_iterator BaseSpec = Decl->bases_begin();
         BaseSpec != Decl->bases_end(); ++BaseSpec) {
      if (Context.getCanonicalType(BaseSpec->getType()) == Base
          || IsDerivedFrom(BaseSpec->getType(), Base))
        return true;
    }
  }

  return false;
}

} // end namespace clang


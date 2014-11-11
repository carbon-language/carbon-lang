//===- ASTCommon.h - Common stuff for ASTReader/ASTWriter -*- C++ -*-=========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines common functions that both ASTReader and ASTWriter use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_SERIALIZATION_ASTCOMMON_H
#define LLVM_CLANG_LIB_SERIALIZATION_ASTCOMMON_H

#include "clang/AST/ASTContext.h"
#include "clang/Serialization/ASTBitCodes.h"

namespace clang {

namespace serialization {

enum DeclUpdateKind {
  UPD_CXX_ADDED_IMPLICIT_MEMBER,
  UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION,
  UPD_CXX_ADDED_ANONYMOUS_NAMESPACE,
  UPD_CXX_ADDED_FUNCTION_DEFINITION,
  UPD_CXX_INSTANTIATED_STATIC_DATA_MEMBER,
  UPD_CXX_INSTANTIATED_CLASS_DEFINITION,
  UPD_CXX_RESOLVED_EXCEPTION_SPEC,
  UPD_CXX_DEDUCED_RETURN_TYPE,
  UPD_DECL_MARKED_USED,
  UPD_MANGLING_NUMBER,
  UPD_STATIC_LOCAL_NUMBER,
  UPD_DECL_MARKED_OPENMP_THREADPRIVATE
};

TypeIdx TypeIdxFromBuiltin(const BuiltinType *BT);

template <typename IdxForTypeTy>
TypeID MakeTypeID(ASTContext &Context, QualType T, IdxForTypeTy IdxForType) {
  if (T.isNull())
    return PREDEF_TYPE_NULL_ID;

  unsigned FastQuals = T.getLocalFastQualifiers();
  T.removeLocalFastQualifiers();

  if (T.hasLocalNonFastQualifiers())
    return IdxForType(T).asTypeID(FastQuals);

  assert(!T.hasLocalQualifiers());

  if (const BuiltinType *BT = dyn_cast<BuiltinType>(T.getTypePtr()))
    return TypeIdxFromBuiltin(BT).asTypeID(FastQuals);

  if (T == Context.AutoDeductTy)
    return TypeIdx(PREDEF_TYPE_AUTO_DEDUCT).asTypeID(FastQuals);
  if (T == Context.AutoRRefDeductTy)
    return TypeIdx(PREDEF_TYPE_AUTO_RREF_DEDUCT).asTypeID(FastQuals);
  if (T == Context.VaListTagTy)
    return TypeIdx(PREDEF_TYPE_VA_LIST_TAG).asTypeID(FastQuals);

  return IdxForType(T).asTypeID(FastQuals);
}

unsigned ComputeHash(Selector Sel);

/// \brief Retrieve the "definitive" declaration that provides all of the
/// visible entries for the given declaration context, if there is one.
///
/// The "definitive" declaration is the only place where we need to look to
/// find information about the declarations within the given declaration
/// context. For example, C++ and Objective-C classes, C structs/unions, and
/// Objective-C protocols, categories, and extensions are all defined in a
/// single place in the source code, so they have definitive declarations
/// associated with them. C++ namespaces, on the other hand, can have
/// multiple definitions.
const DeclContext *getDefinitiveDeclContext(const DeclContext *DC);

/// \brief Determine whether the given declaration kind is redeclarable.
bool isRedeclarableDeclKind(unsigned Kind);

/// \brief Determine whether the given declaration needs an anonymous
/// declaration number.
bool needsAnonymousDeclarationNumber(const NamedDecl *D);

} // namespace serialization

} // namespace clang

#endif

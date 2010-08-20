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

#ifndef LLVM_CLANG_SERIALIZATION_LIB_AST_COMMON_H
#define LLVM_CLANG_SERIALIZATION_LIB_AST_COMMON_H

#include "clang/Serialization/ASTBitCodes.h"

namespace clang {

namespace serialization {

TypeIdx TypeIdxFromBuiltin(const BuiltinType *BT);

template <typename IdxForTypeTy>
TypeID MakeTypeID(QualType T, IdxForTypeTy IdxForType) {
  if (T.isNull())
    return PREDEF_TYPE_NULL_ID;

  unsigned FastQuals = T.getLocalFastQualifiers();
  T.removeFastQualifiers();

  if (T.hasLocalNonFastQualifiers())
    return IdxForType(T).asTypeID(FastQuals);

  assert(!T.hasLocalQualifiers());

  if (const BuiltinType *BT = dyn_cast<BuiltinType>(T.getTypePtr()))
    return TypeIdxFromBuiltin(BT).asTypeID(FastQuals);

  return IdxForType(T).asTypeID(FastQuals);
}

unsigned ComputeHash(Selector Sel);

} // namespace serialization

} // namespace clang

#endif

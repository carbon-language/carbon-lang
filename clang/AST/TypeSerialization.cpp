//===--- TypeSerialization.cpp - Serialization of Decls ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines methods that implement bitcode serialization for Types.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Type.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using namespace clang;

void QualType::Emit(llvm::Serializer& S) const {
  S.EmitPtr(getAsOpaquePtr());
  S.EmitInt(getQualifiers());
}

void QualType::Read(llvm::Deserializer& D) {
  D.ReadPtr(ThePtr);
  ThePtr |= D.ReadInt();
}

/*  FIXME: Either remove this method or complete it.

void Type::Emit(llvm::Serializer& S) {
  switch (getTypeClass()) {
    default:
      assert (false && "Serialization for type class not implemented.");
      break;
      
    case Type::Builtin:
      cast<BuiltinType>(this)->Emit(S);
      break;
  }
}
 */

void Type::EmitTypeInternal(llvm::Serializer& S) const {
  S.Emit(CanonicalType);
}

void Type::ReadTypeInternal(llvm::Deserializer& D) {
  D.Read(CanonicalType);
}

void ComplexType::Emit(llvm::Serializer& S) const {
  EmitTypeInternal(S);
  S.Emit(ElementType);
}

ComplexType* ComplexType::Materialize(llvm::Deserializer& D) {
  ComplexType* T = new ComplexType(QualType(),QualType());
  T->ReadTypeInternal(D);
  D.Read(T->ElementType);
  return T;
}

void PointerType::Emit(llvm::Serializer& S) const {
  EmitTypeInternal(S);
  S.Emit(PointeeType);
}

PointerType* PointerType::Materialize(llvm::Deserializer& D) {
  PointerType* T = new PointerType(QualType(),QualType());
  T->ReadTypeInternal(D);
  D.Read(T->PointeeType);
  return T;
}

void ReferenceType::Emit(llvm::Serializer& S) const {
  EmitTypeInternal(S);
  S.Emit(ReferenceeType);
}

ReferenceType* ReferenceType::Materialize(llvm::Deserializer& D) {
  ReferenceType* T = new ReferenceType(QualType(),QualType());
  T->ReadTypeInternal(D);
  D.Read(T->ReferenceeType);
  return T;
}

void ArrayType::EmitArrayTypeInternal(llvm::Serializer& S) const {
  EmitTypeInternal(S);
  S.Emit(ElementType);
  S.EmitInt(SizeModifier);
  S.EmitInt(IndexTypeQuals);
}

void ArrayType::ReadArrayTypeInternal(llvm::Deserializer& D) {
  ReadTypeInternal(D);
  D.Read(ElementType);
  SizeModifier = static_cast<ArraySizeModifier>(D.ReadInt());
  IndexTypeQuals = D.ReadInt();
}

void ConstantArrayType::Emit(llvm::Serializer& S) const {
#if 0
  // FIXME: APInt serialization
  S.Emit(Size);
#endif
  EmitArrayTypeInternal(S);
}

ConstantArrayType* ConstantArrayType::Materialize(llvm::Deserializer& D) {
#if 0
  llvm::APInt x = S.ReadVal<llvm::APInt>(D);
  
  // "Default" construct the array.
  ConstantArrayType* T =
    new ConstantArrayType(QualType(), QualType(), x, ArrayType::Normal, 0);
  
  // Deserialize the internal values.
  T->ReadArrayTypeInternal(D);

  return T;
#else
  return NULL;
#endif

}

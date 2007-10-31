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
#include "clang/AST/Expr.h"
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

void QualType::EmitOwned(llvm::Serializer& S) const {
  S.EmitInt(getQualifiers());
  S.EmitOwnedPtr(cast<BuiltinType>(getTypePtr()));
}

void QualType::ReadOwned(llvm::Deserializer& D) {
  ThePtr = D.ReadInt();
  ThePtr |= reinterpret_cast<uintptr_t>(D.ReadOwnedPtr<BuiltinType>());
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

void BuiltinType::Emit(llvm::Serializer& S) const {
  S.EmitInt(TypeKind);
}

BuiltinType* BuiltinType::Materialize(llvm::Deserializer& D) {
  Kind k = static_cast<Kind>(D.ReadInt());
  BuiltinType* T = new BuiltinType(k);
  return T;
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
  EmitArrayTypeInternal(S);
  S.Emit(Size);
}

ConstantArrayType* ConstantArrayType::Materialize(llvm::Deserializer& D) {
  // "Default" construct the array type.
  ConstantArrayType* T =
    new ConstantArrayType(QualType(), QualType(), llvm::APInt(), 
                          ArrayType::Normal, 0);
  
  // Deserialize the internal values.
  T->ReadArrayTypeInternal(D);  
  D.Read(T->Size);

  return T;
}

void VariableArrayType::Emit(llvm::Serializer& S) const {
  EmitArrayTypeInternal(S);
  S.EmitOwnedPtr(SizeExpr);
}

VariableArrayType* VariableArrayType::Materialize(llvm::Deserializer& D) {
  // "Default" construct the array type.
  VariableArrayType* T =
    new VariableArrayType(QualType(), QualType(), NULL, ArrayType::Normal, 0);
  
  // Deserialize the internal values.
  T->ReadArrayTypeInternal(D);
  T->SizeExpr = D.ReadOwnedPtr<Expr>();
  
  return T;
}

void VectorType::Emit(llvm::Serializer& S) const {
  EmitTypeInternal(S);
  S.Emit(ElementType);
  S.EmitInt(NumElements);
}

VectorType* VectorType::Materialize(llvm::Deserializer& D) {
  VectorType* T = new VectorType(QualType(),0,QualType());
  T->ReadTypeInternal(D);
  D.Read(T->ElementType);
  T->NumElements = D.ReadInt();
  return T;
}

void FunctionType::EmitFunctionTypeInternal(llvm::Serializer &S) const {
  EmitTypeInternal(S);
  S.EmitBool(SubClassData);
  S.Emit(ResultType);
}

void FunctionType::ReadFunctionTypeInternal(llvm::Deserializer& D) {
  ReadTypeInternal(D);
  SubClassData = D.ReadBool();
  D.Read(ResultType);
}


FunctionTypeNoProto* FunctionTypeNoProto::Materialize(llvm::Deserializer& D) {
  FunctionTypeNoProto* T = new FunctionTypeNoProto(QualType(),QualType());
  T->ReadFunctionTypeInternal(D);
  return T;
}

void FunctionTypeProto::Emit(llvm::Serializer& S) const {
  S.EmitInt(NumArgs);
  EmitFunctionTypeInternal(S);
  
  for (arg_type_iterator i = arg_type_begin(), e = arg_type_end(); i!=e; ++i)
    S.Emit(*i);    
}

FunctionTypeProto* FunctionTypeProto::Materialize(llvm::Deserializer& D) {
  unsigned NumArgs = D.ReadInt();
  
  FunctionTypeProto *FTP = 
  (FunctionTypeProto*)malloc(sizeof(FunctionTypeProto) + 
                             NumArgs*sizeof(QualType));
  
  // Default construct.  Internal fields will be populated using
  // deserialization.
  new (FTP) FunctionTypeProto();
  
  FTP->NumArgs = NumArgs;
  FTP->ReadFunctionTypeInternal(D);
  
  // Fill in the trailing argument array.
  QualType *ArgInfo = reinterpret_cast<QualType *>(FTP+1);;

  for (unsigned i = 0; i != NumArgs; ++i)
    D.Read(ArgInfo[i]);
  
  return FTP;
}

void TypedefType::Emit(llvm::Serializer& S) const {
  EmitTypeInternal(S);
  S.EmitPtr(Decl);
}

TypedefType* TypedefType::Materialize(llvm::Deserializer& D) {
  TypedefType* T = new TypedefType(NULL,QualType());
  T->ReadTypeInternal(D);
  D.ReadPtr(T->Decl);
  return T;
}

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
#include "clang/AST/ASTContext.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using namespace clang;
using llvm::Serializer;
using llvm::Deserializer;
using llvm::SerializedPtrID;


void QualType::Emit(Serializer& S) const {
  S.EmitPtr(getAsOpaquePtr());
  S.EmitInt(getQualifiers());
}

QualType QualType::ReadVal(Deserializer& D) {
  QualType Q;
  D.ReadUIntPtr(Q.ThePtr,false);
  Q.ThePtr |= D.ReadInt();
  return Q;
}

//===----------------------------------------------------------------------===//
// Type Serialization: Dispatch code to handle specific types.
//===----------------------------------------------------------------------===//

void Type::Emit(Serializer& S) const {
  S.EmitInt(getTypeClass());
  S.EmitPtr(this);
  
  if (!isa<BuiltinType>(this))
    EmitImpl(S);
}

void Type::EmitImpl(Serializer& S) const {
  assert (false && "Serializization for type not supported.");
}

void Type::Create(ASTContext& Context, unsigned i, Deserializer& D) {
  Type::TypeClass K = static_cast<Type::TypeClass>(D.ReadInt());
  SerializedPtrID PtrID = D.ReadPtrID();  
  
  switch (K) {
    default:
      assert (false && "Deserialization for type not supported.");
      break;
            
    case Type::Builtin:
      assert (i < Context.getTypes().size());
      assert (isa<BuiltinType>(Context.getTypes()[i]));
      D.RegisterPtr(PtrID,Context.getTypes()[i]); 
      break;
      
    case Type::Complex:
      D.RegisterPtr(PtrID,ComplexType::CreateImpl(Context,D));
      break;
      
    case Type::FunctionProto:
      D.RegisterPtr(PtrID,FunctionTypeProto::CreateImpl(Context,D));
      break;
      
    case Type::Pointer:
      D.RegisterPtr(PtrID,PointerType::CreateImpl(Context,D));
      break;      
  }
}

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

void ComplexType::EmitImpl(Serializer& S) const {
  S.Emit(getElementType());
}

Type* ComplexType::CreateImpl(ASTContext& Context, Deserializer& D) {
  return Context.getComplexType(QualType::ReadVal(D)).getTypePtr();
}

//===----------------------------------------------------------------------===//
// FunctionTypeProto
//===----------------------------------------------------------------------===//

void FunctionTypeProto::EmitImpl(Serializer& S) const {
  S.Emit(getResultType());
  S.EmitBool(isVariadic());
  S.EmitInt(getNumArgs());
  
  for (arg_type_iterator I=arg_type_begin(), E=arg_type_end(); I!=E; ++I)
    S.Emit(*I);
}

Type* FunctionTypeProto::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType ResultType = QualType::ReadVal(D);
  bool isVariadic = D.ReadBool();
  unsigned NumArgs = D.ReadInt();
  
  llvm::SmallVector<QualType,15> Args;

  for (unsigned j = 0; j < NumArgs; ++j)
    Args.push_back(QualType::ReadVal(D));

  return Context.getFunctionType(ResultType,&*Args.begin(), 
                           NumArgs,isVariadic).getTypePtr();
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

void PointerType::EmitImpl(Serializer& S) const {
  S.Emit(getPointeeType());
}

Type* PointerType::CreateImpl(ASTContext& Context, Deserializer& D) {
  return Context.getPointerType(QualType::ReadVal(D)).getTypePtr();
}

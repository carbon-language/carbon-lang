//===--- TypeSerialization.cpp - Serialization of Decls ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines methods that implement bitcode serialization for Types.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using namespace clang;
using llvm::Serializer;
using llvm::Deserializer;
using llvm::SerializedPtrID;


void QualType::Emit(Serializer& S) const {
  S.EmitPtr(getTypePtr());
  S.EmitInt(getCVRQualifiers());
}

QualType QualType::ReadVal(Deserializer& D) {
  uintptr_t Val;
  D.ReadUIntPtr(Val, false);
  return QualType(reinterpret_cast<Type*>(Val), D.ReadInt());
}

void QualType::ReadBackpatch(Deserializer& D) {
  uintptr_t Val;
  D.ReadUIntPtr(Val, false);
  
  Value.setPointer(reinterpret_cast<Type*>(Val));
  Value.setInt(D.ReadInt());
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
      
    case Type::ASQual:
      D.RegisterPtr(PtrID,ASQualType::CreateImpl(Context,D));
      break;
    
    case Type::Complex:
      D.RegisterPtr(PtrID,ComplexType::CreateImpl(Context,D));
      break;
      
    case Type::ConstantArray:
      D.RegisterPtr(PtrID,ConstantArrayType::CreateImpl(Context,D));
      break;
      
    case Type::FunctionNoProto:
      D.RegisterPtr(PtrID,FunctionTypeNoProto::CreateImpl(Context,D));
      break;
      
    case Type::FunctionProto:
      D.RegisterPtr(PtrID,FunctionTypeProto::CreateImpl(Context,D));
      break;
      
    case Type::IncompleteArray:
      D.RegisterPtr(PtrID,IncompleteArrayType::CreateImpl(Context,D));
      break;

    case Type::MemberPointer:
      D.RegisterPtr(PtrID, MemberPointerType::CreateImpl(Context, D));
      break;

    case Type::Pointer:
      D.RegisterPtr(PtrID, PointerType::CreateImpl(Context, D));
      break;

    case Type::BlockPointer:
      D.RegisterPtr(PtrID, BlockPointerType::CreateImpl(Context, D));
      break;

    case Type::Reference:
      D.RegisterPtr(PtrID, ReferenceType::CreateImpl(Context, D));
      break;

    case Type::Tagged:
      D.RegisterPtr(PtrID, TagType::CreateImpl(Context, D));
      break;

    case Type::TypeName:
      D.RegisterPtr(PtrID, TypedefType::CreateImpl(Context, D));
      break;

    case Type::TypeOfExp:
      D.RegisterPtr(PtrID, TypeOfExpr::CreateImpl(Context, D));
      break;

    case Type::TypeOfTyp:
      D.RegisterPtr(PtrID, TypeOfType::CreateImpl(Context, D));
      break;

    case Type::TemplateTypeParm:
      D.RegisterPtr(PtrID,TemplateTypeParmType::CreateImpl(Context, D));
      break;

    case Type::VariableArray:
      D.RegisterPtr(PtrID, VariableArrayType::CreateImpl(Context, D));
      break;
  }
}

//===----------------------------------------------------------------------===//
// ASQualType
//===----------------------------------------------------------------------===//

void ASQualType::EmitImpl(Serializer& S) const {
  S.EmitPtr(getBaseType());
  S.EmitInt(getAddressSpace());
}

Type* ASQualType::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType BaseTy = QualType::ReadVal(D);
  unsigned AddressSpace = D.ReadInt();
  return Context.getASQualType(BaseTy, AddressSpace).getTypePtr();
}

//===----------------------------------------------------------------------===//
// BlockPointerType
//===----------------------------------------------------------------------===//

void BlockPointerType::EmitImpl(Serializer& S) const {
  S.Emit(getPointeeType());
}

Type* BlockPointerType::CreateImpl(ASTContext& Context, Deserializer& D) {
  return Context.getBlockPointerType(QualType::ReadVal(D)).getTypePtr();
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
// ConstantArray
//===----------------------------------------------------------------------===//

void ConstantArrayType::EmitImpl(Serializer& S) const {
  S.Emit(getElementType());
  S.EmitInt(getSizeModifier());
  S.EmitInt(getIndexTypeQualifier());
  S.Emit(Size);
}

Type* ConstantArrayType::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType ElTy = QualType::ReadVal(D);
  ArraySizeModifier am = static_cast<ArraySizeModifier>(D.ReadInt());
  unsigned ITQ = D.ReadInt();

  llvm::APInt Size;
  D.Read(Size);

  return Context.getConstantArrayType(ElTy,Size,am,ITQ).getTypePtr();
}

//===----------------------------------------------------------------------===//
// FunctionTypeNoProto
//===----------------------------------------------------------------------===//

void FunctionTypeNoProto::EmitImpl(Serializer& S) const {
  S.Emit(getResultType());
}

Type* FunctionTypeNoProto::CreateImpl(ASTContext& Context, Deserializer& D) {
  return Context.getFunctionTypeNoProto(QualType::ReadVal(D)).getTypePtr();
}

//===----------------------------------------------------------------------===//
// FunctionTypeProto
//===----------------------------------------------------------------------===//

void FunctionTypeProto::EmitImpl(Serializer& S) const {
  S.Emit(getResultType());
  S.EmitBool(isVariadic());
  S.EmitInt(getTypeQuals());
  S.EmitInt(getNumArgs());
  
  for (arg_type_iterator I=arg_type_begin(), E=arg_type_end(); I!=E; ++I)
    S.Emit(*I);
}

Type* FunctionTypeProto::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType ResultType = QualType::ReadVal(D);
  bool isVariadic = D.ReadBool();
  unsigned TypeQuals = D.ReadInt();
  unsigned NumArgs = D.ReadInt();
  
  llvm::SmallVector<QualType,15> Args;
  
  for (unsigned j = 0; j < NumArgs; ++j)
    Args.push_back(QualType::ReadVal(D));
  
  return Context.getFunctionType(ResultType,&*Args.begin(), 
                                 NumArgs,isVariadic,TypeQuals).getTypePtr();
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

//===----------------------------------------------------------------------===//
// ReferenceType
//===----------------------------------------------------------------------===//

void ReferenceType::EmitImpl(Serializer& S) const {
  S.Emit(getPointeeType());
}

Type* ReferenceType::CreateImpl(ASTContext& Context, Deserializer& D) {
  return Context.getReferenceType(QualType::ReadVal(D)).getTypePtr();
}

//===----------------------------------------------------------------------===//
// MemberPointerType
//===----------------------------------------------------------------------===//

void MemberPointerType::EmitImpl(Serializer& S) const {
  S.Emit(getPointeeType());
  S.Emit(QualType(Class, 0));
}

Type* MemberPointerType::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType Pointee = QualType::ReadVal(D);
  QualType Class = QualType::ReadVal(D);
  return Context.getMemberPointerType(Pointee, Class.getTypePtr()).getTypePtr();
}

//===----------------------------------------------------------------------===//
// TagType
//===----------------------------------------------------------------------===//

void TagType::EmitImpl(Serializer& S) const {
  S.EmitOwnedPtr(getDecl());
}

Type* TagType::CreateImpl(ASTContext& Context, Deserializer& D) {
  std::vector<Type*>& Types = 
    const_cast<std::vector<Type*>&>(Context.getTypes());
  
  TagType* T = new TagType(NULL,QualType());
  Types.push_back(T);
  
  // Deserialize the decl.
  T->decl.setPointer(cast<TagDecl>(D.ReadOwnedPtr<Decl>(Context)));
  T->decl.setInt(0);

  return T;
}

//===----------------------------------------------------------------------===//
// TypedefType
//===----------------------------------------------------------------------===//

void TypedefType::EmitImpl(Serializer& S) const {
  S.Emit(getCanonicalTypeInternal());
  S.EmitPtr(Decl);
}

Type* TypedefType::CreateImpl(ASTContext& Context, Deserializer& D) {
  std::vector<Type*>& Types = 
    const_cast<std::vector<Type*>&>(Context.getTypes());
  
  TypedefType* T = new TypedefType(Type::TypeName, NULL, QualType::ReadVal(D));
  Types.push_back(T);
  
  D.ReadPtr(T->Decl); // May be backpatched.
  return T;
}

//===----------------------------------------------------------------------===//
// TypeOfExpr
//===----------------------------------------------------------------------===//

void TypeOfExpr::EmitImpl(llvm::Serializer& S) const {
  S.EmitOwnedPtr(TOExpr);
}

Type* TypeOfExpr::CreateImpl(ASTContext& Context, Deserializer& D) {
  Expr* E = D.ReadOwnedPtr<Expr>(Context);
  
  std::vector<Type*>& Types = 
    const_cast<std::vector<Type*>&>(Context.getTypes());

  TypeOfExpr* T = new TypeOfExpr(E, Context.getCanonicalType(E->getType()));
  Types.push_back(T);

  return T;
}

//===----------------------------------------------------------------------===//
// TypeOfType
//===----------------------------------------------------------------------===//

void TypeOfType::EmitImpl(llvm::Serializer& S) const {
  S.Emit(TOType);
}

Type* TypeOfType::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType TOType = QualType::ReadVal(D);

  std::vector<Type*>& Types = 
    const_cast<std::vector<Type*>&>(Context.getTypes());

  TypeOfType* T = new TypeOfType(TOType, Context.getCanonicalType(TOType));
  Types.push_back(T);

  return T;
}
  
//===----------------------------------------------------------------------===//
// TemplateTypeParmType
//===----------------------------------------------------------------------===//

void TemplateTypeParmType::EmitImpl(Serializer& S) const {
  S.EmitPtr(getDecl());
}

Type* TemplateTypeParmType::CreateImpl(ASTContext& Context, Deserializer& D) {
  std::vector<Type*>& Types = 
    const_cast<std::vector<Type*>&>(Context.getTypes());
  
  TemplateTypeParmType* T = new TemplateTypeParmType(NULL);
  Types.push_back(T);
  
  D.ReadPtr(T->Decl); // May be backpatched.
  return T;
}
  
//===----------------------------------------------------------------------===//
// VariableArrayType
//===----------------------------------------------------------------------===//

void VariableArrayType::EmitImpl(Serializer& S) const {
  S.Emit(getElementType());
  S.EmitInt(getSizeModifier());
  S.EmitInt(getIndexTypeQualifier());
  S.EmitOwnedPtr(SizeExpr);
}

Type* VariableArrayType::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType ElTy = QualType::ReadVal(D);
  ArraySizeModifier am = static_cast<ArraySizeModifier>(D.ReadInt());
  unsigned ITQ = D.ReadInt();  
  Expr* SizeExpr = D.ReadOwnedPtr<Expr>(Context);
  
  return Context.getVariableArrayType(ElTy,SizeExpr,am,ITQ).getTypePtr();
}

//===----------------------------------------------------------------------===//
// DependentSizedArrayType
//===----------------------------------------------------------------------===//

void DependentSizedArrayType::EmitImpl(Serializer& S) const {
  S.Emit(getElementType());
  S.EmitInt(getSizeModifier());
  S.EmitInt(getIndexTypeQualifier());
  S.EmitOwnedPtr(SizeExpr);
}

Type* DependentSizedArrayType::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType ElTy = QualType::ReadVal(D);
  ArraySizeModifier am = static_cast<ArraySizeModifier>(D.ReadInt());
  unsigned ITQ = D.ReadInt();  
  Expr* SizeExpr = D.ReadOwnedPtr<Expr>(Context);
  
  return Context.getDependentSizedArrayType(ElTy,SizeExpr,am,ITQ).getTypePtr();
}

//===----------------------------------------------------------------------===//
// IncompleteArrayType
//===----------------------------------------------------------------------===//

void IncompleteArrayType::EmitImpl(Serializer& S) const {
  S.Emit(getElementType());
  S.EmitInt(getSizeModifier());
  S.EmitInt(getIndexTypeQualifier());
}

Type* IncompleteArrayType::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType ElTy = QualType::ReadVal(D);
  ArraySizeModifier am = static_cast<ArraySizeModifier>(D.ReadInt());
  unsigned ITQ = D.ReadInt();

  return Context.getIncompleteArrayType(ElTy,am,ITQ).getTypePtr();
}

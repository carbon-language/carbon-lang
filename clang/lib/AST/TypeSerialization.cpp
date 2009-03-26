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
#include "clang/AST/DeclTemplate.h"
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
      
    case Type::ExtQual:
      D.RegisterPtr(PtrID,ExtQualType::CreateImpl(Context,D));
      break;
    
    case Type::Complex:
      D.RegisterPtr(PtrID,ComplexType::CreateImpl(Context,D));
      break;
      
    case Type::ConstantArray:
      D.RegisterPtr(PtrID,ConstantArrayType::CreateImpl(Context,D));
      break;
      
    case Type::FunctionNoProto:
      D.RegisterPtr(PtrID,FunctionNoProtoType::CreateImpl(Context,D));
      break;
      
    case Type::FunctionProto:
      D.RegisterPtr(PtrID,FunctionProtoType::CreateImpl(Context,D));
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

    case Type::LValueReference:
      D.RegisterPtr(PtrID, LValueReferenceType::CreateImpl(Context, D));
      break;

    case Type::RValueReference:
      D.RegisterPtr(PtrID, RValueReferenceType::CreateImpl(Context, D));
      break;

    case Type::Record:
    case Type::Enum:
      // FIXME: Implement this!
      assert(false && "Can't deserialize tag types!");
      break;

    case Type::Typedef:
      D.RegisterPtr(PtrID, TypedefType::CreateImpl(Context, D));
      break;

    case Type::TypeOfExpr:
      D.RegisterPtr(PtrID, TypeOfExprType::CreateImpl(Context, D));
      break;

    case Type::TypeOf:
      D.RegisterPtr(PtrID, TypeOfType::CreateImpl(Context, D));
      break;

    case Type::TemplateTypeParm:
      D.RegisterPtr(PtrID, TemplateTypeParmType::CreateImpl(Context, D));
      break;

    case Type::VariableArray:
      D.RegisterPtr(PtrID, VariableArrayType::CreateImpl(Context, D));
      break;
  }
}

//===----------------------------------------------------------------------===//
// ExtQualType
//===----------------------------------------------------------------------===//

void ExtQualType::EmitImpl(Serializer& S) const {
  S.EmitPtr(getBaseType());
  S.EmitInt(getAddressSpace());
}

Type* ExtQualType::CreateImpl(ASTContext& Context, Deserializer& D) {
  QualType BaseTy = QualType::ReadVal(D);
  unsigned AddressSpace = D.ReadInt();
  return Context.getAddrSpaceQualType(BaseTy, AddressSpace).getTypePtr();
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
// FunctionNoProtoType
//===----------------------------------------------------------------------===//

void FunctionNoProtoType::EmitImpl(Serializer& S) const {
  S.Emit(getResultType());
}

Type* FunctionNoProtoType::CreateImpl(ASTContext& Context, Deserializer& D) {
  return Context.getFunctionNoProtoType(QualType::ReadVal(D)).getTypePtr();
}

//===----------------------------------------------------------------------===//
// FunctionProtoType
//===----------------------------------------------------------------------===//

void FunctionProtoType::EmitImpl(Serializer& S) const {
  S.Emit(getResultType());
  S.EmitBool(isVariadic());
  S.EmitInt(getTypeQuals());
  S.EmitInt(getNumArgs());
  
  for (arg_type_iterator I=arg_type_begin(), E=arg_type_end(); I!=E; ++I)
    S.Emit(*I);
}

Type* FunctionProtoType::CreateImpl(ASTContext& Context, Deserializer& D) {
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

Type* LValueReferenceType::CreateImpl(ASTContext& Context, Deserializer& D) {
  return Context.getLValueReferenceType(QualType::ReadVal(D)).getTypePtr();
}

Type* RValueReferenceType::CreateImpl(ASTContext& Context, Deserializer& D) {
  return Context.getRValueReferenceType(QualType::ReadVal(D)).getTypePtr();
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
  
  // FIXME: This is wrong: we need the subclasses to do the
  // (de-)serialization.
  TagType* T = new TagType(Record, NULL,QualType());
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
  
  TypedefType* T = new TypedefType(Type::Typedef, NULL, QualType::ReadVal(D));
  Types.push_back(T);
  
  D.ReadPtr(T->Decl); // May be backpatched.
  return T;
}

//===----------------------------------------------------------------------===//
// TypeOfExprType
//===----------------------------------------------------------------------===//

void TypeOfExprType::EmitImpl(llvm::Serializer& S) const {
  S.EmitOwnedPtr(TOExpr);
}

Type* TypeOfExprType::CreateImpl(ASTContext& Context, Deserializer& D) {
  Expr* E = D.ReadOwnedPtr<Expr>(Context);
  
  std::vector<Type*>& Types = 
    const_cast<std::vector<Type*>&>(Context.getTypes());

  TypeOfExprType* T 
    = new TypeOfExprType(E, Context.getCanonicalType(E->getType()));
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
  S.EmitInt(Depth);
  S.EmitInt(Index);
  S.EmitPtr(Name);
}

Type* TemplateTypeParmType::CreateImpl(ASTContext& Context, Deserializer& D) {
  unsigned Depth = D.ReadInt();
  unsigned Index = D.ReadInt();
  IdentifierInfo *Name = D.ReadPtr<IdentifierInfo>();
  return Context.getTemplateTypeParmType(Depth, Index, Name).getTypePtr();
}

//===----------------------------------------------------------------------===//
// ClassTemplateSpecializationType
//===----------------------------------------------------------------------===//

void ClassTemplateSpecializationType::EmitImpl(Serializer& S) const {
  S.Emit(getCanonicalTypeInternal());
  S.EmitPtr(Template);
  S.EmitInt(NumArgs);
  // FIXME: Serialize class template specialization types
}

Type* 
ClassTemplateSpecializationType::
CreateImpl(ASTContext& Context, Deserializer& D) {
  llvm::SmallVector<uintptr_t, 16> Args;
  llvm::SmallVector<bool, 16> ArgIsType;

  QualType Canon = QualType::ReadVal(D);
  TemplateDecl *Template = cast<TemplateDecl>(D.ReadPtr<Decl>());
  unsigned NumArgs = D.ReadInt();

  // FIXME: De-serialize class template specialization types
  (void)Template;
  (void)NumArgs;
  return 0;
}

//===----------------------------------------------------------------------===//
// QualifiedNameType
//===----------------------------------------------------------------------===//
void QualifiedNameType::EmitImpl(llvm::Serializer& S) const {
  // FIXME: Serialize the actual components
}

Type* 
QualifiedNameType::CreateImpl(ASTContext& Context, llvm::Deserializer& D) {
  // FIXME: Implement de-serialization
  return 0;
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

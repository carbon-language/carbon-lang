//===--- DeclSerialization.cpp - Serialization of Decls ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines methods that implement bitcode serialization for Decls.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using llvm::Serializer;
using llvm::Deserializer;
using llvm::SerializedPtrID;

using namespace clang;

//===----------------------------------------------------------------------===//
// Decl Serialization: Dispatch code to handle specialized decl types.
//===----------------------------------------------------------------------===//

void Decl::Emit(Serializer& S) const {
  S.EmitInt(getKind());
  EmitImpl(S);
  S.Emit(getLocation());
  S.EmitBool(InvalidDecl);
  // FIXME: HasAttrs?
  S.EmitBool(Implicit);
  S.EmitInt(Access);
  S.EmitPtr(cast_or_null<Decl>(getDeclContext()));  // From Decl.
  S.EmitPtr(cast_or_null<Decl>(getLexicalDeclContext()));  // From Decl.
  S.EmitPtr(NextDeclarator);
  if (const DeclContext *DC = dyn_cast<const DeclContext>(this))
    DC->EmitOutRec(S);
  
  if (getDeclContext() && 
      !getDeclContext()->isFunctionOrMethod()) {
    S.EmitBool(true);
    S.EmitOwnedPtr(NextDeclInContext);
  } else {
    S.EmitBool(false);
    S.EmitPtr(NextDeclInContext);
  }
}

Decl* Decl::Create(Deserializer& D, ASTContext& C) {

  Decl *Dcl;
  Kind k = static_cast<Kind>(D.ReadInt());

  switch (k) {
    default:
      assert (false && "Not implemented.");

    case TranslationUnit:
      Dcl = TranslationUnitDecl::CreateImpl(D, C);
      break;

    case Namespace:
      Dcl = NamespaceDecl::CreateImpl(D, C);
      break;

    case Var:
      Dcl = VarDecl::CreateImpl(D, C);
      break;
      
    case Enum:
      Dcl = EnumDecl::CreateImpl(D, C);
      break;
      
    case EnumConstant:
      Dcl = EnumConstantDecl::CreateImpl(D, C);
      break;
      
    case Field:
      Dcl = FieldDecl::CreateImpl(D, C);
      break;
      
    case ParmVar:
      Dcl = ParmVarDecl::CreateImpl(D, C);
      break;
      
    case OriginalParmVar:
      Dcl = OriginalParmVarDecl::CreateImpl(D, C);
      break;
      
    case Function:
      Dcl = FunctionDecl::CreateImpl(D, C);
      break;

    case OverloadedFunction:
      Dcl = OverloadedFunctionDecl::CreateImpl(D, C);
      break;

    case Record:
      Dcl = RecordDecl::CreateImpl(D, C);
      break;
      
    case Typedef:
      Dcl = TypedefDecl::CreateImpl(D, C);
      break;
      
    case TemplateTypeParm:
      Dcl = TemplateTypeParmDecl::CreateImpl(D, C);
      break;

    case FileScopeAsm:
      Dcl = FileScopeAsmDecl::CreateImpl(D, C);
      break;
  }

  Dcl->Loc = SourceLocation::ReadVal(D);                 // From Decl.
  Dcl->InvalidDecl = D.ReadBool();
  // FIXME: HasAttrs?
  Dcl->Implicit = D.ReadBool();
  Dcl->Access = D.ReadInt();

  assert(Dcl->DeclCtx.getOpaqueValue() == 0);

  const SerializedPtrID &SemaDCPtrID = D.ReadPtrID();
  const SerializedPtrID &LexicalDCPtrID = D.ReadPtrID();

  if (SemaDCPtrID == LexicalDCPtrID) {
    // Allow back-patching.  Observe that we register the variable of the
    // *object* for back-patching. Its actual value will get filled in later.
    uintptr_t X;
    D.ReadUIntPtr(X, SemaDCPtrID); 
    Dcl->DeclCtx.setFromOpaqueValue(reinterpret_cast<void*>(X));
  }
  else {
    MultipleDC *MDC = new MultipleDC();
    Dcl->DeclCtx.setPointer(MDC);
    Dcl->DeclCtx.setInt(true);
    // Allow back-patching.  Observe that we register the variable of the
    // *object* for back-patching. Its actual value will get filled in later.
    D.ReadPtr(MDC->SemanticDC, SemaDCPtrID);
    D.ReadPtr(MDC->LexicalDC, LexicalDCPtrID);
  }
  D.ReadPtr(Dcl->NextDeclarator);
  if (DeclContext *DC = dyn_cast<DeclContext>(Dcl))
    DC->ReadOutRec(D, C);
  bool OwnsNext = D.ReadBool();
  if (OwnsNext)
    Dcl->NextDeclInContext = D.ReadOwnedPtr<Decl>(C);
  else 
    D.ReadPtr(Dcl->NextDeclInContext);
  return Dcl;
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of DeclContext.
//===----------------------------------------------------------------------===//

void DeclContext::EmitOutRec(Serializer& S) const {
  bool Owned = !isFunctionOrMethod();
  S.EmitBool(Owned);
  if (Owned)
    S.EmitOwnedPtr(FirstDecl);
  else
    S.EmitPtr(FirstDecl);
  S.EmitPtr(LastDecl);
}

void DeclContext::ReadOutRec(Deserializer& D, ASTContext& C) {
  bool Owned = D.ReadBool();
  if (Owned)
    FirstDecl = cast_or_null<Decl>(D.ReadOwnedPtr<Decl>(C));
  else
    D.ReadPtr(FirstDecl);
  D.ReadPtr(LastDecl);
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of NamedDecl.
//===----------------------------------------------------------------------===//

void NamedDecl::EmitInRec(Serializer& S) const {
  S.EmitInt(Name.getNameKind());

  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
    S.EmitPtr(Name.getAsIdentifierInfo());
    break;

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    Name.getObjCSelector().Emit(S);
    break;

  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    Name.getCXXNameType().Emit(S);
    break;

  case DeclarationName::CXXOperatorName:
    S.EmitInt(Name.getCXXOverloadedOperator());
    break;

  case DeclarationName::CXXUsingDirective:
    // No extra data to emit
    break;
  }
}

void NamedDecl::ReadInRec(Deserializer& D, ASTContext& C) {
  DeclarationName::NameKind Kind 
    = static_cast<DeclarationName::NameKind>(D.ReadInt());
  switch (Kind) {
  case DeclarationName::Identifier: {
    // Don't allow back-patching.  The IdentifierInfo table must already
    // be loaded.
    Name = D.ReadPtr<IdentifierInfo>();
    break;
  }

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    Name = Selector::ReadVal(D);
    break;

  case DeclarationName::CXXConstructorName:
    Name = C.DeclarationNames.getCXXConstructorName(QualType::ReadVal(D));
    break;
                                                    
  case DeclarationName::CXXDestructorName:
    Name = C.DeclarationNames.getCXXDestructorName(QualType::ReadVal(D));
    break;

  case DeclarationName::CXXConversionFunctionName:
    Name 
      = C.DeclarationNames.getCXXConversionFunctionName(QualType::ReadVal(D));
    break;

  case DeclarationName::CXXOperatorName: {
    OverloadedOperatorKind Op 
      = static_cast<OverloadedOperatorKind>(D.ReadInt());
    Name = C.DeclarationNames.getCXXOperatorName(Op);
    break;
  }

  case DeclarationName::CXXUsingDirective: 
    Name = DeclarationName::getUsingDirectiveName();
    break;
  }
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of ValueDecl.
//===----------------------------------------------------------------------===//

void ValueDecl::EmitInRec(Serializer& S) const {
  NamedDecl::EmitInRec(S);
  S.Emit(getType());                        // From ValueDecl.
}

void ValueDecl::ReadInRec(Deserializer& D, ASTContext& C) {
  NamedDecl::ReadInRec(D, C);
  DeclType = QualType::ReadVal(D);          // From ValueDecl.
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of VarDecl.
//===----------------------------------------------------------------------===//

void VarDecl::EmitInRec(Serializer& S) const {
  ValueDecl::EmitInRec(S);
  S.EmitInt(getStorageClass());             // From VarDecl.
}

void VarDecl::ReadInRec(Deserializer& D, ASTContext& C) {
  ValueDecl::ReadInRec(D, C);
  SClass = static_cast<StorageClass>(D.ReadInt());  // From VarDecl. 
}

void VarDecl::EmitOutRec(Serializer& S) const {
  // Emit this last because it will create a record of its own.
  S.EmitOwnedPtr(getInit());
}

void VarDecl::ReadOutRec(Deserializer& D, ASTContext& C) {
  Init = D.ReadOwnedPtr<Stmt>(C);
}


void VarDecl::EmitImpl(Serializer& S) const {
  VarDecl::EmitInRec(S);
  VarDecl::EmitOutRec(S);
}

void VarDecl::ReadImpl(Deserializer& D, ASTContext& C) {
  ReadInRec(D, C);
  ReadOutRec(D, C);
}

//===----------------------------------------------------------------------===//
//      TranslationUnitDecl Serialization.
//===----------------------------------------------------------------------===//

void TranslationUnitDecl::EmitImpl(llvm::Serializer& S) const
{
}

TranslationUnitDecl* TranslationUnitDecl::CreateImpl(Deserializer& D,
                                                     ASTContext& C) {  
  return new (C) TranslationUnitDecl();
}

//===----------------------------------------------------------------------===//
//      NamespaceDecl Serialization.
//===----------------------------------------------------------------------===//

void NamespaceDecl::EmitImpl(llvm::Serializer& S) const
{
  NamedDecl::EmitInRec(S);
  S.Emit(getLBracLoc());
  S.Emit(getRBracLoc());
}

NamespaceDecl* NamespaceDecl::CreateImpl(Deserializer& D, ASTContext& C) {  
  NamespaceDecl* decl = new (C) NamespaceDecl(0, SourceLocation(), 0);
 
  decl->NamedDecl::ReadInRec(D, C);
  decl->LBracLoc = SourceLocation::ReadVal(D);
  decl->RBracLoc = SourceLocation::ReadVal(D);
  
  return decl;
}

//===----------------------------------------------------------------------===//
//      VarDecl Serialization.
//===----------------------------------------------------------------------===//

VarDecl* VarDecl::CreateImpl(Deserializer& D, ASTContext& C) {  
  VarDecl* decl =
    new (C) VarDecl(Var, 0, SourceLocation(), NULL, QualType(), None);
 
  decl->VarDecl::ReadImpl(D, C);
  return decl;
}

//===----------------------------------------------------------------------===//
//      ParmVarDecl Serialization.
//===----------------------------------------------------------------------===//

void ParmVarDecl::EmitImpl(llvm::Serializer& S) const {
  VarDecl::EmitImpl(S);
  S.EmitInt(getObjCDeclQualifier());        // From ParmVarDecl.
  S.EmitOwnedPtr(getDefaultArg());          // From ParmVarDecl.
}

ParmVarDecl* ParmVarDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  ParmVarDecl* decl = new (C)
    ParmVarDecl(ParmVar,
                0, SourceLocation(), NULL, QualType(), None, NULL);
  
  decl->VarDecl::ReadImpl(D, C);
  decl->objcDeclQualifier = static_cast<ObjCDeclQualifier>(D.ReadInt());
  decl->DefaultArg = D.ReadOwnedPtr<Expr>(C);
  return decl;
}

//===----------------------------------------------------------------------===//
//      OriginalParmVarDecl Serialization.
//===----------------------------------------------------------------------===//

void OriginalParmVarDecl::EmitImpl(llvm::Serializer& S) const {
  ParmVarDecl::EmitImpl(S);
  S.Emit(OriginalType);
}

OriginalParmVarDecl* OriginalParmVarDecl::CreateImpl(
                                              Deserializer& D, ASTContext& C) {
  OriginalParmVarDecl* decl = new (C)
    OriginalParmVarDecl(0, SourceLocation(), NULL, QualType(), 
                                QualType(), None, NULL);
  
  decl->ParmVarDecl::ReadImpl(D, C);
  decl->OriginalType = QualType::ReadVal(D);
  return decl;
}
//===----------------------------------------------------------------------===//
//      EnumDecl Serialization.
//===----------------------------------------------------------------------===//

void EnumDecl::EmitImpl(Serializer& S) const {
  NamedDecl::EmitInRec(S);
  S.EmitBool(isDefinition());
  S.Emit(IntegerType);
}

EnumDecl* EnumDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  EnumDecl* decl = new (C) EnumDecl(0, SourceLocation(), NULL);
  
  decl->NamedDecl::ReadInRec(D, C);
  decl->setDefinition(D.ReadBool());
  decl->IntegerType = QualType::ReadVal(D);
  
  return decl;
}

//===----------------------------------------------------------------------===//
//      EnumConstantDecl Serialization.
//===----------------------------------------------------------------------===//

void EnumConstantDecl::EmitImpl(Serializer& S) const {
  S.Emit(Val);
  ValueDecl::EmitInRec(S);
  S.EmitOwnedPtr(Init);
}
 
EnumConstantDecl* EnumConstantDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  llvm::APSInt val(1);
  D.Read(val);
  
  EnumConstantDecl* decl = new (C)
    EnumConstantDecl(0, SourceLocation(), NULL, QualType(), NULL, val);
  
  decl->ValueDecl::ReadInRec(D, C);
  decl->Init = D.ReadOwnedPtr<Stmt>(C);
  return decl;    
}

//===----------------------------------------------------------------------===//
//      FieldDecl Serialization.
//===----------------------------------------------------------------------===//

void FieldDecl::EmitImpl(Serializer& S) const {
  S.EmitBool(Mutable);
  S.Emit(getType());
  ValueDecl::EmitInRec(S);
  S.EmitOwnedPtr(BitWidth);  
}

FieldDecl* FieldDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  FieldDecl* decl = new (C) FieldDecl(Field, 0, SourceLocation(), NULL, 
                                        QualType(), 0, false);
  decl->Mutable = D.ReadBool();
  decl->ValueDecl::ReadInRec(D, C);
  decl->BitWidth = D.ReadOwnedPtr<Expr>(C);
  return decl;
}

//===----------------------------------------------------------------------===//
//      FunctionDecl Serialization.
//===----------------------------------------------------------------------===//

void FunctionDecl::EmitImpl(Serializer& S) const {
  S.EmitInt(SClass);           // From FunctionDecl.
  S.EmitBool(IsInline);        // From FunctionDecl.
  ValueDecl::EmitInRec(S);
  S.EmitPtr(PreviousDeclaration);
  
  // NOTE: We do not need to serialize out the number of parameters, because
  //  that is encoded in the type (accessed via getNumParams()).
  
  if (ParamInfo != NULL) {
    S.EmitBool(true);
    S.EmitInt(getNumParams());
    S.BatchEmitOwnedPtrs(getNumParams(),&ParamInfo[0], Body);
  }
  else {
    S.EmitBool(false);
    S.EmitOwnedPtr(Body);
  }
}

FunctionDecl* FunctionDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  StorageClass SClass = static_cast<StorageClass>(D.ReadInt());
  bool IsInline = D.ReadBool();
  
  FunctionDecl* decl = new (C)
    FunctionDecl(Function, 0, SourceLocation(), DeclarationName(),
                 QualType(), SClass, IsInline);
  
  decl->ValueDecl::ReadInRec(D, C);
  D.ReadPtr(decl->PreviousDeclaration);

  int numParams = 0;
  bool hasParamDecls = D.ReadBool();
  if (hasParamDecls)
    numParams = D.ReadInt();
    
  decl->ParamInfo = hasParamDecls
                  ? new ParmVarDecl*[numParams] 
                  : NULL;  
  
  if (hasParamDecls)
    D.BatchReadOwnedPtrs(numParams,
                         reinterpret_cast<Decl**>(&decl->ParamInfo[0]),
                         decl->Body, C);
  else
    decl->Body = D.ReadOwnedPtr<Stmt>(C);
  
  return decl;
}

void BlockDecl::EmitImpl(Serializer& S) const {
  // FIXME: what about arguments?
  S.Emit(getCaretLocation());
  S.EmitOwnedPtr(Body);
}

BlockDecl* BlockDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  QualType Q = QualType::ReadVal(D);
  SourceLocation L = SourceLocation::ReadVal(D);
  /*CompoundStmt* BodyStmt = cast<CompoundStmt>(*/D.ReadOwnedPtr<Stmt>(C)/*)*/;
  assert(0 && "Cannot deserialize BlockBlockExpr yet");
  // FIXME: need to handle parameters.
  //return new BlockBlockExpr(L, Q, BodyStmt);
  return 0;
}

//===----------------------------------------------------------------------===//
//      OverloadedFunctionDecl Serialization.
//===----------------------------------------------------------------------===//

void OverloadedFunctionDecl::EmitImpl(Serializer& S) const {
  NamedDecl::EmitInRec(S);

  S.EmitInt(getNumFunctions());
  for (unsigned func = 0; func < getNumFunctions(); ++func)
    S.EmitPtr(Functions[func]);
}

OverloadedFunctionDecl * 
OverloadedFunctionDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  OverloadedFunctionDecl* decl = new (C)
    OverloadedFunctionDecl(0, DeclarationName());
  
  decl->NamedDecl::ReadInRec(D, C);

  unsigned numFunctions = D.ReadInt();
  decl->Functions.reserve(numFunctions);
  for (unsigned func = 0; func < numFunctions; ++func)
    D.ReadPtr(decl->Functions[func]);
  
  return decl;
}

//===----------------------------------------------------------------------===//
//      RecordDecl Serialization.
//===----------------------------------------------------------------------===//

void RecordDecl::EmitImpl(Serializer& S) const {
  S.EmitInt(getTagKind());

  NamedDecl::EmitInRec(S);
  S.EmitBool(isDefinition());
  S.EmitBool(hasFlexibleArrayMember());
  S.EmitBool(isAnonymousStructOrUnion());
}

RecordDecl* RecordDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  TagKind TK = TagKind(D.ReadInt());

  RecordDecl* decl = new (C) RecordDecl(Record, TK, 0, SourceLocation(), NULL);
    
  decl->NamedDecl::ReadInRec(D, C);
  decl->setDefinition(D.ReadBool());
  decl->setHasFlexibleArrayMember(D.ReadBool());
  decl->setAnonymousStructOrUnion(D.ReadBool());
    
  return decl;
}

//===----------------------------------------------------------------------===//
//      TypedefDecl Serialization.
//===----------------------------------------------------------------------===//

void TypedefDecl::EmitImpl(Serializer& S) const {
  S.Emit(UnderlyingType);
  NamedDecl::EmitInRec(S);
}

TypedefDecl* TypedefDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  QualType T = QualType::ReadVal(D);
  
  TypedefDecl* decl = new (C) TypedefDecl(0, SourceLocation(), NULL, T);
  
  decl->NamedDecl::ReadInRec(D, C);

  return decl;
}

//===----------------------------------------------------------------------===//
//      TemplateTypeParmDecl Serialization.
//===----------------------------------------------------------------------===//

void TemplateTypeParmDecl::EmitImpl(Serializer& S) const {
  S.EmitBool(Typename);
  TypeDecl::EmitInRec(S);
}

TemplateTypeParmDecl *
TemplateTypeParmDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  bool Typename = D.ReadBool();
  TemplateTypeParmDecl *decl
    = new (C) TemplateTypeParmDecl(0, SourceLocation(), 0, Typename, 
                                   QualType());
  decl->TypeDecl::ReadInRec(D, C);
  return decl;
}

//===----------------------------------------------------------------------===//
//      NonTypeTemplateParmDecl Serialization.
//===----------------------------------------------------------------------===//
void NonTypeTemplateParmDecl::EmitImpl(Serializer& S) const {
  S.EmitInt(Depth);
  S.EmitInt(Position);
  NamedDecl::Emit(S);
}

NonTypeTemplateParmDecl*
NonTypeTemplateParmDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  unsigned Depth = D.ReadInt();
  unsigned Position = D.ReadInt();
  NonTypeTemplateParmDecl *decl
    = new (C) NonTypeTemplateParmDecl(0, SourceLocation(), Depth, Position,
                                      0, QualType(), SourceLocation());
  decl->NamedDecl::ReadInRec(D, C);
  return decl;
}

//===----------------------------------------------------------------------===//
//      TemplateTemplateParmDecl Serialization.
//===----------------------------------------------------------------------===//
void TemplateTemplateParmDecl::EmitImpl(Serializer& S) const {
  S.EmitInt(Depth);
  S.EmitInt(Position);
  NamedDecl::EmitInRec(S);
}

TemplateTemplateParmDecl*
TemplateTemplateParmDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  unsigned Depth = D.ReadInt();
  unsigned Position = D.ReadInt();
  TemplateTemplateParmDecl *decl
    = new (C) TemplateTemplateParmDecl(0, SourceLocation(), Depth, Position,
                                       0, 0);
  decl->NamedDecl::ReadInRec(D, C);
  return decl;
}

//===----------------------------------------------------------------------===//
//      LinkageSpec Serialization.
//===----------------------------------------------------------------------===//

void LinkageSpecDecl::EmitInRec(Serializer& S) const {
  S.EmitInt(getLanguage());
  S.EmitBool(HadBraces);
}

void LinkageSpecDecl::ReadInRec(Deserializer& D, ASTContext& C) {
  Language = static_cast<LanguageIDs>(D.ReadInt());
  HadBraces = D.ReadBool();
}

//===----------------------------------------------------------------------===//
//      FileScopeAsm Serialization.
//===----------------------------------------------------------------------===//

void FileScopeAsmDecl::EmitImpl(llvm::Serializer& S) const
{
  S.EmitOwnedPtr(AsmString);
}

FileScopeAsmDecl* FileScopeAsmDecl::CreateImpl(Deserializer& D, ASTContext& C) { 
  FileScopeAsmDecl* decl = new (C) FileScopeAsmDecl(0, SourceLocation(), 0);

  decl->AsmString = cast<StringLiteral>(D.ReadOwnedPtr<Expr>(C));
//  D.ReadOwnedPtr(D.ReadOwnedPtr<StringLiteral>())<#T * * Ptr#>, <#bool AutoRegister#>)(decl->AsmString);
  
  return decl;
}

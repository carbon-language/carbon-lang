//===--- ASTImporter.cpp - Importing ASTs from other Contexts ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTImporter class which imports AST nodes from one
//  context into another context.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTImporter.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/TypeVisitor.h"

using namespace clang;

namespace {
  class ASTNodeImporter : public TypeVisitor<ASTNodeImporter, QualType> {
    ASTImporter &Importer;
    
  public:
    explicit ASTNodeImporter(ASTImporter &Importer) : Importer(Importer) { }
    
    using TypeVisitor<ASTNodeImporter, QualType>::Visit;

    // Importing types
    QualType VisitBuiltinType(BuiltinType *T);
    QualType VisitComplexType(ComplexType *T);
    QualType VisitPointerType(PointerType *T);
    QualType VisitBlockPointerType(BlockPointerType *T);
    QualType VisitLValueReferenceType(LValueReferenceType *T);
    QualType VisitRValueReferenceType(RValueReferenceType *T);
    QualType VisitMemberPointerType(MemberPointerType *T);
    QualType VisitConstantArrayType(ConstantArrayType *T);
    QualType VisitIncompleteArrayType(IncompleteArrayType *T);
    QualType VisitVariableArrayType(VariableArrayType *T);
    // FIXME: DependentSizedArrayType
    // FIXME: DependentSizedExtVectorType
    QualType VisitVectorType(VectorType *T);
    QualType VisitExtVectorType(ExtVectorType *T);
    QualType VisitFunctionNoProtoType(FunctionNoProtoType *T);
    QualType VisitFunctionProtoType(FunctionProtoType *T);
    // FIXME: UnresolvedUsingType
    QualType VisitTypedefType(TypedefType *T);
    QualType VisitTypeOfExprType(TypeOfExprType *T);
    // FIXME: DependentTypeOfExprType
    QualType VisitTypeOfType(TypeOfType *T);
    QualType VisitDecltypeType(DecltypeType *T);
    // FIXME: DependentDecltypeType
    QualType VisitRecordType(RecordType *T);
    QualType VisitEnumType(EnumType *T);
    QualType VisitElaboratedType(ElaboratedType *T);
    // FIXME: TemplateTypeParmType
    // FIXME: SubstTemplateTypeParmType
    // FIXME: TemplateSpecializationType
    QualType VisitQualifiedNameType(QualifiedNameType *T);
    // FIXME: TypenameType
    QualType VisitObjCInterfaceType(ObjCInterfaceType *T);
    QualType VisitObjCObjectPointerType(ObjCObjectPointerType *T);
  };
}

//----------------------------------------------------------------------------
// Import Types
//----------------------------------------------------------------------------

QualType ASTNodeImporter::VisitBuiltinType(BuiltinType *T) {
  switch (T->getKind()) {
  case BuiltinType::Void: return Importer.getToContext().VoidTy;
  case BuiltinType::Bool: return Importer.getToContext().BoolTy;
    
  case BuiltinType::Char_U:
    // The context we're importing from has an unsigned 'char'. If we're 
    // importing into a context with a signed 'char', translate to 
    // 'unsigned char' instead.
    if (Importer.getToContext().getLangOptions().CharIsSigned)
      return Importer.getToContext().UnsignedCharTy;
    
    return Importer.getToContext().CharTy;

  case BuiltinType::UChar: return Importer.getToContext().UnsignedCharTy;
    
  case BuiltinType::Char16:
    // FIXME: Make sure that the "to" context supports C++!
    return Importer.getToContext().Char16Ty;
    
  case BuiltinType::Char32: 
    // FIXME: Make sure that the "to" context supports C++!
    return Importer.getToContext().Char32Ty;

  case BuiltinType::UShort: return Importer.getToContext().UnsignedShortTy;
  case BuiltinType::UInt: return Importer.getToContext().UnsignedIntTy;
  case BuiltinType::ULong: return Importer.getToContext().UnsignedLongTy;
  case BuiltinType::ULongLong: 
    return Importer.getToContext().UnsignedLongLongTy;
  case BuiltinType::UInt128: return Importer.getToContext().UnsignedInt128Ty;
    
  case BuiltinType::Char_S:
    // The context we're importing from has an unsigned 'char'. If we're 
    // importing into a context with a signed 'char', translate to 
    // 'unsigned char' instead.
    if (!Importer.getToContext().getLangOptions().CharIsSigned)
      return Importer.getToContext().SignedCharTy;
    
    return Importer.getToContext().CharTy;

  case BuiltinType::SChar: return Importer.getToContext().SignedCharTy;
  case BuiltinType::WChar:
    // FIXME: If not in C++, shall we translate to the C equivalent of
    // wchar_t?
    return Importer.getToContext().WCharTy;
    
  case BuiltinType::Short : return Importer.getToContext().ShortTy;
  case BuiltinType::Int : return Importer.getToContext().IntTy;
  case BuiltinType::Long : return Importer.getToContext().LongTy;
  case BuiltinType::LongLong : return Importer.getToContext().LongLongTy;
  case BuiltinType::Int128 : return Importer.getToContext().Int128Ty;
  case BuiltinType::Float: return Importer.getToContext().FloatTy;
  case BuiltinType::Double: return Importer.getToContext().DoubleTy;
  case BuiltinType::LongDouble: return Importer.getToContext().LongDoubleTy;

  case BuiltinType::NullPtr:
    // FIXME: Make sure that the "to" context supports C++0x!
    return Importer.getToContext().NullPtrTy;
    
  case BuiltinType::Overload: return Importer.getToContext().OverloadTy;
  case BuiltinType::Dependent: return Importer.getToContext().DependentTy;
  case BuiltinType::UndeducedAuto: 
    // FIXME: Make sure that the "to" context supports C++0x!
    return Importer.getToContext().UndeducedAutoTy;

  case BuiltinType::ObjCId:
    // FIXME: Make sure that the "to" context supports Objective-C!
    return Importer.getToContext().ObjCBuiltinIdTy;
    
  case BuiltinType::ObjCClass:
    return Importer.getToContext().ObjCBuiltinClassTy;

  case BuiltinType::ObjCSel:
    return Importer.getToContext().ObjCBuiltinSelTy;
  }
  
  return QualType();
}

QualType ASTNodeImporter::VisitComplexType(ComplexType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getComplexType(ToElementType);
}

QualType ASTNodeImporter::VisitPointerType(PointerType *T) {
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getPointerType(ToPointeeType);
}

QualType ASTNodeImporter::VisitBlockPointerType(BlockPointerType *T) {
  // FIXME: Check for blocks support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getBlockPointerType(ToPointeeType);
}

QualType ASTNodeImporter::VisitLValueReferenceType(LValueReferenceType *T) {
  // FIXME: Check for C++ support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeTypeAsWritten());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getLValueReferenceType(ToPointeeType);
}

QualType ASTNodeImporter::VisitRValueReferenceType(RValueReferenceType *T) {
  // FIXME: Check for C++0x support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeTypeAsWritten());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getRValueReferenceType(ToPointeeType);  
}

QualType ASTNodeImporter::VisitMemberPointerType(MemberPointerType *T) {
  // FIXME: Check for C++ support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();
  
  QualType ClassType = Importer.Import(QualType(T->getClass(), 0));
  return Importer.getToContext().getMemberPointerType(ToPointeeType, 
                                                      ClassType.getTypePtr());
}

QualType ASTNodeImporter::VisitConstantArrayType(ConstantArrayType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getConstantArrayType(ToElementType, 
                                                      T->getSize(),
                                                      T->getSizeModifier(),
                                               T->getIndexTypeCVRQualifiers());
}

QualType ASTNodeImporter::VisitIncompleteArrayType(IncompleteArrayType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getIncompleteArrayType(ToElementType, 
                                                        T->getSizeModifier(),
                                                T->getIndexTypeCVRQualifiers());
}

QualType ASTNodeImporter::VisitVariableArrayType(VariableArrayType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();

  Expr *Size = Importer.Import(T->getSizeExpr());
  if (!Size)
    return QualType();
  
  SourceRange Brackets = Importer.Import(T->getBracketsRange());
  return Importer.getToContext().getVariableArrayType(ToElementType, Size,
                                                      T->getSizeModifier(),
                                                T->getIndexTypeCVRQualifiers(),
                                                      Brackets);
}

QualType ASTNodeImporter::VisitVectorType(VectorType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getVectorType(ToElementType, 
                                               T->getNumElements(),
                                               T->isAltiVec(),
                                               T->isPixel());
}

QualType ASTNodeImporter::VisitExtVectorType(ExtVectorType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getExtVectorType(ToElementType, 
                                                  T->getNumElements());
}

QualType ASTNodeImporter::VisitFunctionNoProtoType(FunctionNoProtoType *T) {
  // FIXME: What happens if we're importing a function without a prototype 
  // into C++? Should we make it variadic?
  QualType ToResultType = Importer.Import(T->getResultType());
  if (ToResultType.isNull())
    return QualType();
  
  return Importer.getToContext().getFunctionNoProtoType(ToResultType,
                                                        T->getNoReturnAttr(), 
                                                        T->getCallConv());
}

QualType ASTNodeImporter::VisitFunctionProtoType(FunctionProtoType *T) {
  QualType ToResultType = Importer.Import(T->getResultType());
  if (ToResultType.isNull())
    return QualType();
  
  // Import argument types
  llvm::SmallVector<QualType, 4> ArgTypes;
  for (FunctionProtoType::arg_type_iterator A = T->arg_type_begin(),
                                         AEnd = T->arg_type_end();
       A != AEnd; ++A) {
    QualType ArgType = Importer.Import(*A);
    if (ArgType.isNull())
      return QualType();
    ArgTypes.push_back(ArgType);
  }
  
  // Import exception types
  llvm::SmallVector<QualType, 4> ExceptionTypes;
  for (FunctionProtoType::exception_iterator E = T->exception_begin(),
                                          EEnd = T->exception_end();
       E != EEnd; ++E) {
    QualType ExceptionType = Importer.Import(*E);
    if (ExceptionType.isNull())
      return QualType();
    ExceptionTypes.push_back(ExceptionType);
  }
       
  return Importer.getToContext().getFunctionType(ToResultType, ArgTypes.data(),
                                                 ArgTypes.size(),
                                                 T->isVariadic(),
                                                 T->getTypeQuals(),
                                                 T->hasExceptionSpec(), 
                                                 T->hasAnyExceptionSpec(),
                                                 ExceptionTypes.size(),
                                                 ExceptionTypes.data(),
                                                 T->getNoReturnAttr(),
                                                 T->getCallConv());
}

QualType ASTNodeImporter::VisitTypedefType(TypedefType *T) {
  TypedefDecl *ToDecl
                 = dyn_cast_or_null<TypedefDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();
  
  return Importer.getToContext().getTypeDeclType(ToDecl);
}

QualType ASTNodeImporter::VisitTypeOfExprType(TypeOfExprType *T) {
  Expr *ToExpr = Importer.Import(T->getUnderlyingExpr());
  if (!ToExpr)
    return QualType();
  
  return Importer.getToContext().getTypeOfExprType(ToExpr);
}

QualType ASTNodeImporter::VisitTypeOfType(TypeOfType *T) {
  QualType ToUnderlyingType = Importer.Import(T->getUnderlyingType());
  if (ToUnderlyingType.isNull())
    return QualType();
  
  return Importer.getToContext().getTypeOfType(ToUnderlyingType);
}

QualType ASTNodeImporter::VisitDecltypeType(DecltypeType *T) {
  Expr *ToExpr = Importer.Import(T->getUnderlyingExpr());
  if (!ToExpr)
    return QualType();
  
  return Importer.getToContext().getDecltypeType(ToExpr);
}

QualType ASTNodeImporter::VisitRecordType(RecordType *T) {
  RecordDecl *ToDecl
    = dyn_cast_or_null<RecordDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getTagDeclType(ToDecl);
}

QualType ASTNodeImporter::VisitEnumType(EnumType *T) {
  EnumDecl *ToDecl
    = dyn_cast_or_null<EnumDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getTagDeclType(ToDecl);
}

QualType ASTNodeImporter::VisitElaboratedType(ElaboratedType *T) {
  QualType ToUnderlyingType = Importer.Import(T->getUnderlyingType());
  if (ToUnderlyingType.isNull())
    return QualType();

  return Importer.getToContext().getElaboratedType(ToUnderlyingType,
                                                   T->getTagKind());
}

QualType ASTNodeImporter::VisitQualifiedNameType(QualifiedNameType *T) {
  NestedNameSpecifier *ToQualifier = Importer.Import(T->getQualifier());
  if (!ToQualifier)
    return QualType();

  QualType ToNamedType = Importer.Import(T->getNamedType());
  if (ToNamedType.isNull())
    return QualType();

  return Importer.getToContext().getQualifiedNameType(ToQualifier, ToNamedType);
}

QualType ASTNodeImporter::VisitObjCInterfaceType(ObjCInterfaceType *T) {
  ObjCInterfaceDecl *Class
    = dyn_cast_or_null<ObjCInterfaceDecl>(Importer.Import(T->getDecl()));
  if (!Class)
    return QualType();

  llvm::SmallVector<ObjCProtocolDecl *, 4> Protocols;
  for (ObjCInterfaceType::qual_iterator P = T->qual_begin(), 
                                     PEnd = T->qual_end();
       P != PEnd; ++P) {
    ObjCProtocolDecl *Protocol
      = dyn_cast_or_null<ObjCProtocolDecl>(Importer.Import(*P));
    if (!Protocol)
      return QualType();
    Protocols.push_back(Protocol);
  }

  return Importer.getToContext().getObjCInterfaceType(Class,
                                                      Protocols.data(),
                                                      Protocols.size());
}

QualType ASTNodeImporter::VisitObjCObjectPointerType(ObjCObjectPointerType *T) {
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();

  llvm::SmallVector<ObjCProtocolDecl *, 4> Protocols;
  for (ObjCObjectPointerType::qual_iterator P = T->qual_begin(), 
                                         PEnd = T->qual_end();
       P != PEnd; ++P) {
    ObjCProtocolDecl *Protocol
      = dyn_cast_or_null<ObjCProtocolDecl>(Importer.Import(*P));
    if (!Protocol)
      return QualType();
    Protocols.push_back(Protocol);
  }

  return Importer.getToContext().getObjCObjectPointerType(ToPointeeType,
                                                          Protocols.data(),
                                                          Protocols.size());
}

ASTImporter::ASTImporter(ASTContext &ToContext, Diagnostic &ToDiags,
                         ASTContext &FromContext, Diagnostic &FromDiags)
  : ToContext(ToContext), FromContext(FromContext),
    ToDiags(ToDiags), FromDiags(FromDiags) { }

QualType ASTImporter::Import(QualType FromT) {
  if (FromT.isNull())
    return QualType();
  
  // Check whether we've already imported this type.  
  llvm::DenseMap<Type *, Type *>::iterator Pos
    = ImportedTypes.find(FromT.getTypePtr());
  if (Pos != ImportedTypes.end())
    return ToContext.getQualifiedType(Pos->second, FromT.getQualifiers());
  
  // Import the type
  ASTNodeImporter Importer(*this);
  QualType ToT = Importer.Visit(FromT.getTypePtr());
  if (ToT.isNull())
    return ToT;
  
  // Record the imported type.
  ImportedTypes[FromT.getTypePtr()] = ToT.getTypePtr();
  
  return ToContext.getQualifiedType(ToT, FromT.getQualifiers());
}

DeclarationName ASTImporter::Import(DeclarationName FromName) {
  if (!FromName)
    return DeclarationName();

  switch (FromName.getNameKind()) {
  case DeclarationName::Identifier:
    return Import(FromName.getAsIdentifierInfo());

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    return Import(FromName.getObjCSelector());

  case DeclarationName::CXXConstructorName: {
    QualType T = Import(FromName.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return ToContext.DeclarationNames.getCXXConstructorName(
                                               ToContext.getCanonicalType(T));
  }

  case DeclarationName::CXXDestructorName: {
    QualType T = Import(FromName.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return ToContext.DeclarationNames.getCXXDestructorName(
                                               ToContext.getCanonicalType(T));
  }

  case DeclarationName::CXXConversionFunctionName: {
    QualType T = Import(FromName.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return ToContext.DeclarationNames.getCXXConversionFunctionName(
                                               ToContext.getCanonicalType(T));
  }

  case DeclarationName::CXXOperatorName:
    return ToContext.DeclarationNames.getCXXOperatorName(
                                          FromName.getCXXOverloadedOperator());

  case DeclarationName::CXXLiteralOperatorName:
    return ToContext.DeclarationNames.getCXXLiteralOperatorName(
                                   Import(FromName.getCXXLiteralIdentifier()));

  case DeclarationName::CXXUsingDirective:
    // FIXME: STATICS!
    return DeclarationName::getUsingDirectiveName();
  }

  // Silence bogus GCC warning
  return DeclarationName();
}

IdentifierInfo *ASTImporter::Import(IdentifierInfo *FromId) {
  if (!FromId)
    return 0;

  return &ToContext.Idents.get(FromId->getName());
}

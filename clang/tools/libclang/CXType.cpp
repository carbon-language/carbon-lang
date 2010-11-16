//===- CXTypes.cpp - Implements 'CXTypes' aspect of libclang ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//
//
// This file implements the 'CXTypes' API hooks in the Clang-C library.
//
//===--------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXCursor.h"
#include "CXString.h"
#include "CXType.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Frontend/ASTUnit.h"

using namespace clang;

static CXTypeKind GetBuiltinTypeKind(const BuiltinType *BT) {
#define BTCASE(K) case BuiltinType::K: return CXType_##K
  switch (BT->getKind()) {
    BTCASE(Void);
    BTCASE(Bool);
    BTCASE(Char_U);
    BTCASE(UChar);
    BTCASE(Char16);
    BTCASE(Char32);
    BTCASE(UShort);
    BTCASE(UInt);
    BTCASE(ULong);
    BTCASE(ULongLong);
    BTCASE(UInt128);
    BTCASE(Char_S);
    BTCASE(SChar);
    BTCASE(WChar);
    BTCASE(Short);
    BTCASE(Int);
    BTCASE(Long);
    BTCASE(LongLong);
    BTCASE(Int128);
    BTCASE(Float);
    BTCASE(Double);
    BTCASE(LongDouble);
    BTCASE(NullPtr);
    BTCASE(Overload);
    BTCASE(Dependent);
    BTCASE(ObjCId);
    BTCASE(ObjCClass);
    BTCASE(ObjCSel);
  default:
    return CXType_Unexposed;
  }
#undef BTCASE
}

static CXTypeKind GetTypeKind(QualType T) {
  Type *TP = T.getTypePtr();
  if (!TP)
    return CXType_Invalid;

#define TKCASE(K) case Type::K: return CXType_##K
  switch (TP->getTypeClass()) {
    case Type::Builtin:
      return GetBuiltinTypeKind(cast<BuiltinType>(TP));
    TKCASE(Complex);
    TKCASE(Pointer);
    TKCASE(BlockPointer);
    TKCASE(LValueReference);
    TKCASE(RValueReference);
    TKCASE(Record);
    TKCASE(Enum);
    TKCASE(Typedef);
    TKCASE(ObjCInterface);
    TKCASE(ObjCObjectPointer);
    TKCASE(FunctionNoProto);
    TKCASE(FunctionProto);
    default:
      return CXType_Unexposed;
  }
#undef TKCASE
}


CXType cxtype::MakeCXType(QualType T, ASTUnit *TU) {
  CXTypeKind TK = GetTypeKind(T);
  CXType CT = { TK, { TK == CXType_Invalid ? 0 : T.getAsOpaquePtr(), TU }};
  return CT;
}

using cxtype::MakeCXType;

static inline QualType GetQualType(CXType CT) {
  return QualType::getFromOpaquePtr(CT.data[0]);
}

static inline ASTUnit* GetASTU(CXType CT) {
  return static_cast<ASTUnit*>(CT.data[1]);
}

extern "C" {

CXType clang_getCursorType(CXCursor C) {
  using namespace cxcursor;
  
  ASTUnit *AU = cxcursor::getCursorASTUnit(C);

  if (clang_isExpression(C.kind)) {
    QualType T = cxcursor::getCursorExpr(C)->getType();
    return MakeCXType(T, AU);
  }

  if (clang_isDeclaration(C.kind)) {
    Decl *D = cxcursor::getCursorDecl(C);

    if (TypeDecl *TD = dyn_cast<TypeDecl>(D))
      return MakeCXType(QualType(TD->getTypeForDecl(), 0), AU);
    if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(D))
      return MakeCXType(QualType(ID->getTypeForDecl(), 0), AU);
    if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
      return MakeCXType(VD->getType(), AU);
    if (ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(D))
      return MakeCXType(PD->getType(), AU);
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      return MakeCXType(FD->getType(), AU);
    return MakeCXType(QualType(), AU);
  }
  
  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef:
      return MakeCXType(
                QualType(getCursorObjCSuperClassRef(C).first->getTypeForDecl(), 
                         0), 
                        AU);
      
    case CXCursor_ObjCClassRef:
      return MakeCXType(
                      QualType(getCursorObjCClassRef(C).first->getTypeForDecl(), 
                               0), 
                        AU);
      
    case CXCursor_TypeRef:
      return MakeCXType(QualType(getCursorTypeRef(C).first->getTypeForDecl(), 
                                 0), 
                        AU);
      
    case CXCursor_CXXBaseSpecifier:
      return cxtype::MakeCXType(getCursorCXXBaseSpecifier(C)->getType(), AU);
      
    case CXCursor_ObjCProtocolRef:        
    case CXCursor_TemplateRef:
    case CXCursor_NamespaceRef:
    case CXCursor_MemberRef:
    case CXCursor_OverloadedDeclRef:      
    default:
      break;
    }
    
    return MakeCXType(QualType(), AU);
  }

  return MakeCXType(QualType(), AU);
}

CXType clang_getCanonicalType(CXType CT) {
  if (CT.kind == CXType_Invalid)
    return CT;

  QualType T = GetQualType(CT);

  if (T.isNull())
    return MakeCXType(QualType(), GetASTU(CT));

  ASTUnit *AU = GetASTU(CT);
  return MakeCXType(AU->getASTContext().getCanonicalType(T), AU);
}

CXType clang_getPointeeType(CXType CT) {
  QualType T = GetQualType(CT);
  Type *TP = T.getTypePtr();
  
  if (!TP)
    return MakeCXType(QualType(), GetASTU(CT));
  
  switch (TP->getTypeClass()) {
    case Type::Pointer:
      T = cast<PointerType>(TP)->getPointeeType();
      break;
    case Type::BlockPointer:
      T = cast<BlockPointerType>(TP)->getPointeeType();
      break;
    case Type::LValueReference:
    case Type::RValueReference:
      T = cast<ReferenceType>(TP)->getPointeeType();
      break;
    case Type::ObjCObjectPointer:
      T = cast<ObjCObjectPointerType>(TP)->getPointeeType();
      break;
    default:
      T = QualType();
      break;
  }
  return MakeCXType(T, GetASTU(CT));
}

CXCursor clang_getTypeDeclaration(CXType CT) {
  if (CT.kind == CXType_Invalid)
    return cxcursor::MakeCXCursorInvalid(CXCursor_NoDeclFound);

  QualType T = GetQualType(CT);
  Type *TP = T.getTypePtr();

  if (!TP)
    return cxcursor::MakeCXCursorInvalid(CXCursor_NoDeclFound);

  Decl *D = 0;

try_again:
  switch (TP->getTypeClass()) {
  case Type::Typedef:
    D = cast<TypedefType>(TP)->getDecl();
    break;
  case Type::ObjCObject:
    D = cast<ObjCObjectType>(TP)->getInterface();
    break;
  case Type::ObjCInterface:
    D = cast<ObjCInterfaceType>(TP)->getDecl();
    break;
  case Type::Record:
  case Type::Enum:
    D = cast<TagType>(TP)->getDecl();
    break;
  case Type::TemplateSpecialization:
    if (const RecordType *Record = TP->getAs<RecordType>())
      D = Record->getDecl();
    else
      D = cast<TemplateSpecializationType>(TP)->getTemplateName()
                                                         .getAsTemplateDecl();
    break;
      
  case Type::InjectedClassName:
    D = cast<InjectedClassNameType>(TP)->getDecl();
    break;

  // FIXME: Template type parameters!      

  case Type::Elaborated:
    TP = cast<ElaboratedType>(TP)->getNamedType().getTypePtr();
    goto try_again;
    
  default:
    break;
  }

  if (!D)
    return cxcursor::MakeCXCursorInvalid(CXCursor_NoDeclFound);

  return cxcursor::MakeCXCursor(D, GetASTU(CT));
}

CXString clang_getTypeKindSpelling(enum CXTypeKind K) {
  const char *s = 0;
#define TKIND(X) case CXType_##X: s = ""  #X  ""; break
  switch (K) {
    TKIND(Invalid);
    TKIND(Unexposed);
    TKIND(Void);
    TKIND(Bool);
    TKIND(Char_U);
    TKIND(UChar);
    TKIND(Char16);
    TKIND(Char32);  
    TKIND(UShort);
    TKIND(UInt);
    TKIND(ULong);
    TKIND(ULongLong);
    TKIND(UInt128);
    TKIND(Char_S);
    TKIND(SChar);
    TKIND(WChar);
    TKIND(Short);
    TKIND(Int);
    TKIND(Long);
    TKIND(LongLong);
    TKIND(Int128);
    TKIND(Float);
    TKIND(Double);
    TKIND(LongDouble);
    TKIND(NullPtr);
    TKIND(Overload);
    TKIND(Dependent);
    TKIND(ObjCId);
    TKIND(ObjCClass);
    TKIND(ObjCSel);
    TKIND(Complex);
    TKIND(Pointer);
    TKIND(BlockPointer);
    TKIND(LValueReference);
    TKIND(RValueReference);
    TKIND(Record);
    TKIND(Enum);
    TKIND(Typedef);
    TKIND(ObjCInterface);
    TKIND(ObjCObjectPointer);
    TKIND(FunctionNoProto);
    TKIND(FunctionProto);
  }
#undef TKIND
  return cxstring::createCXString(s);
}

unsigned clang_equalTypes(CXType A, CXType B) {
  return A.data[0] == B.data[0] && A.data[1] == B.data[1];;
}

CXType clang_getResultType(CXType X) {
  QualType T = GetQualType(X);
  if (!T.getTypePtr())
    return MakeCXType(QualType(), GetASTU(X));
  
  if (const FunctionType *FD = T->getAs<FunctionType>())
    return MakeCXType(FD->getResultType(), GetASTU(X));
  
  return MakeCXType(QualType(), GetASTU(X));
}

CXType clang_getCursorResultType(CXCursor C) {
  if (clang_isDeclaration(C.kind)) {
    Decl *D = cxcursor::getCursorDecl(C);
    if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
      return MakeCXType(MD->getResultType(), cxcursor::getCursorASTUnit(C));

    return clang_getResultType(clang_getCursorType(C));
  }

  return MakeCXType(QualType(), cxcursor::getCursorASTUnit(C));
}

unsigned clang_isPODType(CXType X) {
  QualType T = GetQualType(X);
  if (!T.getTypePtr())
    return 0;
  return T->isPODType() ? 1 : 0;
}

} // end: extern "C"

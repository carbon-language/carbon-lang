//===- CXCursor.cpp - Routines for manipulating CXCursors -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXCursors. It should be the
// only file that has internal knowledge of the encoding of the data in
// CXCursor.
//
//===----------------------------------------------------------------------===//

#include "CXCursor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

CXCursor cxcursor::MakeCXCursor(CXCursorKind K, Decl *D) {
  CXCursor C = { K, { D, 0, 0 } };
  return C;  
}

CXCursor cxcursor::MakeCXCursor(CXCursorKind K, Decl *D, Stmt *S) {
  assert(clang_isReference(K));
  CXCursor C = { K, { D, S, 0 } };
  return C;  
}

static CXCursorKind GetCursorKind(Decl *D) {
  switch (D->getKind()) {
    case Decl::Enum:               return CXCursor_EnumDecl;
    case Decl::EnumConstant:       return CXCursor_EnumConstantDecl;
    case Decl::Field:              return CXCursor_FieldDecl;
    case Decl::Function:  
      return cast<FunctionDecl>(D)->isThisDeclarationADefinition()
              ? CXCursor_FunctionDefn : CXCursor_FunctionDecl;
    case Decl::ObjCCategory:       return CXCursor_ObjCCategoryDecl;
    case Decl::ObjCCategoryImpl:   return CXCursor_ObjCCategoryDefn;
    case Decl::ObjCClass:
      // FIXME
      return CXCursor_NotImplemented;
    case Decl::ObjCImplementation: return CXCursor_ObjCClassDefn;
    case Decl::ObjCInterface:      return CXCursor_ObjCInterfaceDecl;
    case Decl::ObjCIvar:           return CXCursor_ObjCIvarDecl; 
    case Decl::ObjCMethod:
      return cast<ObjCMethodDecl>(D)->isInstanceMethod()
              ? CXCursor_ObjCInstanceMethodDecl : CXCursor_ObjCClassMethodDecl;
    case Decl::ObjCProperty:       return CXCursor_ObjCPropertyDecl;
    case Decl::ObjCProtocol:       return CXCursor_ObjCProtocolDecl;
    case Decl::ParmVar:            return CXCursor_ParmDecl;
    case Decl::Typedef:            return CXCursor_TypedefDecl;
    case Decl::Var:                return CXCursor_VarDecl;
    default:
      if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
        switch (TD->getTagKind()) {
          case TagDecl::TK_struct: return CXCursor_StructDecl;
          case TagDecl::TK_class:  return CXCursor_ClassDecl;
          case TagDecl::TK_union:  return CXCursor_UnionDecl;
          case TagDecl::TK_enum:   return CXCursor_EnumDecl;
        }
      }
  }
  
  llvm_unreachable("Invalid Decl");
  return CXCursor_NotImplemented;  
}

CXCursor cxcursor::MakeCXCursor(Decl *D) {
  return MakeCXCursor(GetCursorKind(D), D);
}

CXCursor cxcursor::MakeCursorObjCSuperClassRef(ObjCInterfaceDecl *Super, 
                                         SourceLocation Loc) {
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCSuperClassRef, { Super, RawLoc, 0 } };
  return C;    
}

std::pair<ObjCInterfaceDecl *, SourceLocation> 
cxcursor::getCursorObjCSuperClassRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCSuperClassRef);
  return std::make_pair(static_cast<ObjCInterfaceDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorObjCProtocolRef(ObjCProtocolDecl *Super, 
                                             SourceLocation Loc) {
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCProtocolRef, { Super, RawLoc, 0 } };
  return C;    
}

std::pair<ObjCProtocolDecl *, SourceLocation> 
cxcursor::getCursorObjCProtocolRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCProtocolRef);
  return std::make_pair(static_cast<ObjCProtocolDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

CXCursor cxcursor::MakeCursorObjCClassRef(ObjCInterfaceDecl *Class, 
                                         SourceLocation Loc) {
  void *RawLoc = reinterpret_cast<void *>(Loc.getRawEncoding());
  CXCursor C = { CXCursor_ObjCClassRef, { Class, RawLoc, 0 } };
  return C;    
}

std::pair<ObjCInterfaceDecl *, SourceLocation> 
cxcursor::getCursorObjCClassRef(CXCursor C) {
  assert(C.kind == CXCursor_ObjCClassRef);
  return std::make_pair(static_cast<ObjCInterfaceDecl *>(C.data[0]),
           SourceLocation::getFromRawEncoding(
                                      reinterpret_cast<uintptr_t>(C.data[1])));
}

Decl *cxcursor::getCursorDecl(CXCursor Cursor) {
  return (Decl *)Cursor.data[0];
}

Expr *cxcursor::getCursorExpr(CXCursor Cursor) {
  return dyn_cast_or_null<Expr>(getCursorStmt(Cursor));
}

Stmt *cxcursor::getCursorStmt(CXCursor Cursor) {
  if (Cursor.kind == CXCursor_ObjCSuperClassRef ||
      Cursor.kind == CXCursor_ObjCProtocolRef ||
      Cursor.kind == CXCursor_ObjCClassRef)
    return 0;

  return (Stmt *)Cursor.data[1];
}

Decl *cxcursor::getCursorReferringDecl(CXCursor Cursor) {
  return (Decl *)Cursor.data[2];
}

bool cxcursor::operator==(CXCursor X, CXCursor Y) {
  return X.kind == Y.kind && X.data[0] == Y.data[0] && X.data[1] == Y.data[1] &&
         X.data[2] == Y.data[2];
}

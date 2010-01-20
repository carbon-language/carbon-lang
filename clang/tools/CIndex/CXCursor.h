//===- CXCursor.h - Routines for manipulating CXCursors -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXCursors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CXCURSOR_H
#define LLVM_CLANG_CXCursor_H

#include "clang-c/Index.h"
#include "clang/Basic/SourceLocation.h"
#include <utility>

namespace clang {

class ASTContext;
class ASTUnit;
class Decl;
class Expr;
class NamedDecl;
class ObjCInterfaceDecl;
class ObjCProtocolDecl;
class Stmt;

namespace cxcursor {
  
CXCursor MakeCXCursorInvalid(CXCursorKind K);
CXCursor MakeCXCursor(clang::Stmt *S, clang::Decl *Parent, ASTUnit *TU);
CXCursor MakeCXCursor(clang::Decl *D, ASTUnit *TU);

/// \brief Create an Objective-C superclass reference at the given location.
CXCursor MakeCursorObjCSuperClassRef(ObjCInterfaceDecl *Super, 
                                     SourceLocation Loc, 
                                     ASTUnit *TU);

/// \brief Unpack an ObjCSuperClassRef cursor into the interface it references
/// and optionally the location where the reference occurred.
std::pair<ObjCInterfaceDecl *, SourceLocation> 
  getCursorObjCSuperClassRef(CXCursor C);

/// \brief Create an Objective-C protocol reference at the given location.
CXCursor MakeCursorObjCProtocolRef(ObjCProtocolDecl *Proto, SourceLocation Loc, 
                                   ASTUnit *TU);

/// \brief Unpack an ObjCProtocolRef cursor into the protocol it references
/// and optionally the location where the reference occurred.
std::pair<ObjCProtocolDecl *, SourceLocation> 
  getCursorObjCProtocolRef(CXCursor C);

/// \brief Create an Objective-C class reference at the given location.
CXCursor MakeCursorObjCClassRef(ObjCInterfaceDecl *Class, SourceLocation Loc, 
                                ASTUnit *TU);

/// \brief Unpack an ObjCClassRef cursor into the class it references
/// and optionally the location where the reference occurred.
std::pair<ObjCInterfaceDecl *, SourceLocation> 
  getCursorObjCClassRef(CXCursor C);

Decl *getCursorDecl(CXCursor Cursor);
Expr *getCursorExpr(CXCursor Cursor);
Stmt *getCursorStmt(CXCursor Cursor);
ASTContext &getCursorContext(CXCursor Cursor);
ASTUnit *getCursorASTUnit(CXCursor Cursor);
  
bool operator==(CXCursor X, CXCursor Y);
  
inline bool operator!=(CXCursor X, CXCursor Y) {
  return !(X == Y);
}

}} // end namespace: clang::cxcursor

#endif

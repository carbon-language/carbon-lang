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
#define LLVM_CLANG_CXCURSOR_H

#include "clang-c/Index.h"
#include "clang/Basic/SourceLocation.h"
#include <utility>

namespace clang {

class ASTContext;
class ASTUnit;
class Attr;
class Decl;
class Expr;
class MacroDefinition;
class MacroInstantiation;
class NamedDecl;
class ObjCInterfaceDecl;
class ObjCProtocolDecl;
class Stmt;
class TypeDecl;

namespace cxcursor {
  
CXCursor MakeCXCursor(const clang::Attr *A, clang::Decl *Parent, ASTUnit *TU);
CXCursor MakeCXCursor(clang::Decl *D, ASTUnit *TU);
CXCursor MakeCXCursor(clang::Stmt *S, clang::Decl *Parent, ASTUnit *TU);
CXCursor MakeCXCursorInvalid(CXCursorKind K);

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

/// \brief Create a type reference at the given location.
CXCursor MakeCursorTypeRef(TypeDecl *Type, SourceLocation Loc, ASTUnit *TU);

/// \brief Unpack a TypeRef cursor into the class it references
/// and optionally the location where the reference occurred.
std::pair<TypeDecl *, SourceLocation> getCursorTypeRef(CXCursor C);

/// \brief Create a preprocessing directive cursor.
CXCursor MakePreprocessingDirectiveCursor(SourceRange Range, ASTUnit *TU);

/// \brief Unpack a given preprocessing directive to retrieve its source range.
SourceRange getCursorPreprocessingDirective(CXCursor C);

/// \brief Create a macro definition cursor.
CXCursor MakeMacroDefinitionCursor(MacroDefinition *, ASTUnit *TU);

/// \brief Unpack a given macro definition cursor to retrieve its
/// source range.
MacroDefinition *getCursorMacroDefinition(CXCursor C);

/// \brief Create a macro instantiation cursor.
CXCursor MakeMacroInstantiationCursor(MacroInstantiation *, ASTUnit *TU);

/// \brief Unpack a given macro instantiation cursor to retrieve its
/// source range.
MacroInstantiation *getCursorMacroInstantiation(CXCursor C);

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

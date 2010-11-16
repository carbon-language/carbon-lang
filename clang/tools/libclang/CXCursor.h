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
#include "llvm/ADT/PointerUnion.h"
#include <utility>

namespace clang {

class ASTContext;
class ASTUnit;
class Attr;
class CXXBaseSpecifier;
class Decl;
class Expr;
class FieldDecl;
class InclusionDirective;
class LabelStmt;
class MacroDefinition;
class MacroInstantiation;
class NamedDecl;
class ObjCInterfaceDecl;
class ObjCProtocolDecl;
class OverloadedTemplateStorage;
class OverloadExpr;
class Stmt;
class TemplateDecl;
class TemplateName;
class TypeDecl;
  
namespace cxcursor {
  
CXCursor MakeCXCursor(const clang::Attr *A, clang::Decl *Parent,
                      CXTranslationUnit TU);
CXCursor MakeCXCursor(clang::Decl *D, CXTranslationUnit TU,
                      bool FirstInDeclGroup = true);
CXCursor MakeCXCursor(clang::Stmt *S, clang::Decl *Parent,
                      CXTranslationUnit TU);
CXCursor MakeCXCursorInvalid(CXCursorKind K);

/// \brief Create an Objective-C superclass reference at the given location.
CXCursor MakeCursorObjCSuperClassRef(ObjCInterfaceDecl *Super, 
                                     SourceLocation Loc, 
                                     CXTranslationUnit TU);

/// \brief Unpack an ObjCSuperClassRef cursor into the interface it references
/// and optionally the location where the reference occurred.
std::pair<ObjCInterfaceDecl *, SourceLocation> 
  getCursorObjCSuperClassRef(CXCursor C);

/// \brief Create an Objective-C protocol reference at the given location.
CXCursor MakeCursorObjCProtocolRef(ObjCProtocolDecl *Proto, SourceLocation Loc, 
                                   CXTranslationUnit TU);

/// \brief Unpack an ObjCProtocolRef cursor into the protocol it references
/// and optionally the location where the reference occurred.
std::pair<ObjCProtocolDecl *, SourceLocation> 
  getCursorObjCProtocolRef(CXCursor C);

/// \brief Create an Objective-C class reference at the given location.
CXCursor MakeCursorObjCClassRef(ObjCInterfaceDecl *Class, SourceLocation Loc, 
                                CXTranslationUnit TU);

/// \brief Unpack an ObjCClassRef cursor into the class it references
/// and optionally the location where the reference occurred.
std::pair<ObjCInterfaceDecl *, SourceLocation> 
  getCursorObjCClassRef(CXCursor C);

/// \brief Create a type reference at the given location.
CXCursor MakeCursorTypeRef(TypeDecl *Type, SourceLocation Loc,
                           CXTranslationUnit TU);
                               
/// \brief Unpack a TypeRef cursor into the class it references
/// and optionally the location where the reference occurred.
std::pair<TypeDecl *, SourceLocation> getCursorTypeRef(CXCursor C);

/// \brief Create a reference to a template at the given location.
CXCursor MakeCursorTemplateRef(TemplateDecl *Template, SourceLocation Loc,
                               CXTranslationUnit TU);

/// \brief Unpack a TemplateRef cursor into the template it references and
/// the location where the reference occurred.
std::pair<TemplateDecl *, SourceLocation> getCursorTemplateRef(CXCursor C);

/// \brief Create a reference to a namespace or namespace alias at the given 
/// location.
CXCursor MakeCursorNamespaceRef(NamedDecl *NS, SourceLocation Loc,
                                CXTranslationUnit TU);

/// \brief Unpack a NamespaceRef cursor into the namespace or namespace alias
/// it references and the location where the reference occurred.
std::pair<NamedDecl *, SourceLocation> getCursorNamespaceRef(CXCursor C);

/// \brief Create a reference to a field at the given location.
CXCursor MakeCursorMemberRef(FieldDecl *Field, SourceLocation Loc, 
                             CXTranslationUnit TU);
  
/// \brief Unpack a MemberRef cursor into the field it references and the 
/// location where the reference occurred.
std::pair<FieldDecl *, SourceLocation> getCursorMemberRef(CXCursor C);

/// \brief Create a CXX base specifier cursor.
CXCursor MakeCursorCXXBaseSpecifier(CXXBaseSpecifier *B,
                                    CXTranslationUnit TU);

/// \brief Unpack a CXXBaseSpecifier cursor into a CXXBaseSpecifier.
CXXBaseSpecifier *getCursorCXXBaseSpecifier(CXCursor C);

/// \brief Create a preprocessing directive cursor.
CXCursor MakePreprocessingDirectiveCursor(SourceRange Range,
                                          CXTranslationUnit TU);

/// \brief Unpack a given preprocessing directive to retrieve its source range.
SourceRange getCursorPreprocessingDirective(CXCursor C);

/// \brief Create a macro definition cursor.
CXCursor MakeMacroDefinitionCursor(MacroDefinition *, CXTranslationUnit TU);

/// \brief Unpack a given macro definition cursor to retrieve its
/// source range.
MacroDefinition *getCursorMacroDefinition(CXCursor C);

/// \brief Create a macro instantiation cursor.
CXCursor MakeMacroInstantiationCursor(MacroInstantiation *,
                                      CXTranslationUnit TU);

/// \brief Unpack a given macro instantiation cursor to retrieve its
/// source range.
MacroInstantiation *getCursorMacroInstantiation(CXCursor C);

/// \brief Create an inclusion directive cursor.
CXCursor MakeInclusionDirectiveCursor(InclusionDirective *,
                                      CXTranslationUnit TU);

/// \brief Unpack a given inclusion directive cursor to retrieve its
/// source range.
InclusionDirective *getCursorInclusionDirective(CXCursor C);

/// \brief Create a label reference at the given location.
CXCursor MakeCursorLabelRef(LabelStmt *Label, SourceLocation Loc,
                            CXTranslationUnit TU);

/// \brief Unpack a label reference into the label statement it refers to and
/// the location of the reference.
std::pair<LabelStmt *, SourceLocation> getCursorLabelRef(CXCursor C);

/// \brief Create a overloaded declaration reference cursor for an expression.
CXCursor MakeCursorOverloadedDeclRef(OverloadExpr *E, CXTranslationUnit TU);

/// \brief Create a overloaded declaration reference cursor for a declaration.
CXCursor MakeCursorOverloadedDeclRef(Decl *D, SourceLocation Location,
                                     CXTranslationUnit TU);

/// \brief Create a overloaded declaration reference cursor for a template name.
CXCursor MakeCursorOverloadedDeclRef(TemplateName Template, 
                                     SourceLocation Location,
                                     CXTranslationUnit TU);

/// \brief Internal storage for an overloaded declaration reference cursor;
typedef llvm::PointerUnion3<OverloadExpr *, Decl *, 
                            OverloadedTemplateStorage *>
  OverloadedDeclRefStorage;
  
/// \brief Unpack an overloaded declaration reference into an expression,
/// declaration, or template name along with the source location.
std::pair<OverloadedDeclRefStorage, SourceLocation>
  getCursorOverloadedDeclRef(CXCursor C);
  
Decl *getCursorDecl(CXCursor Cursor);
Expr *getCursorExpr(CXCursor Cursor);
Stmt *getCursorStmt(CXCursor Cursor);
Attr *getCursorAttr(CXCursor Cursor);

ASTContext &getCursorContext(CXCursor Cursor);
ASTUnit *getCursorASTUnit(CXCursor Cursor);
CXTranslationUnit getCursorTU(CXCursor Cursor);
  
bool operator==(CXCursor X, CXCursor Y);
  
inline bool operator!=(CXCursor X, CXCursor Y) {
  return !(X == Y);
}

/// \brief Return true if the cursor represents a declaration that is the
/// first in a declaration group.
bool isFirstInDeclGroup(CXCursor C);

}} // end namespace: clang::cxcursor

#endif

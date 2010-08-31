//===- CIndexCXX.cpp - Clang-C Source Indexing Library --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the libclang support for C++ cursors.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXCursor.h"
#include "CXType.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"

using namespace clang;
using namespace clang::cxstring;

extern "C" {

unsigned clang_isVirtualBase(CXCursor C) {
  if (C.kind != CXCursor_CXXBaseSpecifier)
    return 0;
  
  CXXBaseSpecifier *B = cxcursor::getCursorCXXBaseSpecifier(C);
  return B->isVirtual();
}

enum CX_CXXAccessSpecifier clang_getCXXAccessSpecifier(CXCursor C) {
  if (C.kind != CXCursor_CXXBaseSpecifier)
    return CX_CXXInvalidAccessSpecifier;
  
  CXXBaseSpecifier *B = cxcursor::getCursorCXXBaseSpecifier(C);
  switch (B->getAccessSpecifier()) {
    case AS_public: return CX_CXXPublic;
    case AS_protected: return CX_CXXProtected;
    case AS_private: return CX_CXXPrivate;
    case AS_none: return CX_CXXInvalidAccessSpecifier;
  }
  
  // FIXME: Clang currently thinks this is reachable.
  return CX_CXXInvalidAccessSpecifier;
}

enum CXCursorKind clang_getTemplateCursorKind(CXCursor C) {
  using namespace clang::cxcursor;
  
  switch (C.kind) {
  case CXCursor_ClassTemplate: 
  case CXCursor_FunctionTemplate:
    if (TemplateDecl *Template
                           = dyn_cast_or_null<TemplateDecl>(getCursorDecl(C)))
      return MakeCXCursor(Template->getTemplatedDecl(), 
                          getCursorASTUnit(C)).kind;
    break;
      
  case CXCursor_ClassTemplatePartialSpecialization:
    if (ClassTemplateSpecializationDecl *PartialSpec
          = dyn_cast_or_null<ClassTemplatePartialSpecializationDecl>(
                                                            getCursorDecl(C))) {
      switch (PartialSpec->getTagKind()) {
      case TTK_Class: return CXCursor_ClassDecl;
      case TTK_Struct: return CXCursor_StructDecl;
      case TTK_Union: return CXCursor_UnionDecl;
      case TTK_Enum: return CXCursor_NoDeclFound;
      }
    }
    break;
      
  default:
    break;
  }
  
  return CXCursor_NoDeclFound;
}

} // end extern "C"

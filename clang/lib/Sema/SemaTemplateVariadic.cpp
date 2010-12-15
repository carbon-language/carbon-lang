//===------- SemaTemplateVariadic.cpp - C++ Variadic Templates ------------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements semantic analysis for C++0x variadic templates.
//===----------------------------------------------------------------------===/

#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeLoc.h"

using namespace clang;

bool Sema::DiagnoseUnexpandedParameterPack(SourceLocation Loc, 
                                           TypeSourceInfo *T,
                                         UnexpandedParameterPackContext UPPC) {
  // C++0x [temp.variadic]p5:
  //   An appearance of a name of a parameter pack that is not expanded is 
  //   ill-formed.
  if (!T->getType()->containsUnexpandedParameterPack())
    return false;

  // FIXME: Provide the names and locations of the unexpanded parameter packs.
  Diag(Loc, diag::err_unexpanded_parameter_pack)
    << (int)UPPC << T->getTypeLoc().getSourceRange();
  return true;
}

bool Sema::DiagnoseUnexpandedParameterPack(Expr *E,
                                           UnexpandedParameterPackContext UPPC) {
  // C++0x [temp.variadic]p5:
  //   An appearance of a name of a parameter pack that is not expanded is 
  //   ill-formed.
  if (!E->containsUnexpandedParameterPack())
    return false;

  // FIXME: Provide the names and locations of the unexpanded parameter packs.
  Diag(E->getSourceRange().getBegin(), diag::err_unexpanded_parameter_pack)
    << (int)UPPC << E->getSourceRange();
  return true;
}

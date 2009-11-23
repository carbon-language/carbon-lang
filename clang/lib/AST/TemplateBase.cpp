//===--- TemplateBase.cpp - Common template AST class implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements common classes used throughout C++ template
// representations.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FoldingSet.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeLoc.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// TemplateArgument Implementation
//===----------------------------------------------------------------------===//

/// \brief Construct a template argument pack.
void TemplateArgument::setArgumentPack(TemplateArgument *args, unsigned NumArgs,
                                       bool CopyArgs) {
  assert(isNull() && "Must call setArgumentPack on a null argument");

  Kind = Pack;
  Args.NumArgs = NumArgs;
  Args.CopyArgs = CopyArgs;
  if (!Args.CopyArgs) {
    Args.Args = args;
    return;
  }

  // FIXME: Allocate in ASTContext
  Args.Args = new TemplateArgument[NumArgs];
  for (unsigned I = 0; I != Args.NumArgs; ++I)
    Args.Args[I] = args[I];
}

void TemplateArgument::Profile(llvm::FoldingSetNodeID &ID,
                               ASTContext &Context) const {
  ID.AddInteger(Kind);
  switch (Kind) {
  case Null:
    break;

  case Type:
    getAsType().Profile(ID);
    break;

  case Declaration:
    ID.AddPointer(getAsDecl()? getAsDecl()->getCanonicalDecl() : 0);
    break;

  case Template:
    if (TemplateTemplateParmDecl *TTP
          = dyn_cast_or_null<TemplateTemplateParmDecl>(
                                       getAsTemplate().getAsTemplateDecl())) {
      ID.AddBoolean(true);
      ID.AddInteger(TTP->getDepth());
      ID.AddInteger(TTP->getPosition());
    } else {
      ID.AddBoolean(false);
      ID.AddPointer(Context.getCanonicalTemplateName(getAsTemplate())
                      .getAsVoidPointer());
    }
    break;
      
  case Integral:
    getAsIntegral()->Profile(ID);
    getIntegralType().Profile(ID);
    break;

  case Expression:
    getAsExpr()->Profile(ID, Context, true);
    break;

  case Pack:
    ID.AddInteger(Args.NumArgs);
    for (unsigned I = 0; I != Args.NumArgs; ++I)
      Args.Args[I].Profile(ID, Context);
  }
}

//===----------------------------------------------------------------------===//
// TemplateArgumentLoc Implementation
//===----------------------------------------------------------------------===//

SourceRange TemplateArgumentLoc::getSourceRange() const {
  switch (Argument.getKind()) {
  case TemplateArgument::Expression:
    return getSourceExpression()->getSourceRange();
      
  case TemplateArgument::Declaration:
    return getSourceDeclExpression()->getSourceRange();
      
  case TemplateArgument::Type:
    return getSourceDeclaratorInfo()->getTypeLoc().getFullSourceRange();
      
  case TemplateArgument::Template:
    if (getTemplateQualifierRange().isValid())
      return SourceRange(getTemplateQualifierRange().getBegin(),
                         getTemplateNameLoc());
    return SourceRange(getTemplateNameLoc());
      
  case TemplateArgument::Integral:
  case TemplateArgument::Pack:
  case TemplateArgument::Null:
    return SourceRange();
  }

  // Silence bonus gcc warning.
  return SourceRange();
}

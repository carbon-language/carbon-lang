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
#include "clang/AST/Expr.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// TemplateArgument Implementation
//===----------------------------------------------------------------------===//

TemplateArgument::TemplateArgument(Expr *E) : Kind(Expression) {
  TypeOrValue = reinterpret_cast<uintptr_t>(E);
  StartLoc = E->getSourceRange().getBegin();
}

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

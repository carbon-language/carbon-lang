//===--- SemaAttr.cpp - Semantic Analysis for Attributes ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for non-trivial attributes.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/Expr.h"
using namespace clang;

void Sema::ActOnPragmaPack(PragmaPackKind Kind, IdentifierInfo *Name, 
                           ExprTy *alignment, SourceLocation PragmaLoc, 
                           SourceLocation LParenLoc, SourceLocation RParenLoc) {
  Expr *Alignment = static_cast<Expr *>(alignment);

  // If specified then alignment must be a "small" power of two.
  unsigned AlignmentVal = 0;
  if (Alignment) {
    llvm::APSInt Val;
    if (!Alignment->isIntegerConstantExpr(Val, Context) ||
        !Val.isPowerOf2() ||
        Val.getZExtValue() > 16) {
      Diag(PragmaLoc, diag::warn_pragma_pack_invalid_alignment);
      Alignment->Destroy(Context);
      return; // Ignore
    }

    AlignmentVal = (unsigned) Val.getZExtValue();
  }

  switch (Kind) {
  case Action::PPK_Default: // pack([n])
    PackContext.setAlignment(AlignmentVal);
    break;

  case Action::PPK_Show: // pack(show)
    // Show the current alignment, making sure to show the right value
    // for the default.
    AlignmentVal = PackContext.getAlignment();
    // FIXME: This should come from the target.
    if (AlignmentVal == 0)
      AlignmentVal = 8;
    Diag(PragmaLoc, diag::warn_pragma_pack_show) << AlignmentVal;
    break;

  case Action::PPK_Push: // pack(push [, id] [, [n])
    PackContext.push(Name);
    // Set the new alignment if specified.
    if (Alignment)
      PackContext.setAlignment(AlignmentVal);    
    break;

  case Action::PPK_Pop: // pack(pop [, id] [,  n])
    // MSDN, C/C++ Preprocessor Reference > Pragma Directives > pack:
    // "#pragma pack(pop, identifier, n) is undefined"
    if (Alignment && Name)
      Diag(PragmaLoc, diag::warn_pragma_pack_pop_identifer_and_alignment);    
    
    // Do the pop.
    if (!PackContext.pop(Name)) {
      // If a name was specified then failure indicates the name
      // wasn't found. Otherwise failure indicates the stack was
      // empty.
      Diag(PragmaLoc, diag::warn_pragma_pack_pop_failed)
        << (Name ? "no record matching name" : "stack empty");

      // FIXME: Warn about popping named records as MSVC does.
    } else {
      // Pop succeeded, set the new alignment if specified.
      if (Alignment)
        PackContext.setAlignment(AlignmentVal);
    }
    break;

  default:
    assert(0 && "Invalid #pragma pack kind.");
  }
}

bool PragmaPackStack::pop(IdentifierInfo *Name) {
  if (Stack.empty())
    return false;

  // If name is empty just pop top.
  if (!Name) {
    Alignment = Stack.back().first;
    Stack.pop_back();
    return true;
  } 

  // Otherwise, find the named record.
  for (unsigned i = Stack.size(); i != 0; ) {
    --i;
    if (Stack[i].second == Name) {
      // Found it, pop up to and including this record.
      Alignment = Stack[i].first;
      Stack.erase(Stack.begin() + i, Stack.end());
      return true;
    }
  }

  return false;
}

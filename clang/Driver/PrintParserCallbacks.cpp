//===--- PrintParserActions.cpp - Implement -parse-print-callbacks mode ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code simply runs the preprocessor on the input file and prints out the
// result.  This is the traditional behavior of the -E option.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/Lex/IdentifierTable.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/SemaDecl.h"
#include <iostream>

using namespace llvm;
using namespace clang;

namespace {
  class ParserPrintActions : public MinimalAction {
    
    /// ParseDeclarator - This callback is invoked when a declarator is parsed
    /// and 'Init' specifies the initializer if any.  This is for things like:
    /// "int X = 4" or "typedef int foo".
    virtual DeclTy *ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init,
                                    DeclTy *LastInGroup) {
      std::cout << "ParseDeclarator ";
      if (IdentifierInfo *II = D.getIdentifier()) {
        std::cout << "'" << II->getName() << "'";
      } else {
        std::cout << "<anon>";
      }
      std::cout << "\n";
      
      // Pass up to EmptyActions so that the symbol table is maintained right.
      return MinimalAction::ParseDeclarator(S, D, Init, LastInGroup);
    }
    
    /// PopScope - This callback is called immediately before the specified scope
    /// is popped and deleted.
    virtual void PopScope(SourceLocation Loc, Scope *S) {
      std::cout << "PopScope\n";
      
      // Pass up to EmptyActions so that the symbol table is maintained right.
      MinimalAction::PopScope(Loc, S);
    }
  };
}

MinimalAction *llvm::clang::CreatePrintParserActionsAction() {
  return new ParserPrintActions();
}

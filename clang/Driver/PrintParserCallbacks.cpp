//===--- PrintParserActions.cpp - Implement -parse-print-callbacks mode ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This code simply runs the preprocessor on the input file and prints out the
// result.  This is the traditional behavior of the -E option.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/Support/Streams.h"
using namespace clang;

namespace {
  class ParserPrintActions : public MinimalAction {
    
  public:
    ParserPrintActions(IdentifierTable &IT) : MinimalAction(IT) {}
    
    /// ActOnDeclarator - This callback is invoked when a declarator is parsed
    /// and 'Init' specifies the initializer if any.  This is for things like:
    /// "int X = 4" or "typedef int foo".
    virtual DeclTy *ActOnDeclarator(Scope *S, Declarator &D,
                                    DeclTy *LastInGroup) {
      llvm::cout << "ActOnDeclarator ";
      if (IdentifierInfo *II = D.getIdentifier()) {
        llvm::cout << "'" << II->getName() << "'";
      } else {
        llvm::cout << "<anon>";
      }
      llvm::cout << "\n";
      
      // Pass up to EmptyActions so that the symbol table is maintained right.
      return MinimalAction::ActOnDeclarator(S, D, LastInGroup);
    }
    
    /// ActOnPopScope - This callback is called immediately before the specified
    /// scope is popped and deleted.
    virtual void ActOnPopScope(SourceLocation Loc, Scope *S) {
      llvm::cout << "ActOnPopScope\n";
      
      // Pass up to EmptyActions so that the symbol table is maintained right.
      MinimalAction::ActOnPopScope(Loc, S);
    }
  };
}

MinimalAction *clang::CreatePrintParserActionsAction(IdentifierTable &IT) {
  return new ParserPrintActions(IT);
}

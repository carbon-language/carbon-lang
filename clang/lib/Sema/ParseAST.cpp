//===--- ParseAST.cpp - Provide the clang::ParseAST method ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the clang::ParseAST method.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/OwningPtr.h>
#include "clang/Sema/ParseAST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TranslationUnit.h"
#include "Sema.h"
#include "clang/Parse/Parser.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Public interface to the file
//===----------------------------------------------------------------------===//

/// ParseAST - Parse the entire file specified, notifying the ASTConsumer as
/// the file is parsed.
///
/// \param TU If 0, then memory used for AST elements will be allocated only
/// for the duration of the ParseAST() call. In this case, the client should
/// not access any AST elements after ParseAST() returns.
void clang::ParseAST(Preprocessor &PP, ASTConsumer *Consumer,
                     TranslationUnit *TU, bool PrintStats) {
  // Collect global stats on Decls/Stmts (until we have a module streamer).
  if (PrintStats) {
    Decl::CollectingStats(true);
    Stmt::CollectingStats(true);
  }

  llvm::OwningPtr<ASTContext> ContextOwner;
  llvm::OwningPtr<TranslationUnit> TranslationUnitOwner;
  if (TU == 0) {
    ASTContext *Context = new ASTContext(PP.getLangOptions(),
                                         PP.getSourceManager(),
                                         PP.getTargetInfo(),
                                         PP.getIdentifierTable(),
                                         PP.getSelectorTable());
    ContextOwner.reset(Context);
    TU = new TranslationUnit(*Context);
    TranslationUnitOwner.reset(TU);
  }

  Sema S(PP, TU->getContext(), *Consumer);
  Parser P(PP, S);
  PP.EnterMainSourceFile();
    
  // Initialize the parser.
  P.Initialize();
  
  Consumer->InitializeTU(*TU);
  
  Parser::DeclTy *ADecl;
  
  while (!P.ParseTopLevelDecl(ADecl)) {  // Not end of file.
    // If we got a null return and something *was* parsed, ignore it.  This
    // is due to a top-level semicolon, an action override, or a parse error
    // skipping something.
    if (ADecl) {
      Decl* D = static_cast<Decl*>(ADecl);      
      Consumer->HandleTopLevelDecl(D);
    }
  };
  
  Consumer->HandleTranslationUnit(*TU);

  if (PrintStats) {
    fprintf(stderr, "\nSTATISTICS:\n");
    P.getActions().PrintStats();
    TU->getContext().PrintStats();
    Decl::PrintStats();
    Stmt::PrintStats();
    Consumer->PrintStats();
    
    Decl::CollectingStats(false);
    Stmt::CollectingStats(false);
  }
}

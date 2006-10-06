//===--- ASTStreamer.cpp - Provide streaming interface to ASTs ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ASTStreamer interface.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTStreamer.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/Parser.h"
using namespace llvm;
using namespace clang;

/// Interface to the Builder.cpp file.
///
Action *CreateASTBuilderActions(Preprocessor &PP, bool FullLocInfo);


namespace {
  class ASTStreamer {
    Parser P;
  public:
    ASTStreamer(Preprocessor &PP, unsigned MainFileID, bool FullLocInfo)
      : P(PP, *CreateASTBuilderActions(PP, FullLocInfo)) {
      PP.EnterSourceFile(MainFileID, 0, true);
      
      // Initialize the parser.
      P.Initialize();
    }
    
    /// ReadTopLevelDecl - Parse and return the next top-level declaration.
    Decl *ReadTopLevelDecl() {
      Parser::DeclTy *Result;
      if (P.ParseTopLevelDecl(Result))
        return 0;
      Result = (Decl*)1; // FIXME!
      return (Decl*)Result;
    }
    
    ~ASTStreamer() {
      P.Finalize();
      delete &P.getActions();
    }
  };
}



//===----------------------------------------------------------------------===//
// Public interface to the file
//===----------------------------------------------------------------------===//

/// ASTStreamer_Init - Create an ASTStreamer with the specified preprocessor
/// and FileID.
ASTStreamerTy *llvm::clang::ASTStreamer_Init(Preprocessor &PP, 
                                             unsigned MainFileID,
                                             bool FullLocInfo) {
  return new ASTStreamer(PP, MainFileID, FullLocInfo);
}

/// ASTStreamer_ReadTopLevelDecl - Parse and return one top-level declaration. This
/// returns null at end of file.
Decl *llvm::clang::ASTStreamer_ReadTopLevelDecl(ASTStreamerTy *Streamer) {
  return static_cast<ASTStreamer*>(Streamer)->ReadTopLevelDecl();
}

/// ASTStreamer_Terminate - Gracefully shut down the streamer.
///
void llvm::clang::ASTStreamer_Terminate(ASTStreamerTy *Streamer) {
  delete static_cast<ASTStreamer*>(Streamer);
}

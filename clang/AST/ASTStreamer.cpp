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
#include "clang/AST/ASTBuilder.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/Parser.h"
using namespace llvm;
using namespace clang;

namespace {
  class ASTStreamer {
    Parser P;
    std::vector<Decl*> LastInGroupList;
  public:
    ASTStreamer(Preprocessor &PP, unsigned MainFileID)
      : P(PP, *new Sema(PP, LastInGroupList)) {
      PP.EnterSourceFile(MainFileID, 0, true);
      
      // Initialize the parser.
      P.Initialize();
    }
    
    /// ReadTopLevelDecl - Parse and return the next top-level declaration.
    Decl *ReadTopLevelDecl() {
      Parser::DeclTy *Result;
      
      /// If the previous time through we read something like 'int X, Y', return
      /// the next declarator.
      if (!LastInGroupList.empty()) {
        Result = LastInGroupList.back();
        LastInGroupList.pop_back();
        return (Decl*)Result;
      }

      do {
        if (P.ParseTopLevelDecl(Result))
          return 0;  // End of file.

        // If we got a null return and something *was* parsed, try again.  This
        // is due to a top-level semicolon, an action override, or a parse error
        // skipping something.
      } while (Result == 0);

      // If we parsed a declspec with multiple declarators, reverse the list and
      // return the first one.
      if (!LastInGroupList.empty()) {
        LastInGroupList.push_back((Decl*)Result);
        std::reverse(LastInGroupList.begin(), LastInGroupList.end());
        Result = LastInGroupList.back();
        LastInGroupList.pop_back();
      }

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
                                             unsigned MainFileID) {
  return new ASTStreamer(PP, MainFileID);
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

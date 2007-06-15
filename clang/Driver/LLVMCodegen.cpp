//===--- LLVMCodegen.cpp - Emit LLVM Code from ASTs -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This builds an AST and converts it to LLVM Code.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Sema/ASTStreamer.h"
#include "clang/AST/AST.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/Module.h"
#include <iostream>
using namespace clang;

//===----------------------------------------------------------------------===//
// LLVM Emission
//===----------------------------------------------------------------------===//

void clang::EmitLLVMFromASTs(Preprocessor &PP, unsigned MainFileID,
                             bool PrintStats) {
  Diagnostic &Diags = PP.getDiagnostics();
  // Create the streamer to read the file.
  ASTContext Context(PP.getTargetInfo(), PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);
  
  // Create the module to codegen into.
  llvm::Module M("foo");
  
  CodeGen::BuilderTy *Builder = CodeGen::Init(Context, M);
  
  while (Decl *D = ASTStreamer_ReadTopLevelDecl(Streamer)) {
    // If an error occurred, stop code generation, but continue parsing and
    // semantic analysis (to ensure all warnings and errors are emitted).
    if (Diags.hasErrorOccurred())
      continue;
    
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      CodeGen::CodeGenFunction(Builder, FD);
    } else if (isa<TypedefDecl>(D)) {
      std::cerr << "Read top-level typedef decl: '" << D->getName() << "'\n";
    } else {
      std::cerr << "Read top-level variable decl: '" << D->getName() << "'\n";
    }
  }
  
  if (PrintStats) {
    std::cerr << "\nSTATISTICS:\n";
    CodeGen::PrintStats(Builder);
    ASTStreamer_PrintStats(Streamer);
    Context.PrintStats();
  }
  
  CodeGen::Terminate(Builder);
  ASTStreamer_Terminate(Streamer);
  
  // Print the generated code.
  M.print(std::cout);
}


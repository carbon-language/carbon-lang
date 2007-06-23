//===--- ASTStreamers.cpp - ASTStreamer Drivers ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ASTStreamer drivers.
//
//===----------------------------------------------------------------------===//

#include "ASTStreamers.h"
#include "clang/AST/AST.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/ASTStreamer.h"

void clang::BuildASTs(Preprocessor &PP, unsigned MainFileID, bool Stats) {
  // collect global stats on Decls/Stmts (until we have a module streamer)
  if (Stats) {
    Decl::CollectingStats(true);
    Stmt::CollectingStats(true);
  }

  ASTContext Context(PP.getTargetInfo(), PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);

  while (ASTStreamer_ReadTopLevelDecl(Streamer))
    /* keep reading */;

  if (Stats) {
    fprintf(stderr, "\nSTATISTICS:\n");
    ASTStreamer_PrintStats(Streamer);
    Context.PrintStats();
    Decl::PrintStats();
    Stmt::PrintStats();
  }
  
  ASTStreamer_Terminate(Streamer);
}

void clang::PrintFunctionDecl(FunctionDecl *FD) {
  bool HasBody = FD->getBody();
  
  std::string Proto = FD->getName();
  FunctionType *AFT = cast<FunctionType>(FD->getType());

  if (FunctionTypeProto *FT = dyn_cast<FunctionTypeProto>(AFT)) {
    Proto += "(";
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
      if (i) Proto += ", ";
      std::string ParamStr;
      if (HasBody) ParamStr = FD->getParamDecl(i)->getName();
      
      FT->getArgType(i).getAsStringInternal(ParamStr);
      Proto += ParamStr;
    }
    
    if (FT->isVariadic()) {
      if (FD->getNumParams()) Proto += ", ";
      Proto += "...";
    }
    Proto += ")";
  } else {
    assert(isa<FunctionTypeNoProto>(AFT));
    Proto += "()";
  }

  AFT->getResultType().getAsStringInternal(Proto);
  fprintf(stderr, "\n%s", Proto.c_str());
  
  if (FD->getBody()) {
    fprintf(stderr, " ");
    FD->getBody()->dump();
    fprintf(stderr, "\n");
  } else {
    fprintf(stderr, ";\n");
  }
}

void clang::PrintTypeDefDecl(TypedefDecl *TD) {
  std::string S = TD->getName();
  TD->getUnderlyingType().getAsStringInternal(S);
  fprintf(stderr, "typedef %s;\n", S.c_str());
}

void clang::PrintASTs(Preprocessor &PP, unsigned MainFileID, bool Stats) {
  ASTContext Context(PP.getTargetInfo(), PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);
  
  while (Decl *D = ASTStreamer_ReadTopLevelDecl(Streamer)) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      PrintFunctionDecl(FD);
    } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
      PrintTypeDefDecl(TD);
    } else {
      fprintf(stderr, "Read top-level variable decl: '%s'\n", D->getName());
    }
  }
  
  if (Stats) {
    fprintf(stderr, "\nSTATISTICS:\n");
    ASTStreamer_PrintStats(Streamer);
    Context.PrintStats();
  }
  
  ASTStreamer_Terminate(Streamer);
}

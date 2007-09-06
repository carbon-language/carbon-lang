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
#include "clang/AST/CFG.h"
#include "clang/Analysis/LiveVariables.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/ASTStreamer.h"
using namespace clang;

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



static void PrintFunctionDeclStart(FunctionDecl *FD) {
  bool HasBody = FD->getBody();
  
  fprintf(stderr, "\n");

  switch (FD->getStorageClass()) {
  default: assert(0 && "Unknown storage class");
  case FunctionDecl::None: break;
  case FunctionDecl::Extern: fprintf(stderr, "extern "); break;
  case FunctionDecl::Static: fprintf(stderr, "static "); break;
  }
  
  if (FD->isInline())
    fprintf(stderr, "inline ");
  
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
  fprintf(stderr, "%s", Proto.c_str());
  
  if (!FD->getBody())
    fprintf(stderr, ";\n");
  // Doesn't print the body.
}

static void PrintTypeDefDecl(TypedefDecl *TD) {
  std::string S = TD->getName();
  TD->getUnderlyingType().getAsStringInternal(S);
  fprintf(stderr, "typedef %s;\n", S.c_str());
}

void clang::PrintASTs(Preprocessor &PP, unsigned MainFileID, bool Stats) {
  ASTContext Context(PP.getTargetInfo(), PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);
  
  while (Decl *D = ASTStreamer_ReadTopLevelDecl(Streamer)) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      PrintFunctionDeclStart(FD);

      if (FD->getBody()) {
        fprintf(stderr, " ");
        FD->getBody()->dumpPretty();
        fprintf(stderr, "\n");
      }
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

void clang::DumpASTs(Preprocessor &PP, unsigned MainFileID, bool Stats) {
  ASTContext Context(PP.getTargetInfo(), PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);
  
  while (Decl *D = ASTStreamer_ReadTopLevelDecl(Streamer)) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      PrintFunctionDeclStart(FD);
      
      if (FD->getBody()) {
        fprintf(stderr, "\n");
        FD->getBody()->dumpAll(PP.getSourceManager());
        fprintf(stderr, "\n");
      }
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

void clang::DumpCFGs(Preprocessor &PP, unsigned MainFileID,
                     bool Stats, bool use_graphviz) 
{
  ASTContext Context(PP.getTargetInfo(), PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);
  
  while (Decl *D = ASTStreamer_ReadTopLevelDecl(Streamer)) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {      
      if (FD->getBody()) {
        PrintFunctionDeclStart(FD);
        fprintf(stderr,"\n");
        if (CFG* C = CFG::buildCFG(FD->getBody())) {
          if (use_graphviz) C->viewCFG(); else C->dump();
        }
        else
          fprintf(stderr," Error processing CFG.\n");
      }
    }
  }
  
  if (Stats) {
    fprintf(stderr, "\nSTATISTICS:\n");
    ASTStreamer_PrintStats(Streamer);
    Context.PrintStats();
  }
  
  ASTStreamer_Terminate(Streamer);
}

void clang::AnalyzeLiveVariables(Preprocessor &PP, unsigned MainFileID)
{
  ASTContext Context(PP.getTargetInfo(), PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);
  
  while (Decl *D = ASTStreamer_ReadTopLevelDecl(Streamer)) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {      
      if (FD->getBody()) {
        PrintFunctionDeclStart(FD);
        fprintf(stderr,"\n");
        if (CFG* C = CFG::buildCFG(FD->getBody())) {
          LiveVariables L;
          L.runOnCFG(*C);
          L.dumpBlockLiveness(PP.getSourceManager());
        }
        else
          fprintf(stderr," Error processing CFG.\n");
      }
    }
  }
      
  ASTStreamer_Terminate(Streamer);
}



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
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/ASTStreamer.h"
using namespace clang;

void clang::BuildASTs(Preprocessor &PP, unsigned MainFileID, bool Stats) {
  // collect global stats on Decls/Stmts (until we have a module streamer)
  if (Stats) {
    Decl::CollectingStats(true);
    Stmt::CollectingStats(true);
  }

  ASTContext Context(PP.getSourceManager(), PP.getTargetInfo(),
                     PP.getIdentifierTable());
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

static void PrintObjcInterfaceDecl(ObjcInterfaceDecl *OID) {
  std::string S = OID->getName();
  fprintf(stderr, "@interface %s;\n", S.c_str());
  // FIXME: implement the rest...
}

void clang::PrintASTs(Preprocessor &PP, unsigned MainFileID, bool Stats) {
  ASTContext Context(PP.getSourceManager(), PP.getTargetInfo(),
                     PP.getIdentifierTable());
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
    } else if (ObjcInterfaceDecl *OID = dyn_cast<ObjcInterfaceDecl>(D)) {
      PrintObjcInterfaceDecl(OID);
    } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
      fprintf(stderr, "Read top-level variable decl: '%s'\n", SD->getName());
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
  ASTContext Context(PP.getSourceManager(), PP.getTargetInfo(),
                     PP.getIdentifierTable());
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
    } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
      fprintf(stderr, "Read top-level variable decl: '%s'\n", SD->getName());
    }
  }
  
  if (Stats) {
    fprintf(stderr, "\nSTATISTICS:\n");
    ASTStreamer_PrintStats(Streamer);
    Context.PrintStats();
  }
  
  ASTStreamer_Terminate(Streamer);
}

//===----------------------------------------------------------------------===//
// CFGVisitor & VisitCFGs - Boilerplate interface and logic to visit
//   the CFGs for all function definitions.

namespace {

class CFGVisitor {
public:
  virtual ~CFGVisitor() {}
  virtual void VisitCFG(CFG& C) = 0;
  virtual bool printFuncDeclStart() { return true; }
};

} // end anonymous namespace

static void VisitCFGs(CFGVisitor& Visitor, Preprocessor& PP, 
                      unsigned MainFileID, bool Stats) {

  bool printFDecl = Visitor.printFuncDeclStart();
  ASTContext Context(PP.getSourceManager(), PP.getTargetInfo(),
                     PP.getIdentifierTable());
  ASTStreamerTy *Streamer = ASTStreamer_Init(PP, Context, MainFileID);
  
  while (Decl *D = ASTStreamer_ReadTopLevelDecl(Streamer)) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      if (FD->getBody()) {
        
        if (printFDecl) {
          PrintFunctionDeclStart(FD);          
          fprintf(stderr,"\n");
        }
        
        if (CFG* C = CFG::buildCFG(FD->getBody())) {
          Visitor.VisitCFG(*C);
          delete C;
        }
        else
          fprintf(stderr," Error processing CFG.\n");          
      }
  }
  
  if (Stats) {
    fprintf(stderr, "\nSTATISTICS:\n");
    ASTStreamer_PrintStats(Streamer);
    Context.PrintStats();
  }
  
  ASTStreamer_Terminate(Streamer);
}

//===----------------------------------------------------------------------===//
// DumpCFGs - Dump CFGs to stderr or visualize with Graphviz

namespace {
  class CFGDumper : public CFGVisitor {
    const bool UseGraphviz;
  public:
    CFGDumper(bool use_graphviz) : UseGraphviz(use_graphviz) {}
    
    virtual void VisitCFG(CFG& C) {
      if (UseGraphviz) C.viewCFG();
      else C.dump();
    }    
  }; 
} // end anonymous namespace 
  
void clang::DumpCFGs(Preprocessor &PP, unsigned MainFileID,
                     bool Stats, bool use_graphviz) {
  CFGDumper Visitor(use_graphviz);
  VisitCFGs(Visitor,PP,MainFileID,Stats);
}

//===----------------------------------------------------------------------===//
// AnalyzeLiveVariables - perform live variable analysis and dump results

namespace {
  class LivenessVisitor : public CFGVisitor {
    Preprocessor& PP;
  public:
    LivenessVisitor(Preprocessor& pp) : PP(pp) {}
    
    virtual void VisitCFG(CFG& C) {
      LiveVariables L;
      L.runOnCFG(C);
      L.dumpBlockLiveness(PP.getSourceManager());    
      L.dumpVarLiveness(PP.getSourceManager());
    }
  };
} // end anonymous namespace
  
void clang::AnalyzeLiveVariables(Preprocessor &PP, unsigned MainFileID) {
  LivenessVisitor Visitor(PP);
  VisitCFGs(Visitor,PP,MainFileID,false);
}

//===----------------------------------------------------------------------===//
// RunDeadStores - run checker to locate dead stores in a function

namespace {
  class DeadStoreVisitor : public CFGVisitor {
    Preprocessor& PP;
  public:
    DeadStoreVisitor(Preprocessor& pp) : PP(pp) {}
    virtual void VisitCFG(CFG& C) { CheckDeadStores(C,PP); }
    virtual bool printFuncDeclStart() { return false; }
  }; 
} // end anonymous namespace

void clang::RunDeadStoresCheck(Preprocessor &PP,unsigned MainFileID,bool Stats){
  DeadStoreVisitor Visitor(PP);
  VisitCFGs(Visitor,PP,MainFileID,Stats);
}

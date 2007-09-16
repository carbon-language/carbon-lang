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
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/CFG.h"
#include "clang/Analysis/LiveVariables.h"
#include "clang/Analysis/LocalCheckers.h"
using namespace clang;


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

namespace {
  class ASTPrinter : public ASTConsumer {
    virtual void HandleTopLevelDecl(Decl *D) {
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
  };
}

ASTConsumer *clang::CreateASTPrinter() { return new ASTPrinter(); }

namespace {
  class ASTDumper : public ASTConsumer {
    SourceManager *SM;
  public:
    void Initialize(ASTContext &Context, unsigned MainFileID) {
      SM = &Context.SourceMgr;
    }
    
    virtual void HandleTopLevelDecl(Decl *D) {
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        PrintFunctionDeclStart(FD);
        
        if (FD->getBody()) {
          fprintf(stderr, "\n");
          FD->getBody()->dumpAll(*SM);
          fprintf(stderr, "\n");
        }
      } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
        PrintTypeDefDecl(TD);
      } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
        fprintf(stderr, "Read top-level variable decl: '%s'\n", SD->getName());
      }
    }
  };
}

ASTConsumer *clang::CreateASTDumper() { return new ASTDumper(); }

//===----------------------------------------------------------------------===//
// CFGVisitor & VisitCFGs - Boilerplate interface and logic to visit
//   the CFGs for all function definitions.

namespace {

class CFGVisitor : public ASTConsumer {
public:
  // CFG Visitor interface to be implemented by subclass.
  virtual void VisitCFG(CFG& C) = 0;
  virtual bool printFuncDeclStart() { return true; }
  
  virtual void HandleTopLevelDecl(Decl *D);
};

} // end anonymous namespace

void CFGVisitor::HandleTopLevelDecl(Decl *D) {
  FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD || !FD->getBody())
    return;
      
  if (printFuncDeclStart()) {
    PrintFunctionDeclStart(FD);          
    fprintf(stderr,"\n");
  }
    
  if (CFG *C = CFG::buildCFG(FD->getBody())) {
    VisitCFG(*C);
    delete C;
  } else
    fprintf(stderr, " Error processing CFG.\n");          
}

//===----------------------------------------------------------------------===//
// DumpCFGs - Dump CFGs to stderr or visualize with Graphviz

namespace {
  class CFGDumper : public CFGVisitor {
    const bool UseGraphviz;
  public:
    CFGDumper(bool use_graphviz) : UseGraphviz(use_graphviz) {}
    
    virtual void VisitCFG(CFG &C) {
      if (UseGraphviz)
        C.viewCFG();
      else
        C.dump();
    }
  }; 
} // end anonymous namespace 
  
ASTConsumer *clang::CreateCFGDumper(bool ViewGraphs) {
  return new CFGDumper(ViewGraphs);
}

//===----------------------------------------------------------------------===//
// AnalyzeLiveVariables - perform live variable analysis and dump results

namespace {
  class LivenessVisitor : public CFGVisitor {
    SourceManager *SM;
  public:
    virtual void Initialize(ASTContext &Context, unsigned MainFileID) {
      SM = &Context.SourceMgr;
    }

    virtual void VisitCFG(CFG& C) {
      LiveVariables L;
      L.runOnCFG(C);
      L.dumpBlockLiveness(*SM);    
      L.dumpVarLiveness(*SM);
    }
  };
} // end anonymous namespace
  
ASTConsumer *clang::CreateLiveVarAnalyzer() {
  return new LivenessVisitor();
}

//===----------------------------------------------------------------------===//
// RunDeadStores - run checker to locate dead stores in a function

namespace {
  class DeadStoreVisitor : public CFGVisitor {
    Diagnostic &Diags;
    ASTContext *Ctx;
  public:
    DeadStoreVisitor(Diagnostic &diags) : Diags(diags) {}
    virtual void Initialize(ASTContext &Context, unsigned MainFileID) {
      Ctx = &Context;
    }
    
    virtual void VisitCFG(CFG& C) { CheckDeadStores(C, *Ctx, Diags); }
    virtual bool printFuncDeclStart() { return false; }
  }; 
} // end anonymous namespace

ASTConsumer *clang::CreateDeadStoreChecker(Diagnostic &Diags) {
  return new DeadStoreVisitor(Diags);
}

//===----------------------------------------------------------------------===//
// LLVM Emitter

#include "clang/Basic/Diagnostic.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "llvm/Module.h"
#include <iostream>

namespace {
  class LLVMEmitter : public ASTConsumer {
    Diagnostic &Diags;
    llvm::Module *M;
    ASTContext *Ctx;
    CodeGen::BuilderTy *Builder;
  public:
    LLVMEmitter(Diagnostic &diags) : Diags(diags) {}
    virtual void Initialize(ASTContext &Context, unsigned MainFileID) {
      Ctx = &Context;
      M = new llvm::Module("foo");
      Builder = CodeGen::Init(Context, *M);
    }
    
    virtual void HandleTopLevelDecl(Decl *D) {
      // If an error occurred, stop code generation, but continue parsing and
      // semantic analysis (to ensure all warnings and errors are emitted).
      if (Diags.hasErrorOccurred())
        return;
      
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        CodeGen::CodeGenFunction(Builder, FD);
      } else if (FileVarDecl *FVD = dyn_cast<FileVarDecl>(D)) {
        CodeGen::CodeGenGlobalVar(Builder, FVD);
      } else {
        assert(isa<TypedefDecl>(D) && "Only expected typedefs here");
        // don't codegen for now, eventually pass down for debug info.
        //std::cerr << "Read top-level typedef decl: '" << D->getName() << "'\n";
      }
    }
    
    ~LLVMEmitter() {
      CodeGen::Terminate(Builder);
      
      // Print the generated code.
      M->print(std::cout);
      delete M;
    }
  }; 
} // end anonymous namespace

ASTConsumer *clang::CreateLLVMEmitter(Diagnostic &Diags) {
  return new LLVMEmitter(Diags);
}


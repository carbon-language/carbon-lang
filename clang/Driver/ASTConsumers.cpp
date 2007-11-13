//===--- ASTConsumers.cpp - ASTConsumer implementations -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AST Consumer Implementations.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
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

static void PrintObjcMethodDecl(ObjcMethodDecl *OMD) {
  if (OMD->isInstance())
    fprintf(stderr, "\n- ");
  else 
    fprintf(stderr, "\n+ ");
  if (!OMD->getResultType().isNull())
    fprintf(stderr, "(%s) ", OMD->getResultType().getAsString().c_str());
  // FIXME: just print original selector name!
  fprintf(stderr, "%s ", OMD->getSelector().getName().c_str());
  
  for (int i = 0; i < OMD->getNumParams(); i++) {
    ParmVarDecl *PDecl = OMD->getParamDecl(i);
    // FIXME: selector is missing here!
    fprintf(stderr, " :(%s) %s", PDecl->getType().getAsString().c_str(), 
            PDecl->getName()); 
  }
}

static void PrintObjcImplementationDecl(ObjcImplementationDecl *OID) {
  std::string I = OID->getName();
  ObjcInterfaceDecl *SID = OID->getSuperClass();
  if (SID) {
    std::string S = SID->getName();
    fprintf(stderr, "@implementation %s : %s", I.c_str(), S.c_str());
  }
  else
    fprintf(stderr, "@implementation %s", I.c_str());
  
  for (int i = 0; i < OID->getNumInstanceMethods(); i++) {
    PrintObjcMethodDecl(OID->getInstanceMethods()[i]);
    ObjcMethodDecl *OMD = OID->getInstanceMethods()[i];
    if (OMD->getBody()) {
      fprintf(stderr, " ");
      OMD->getBody()->dumpPretty();
      fprintf(stderr, "\n");
    }
  }
  
  for (int i = 0; i < OID->getNumClassMethods(); i++) {
    PrintObjcMethodDecl(OID->getClassMethods()[i]);
    ObjcMethodDecl *OMD = OID->getClassMethods()[i];
    if (OMD->getBody()) {
      fprintf(stderr, " ");
      OMD->getBody()->dumpPretty();
      fprintf(stderr, "\n");
    }
  }
  
  fprintf(stderr,"@end\n");
}


static void PrintObjcInterfaceDecl(ObjcInterfaceDecl *OID) {
  std::string I = OID->getName();
  ObjcInterfaceDecl *SID = OID->getSuperClass();
  if (SID) {
    std::string S = SID->getName();
    fprintf(stderr, "@interface %s : %s", I.c_str(), S.c_str());
  }
  else
    fprintf(stderr, "@interface %s", I.c_str());
  // Protocols?
  int count = OID->getNumIntfRefProtocols();
  if (count > 0) {
    ObjcProtocolDecl **refProtocols = OID->getReferencedProtocols();
    for (int i = 0; i < count; i++)
      fprintf(stderr, "%c%s", (i == 0 ? '<' : ','), 
              refProtocols[i]->getName());
  }
  if (count > 0)
    fprintf(stderr, ">\n");
  else
    fprintf(stderr, "\n");
  
  int NumIvars = OID->getNumInstanceVariables();
  if (NumIvars > 0) {
    ObjcIvarDecl **Ivars = OID->getInstanceVariables();
    fprintf(stderr,"{");
    for (int i = 0; i < NumIvars; i++) {
      fprintf(stderr, "\t%s %s;\n", Ivars[i]->getType().getAsString().c_str(),
              Ivars[i]->getName());
    }
    fprintf(stderr, "}\n");
  }
  
  int NumProperties = OID->getNumPropertyDecl();
  if (NumProperties > 0) {
    for (int i = 0; i < NumProperties; i++) {
      ObjcPropertyDecl *PDecl = OID->getPropertyDecl()[i];
      fprintf(stderr, "@property");
      if (PDecl->getPropertyAttributes() != ObjcPropertyDecl::OBJC_PR_noattr) {
        bool first = true;
        fprintf(stderr, " (");
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_readonly)
        {
          fprintf(stderr, "%creadonly", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_getter)
        {
          fprintf(stderr, "%cgetter = %s", first ? ' ' : ','
                  , PDecl->getGetterName()->getName());
          first = false;
        }
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_setter)
        {
          fprintf(stderr, "%csetter = %s:", first ? ' ' : ','
                  , PDecl->getSetterName()->getName());
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_assign)
        {
          fprintf(stderr, "%cassign", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_readwrite)
        {
          fprintf(stderr, "%creadwrite", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_retain)
        {
          fprintf(stderr, "%cretain", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_copy)
        {
          fprintf(stderr, "%ccopy", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_nonatomic)
        {
          fprintf(stderr, "%cnonatomic", first ? ' ' : ',');
          first = false;
        }
        fprintf(stderr, " )");
      }
      ObjcIvarDecl **IDecl = PDecl->getPropertyDecls();
      fprintf(stderr, " %s %s", IDecl[0]->getType().getAsString().c_str(),
              IDecl[0]->getName());

      for (int j = 1; j < PDecl->getNumPropertyDecls(); j++) {
        fprintf(stderr, ", %s", IDecl[j]->getName());
      }
      fprintf(stderr, ";\n");
    }
  }
  fprintf(stderr,"@end\n");
  // FIXME: implement the rest...
}

static void PrintObjcProtocolDecl(ObjcProtocolDecl *PID) {
  std::string S = PID->getName();
  fprintf(stderr, "@protocol %s;\n", S.c_str());
  // FIXME: implement the rest...
}

static void PrintObjcCategoryImplDecl(ObjcCategoryImplDecl *PID) {
  std::string S = PID->getName();
  std::string I = PID->getClassInterface()->getName();
  fprintf(stderr, "@implementation %s(%s);\n", I.c_str(), S.c_str());
  // FIXME: implement the rest...
}

static void PrintObjcCategoryDecl(ObjcCategoryDecl *PID) {
  std::string S = PID->getName();
  std::string I = PID->getClassInterface()->getName();
  fprintf(stderr, "@interface %s(%s);\n", I.c_str(), S.c_str());
  // FIXME: implement the rest...
}

static void PrintObjcCompatibleAliasDecl(ObjcCompatibleAliasDecl *AID) {
  std::string A = AID->getName();
  std::string I = AID->getClassInterface()->getName();
  fprintf(stderr, "@compatibility_alias %s %s;\n", A.c_str(), I.c_str());
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
      } else if (isa<ObjcMethodDecl>(D)) {
	    // Do nothing, methods definitions are printed in
		// PrintObjcImplementationDecl.
      } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
        PrintTypeDefDecl(TD);
      } else if (ObjcInterfaceDecl *OID = dyn_cast<ObjcInterfaceDecl>(D)) {
        PrintObjcInterfaceDecl(OID);
      } else if (ObjcProtocolDecl *PID = dyn_cast<ObjcProtocolDecl>(D)) {
        PrintObjcProtocolDecl(PID);
      } else if (ObjcForwardProtocolDecl *OFPD = 
                     dyn_cast<ObjcForwardProtocolDecl>(D)) {
        fprintf(stderr, "@protocol ");
        for (unsigned i = 0, e = OFPD->getNumForwardDecls(); i != e; ++i) {
          const ObjcProtocolDecl *D = OFPD->getForwardProtocolDecl(i);
          if (i) fprintf(stderr, ", ");
          fprintf(stderr, "%s", D->getName());
        }
        fprintf(stderr, ";\n");
      } else if (ObjcImplementationDecl *OID = 
                   dyn_cast<ObjcImplementationDecl>(D)) {
        PrintObjcImplementationDecl(OID);
      } else if (ObjcCategoryImplDecl *OID = 
                 dyn_cast<ObjcCategoryImplDecl>(D)) {
        PrintObjcCategoryImplDecl(OID);
      } else if (ObjcCategoryDecl *OID = 
                 dyn_cast<ObjcCategoryDecl>(D)) {
        PrintObjcCategoryDecl(OID);
      } else if (ObjcCompatibleAliasDecl *OID = 
                 dyn_cast<ObjcCompatibleAliasDecl>(D)) {
        PrintObjcCompatibleAliasDecl(OID);
      } else if (isa<ObjcClassDecl>(D)) {
        fprintf(stderr, "@class [printing todo]\n");
      } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
        fprintf(stderr, "Read top-level variable decl: '%s'\n", SD->getName());
      } else {
        assert(0 && "Unknown decl type!");
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
      } else if (ObjcInterfaceDecl *OID = dyn_cast<ObjcInterfaceDecl>(D)) {
        fprintf(stderr, "Read objc interface '%s'\n", OID->getName());
      } else if (ObjcProtocolDecl *OPD = dyn_cast<ObjcProtocolDecl>(D)) {
        fprintf(stderr, "Read objc protocol '%s'\n", OPD->getName());
      } else if (ObjcCategoryDecl *OCD = dyn_cast<ObjcCategoryDecl>(D)) {
        fprintf(stderr, "Read objc category '%s'\n", OCD->getName());
      } else if (isa<ObjcForwardProtocolDecl>(D)) {
        fprintf(stderr, "Read objc fwd protocol decl\n");
      } else if (isa<ObjcClassDecl>(D)) {
        fprintf(stderr, "Read objc fwd class decl\n");
      } else {
        assert(0 && "Unknown decl type!");
      }
    }
  };
}

ASTConsumer *clang::CreateASTDumper() { return new ASTDumper(); }

namespace {
  class ASTViewer : public ASTConsumer {
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
          FD->getBody()->viewAST();
          fprintf(stderr, "\n");
        }
      }
    }
  };
}

ASTConsumer *clang::CreateASTViewer() { return new ASTViewer(); }


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
    
  CFG *C = CFG::buildCFG(FD->getBody());
  VisitCFG(*C);
  delete C;
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
      LiveVariables L(C);
      L.runOnCFG(C);
      L.dumpBlockLiveness(*SM);
    }
  };
} // end anonymous namespace
  
ASTConsumer *clang::CreateLiveVarAnalyzer() {
  return new LivenessVisitor();
}

//===----------------------------------------------------------------------===//
// DeadStores - run checker to locate dead stores in a function

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
// Unitialized Values - run checker to flag potential uses of uninitalized
//  variables.

namespace {
  class UninitValsVisitor : public CFGVisitor {
    Diagnostic &Diags;
    ASTContext *Ctx;
  public:
    UninitValsVisitor(Diagnostic &diags) : Diags(diags) {}
    virtual void Initialize(ASTContext &Context, unsigned MainFileID) {
      Ctx = &Context;
    }
    
    virtual void VisitCFG(CFG& C) { CheckUninitializedValues(C, *Ctx, Diags); }
    virtual bool printFuncDeclStart() { return false; }
  }; 
} // end anonymous namespace

ASTConsumer *clang::CreateUnitValsChecker(Diagnostic &Diags) {
  return new UninitValsVisitor(Diags);
}

//===----------------------------------------------------------------------===//
// LLVM Emitter

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include <iostream>

namespace {
  class LLVMEmitter : public ASTConsumer {
    Diagnostic &Diags;
    llvm::Module *M;
    const llvm::TargetData *TD;
    ASTContext *Ctx;
    CodeGen::CodeGenModule *Builder;
  public:
    LLVMEmitter(Diagnostic &diags) : Diags(diags) {}
    virtual void Initialize(ASTContext &Context, unsigned MainFileID) {
      Ctx = &Context;
      M = new llvm::Module("foo");
      M->setTargetTriple(Ctx->Target.getTargetTriple());
      TD = new llvm::TargetData(Ctx->Target.getTargetDescription());
      Builder = CodeGen::Init(Context, *M, *TD);
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


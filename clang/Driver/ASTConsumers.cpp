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

//===----------------------------------------------------------------------===//
/// DeclPrinter - Utility class for printing top-level decls.

namespace {
  class DeclPrinter {
  public:
    FILE* FP;

    DeclPrinter(FILE* fp) : FP(fp ? fp : stderr) {}
    
    void PrintFunctionDeclStart(FunctionDecl *FD);    
    void PrintTypeDefDecl(TypedefDecl *TD);    
    void PrintObjcMethodDecl(ObjcMethodDecl *OMD);    
    void PrintObjcImplementationDecl(ObjcImplementationDecl *OID);
    void PrintObjcInterfaceDecl(ObjcInterfaceDecl *OID);
    void PrintObjcProtocolDecl(ObjcProtocolDecl *PID);  
    void PrintObjcCategoryImplDecl(ObjcCategoryImplDecl *PID);    
    void PrintObjcCategoryDecl(ObjcCategoryDecl *PID);    
    void PrintObjcCompatibleAliasDecl(ObjcCompatibleAliasDecl *AID);
  };
} // end anonymous namespace

void DeclPrinter::PrintFunctionDeclStart(FunctionDecl *FD) {
  bool HasBody = FD->getBody();
  
  fprintf(FP, "\n");

  switch (FD->getStorageClass()) {
  default: assert(0 && "Unknown storage class");
  case FunctionDecl::None: break;
  case FunctionDecl::Extern: fprintf(FP, "extern "); break;
  case FunctionDecl::Static: fprintf(FP, "static "); break;
  }
  
  if (FD->isInline())
    fprintf(FP, "inline ");
  
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
  fprintf(FP, "%s", Proto.c_str());
  
  if (!FD->getBody())
    fprintf(FP, ";\n");
  // Doesn't print the body.
}

void DeclPrinter::PrintTypeDefDecl(TypedefDecl *TD) {
  std::string S = TD->getName();
  TD->getUnderlyingType().getAsStringInternal(S);
  fprintf(FP, "typedef %s;\n", S.c_str());
}

void DeclPrinter::PrintObjcMethodDecl(ObjcMethodDecl *OMD) {
  if (OMD->isInstance())
    fprintf(FP, "\n- ");
  else 
    fprintf(FP, "\n+ ");
  if (!OMD->getResultType().isNull())
    fprintf(FP, "(%s) ", OMD->getResultType().getAsString().c_str());
  // FIXME: just print original selector name!
  fprintf(FP, "%s ", OMD->getSelector().getName().c_str());
  
  for (int i = 0; i < OMD->getNumParams(); i++) {
    ParmVarDecl *PDecl = OMD->getParamDecl(i);
    // FIXME: selector is missing here!
    fprintf(FP, " :(%s) %s", PDecl->getType().getAsString().c_str(), 
            PDecl->getName()); 
  }
}

void DeclPrinter::PrintObjcImplementationDecl(ObjcImplementationDecl *OID) {
  std::string I = OID->getName();
  ObjcInterfaceDecl *SID = OID->getSuperClass();
  if (SID) {
    std::string S = SID->getName();
    fprintf(FP, "@implementation %s : %s", I.c_str(), S.c_str());
  }
  else
    fprintf(FP, "@implementation %s", I.c_str());
  
  for (int i = 0; i < OID->getNumInstanceMethods(); i++) {
    PrintObjcMethodDecl(OID->getInstanceMethods()[i]);
    ObjcMethodDecl *OMD = OID->getInstanceMethods()[i];
    if (OMD->getBody()) {
      fprintf(FP, " ");
      OMD->getBody()->dumpPretty();
      fprintf(FP, "\n");
    }
  }
  
  for (int i = 0; i < OID->getNumClassMethods(); i++) {
    PrintObjcMethodDecl(OID->getClassMethods()[i]);
    ObjcMethodDecl *OMD = OID->getClassMethods()[i];
    if (OMD->getBody()) {
      fprintf(FP, " ");
      OMD->getBody()->dumpPretty();
      fprintf(FP, "\n");
    }
  }
  
  fprintf(FP,"@end\n");
}


void DeclPrinter::PrintObjcInterfaceDecl(ObjcInterfaceDecl *OID) {
  std::string I = OID->getName();
  ObjcInterfaceDecl *SID = OID->getSuperClass();
  if (SID) {
    std::string S = SID->getName();
    fprintf(FP, "@interface %s : %s", I.c_str(), S.c_str());
  }
  else
    fprintf(FP, "@interface %s", I.c_str());
  // Protocols?
  int count = OID->getNumIntfRefProtocols();
  if (count > 0) {
    ObjcProtocolDecl **refProtocols = OID->getReferencedProtocols();
    for (int i = 0; i < count; i++)
      fprintf(FP, "%c%s", (i == 0 ? '<' : ','), 
              refProtocols[i]->getName());
  }
  if (count > 0)
    fprintf(FP, ">\n");
  else
    fprintf(FP, "\n");
  
  int NumIvars = OID->getNumInstanceVariables();
  if (NumIvars > 0) {
    ObjcIvarDecl **Ivars = OID->getInstanceVariables();
    fprintf(FP,"{");
    for (int i = 0; i < NumIvars; i++) {
      fprintf(FP, "\t%s %s;\n", Ivars[i]->getType().getAsString().c_str(),
              Ivars[i]->getName());
    }
    fprintf(FP, "}\n");
  }
  
  int NumProperties = OID->getNumPropertyDecl();
  if (NumProperties > 0) {
    for (int i = 0; i < NumProperties; i++) {
      ObjcPropertyDecl *PDecl = OID->getPropertyDecl()[i];
      fprintf(FP, "@property");
      if (PDecl->getPropertyAttributes() != ObjcPropertyDecl::OBJC_PR_noattr) {
        bool first = true;
        fprintf(FP, " (");
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_readonly)
        {
          fprintf(FP, "%creadonly", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_getter)
        {
          fprintf(FP, "%cgetter = %s", first ? ' ' : ','
                  , PDecl->getGetterName()->getName());
          first = false;
        }
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_setter)
        {
          fprintf(FP, "%csetter = %s:", first ? ' ' : ','
                  , PDecl->getSetterName()->getName());
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_assign)
        {
          fprintf(FP, "%cassign", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_readwrite)
        {
          fprintf(FP, "%creadwrite", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_retain)
        {
          fprintf(FP, "%cretain", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_copy)
        {
          fprintf(FP, "%ccopy", first ? ' ' : ',');
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_nonatomic)
        {
          fprintf(FP, "%cnonatomic", first ? ' ' : ',');
          first = false;
        }
        fprintf(FP, " )");
      }
      ObjcIvarDecl **IDecl = PDecl->getPropertyDecls();
      fprintf(FP, " %s %s", IDecl[0]->getType().getAsString().c_str(),
              IDecl[0]->getName());

      for (int j = 1; j < PDecl->getNumPropertyDecls(); j++) {
        fprintf(FP, ", %s", IDecl[j]->getName());
      }
      fprintf(FP, ";\n");
    }
  }
  fprintf(FP,"@end\n");
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjcProtocolDecl(ObjcProtocolDecl *PID) {
  std::string S = PID->getName();
  fprintf(FP, "@protocol %s;\n", S.c_str());
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjcCategoryImplDecl(ObjcCategoryImplDecl *PID) {
  std::string S = PID->getName();
  std::string I = PID->getClassInterface()->getName();
  fprintf(FP, "@implementation %s(%s);\n", I.c_str(), S.c_str());
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjcCategoryDecl(ObjcCategoryDecl *PID) {
  std::string S = PID->getName();
  std::string I = PID->getClassInterface()->getName();
  fprintf(FP, "@interface %s(%s);\n", I.c_str(), S.c_str());
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjcCompatibleAliasDecl(ObjcCompatibleAliasDecl *AID) {
  std::string A = AID->getName();
  std::string I = AID->getClassInterface()->getName();
  fprintf(FP, "@compatibility_alias %s %s;\n", A.c_str(), I.c_str());
}

//===----------------------------------------------------------------------===//
/// ASTPrinter - Pretty-printer of ASTs

namespace {
  class ASTPrinter : public ASTConsumer, public DeclPrinter {
  public:
    ASTPrinter(FILE* F = NULL) : DeclPrinter(F) {}
    
    virtual void HandleTopLevelDecl(Decl *D) {
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        PrintFunctionDeclStart(FD);
        
        if (FD->getBody()) {
          fprintf(FP, " ");
          FD->getBody()->dumpPretty();
          fprintf(FP, "\n");
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
        fprintf(FP, "@protocol ");
        for (unsigned i = 0, e = OFPD->getNumForwardDecls(); i != e; ++i) {
          const ObjcProtocolDecl *D = OFPD->getForwardProtocolDecl(i);
          if (i) fprintf(FP, ", ");
          fprintf(FP, "%s", D->getName());
        }
        fprintf(FP, ";\n");
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
        fprintf(FP, "@class [printing todo]\n");
      } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
        fprintf(FP, "Read top-level variable decl: '%s'\n", SD->getName());
      } else {
        assert(0 && "Unknown decl type!");
      }
    }
  };
}

ASTConsumer *clang::CreateASTPrinter(FILE* fp) { return new ASTPrinter(fp); }

//===----------------------------------------------------------------------===//
/// ASTDumper - Low-level dumper of ASTs

namespace {
  class ASTDumper : public ASTConsumer, public DeclPrinter {
    SourceManager *SM;
  public:
    ASTDumper(FILE* fp = NULL) : DeclPrinter(fp) {}
    
    void Initialize(ASTContext &Context, unsigned MainFileID) {
      SM = &Context.SourceMgr;
    }
    
    virtual void HandleTopLevelDecl(Decl *D) {
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        PrintFunctionDeclStart(FD);
        
        if (FD->getBody()) {
          fprintf(FP, "\n");
          FD->getBody()->dumpAll(*SM);
          fprintf(FP, "\n");
        }
      } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
        PrintTypeDefDecl(TD);
      } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
        fprintf(FP, "Read top-level variable decl: '%s'\n", SD->getName());
      } else if (ObjcInterfaceDecl *OID = dyn_cast<ObjcInterfaceDecl>(D)) {
        fprintf(FP, "Read objc interface '%s'\n", OID->getName());
      } else if (ObjcProtocolDecl *OPD = dyn_cast<ObjcProtocolDecl>(D)) {
        fprintf(FP, "Read objc protocol '%s'\n", OPD->getName());
      } else if (ObjcCategoryDecl *OCD = dyn_cast<ObjcCategoryDecl>(D)) {
        fprintf(FP, "Read objc category '%s'\n", OCD->getName());
      } else if (isa<ObjcForwardProtocolDecl>(D)) {
        fprintf(FP, "Read objc fwd protocol decl\n");
      } else if (isa<ObjcClassDecl>(D)) {
        fprintf(FP, "Read objc fwd class decl\n");
      } else {
        assert(0 && "Unknown decl type!");
      }
    }
  };
}

ASTConsumer *clang::CreateASTDumper() { return new ASTDumper(); }

//===----------------------------------------------------------------------===//
/// ASTViewer - AST Visualization

namespace {
  class ASTViewer : public ASTConsumer {
    SourceManager *SM;
  public:
    void Initialize(ASTContext &Context, unsigned MainFileID) {
      SM = &Context.SourceMgr;
    }
    
    virtual void HandleTopLevelDecl(Decl *D) {
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        DeclPrinter(stderr).PrintFunctionDeclStart(FD);
        
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
    DeclPrinter(stderr).PrintFunctionDeclStart(FD);          
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
        assert(isa<TypeDecl>(D) && "Only expected type decls here");
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


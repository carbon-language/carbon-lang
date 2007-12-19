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
#include "clang/AST/TranslationUnit.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/CFG.h"
#include "clang/Analysis/LiveVariables.h"
#include "clang/Analysis/LocalCheckers.h"
#include "llvm/Support/Streams.h"

using namespace clang;

//===----------------------------------------------------------------------===//
/// DeclPrinter - Utility class for printing top-level decls.

namespace {
  class DeclPrinter {
  public:
    std::ostream& Out;

    DeclPrinter(std::ostream* out) : Out(out ? *out : *llvm::cerr.stream()) {}    
    DeclPrinter() : Out(*llvm::cerr.stream()) {}
    
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
  
  Out << '\n';

  switch (FD->getStorageClass()) {
  default: assert(0 && "Unknown storage class");
  case FunctionDecl::None: break;
  case FunctionDecl::Extern: Out << "extern "; break;
  case FunctionDecl::Static: Out << "static "; break;
  }
  
  if (FD->isInline())
    Out << "inline ";
  
  std::string Proto = FD->getName();
  const FunctionType *AFT = FD->getType()->getAsFunctionType();

  if (const FunctionTypeProto *FT = dyn_cast<FunctionTypeProto>(AFT)) {
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
  Out << Proto;
  
  if (!FD->getBody())
    Out << ";\n";
  // Doesn't print the body.
}

void DeclPrinter::PrintTypeDefDecl(TypedefDecl *TD) {
  std::string S = TD->getName();
  TD->getUnderlyingType().getAsStringInternal(S);
  Out << "typedef " << S << ";\n";
}

void DeclPrinter::PrintObjcMethodDecl(ObjcMethodDecl *OMD) {
  if (OMD->isInstance())
    Out << "\n- ";
  else 
    Out << "\n+ ";
  if (!OMD->getResultType().isNull())
    Out << '(' << OMD->getResultType().getAsString() << ") ";
  // FIXME: just print original selector name!
  Out << OMD->getSelector().getName();
  
  for (int i = 0; i < OMD->getNumParams(); i++) {
    ParmVarDecl *PDecl = OMD->getParamDecl(i);
    // FIXME: selector is missing here!    
    Out << " :(" << PDecl->getType().getAsString() << ") " << PDecl->getName(); 
  }
}

void DeclPrinter::PrintObjcImplementationDecl(ObjcImplementationDecl *OID) {
  std::string I = OID->getName();
  ObjcInterfaceDecl *SID = OID->getSuperClass();

  if (SID)
    Out << "@implementation " << I << " : " << SID->getName();
  else
    Out << "@implementation " << I;
  
  for (ObjcImplementationDecl::instmeth_iterator I = OID->instmeth_begin(),
       E = OID->instmeth_end(); I != E; ++I) {
    ObjcMethodDecl *OMD = *I;
    PrintObjcMethodDecl(OMD);
    if (OMD->getBody()) {
      Out << ' ';
      OMD->getBody()->printPretty(Out);
      Out << '\n';
    }
  }
  
  for (ObjcImplementationDecl::classmeth_iterator I = OID->classmeth_begin(),
       E = OID->classmeth_end(); I != E; ++I) {
    ObjcMethodDecl *OMD = *I;
    PrintObjcMethodDecl(OMD);
    if (OMD->getBody()) {
      Out << ' ';
      OMD->getBody()->printPretty(Out);
      Out << '\n';
    }
  }
  
  Out << "@end\n";
}


void DeclPrinter::PrintObjcInterfaceDecl(ObjcInterfaceDecl *OID) {
  std::string I = OID->getName();
  ObjcInterfaceDecl *SID = OID->getSuperClass();

  if (SID)
    Out << "@interface " << I << " : " << SID->getName();
  else
    Out << "@interface " << I;
  
  // Protocols?
  int count = OID->getNumIntfRefProtocols();

  if (count > 0) {
    ObjcProtocolDecl **refProtocols = OID->getReferencedProtocols();
    for (int i = 0; i < count; i++)
      Out << (i == 0 ? '<' : ',') << refProtocols[i]->getName();
  }
  
  if (count > 0)
    Out << ">\n";
  else
    Out << '\n';
  
  if (OID->getNumInstanceVariables() > 0) {
    Out << '{';
    for (ObjcInterfaceDecl::ivar_iterator I = OID->ivar_begin(),
         E = OID->ivar_end(); I != E; ++I) {
      Out << '\t' << (*I)->getType().getAsString()
          << ' '  << (*I)->getName() << ";\n";      
    }
    Out << "}\n";
  }
  
  int NumProperties = OID->getNumPropertyDecl();
  if (NumProperties > 0) {
    for (int i = 0; i < NumProperties; i++) {
      ObjcPropertyDecl *PDecl = OID->getPropertyDecl()[i];
      Out << "@property";
      if (PDecl->getPropertyAttributes() != ObjcPropertyDecl::OBJC_PR_noattr) {
        bool first = true;
        Out << " (";
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_readonly)
        {
          Out << (first ? ' ' : ',') << "readonly";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_getter)
        {
          Out << (first ? ' ' : ',') << "getter = "
              << PDecl->getGetterName()->getName();
          first = false;
        }
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_setter)
        {
          Out << (first ? ' ' : ',') << "setter = "
              << PDecl->getSetterName()->getName();
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_assign)
        {
          Out << (first ? ' ' : ',') << "assign";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_readwrite)
        {
          Out << (first ? ' ' : ',') << "readwrite";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_retain)
        {
          Out << (first ? ' ' : ',') << "retain";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_copy)
        {
          Out << (first ? ' ' : ',') << "copy";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjcPropertyDecl::OBJC_PR_nonatomic)
        {
          Out << (first ? ' ' : ',') << "nonatomic";
          first = false;
        }
        Out << " )";
      }
      
      ObjcIvarDecl **IDecl = PDecl->getPropertyDecls();
      
      Out << ' ' << IDecl[0]->getType().getAsString()
          << ' ' << IDecl[0]->getName();
      
      for (int j = 1; j < PDecl->getNumPropertyDecls(); j++)
        Out << ", " << IDecl[j]->getName();

      Out << ";\n";
    }
  }
  
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjcProtocolDecl(ObjcProtocolDecl *PID) {
  Out << "@protocol " << PID->getName() << '\n';
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjcCategoryImplDecl(ObjcCategoryImplDecl *PID) {
  Out << "@implementation "
      << PID->getClassInterface()->getName()
      << '(' << PID->getName() << ");\n";  
  
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjcCategoryDecl(ObjcCategoryDecl *PID) {
  Out << "@interface " 
      << PID->getClassInterface()->getName()
      << '(' << PID->getName() << ");\n";
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjcCompatibleAliasDecl(ObjcCompatibleAliasDecl *AID) {
  Out << "@compatibility_alias " << AID->getName() 
      << ' ' << AID->getClassInterface()->getName() << ";\n";  
}

//===----------------------------------------------------------------------===//
/// ASTPrinter - Pretty-printer of ASTs

namespace {
  class ASTPrinter : public ASTConsumer, public DeclPrinter {
  public:
    ASTPrinter(std::ostream* o = NULL) : DeclPrinter(o) {}
    
    virtual void HandleTopLevelDecl(Decl *D) {
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        PrintFunctionDeclStart(FD);
        
        if (FD->getBody()) {
          Out << ' ';
          FD->getBody()->printPretty(Out);
          Out << '\n';
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
        Out << "@protocol ";
        for (unsigned i = 0, e = OFPD->getNumForwardDecls(); i != e; ++i) {
          const ObjcProtocolDecl *D = OFPD->getForwardProtocolDecl(i);
          if (i) Out << ", ";
          Out << D->getName();
        }
        Out << ";\n";
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
        Out << "@class [printing todo]\n";
      } else if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
        Out << "Read top-level tag decl: '" << TD->getName() << "'\n";
      } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
        Out << "Read top-level variable decl: '" << SD->getName() << "'\n";
      } else {
        assert(0 && "Unknown decl type!");
      }
    }
  };
}

ASTConsumer *clang::CreateASTPrinter(std::ostream* out) {
  return new ASTPrinter(out);
}

//===----------------------------------------------------------------------===//
/// ASTDumper - Low-level dumper of ASTs

namespace {
  class ASTDumper : public ASTConsumer, public DeclPrinter {
    SourceManager *SM;
  public:
    ASTDumper() : DeclPrinter() {}
    
    void Initialize(ASTContext &Context) {
      SM = &Context.getSourceManager();
    }
    
    virtual void HandleTopLevelDecl(Decl *D) {
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        PrintFunctionDeclStart(FD);
        
        if (FD->getBody()) {
          Out << '\n';
          // FIXME: convert dumper to use std::ostream?
          FD->getBody()->dumpAll(*SM);
          Out << '\n';
        }
      } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
        PrintTypeDefDecl(TD);
      } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
        Out << "Read top-level variable decl: '" << SD->getName() << "'\n";
      } else if (ObjcInterfaceDecl *OID = dyn_cast<ObjcInterfaceDecl>(D)) {
        Out << "Read objc interface '" << OID->getName() << "'\n";
      } else if (ObjcProtocolDecl *OPD = dyn_cast<ObjcProtocolDecl>(D)) {
        Out << "Read objc protocol '" << OPD->getName() << "'\n";
      } else if (ObjcCategoryDecl *OCD = dyn_cast<ObjcCategoryDecl>(D)) {
        Out << "Read objc category '" << OCD->getName() << "'\n";
      } else if (isa<ObjcForwardProtocolDecl>(D)) {
        Out << "Read objc fwd protocol decl\n";
      } else if (isa<ObjcClassDecl>(D)) {
        Out << "Read objc fwd class decl\n";
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
    void Initialize(ASTContext &Context) {
      SM = &Context.getSourceManager();
    }
    
    virtual void HandleTopLevelDecl(Decl *D) {
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        DeclPrinter().PrintFunctionDeclStart(FD);
        
        if (FD->getBody()) {
          llvm::cerr << '\n';
          FD->getBody()->viewAST();
          llvm::cerr << '\n';
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
    DeclPrinter().PrintFunctionDeclStart(FD);
    llvm::cerr << '\n';
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
    virtual void Initialize(ASTContext &Context) {
      SM = &Context.getSourceManager();
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
    virtual void Initialize(ASTContext &Context) {
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
    virtual void Initialize(ASTContext &Context) {
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

namespace {
  class LLVMEmitter : public ASTConsumer {
    Diagnostic &Diags;
    llvm::Module *M;
    const llvm::TargetData *TD;
    ASTContext *Ctx;
    const LangOptions &Features;
    CodeGen::CodeGenModule *Builder;
  public:
    LLVMEmitter(Diagnostic &diags, const LangOptions &LO) 
      : Diags(diags)
      , Features(LO) {}
    virtual void Initialize(ASTContext &Context) {
      Ctx = &Context;
      M = new llvm::Module("foo");
      M->setTargetTriple(Ctx->Target.getTargetTriple());
      M->setDataLayout(Ctx->Target.getTargetDescription());
      TD = new llvm::TargetData(Ctx->Target.getTargetDescription());
      Builder = CodeGen::Init(Context, Features, *M, *TD, Diags);
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
        //std::cerr << "Read top-level typedef decl: '"
        //    << D->getName() << "'\n";
      }
    }
    
    ~LLVMEmitter() {
      CodeGen::Terminate(Builder);
      
      // Print the generated code.
      M->print(llvm::cout.stream());
      delete M;
    }
  }; 
} // end anonymous namespace

ASTConsumer *clang::CreateLLVMEmitter(Diagnostic &Diags, 
                                      const LangOptions &Features) {
  return new LLVMEmitter(Diags, Features);
}

//===----------------------------------------------------------------------===//
// AST Serializer

namespace {

class ASTSerializer : public ASTConsumer {
protected:
  Diagnostic &Diags;
  TranslationUnit TU;
public:
  ASTSerializer(Diagnostic& diags, const LangOptions& LO)
    : Diags(diags), TU(LO) {}
  
  virtual void Initialize(ASTContext &Context) {
    TU.setContext(&Context);
  }
  
  virtual void HandleTopLevelDecl(Decl *D) {
    if (Diags.hasErrorOccurred())
      return;
    
    TU.AddTopLevelDecl(D);
  }
};
    
class SingleFileSerializer : public ASTSerializer {
  const llvm::sys::Path FName;
public:
  SingleFileSerializer(const llvm::sys::Path& F, Diagnostic &diags,
                          const LangOptions &LO)
  : ASTSerializer(diags,LO), FName(F) {}    
  
  ~SingleFileSerializer() {
    EmitASTBitcodeFile(TU,FName);
  }
};

class BuildSerializer : public ASTSerializer {
  llvm::sys::Path EmitDir;  
public:
  BuildSerializer(const llvm::sys::Path& dir, Diagnostic &diags,
                  const LangOptions &LO)
  : ASTSerializer(diags,LO), EmitDir(dir) {}
  
  ~BuildSerializer() { assert (false && "not implemented."); }
};
  
  
} // end anonymous namespace


ASTConsumer* clang::CreateASTSerializer(const std::string& InFile,
                                        const std::string& OutputFile,
                                        Diagnostic &Diags,
                                        const LangOptions &Features) {
  
  if (OutputFile.size()) {
    // The user specified an AST-emission directory.  Determine if the path
    // is absolute.    
    llvm::sys::Path EmitDir(OutputFile);
    
    if (!EmitDir.isAbsolute()) {
      llvm::cerr << 
        "error: Output directory for --serialize must be an absolute path.\n";
      
      return NULL;
    }
    
    // Create the directory if it does not exist.
    EmitDir.createDirectoryOnDisk(true);
    if (!EmitDir.canWrite() || !EmitDir.isDirectory()) {
      llvm::cerr <<
        "error: Could not create output directory for --serialize.\n";
      
      return NULL;
    }
    
    return new BuildSerializer(EmitDir, Diags, Features);
  }

  // The user did not specify an output directory for serialized ASTs.
  // Serialize the translation to a single file whose name is the same
  // as the input file with the ".ast" extension appended.
  
  llvm::sys::Path FName(InFile.c_str());
  FName.appendComponent("ast");
  return new SingleFileSerializer(FName, Diags, Features);  
}

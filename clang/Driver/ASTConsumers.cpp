//===--- ASTConsumers.cpp - ASTConsumer implementations -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AST Consumer Implementations.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
#include "clang/AST/TranslationUnit.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/CFG.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/Analyses/GRConstants.h"
#include "clang/Analysis/LocalCheckers.h"
#include "llvm/Support/Streams.h"
#include <fstream>

using namespace clang;

//===----------------------------------------------------------------------===//
/// DeclPrinter - Utility class for printing top-level decls.

namespace {
  class DeclPrinter {
  public:
    std::ostream& Out;

    DeclPrinter(std::ostream* out) : Out(out ? *out : *llvm::cerr.stream()) {}
    DeclPrinter() : Out(*llvm::cerr.stream()) {}
    
    void PrintDecl(Decl *D);
    void PrintFunctionDeclStart(FunctionDecl *FD);    
    void PrintTypeDefDecl(TypedefDecl *TD);    
    void PrintLinkageSpec(LinkageSpecDecl *LS);
    void PrintObjCMethodDecl(ObjCMethodDecl *OMD);    
    void PrintObjCImplementationDecl(ObjCImplementationDecl *OID);
    void PrintObjCInterfaceDecl(ObjCInterfaceDecl *OID);
    void PrintObjCProtocolDecl(ObjCProtocolDecl *PID);  
    void PrintObjCCategoryImplDecl(ObjCCategoryImplDecl *PID);    
    void PrintObjCCategoryDecl(ObjCCategoryDecl *PID);    
    void PrintObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *AID);
  };
} // end anonymous namespace

void DeclPrinter:: PrintDecl(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    PrintFunctionDeclStart(FD);

    if (FD->getBody()) {
      Out << ' ';
      FD->getBody()->printPretty(Out);
      Out << '\n';
    }
  } else if (isa<ObjCMethodDecl>(D)) {
    // Do nothing, methods definitions are printed in
    // PrintObjCImplementationDecl.
  } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    PrintTypeDefDecl(TD);
  } else if (ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(D)) {
    PrintObjCInterfaceDecl(OID);
  } else if (ObjCProtocolDecl *PID = dyn_cast<ObjCProtocolDecl>(D)) {
    PrintObjCProtocolDecl(PID);
  } else if (ObjCForwardProtocolDecl *OFPD = 
	     dyn_cast<ObjCForwardProtocolDecl>(D)) {
    Out << "@protocol ";
    for (unsigned i = 0, e = OFPD->getNumForwardDecls(); i != e; ++i) {
      const ObjCProtocolDecl *D = OFPD->getForwardProtocolDecl(i);
      if (i) Out << ", ";
      Out << D->getName();
    }
    Out << ";\n";
  } else if (ObjCImplementationDecl *OID = 
	     dyn_cast<ObjCImplementationDecl>(D)) {
    PrintObjCImplementationDecl(OID);
  } else if (ObjCCategoryImplDecl *OID = 
	     dyn_cast<ObjCCategoryImplDecl>(D)) {
    PrintObjCCategoryImplDecl(OID);
  } else if (ObjCCategoryDecl *OID = 
	     dyn_cast<ObjCCategoryDecl>(D)) {
    PrintObjCCategoryDecl(OID);
  } else if (ObjCCompatibleAliasDecl *OID = 
	     dyn_cast<ObjCCompatibleAliasDecl>(D)) {
    PrintObjCCompatibleAliasDecl(OID);
  } else if (isa<ObjCClassDecl>(D)) {
    Out << "@class [printing todo]\n";
  } else if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    Out << "Read top-level tag decl: '" << TD->getName() << "'\n";
  } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D)) {
    Out << "Read top-level variable decl: '" << SD->getName() << "'\n";
  } else if (LinkageSpecDecl *LSD = dyn_cast<LinkageSpecDecl>(D)) {
    PrintLinkageSpec(LSD);
  } else {
    assert(0 && "Unknown decl type!");
  }
}

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

void DeclPrinter::PrintLinkageSpec(LinkageSpecDecl *LS) {
  const char *l;
  if (LS->getLanguage() == LinkageSpecDecl::lang_c)
    l = "C";
  else if (LS->getLanguage() == LinkageSpecDecl::lang_cxx)
    l = "C++";
  else assert(0 && "unknown language in linkage specification");
  Out << "extern \"" << l << "\" { ";
  PrintDecl(LS->getDecl());
  Out << "}\n";
}

void DeclPrinter::PrintObjCMethodDecl(ObjCMethodDecl *OMD) {
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

void DeclPrinter::PrintObjCImplementationDecl(ObjCImplementationDecl *OID) {
  std::string I = OID->getName();
  ObjCInterfaceDecl *SID = OID->getSuperClass();

  if (SID)
    Out << "@implementation " << I << " : " << SID->getName();
  else
    Out << "@implementation " << I;
  
  for (ObjCImplementationDecl::instmeth_iterator I = OID->instmeth_begin(),
       E = OID->instmeth_end(); I != E; ++I) {
    ObjCMethodDecl *OMD = *I;
    PrintObjCMethodDecl(OMD);
    if (OMD->getBody()) {
      Out << ' ';
      OMD->getBody()->printPretty(Out);
      Out << '\n';
    }
  }
  
  for (ObjCImplementationDecl::classmeth_iterator I = OID->classmeth_begin(),
       E = OID->classmeth_end(); I != E; ++I) {
    ObjCMethodDecl *OMD = *I;
    PrintObjCMethodDecl(OMD);
    if (OMD->getBody()) {
      Out << ' ';
      OMD->getBody()->printPretty(Out);
      Out << '\n';
    }
  }
  
  Out << "@end\n";
}


void DeclPrinter::PrintObjCInterfaceDecl(ObjCInterfaceDecl *OID) {
  std::string I = OID->getName();
  ObjCInterfaceDecl *SID = OID->getSuperClass();

  if (SID)
    Out << "@interface " << I << " : " << SID->getName();
  else
    Out << "@interface " << I;
  
  // Protocols?
  int count = OID->getNumIntfRefProtocols();

  if (count > 0) {
    ObjCProtocolDecl **refProtocols = OID->getReferencedProtocols();
    for (int i = 0; i < count; i++)
      Out << (i == 0 ? '<' : ',') << refProtocols[i]->getName();
  }
  
  if (count > 0)
    Out << ">\n";
  else
    Out << '\n';
  
  if (OID->getNumInstanceVariables() > 0) {
    Out << '{';
    for (ObjCInterfaceDecl::ivar_iterator I = OID->ivar_begin(),
         E = OID->ivar_end(); I != E; ++I) {
      Out << '\t' << (*I)->getType().getAsString()
          << ' '  << (*I)->getName() << ";\n";      
    }
    Out << "}\n";
  }
  
  int NumProperties = OID->getNumPropertyDecl();
  if (NumProperties > 0) {
    for (int i = 0; i < NumProperties; i++) {
      ObjCPropertyDecl *PDecl = OID->getPropertyDecl()[i];
      Out << "@property";
      if (PDecl->getPropertyAttributes() != ObjCPropertyDecl::OBJC_PR_noattr) {
        bool first = true;
        Out << " (";
        if (PDecl->getPropertyAttributes() & 
            ObjCPropertyDecl::OBJC_PR_readonly) {
          Out << (first ? ' ' : ',') << "readonly";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_getter) {
          Out << (first ? ' ' : ',') << "getter = "
              << PDecl->getGetterName()->getName();
          first = false;
        }
        if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_setter) {
          Out << (first ? ' ' : ',') << "setter = "
              << PDecl->getSetterName()->getName();
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_assign) {
          Out << (first ? ' ' : ',') << "assign";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() &
            ObjCPropertyDecl::OBJC_PR_readwrite) {
          Out << (first ? ' ' : ',') << "readwrite";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_retain) {
          Out << (first ? ' ' : ',') << "retain";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_copy) {
          Out << (first ? ' ' : ',') << "copy";
          first = false;
        }
        
        if (PDecl->getPropertyAttributes() & 
            ObjCPropertyDecl::OBJC_PR_nonatomic) {
          Out << (first ? ' ' : ',') << "nonatomic";
          first = false;
        }
        Out << " )";
      }
      
      ObjCIvarDecl **IDecl = PDecl->getPropertyDecls();
      
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

void DeclPrinter::PrintObjCProtocolDecl(ObjCProtocolDecl *PID) {
  Out << "@protocol " << PID->getName() << '\n';
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCategoryImplDecl(ObjCCategoryImplDecl *PID) {
  Out << "@implementation "
      << PID->getClassInterface()->getName()
      << '(' << PID->getName() << ");\n";  
  
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCategoryDecl(ObjCCategoryDecl *PID) {
  Out << "@interface " 
      << PID->getClassInterface()->getName()
      << '(' << PID->getName() << ");\n";
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *AID) {
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
      PrintDecl(D);
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
      } else if (ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(D)) {
        Out << "Read objc interface '" << OID->getName() << "'\n";
      } else if (ObjCProtocolDecl *OPD = dyn_cast<ObjCProtocolDecl>(D)) {
        Out << "Read objc protocol '" << OPD->getName() << "'\n";
      } else if (ObjCCategoryDecl *OCD = dyn_cast<ObjCCategoryDecl>(D)) {
        Out << "Read objc category '" << OCD->getName() << "'\n";
      } else if (isa<ObjCForwardProtocolDecl>(D)) {
        Out << "Read objc fwd protocol decl\n";
      } else if (isa<ObjCClassDecl>(D)) {
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
// GRConstants - Perform intra-procedural, path-sensitive constant propagation.

namespace {
  class GRConstantsVisitor : public CFGVisitor {
  public:
    virtual void Initialize(ASTContext &Context) {}
    
    virtual void VisitCFG(CFG& C) {
      RunGRConstants(C);
    }
  };
} // end anonymous namespace

ASTConsumer *clang::CreateGRConstants() {
  return new GRConstantsVisitor();
}

//===----------------------------------------------------------------------===//
// LLVM Emitter

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Bitcode/ReaderWriter.h"

namespace {
  class CodeGenerator : public ASTConsumer {
    Diagnostic &Diags;
    const llvm::TargetData *TD;
    ASTContext *Ctx;
    const LangOptions &Features;
  protected:
    llvm::Module *M;
    CodeGen::CodeGenModule *Builder;
  public:
    CodeGenerator(Diagnostic &diags, const LangOptions &LO)
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
      } else if (LinkageSpecDecl *LSD = dyn_cast<LinkageSpecDecl>(D)) {
        CodeGen::CodeGenLinkageSpec(Builder, LSD);
      } else {
        assert(isa<TypeDecl>(D) && "Only expected type decls here");
        // don't codegen for now, eventually pass down for debug info.
        //std::cerr << "Read top-level typedef decl: '"
        //    << D->getName() << "'\n";
      }
    }
  };
}

namespace {
  class LLVMEmitter : public CodeGenerator {
  public:
    LLVMEmitter(Diagnostic &diags, const LangOptions &LO)
    : CodeGenerator(diags,LO) {}

    ~LLVMEmitter() {
      CodeGen::Terminate(Builder);
      
      // Print the generated code.
      M->print(llvm::cout.stream());
      delete M;
    }
  };
}

ASTConsumer *clang::CreateLLVMEmitter(Diagnostic &Diags, 
                                      const LangOptions &Features) {
  return new LLVMEmitter(Diags, Features);
}

namespace {
  class BCWriter : public CodeGenerator {
  public:
    std::ostream& Out;

    BCWriter(std::ostream* out, Diagnostic &diags, const LangOptions &LO)
    : CodeGenerator(diags,LO)
    , Out(*out) {}

    ~BCWriter() {
      CodeGen::Terminate(Builder);
      llvm::WriteBitcodeToFile(M, Out);
      delete M;
    }
  };
}

ASTConsumer *clang::CreateBCWriter(const std::string& InFile,
                                   const std::string& OutputFile,
                                   Diagnostic &Diags,
                                   const LangOptions &Features) {
  std::string FileName = OutputFile;
  
  std::ostream *Out;
  if (OutputFile == "-")
    Out = llvm::cout.stream();
  else if (!OutputFile.size()) {
    if (InFile == "-")
      Out = llvm::cout.stream();
    else {
      llvm::sys::Path Path(InFile);
      Path.eraseSuffix();
      Path.appendSuffix("bc");
      FileName = Path.toString();
      Out = new std::ofstream(FileName.c_str(), 
                              std::ios_base::binary|std::ios_base::out);
    }
  } else {
    Out = new std::ofstream(FileName.c_str(), 
                            std::ios_base::binary|std::ios_base::out);
  }

  return new BCWriter(Out, Diags, Features);
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
  
  ~BuildSerializer() {
    SourceManager& SourceMgr = TU.getASTContext()->getSourceManager();
    unsigned ID = SourceMgr.getMainFileID();
    assert (ID && "MainFileID not set!");
    const FileEntry* FE = SourceMgr.getFileEntryForID(ID);
    assert (FE && "No FileEntry for main file.");
    
    // FIXME: This is not portable to Windows.
    // FIXME: This logic should probably be moved elsewhere later.
        
    llvm::sys::Path FName(EmitDir);
    
    std::vector<char> buf;
    buf.reserve(strlen(FE->getName())+100);    
    
    sprintf(&buf[0], "dev_%llx", (uint64_t) FE->getDevice());
    FName.appendComponent(&buf[0]);
    FName.createDirectoryOnDisk(true);
    if (!FName.canWrite() || !FName.isDirectory()) {
      assert (false && "Could not create 'device' serialization directory.");
      return;
    }
            
    sprintf(&buf[0], "%s-%llX.ast", FE->getName(), (uint64_t) FE->getInode());
    FName.appendComponent(&buf[0]);    
    EmitASTBitcodeFile(TU,FName);
    
    // Now emit the sources.
    
  }
};
  
  
} // end anonymous namespace


ASTConsumer* clang::CreateASTSerializer(const std::string& InFile,
                                        const std::string& OutputFile,
                                        Diagnostic &Diags,
                                        const LangOptions &Features) {
  
  if (OutputFile.size()) {
    if (InFile == "-") {
      llvm::cerr <<
        "error: Cannot use --serialize with -o for source read from STDIN.\n";
      return NULL;
    }
    
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
    
    // FIXME: We should probably only allow using BuildSerializer when
    // the ASTs come from parsed source files, and not from .ast files.
    return new BuildSerializer(EmitDir, Diags, Features);
  }

  // The user did not specify an output directory for serialized ASTs.
  // Serialize the translation to a single file whose name is the same
  // as the input file with the ".ast" extension appended.
  
  llvm::sys::Path FName(InFile.c_str());
  FName.appendSuffix("ast");
  return new SingleFileSerializer(FName, Diags, Features);  
}

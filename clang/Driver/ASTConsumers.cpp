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
#include "HTMLDiagnostics.h"
#include "clang/AST/TranslationUnit.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/CFG.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/Timer.h"
#include "llvm/ADT/OwningPtr.h"

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
    void PrintObjCPropertyDecl(ObjCPropertyDecl *PD);
    void PrintObjCPropertyImplDecl(ObjCPropertyImplDecl *PID);
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
  } else if (FileScopeAsmDecl *AD = dyn_cast<FileScopeAsmDecl>(D)) {
    Out << "asm(";
    AD->getAsmString()->printPretty(Out);
    Out << ")\n";
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
  case FunctionDecl::PrivateExtern: Out << "__private_extern__ "; break;
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
  else {
    assert(LS->getLanguage() == LinkageSpecDecl::lang_cxx &&
           "unknown language in linkage specification");
    l = "C++";
  }
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
  
  for (unsigned i = 0, e = OMD->getNumParams(); i != e; ++i) {
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
  
  for (ObjCImplementationDecl::propimpl_iterator I = OID->propimpl_begin(),
       E = OID->propimpl_end(); I != E; ++I)
    PrintObjCPropertyImplDecl(*I);
  
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
  
  if (OID->ivar_size() > 0) {
    Out << '{';
    for (ObjCInterfaceDecl::ivar_iterator I = OID->ivar_begin(),
         E = OID->ivar_end(); I != E; ++I) {
      Out << '\t' << (*I)->getType().getAsString()
          << ' '  << (*I)->getName() << ";\n";      
    }
    Out << "}\n";
  }
  
  for (ObjCInterfaceDecl::classprop_iterator I = OID->classprop_begin(),
       E = OID->classprop_end(); I != E; ++I)
    PrintObjCPropertyDecl(*I);
  bool eol_needed = false;
  for (ObjCInterfaceDecl::classmeth_iterator I = OID->classmeth_begin(),
       E = OID->classmeth_end(); I != E; ++I)
    eol_needed = true, PrintObjCMethodDecl(*I);
  
  for (ObjCInterfaceDecl::instmeth_iterator I = OID->instmeth_begin(),
       E = OID->instmeth_end(); I != E; ++I)
    eol_needed = true, PrintObjCMethodDecl(*I);
  
  Out << (eol_needed ? "\n@end\n" : "@end\n");
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCProtocolDecl(ObjCProtocolDecl *PID) {
  Out << "@protocol " << PID->getName() << '\n';
  
  for (ObjCProtocolDecl::classprop_iterator I = PID->classprop_begin(),
       E = PID->classprop_end(); I != E; ++I)
    PrintObjCPropertyDecl(*I);
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCategoryImplDecl(ObjCCategoryImplDecl *PID) {
  Out << "@implementation "
      << PID->getClassInterface()->getName()
      << '(' << PID->getName() << ");\n";  
  for (ObjCCategoryImplDecl::propimpl_iterator I = PID->propimpl_begin(),
       E = PID->propimpl_end(); I != E; ++I)
    PrintObjCPropertyImplDecl(*I);
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCategoryDecl(ObjCCategoryDecl *PID) {
  Out << "@interface " 
      << PID->getClassInterface()->getName()
      << '(' << PID->getName() << ");\n";
  // Output property declarations.
  for (ObjCCategoryDecl::classprop_iterator I = PID->classprop_begin(),
       E = PID->classprop_end(); I != E; ++I)
    PrintObjCPropertyDecl(*I);
  Out << "@end\n";
  
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *AID) {
  Out << "@compatibility_alias " << AID->getName() 
      << ' ' << AID->getClassInterface()->getName() << ";\n";  
}

/// PrintObjCPropertyDecl - print a property declaration.
///
void DeclPrinter::PrintObjCPropertyDecl(ObjCPropertyDecl *PDecl) {
  if (PDecl->getPropertyImplementation() == ObjCPropertyDecl::Required)
    Out << "@required\n";
  else if (PDecl->getPropertyImplementation() == ObjCPropertyDecl::Optional)
    Out << "@optional\n";
  
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
    << PDecl->getGetterName().getName();
    first = false;
  }
  if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_setter) {
    Out << (first ? ' ' : ',') << "setter = "
    << PDecl->getSetterName().getName();
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
  Out << ' ' << PDecl->getType().getAsString()
  << ' ' << PDecl->getName();
    
  Out << ";\n";
}

/// PrintObjCPropertyImplDecl - Print an objective-c property implementation
/// declaration syntax.
///
void DeclPrinter::PrintObjCPropertyImplDecl(ObjCPropertyImplDecl *PID) {
  if (PID->getPropertyImplementation() == 
      ObjCPropertyImplDecl::OBJC_PR_IMPL_SYNTHSIZE)
    Out << "\n@synthesize ";
  else
    Out << "\n@dynamic ";
  Out << PID->getPropertyDecl()->getName();
  if (PID->getPropertyIvarDecl())
    Out << "=" << PID->getPropertyIvarDecl()->getName();
  Out << ";\n";
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
      } else if (isa<FileScopeAsmDecl>(D)) {
        Out << "Read file scope asm decl\n";
      } else if (ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(D)) {
        Out << "Read objc method decl: '" << MD->getSelector().getName()
            << "'\n";
        if (MD->getBody()) {
          // FIXME: convert dumper to use std::ostream?
          MD->getBody()->dumpAll(*SM);
          Out << '\n';
        }
      } else if (isa<ObjCImplementationDecl>(D)) {
        Out << "Read objc implementation decl\n";
      }
      else {
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
      else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
        DeclPrinter().PrintObjCMethodDecl(MD);
        
        if (MD->getBody()) {
          llvm::cerr << '\n';
          MD->getBody()->viewAST();
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
  std::string FName;
public:
  CFGVisitor(const std::string& fname) : FName(fname) {}
  CFGVisitor() : FName("") {}
  
  // CFG Visitor interface to be implemented by subclass.
  virtual void VisitCFG(CFG& C, Decl& CD) = 0;
  virtual bool printFuncDeclStart() { return true; }
  
  virtual void HandleTopLevelDecl(Decl *D);
};

} // end anonymous namespace

void CFGVisitor::HandleTopLevelDecl(Decl *D) {
  
  CFG *C = NULL;
  
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {

    if (!FD->getBody())
      return;
  
    if (FName.size() > 0 && FName != FD->getIdentifier()->getName())
      return;
      
    if (printFuncDeclStart()) {
      DeclPrinter().PrintFunctionDeclStart(FD);
      llvm::cerr << '\n';
    }
      
    C = CFG::buildCFG(FD->getBody());
  }
  else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    
    if (!MD->getBody())
      return;
      
    if (FName.size() > 0 && FName != MD->getSelector().getName())
      return;
    
    if (printFuncDeclStart()) {
      DeclPrinter().PrintObjCMethodDecl(MD);
      llvm::cerr << '\n';
    }
    
    C = CFG::buildCFG(MD->getBody());
  }
  
  if (C) {  
    VisitCFG(*C, *D);
    delete C;
  }
}

//===----------------------------------------------------------------------===//
// DumpCFGs - Dump CFGs to stderr or visualize with Graphviz

namespace {
  class CFGDumper : public CFGVisitor {
    const bool UseGraphviz;
  public:
    CFGDumper(bool use_graphviz, const std::string& fname) 
     : CFGVisitor(fname), UseGraphviz(use_graphviz) {}
    
    virtual void VisitCFG(CFG& C, Decl&) {
      if (UseGraphviz)
        C.viewCFG();
      else
        C.dump();
    }
  }; 
} // end anonymous namespace 
  
ASTConsumer *clang::CreateCFGDumper(bool ViewGraphs, const std::string& FName) {
  return new CFGDumper(ViewGraphs, FName);
}

//===----------------------------------------------------------------------===//
// AnalyzeLiveVariables - perform live variable analysis and dump results

namespace {
  class LivenessVisitor : public CFGVisitor {
    SourceManager *SM;
  public:
    LivenessVisitor(const std::string& fname) : CFGVisitor(fname) {}
    
    virtual void Initialize(ASTContext &Context) {
      SM = &Context.getSourceManager();
    }

    virtual void VisitCFG(CFG& C, Decl& CD) {
      LiveVariables L(C);
      L.runOnCFG(C);
      L.dumpBlockLiveness(*SM);
    }
  };
} // end anonymous namespace
  
ASTConsumer *clang::CreateLiveVarAnalyzer(const std::string& fname) {
  return new LivenessVisitor(fname);
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
    
    virtual void VisitCFG(CFG& C, Decl& CD) {
      CheckDeadStores(C, *Ctx, Diags);
    }
    
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
    
    virtual void VisitCFG(CFG& C, Decl&) { 
      CheckUninitializedValues(C, *Ctx, Diags);
    }
    
    virtual bool printFuncDeclStart() { return false; }
  }; 
} // end anonymous namespace

ASTConsumer *clang::CreateUnitValsChecker(Diagnostic &Diags) {
  return new UninitValsVisitor(Diags);
}

//===----------------------------------------------------------------------===//
// CheckerConsumer - Generic Driver for running intra-procedural path-sensitive
//  analyses.

namespace {
  
class CheckerConsumer : public CFGVisitor {
protected:
  Diagnostic &Diags;
  ASTContext* Ctx;
  Preprocessor* PP;
  PreprocessorFactory* PPF;
  const std::string& HTMLDir;
  bool Visualize;
  bool TrimGraph;
  llvm::OwningPtr<PathDiagnosticClient> PD;
  bool AnalyzeAll;
public:
  CheckerConsumer(Diagnostic &diags, Preprocessor* pp, PreprocessorFactory* ppf,
                  const std::string& fname,
                  const std::string& htmldir,
                  bool visualize, bool trim, bool analyzeAll)
    : CFGVisitor(fname), Diags(diags), PP(pp), PPF(ppf), HTMLDir(htmldir),
      Visualize(visualize), TrimGraph(trim), AnalyzeAll(analyzeAll) {}
  
  virtual void Initialize(ASTContext &Context) { Ctx = &Context; }    
  virtual void VisitCFG(CFG& C, Decl&);
  virtual bool printFuncDeclStart() { return false; }

  virtual const char* getCheckerName() = 0;
  virtual void getTransferFunctions(std::vector<GRTransferFuncs*>& TFs) = 0;
};
} // end anonymous namespace

void CheckerConsumer::VisitCFG(CFG& C, Decl& CD) {
  
  if (Diags.hasErrorOccurred())
    return;
  
  SourceLocation Loc = CD.getLocation();
  
  if (!Loc.isFileID())
    return;
  
  if (!AnalyzeAll && !Ctx->getSourceManager().isFromMainFile(Loc))
    return;
  
  // Lazily create the diagnostic client.
  
  if (!HTMLDir.empty() && PD.get() == NULL)
    PD.reset(CreateHTMLDiagnosticClient(HTMLDir, PP, PPF));
  

  if (!Visualize) {
    
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(&CD)) {
      llvm::cerr << "ANALYZE: "
      << Ctx->getSourceManager().getSourceName(FD->getLocation())
      << ' '
      << FD->getIdentifier()->getName()
      << '\n';
    }
    else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(&CD)) {
      llvm::cerr << "ANALYZE (ObjC Method): "
      << Ctx->getSourceManager().getSourceName(MD->getLocation())
      << " '"
      << MD->getSelector().getName() << "'\n";
    }
  }
  else
    llvm::cerr << '\n';    
  
  std::vector<GRTransferFuncs*> TFs;
  getTransferFunctions(TFs);
  
  while (!TFs.empty()) {

    // Construct the analysis engine.
    GRExprEngine Eng(C, CD, *Ctx);
    
    // Set base transfer functions.
    llvm::OwningPtr<GRTransferFuncs> TF(TFs.back());
    TFs.pop_back();
      
    Eng.setTransferFunctions(TF.get());
    
    // Execute the worklist algorithm.
    Eng.ExecuteWorkList();
    
    // Display warnings.
    Eng.EmitWarnings(Diags, PD.get());
    
  #ifndef NDEBUG
    if (Visualize) Eng.ViewGraph(TrimGraph);
  #endif
  }
}

//===----------------------------------------------------------------------===//
// GRSimpleVals - Perform intra-procedural, path-sensitive constant propagation.

namespace {
class GRSimpleValsVisitor : public CheckerConsumer {
public:
  GRSimpleValsVisitor(Diagnostic &diags, Preprocessor* pp,
                      PreprocessorFactory* ppf,
                      const std::string& fname, const std::string& htmldir,
                      bool visualize, bool trim, bool analyzeAll)
  : CheckerConsumer(diags, pp, ppf, fname, htmldir, visualize,
                    trim, analyzeAll) {}

  virtual const char* getCheckerName() { return "GRSimpleVals"; }
  
  virtual void getTransferFunctions(std::vector<GRTransferFuncs*>& TFs) {
    return TFs.push_back(MakeGRSimpleValsTF());
  }
};
} // end anonymous namespace

ASTConsumer* clang::CreateGRSimpleVals(Diagnostic &Diags,
                                       Preprocessor* PP,
                                       PreprocessorFactory* PPF,
                                       const std::string& FunctionName,
                                       const std::string& HTMLDir,
                                       bool Visualize, bool TrimGraph,
                                       bool AnalyzeAll) {
  
  return new GRSimpleValsVisitor(Diags, PP, PPF, FunctionName, HTMLDir,
                                 Visualize, TrimGraph, AnalyzeAll);
}


//===----------------------------------------------------------------------===//
// Core Foundation Reference Counting Checker

namespace {
class CFRefCountCheckerVisitor : public CheckerConsumer {
  const LangOptions& LangOpts;
public:
  CFRefCountCheckerVisitor(Diagnostic &diags, Preprocessor* pp,
                           PreprocessorFactory* ppf,
                           const LangOptions& lopts,
                           const std::string& fname,
                           const std::string& htmldir,
                           bool visualize, bool trim, bool analyzeAll)
  : CheckerConsumer(diags, pp, ppf, fname, htmldir, visualize,
                    trim, analyzeAll), LangOpts(lopts) {}
  
  virtual const char* getCheckerName() { return "CFRefCountChecker"; }
  
  virtual void getTransferFunctions(std::vector<GRTransferFuncs*>& TFs) {
    switch (LangOpts.getGCMode()) {
      case LangOptions::NonGC:
        TFs.push_back(MakeCFRefCountTF(*Ctx, false, true, LangOpts));
        break;
        
      case LangOptions::GCOnly:
        TFs.push_back(MakeCFRefCountTF(*Ctx, true, true, LangOpts));
        break;
        
      case LangOptions::HybridGC:
        TFs.push_back(MakeCFRefCountTF(*Ctx, false, true, LangOpts));
        TFs.push_back(MakeCFRefCountTF(*Ctx, true, false, LangOpts));
        break;
    }
  }
};
} // end anonymous namespace

ASTConsumer* clang::CreateCFRefChecker(Diagnostic &Diags,
                                       Preprocessor* PP,
                                       PreprocessorFactory* PPF,
                                       const LangOptions& LangOpts,
                                       const std::string& FunctionName,
                                       const std::string& HTMLDir,
                                       bool Visualize, bool TrimGraph,
                                       bool AnalyzeAll) {
  
  return new CFRefCountCheckerVisitor(Diags, PP, PPF, LangOpts, FunctionName,
                                      HTMLDir, Visualize, TrimGraph,
                                      AnalyzeAll);
}

//===----------------------------------------------------------------------===//
// AST Serializer

namespace {

class ASTSerializer : public ASTConsumer {
protected:
  Diagnostic &Diags;
  const LangOptions& lang;
  TranslationUnit* TU;

public:
  ASTSerializer(Diagnostic& diags, const LangOptions& LO)
    : Diags(diags), lang(LO), TU(0) {}
    
  virtual ~ASTSerializer() { delete TU; }
  
  virtual void Initialize(ASTContext &Context) {
    if (!TU) TU = new TranslationUnit(Context, lang);
  }
  
  virtual void HandleTopLevelDecl(Decl *D) {
    if (Diags.hasErrorOccurred())
      return;
    
    if (TU) TU->AddTopLevelDecl(D);
  }
};
    
class SingleFileSerializer : public ASTSerializer {
  const llvm::sys::Path FName;
public:
  SingleFileSerializer(const llvm::sys::Path& F, Diagnostic &diags,
                          const LangOptions &LO)
  : ASTSerializer(diags,LO), FName(F) {}    
  
  ~SingleFileSerializer() {
    EmitASTBitcodeFile(TU, FName);
  }
};

class BuildSerializer : public ASTSerializer {
  llvm::sys::Path EmitDir;  
public:
  BuildSerializer(const llvm::sys::Path& dir, Diagnostic &diags,
                  const LangOptions &LO)
  : ASTSerializer(diags,LO), EmitDir(dir) {}
  
  ~BuildSerializer() {

    if (!TU)
      return;
    
    SourceManager& SourceMgr = TU->getContext().getSourceManager();
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
    EmitASTBitcodeFile(TU, FName);
    
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

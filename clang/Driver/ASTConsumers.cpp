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
#include "clang/Driver/HTMLDiagnostics.h"
#include "clang/AST/TranslationUnit.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "llvm/Module.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

//===----------------------------------------------------------------------===//
/// DeclPrinter - Utility class for printing top-level decls.

namespace {
  class DeclPrinter {
  public:
    llvm::raw_ostream& Out;

    DeclPrinter(llvm::raw_ostream* out) : Out(out ? *out : llvm::errs()) {}
    DeclPrinter() : Out(llvm::errs()) {}
    virtual ~DeclPrinter();
    
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

DeclPrinter::~DeclPrinter() {
  Out.flush();
}

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
  } else if (ObjCClassDecl *OFCD = dyn_cast<ObjCClassDecl>(D)) {
    Out << "@class ";
    ObjCInterfaceDecl **ForwardDecls = OFCD->getForwardDecls();
    for (unsigned i = 0, e = OFCD->getNumForwardDecls(); i != e; ++i) {
      const ObjCInterfaceDecl *D = ForwardDecls[i];
      if (i) Out << ", ";
      Out << D->getName();
    }
    Out << ";\n";
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
    Out << '(' << OMD->getResultType().getAsString() << ")";
  
  std::string name = OMD->getSelector().getName();
  std::string::size_type pos, lastPos = 0;
  for (unsigned i = 0, e = OMD->getNumParams(); i != e; ++i) {
    ParmVarDecl *PDecl = OMD->getParamDecl(i);
    // FIXME: selector is missing here!    
    pos = name.find_first_of(":", lastPos);
    Out << " " << name.substr(lastPos, pos - lastPos);
    Out << ":(" << PDecl->getType().getAsString() << ")" << PDecl->getName(); 
    lastPos = pos + 1;
  }
    
  if (OMD->getNumParams() == 0)
    Out << " " << name;
    
  if (OMD->isVariadic())
      Out << ", ...";
  
  Out << ";";
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
  const ObjCList<ObjCProtocolDecl> &Protocols = OID->getReferencedProtocols();
  if (!Protocols.empty()) {
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
         E = Protocols.end(); I != E; ++I)
      Out << (I == Protocols.begin() ? '<' : ',') << (*I)->getName();
  }
  
  if (!Protocols.empty())
    Out << ">";
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
  if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize)
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
    ASTPrinter(llvm::raw_ostream* o = NULL) : DeclPrinter(o) {}
    
    virtual void HandleTopLevelDecl(Decl *D) {
      PrintDecl(D);
    }
  };
}

ASTConsumer *clang::CreateASTPrinter(llvm::raw_ostream* out) {
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
      } else if (isa<ObjCCategoryImplDecl>(D)) {
        Out << "Read objc category implementation decl\n";
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
/// InheritanceViewer - C++ Inheritance Visualization

namespace {
class InheritanceViewer : public ASTConsumer {
  const std::string clsname;
public:
  InheritanceViewer(const std::string& cname) : clsname(cname) {}
  
  void HandleTranslationUnit(TranslationUnit& TU) {
    ASTContext& C = TU.getContext();
    for (ASTContext::type_iterator I=C.types_begin(),E=C.types_end(); I!=E; ++I)
      if (CXXRecordType *T = dyn_cast<CXXRecordType>(*I)) {
        CXXRecordDecl* D = T->getDecl();
        // FIXME: This lookup needs to be generalized to handle namespaces and
        // (when we support them) templates.
        if (D->getName() == clsname) {
          D->viewInheritance(C);      
        }
      }
  }
}; 
}

ASTConsumer *clang::CreateInheritanceViewer(const std::string& clsname) {
  return new InheritanceViewer(clsname);
}

//===----------------------------------------------------------------------===//
// AST Serializer

namespace {

class ASTSerializer : public ASTConsumer {
protected:
  Diagnostic& Diags;

public:
  ASTSerializer(Diagnostic& diags) : Diags(diags) {}
};

class SingleFileSerializer : public ASTSerializer {
  const llvm::sys::Path FName;
public:
  SingleFileSerializer(const llvm::sys::Path& F, Diagnostic& diags)
    : ASTSerializer(diags), FName(F) {}    
  
  virtual void HandleTranslationUnit(TranslationUnit& TU) {
    if (Diags.hasErrorOccurred())
      return;
    EmitASTBitcodeFile(&TU, FName);
  }
};

class BuildSerializer : public ASTSerializer {
  llvm::sys::Path EmitDir;  
public:
  BuildSerializer(const llvm::sys::Path& dir, Diagnostic& diags)
    : ASTSerializer(diags), EmitDir(dir) {}
  
  virtual void HandleTranslationUnit(TranslationUnit& TU) {
    if (Diags.hasErrorOccurred())
      return;
    
    SourceManager& SourceMgr = TU.getContext().getSourceManager();
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
    EmitASTBitcodeFile(&TU, FName);
    
    // Now emit the sources.
    
  }
};
  
  
} // end anonymous namespace


ASTConsumer* clang::CreateASTSerializer(const std::string& InFile,
                                        const std::string& OutputFile,
                                        Diagnostic &Diags) {
  
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
    return new BuildSerializer(EmitDir, Diags);
  }

  // The user did not specify an output directory for serialized ASTs.
  // Serialize the translation to a single file whose name is the same
  // as the input file with the ".ast" extension appended.
  
  llvm::sys::Path FName(InFile.c_str());
  FName.appendSuffix("ast");
  return new SingleFileSerializer(FName, Diags);
}

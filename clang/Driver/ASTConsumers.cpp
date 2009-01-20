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
#include "clang/Driver/PathDiagnosticClients.h"
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
      Out << D->getNameAsString();
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
      Out << D->getNameAsString();
    }
    Out << ";\n";
  } else if (EnumDecl *ED = dyn_cast<EnumDecl>(D)) {
    Out << "enum " << ED->getNameAsString() << " {\n";
    for (EnumDecl::enumerator_iterator E = ED->enumerator_begin(),
                                    EEnd = ED->enumerator_end();
         E != EEnd; ++E)
      Out << "  " << (*E)->getNameAsString() << ",\n";
    Out << "};\n";
  } else if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    Out << "Read top-level tag decl: '" << TD->getNameAsString() << "'\n";
  } else if (LinkageSpecDecl *LSD = dyn_cast<LinkageSpecDecl>(D)) {
    PrintLinkageSpec(LSD);
  } else if (FileScopeAsmDecl *AD = dyn_cast<FileScopeAsmDecl>(D)) {
    Out << "asm(";
    AD->getAsmString()->printPretty(Out);
    Out << ")\n";
  } else if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    Out << "Read top-level variable decl: '" << ND->getNameAsString() << "'\n";
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
  
  std::string Proto = FD->getNameAsString();
  const FunctionType *AFT = FD->getType()->getAsFunctionType();

  if (const FunctionTypeProto *FT = dyn_cast<FunctionTypeProto>(AFT)) {
    Proto += "(";
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
      if (i) Proto += ", ";
      std::string ParamStr;
      if (HasBody) ParamStr = FD->getParamDecl(i)->getNameAsString();
      
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
  std::string S = TD->getNameAsString();
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

  Out << "extern \"" << l << "\" ";
  if (LS->hasBraces()) 
    Out << "{\n";

  for (LinkageSpecDecl::decl_iterator D = LS->decls_begin(), 
                                   DEnd = LS->decls_end();
       D != DEnd; ++D)
    PrintDecl(*D);

  if (LS->hasBraces())
    Out << "}";
  Out << "\n";
}

void DeclPrinter::PrintObjCMethodDecl(ObjCMethodDecl *OMD) {
  if (OMD->isInstanceMethod())
    Out << "\n- ";
  else 
    Out << "\n+ ";
  if (!OMD->getResultType().isNull())
    Out << '(' << OMD->getResultType().getAsString() << ")";
  
  std::string name = OMD->getSelector().getAsString();
  std::string::size_type pos, lastPos = 0;
  for (unsigned i = 0, e = OMD->getNumParams(); i != e; ++i) {
    ParmVarDecl *PDecl = OMD->getParamDecl(i);
    // FIXME: selector is missing here!    
    pos = name.find_first_of(":", lastPos);
    Out << " " << name.substr(lastPos, pos - lastPos);
    Out << ":(" << PDecl->getType().getAsString() << ")"
        << PDecl->getNameAsString(); 
    lastPos = pos + 1;
  }
    
  if (OMD->getNumParams() == 0)
    Out << " " << name;
    
  if (OMD->isVariadic())
      Out << ", ...";
  
  Out << ";";
}

void DeclPrinter::PrintObjCImplementationDecl(ObjCImplementationDecl *OID) {
  std::string I = OID->getNameAsString();
  ObjCInterfaceDecl *SID = OID->getSuperClass();

  if (SID)
    Out << "@implementation " << I << " : " << SID->getNameAsString();
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
  std::string I = OID->getNameAsString();
  ObjCInterfaceDecl *SID = OID->getSuperClass();

  if (SID)
    Out << "@interface " << I << " : " << SID->getNameAsString();
  else
    Out << "@interface " << I;
  
  // Protocols?
  const ObjCList<ObjCProtocolDecl> &Protocols = OID->getReferencedProtocols();
  if (!Protocols.empty()) {
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
         E = Protocols.end(); I != E; ++I)
      Out << (I == Protocols.begin() ? '<' : ',') << (*I)->getNameAsString();
  }
  
  if (!Protocols.empty())
    Out << ">";
  Out << '\n';
  
  if (OID->ivar_size() > 0) {
    Out << '{';
    for (ObjCInterfaceDecl::ivar_iterator I = OID->ivar_begin(),
         E = OID->ivar_end(); I != E; ++I) {
      Out << '\t' << (*I)->getType().getAsString()
          << ' '  << (*I)->getNameAsString() << ";\n";      
    }
    Out << "}\n";
  }
  
  for (ObjCInterfaceDecl::prop_iterator I = OID->prop_begin(),
       E = OID->prop_end(); I != E; ++I)
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
  Out << "@protocol " << PID->getNameAsString() << '\n';
  
  for (ObjCProtocolDecl::prop_iterator I = PID->prop_begin(),
       E = PID->prop_end(); I != E; ++I)
    PrintObjCPropertyDecl(*I);
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCategoryImplDecl(ObjCCategoryImplDecl *PID) {
  Out << "@implementation "
      << PID->getClassInterface()->getNameAsString()
      << '(' << PID->getNameAsString() << ");\n";  
  for (ObjCCategoryImplDecl::propimpl_iterator I = PID->propimpl_begin(),
       E = PID->propimpl_end(); I != E; ++I)
    PrintObjCPropertyImplDecl(*I);
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCategoryDecl(ObjCCategoryDecl *PID) {
  Out << "@interface " 
      << PID->getClassInterface()->getNameAsString()
      << '(' << PID->getNameAsString() << ");\n";
  // Output property declarations.
  for (ObjCCategoryDecl::prop_iterator I = PID->prop_begin(),
       E = PID->prop_end(); I != E; ++I)
    PrintObjCPropertyDecl(*I);
  Out << "@end\n";
  
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *AID) {
  Out << "@compatibility_alias " << AID->getNameAsString() 
      << ' ' << AID->getClassInterface()->getNameAsString() << ";\n";  
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
        << PDecl->getGetterName().getAsString();
    first = false;
  }
  if (PDecl->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_setter) {
    Out << (first ? ' ' : ',') << "setter = "
        << PDecl->getSetterName().getAsString();
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
  << ' ' << PDecl->getNameAsString();
    
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
  Out << PID->getPropertyDecl()->getNameAsString();
  if (PID->getPropertyIvarDecl())
    Out << "=" << PID->getPropertyIvarDecl()->getNameAsString();
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
      } else if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
        Out << "Read top-level variable decl: '" << ND->getNameAsString()
            << "'\n";
      } else if (ObjCInterfaceDecl *OID = dyn_cast<ObjCInterfaceDecl>(D)) {
        Out << "Read objc interface '" << OID->getNameAsString() << "'\n";
      } else if (ObjCProtocolDecl *OPD = dyn_cast<ObjCProtocolDecl>(D)) {
        Out << "Read objc protocol '" << OPD->getNameAsString() << "'\n";
      } else if (ObjCCategoryDecl *OCD = dyn_cast<ObjCCategoryDecl>(D)) {
        Out << "Read objc category '" << OCD->getNameAsString() << "'\n";
      } else if (isa<ObjCForwardProtocolDecl>(D)) {
        Out << "Read objc fwd protocol decl\n";
      } else if (isa<ObjCClassDecl>(D)) {
        Out << "Read objc fwd class decl\n";
      } else if (isa<FileScopeAsmDecl>(D)) {
        Out << "Read file scope asm decl\n";
      } else if (ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(D)) {
        Out << "Read objc method decl: '" << MD->getSelector().getAsString()
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
      } else if (isa<LinkageSpecDecl>(D)) {
        Out << "Read linkage spec decl\n";
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
/// DeclContextPrinter - Decl and DeclContext Visualization

namespace {

class DeclContextPrinter : public ASTConsumer {
  llvm::raw_ostream& Out;
public:
  DeclContextPrinter() : Out(llvm::errs()) {}

  void HandleTranslationUnit(TranslationUnit& TU) {
    TranslationUnitDecl* TUD = TU.getContext().getTranslationUnitDecl();
    PrintDeclContext(TUD, 4);
  }

  void PrintDeclContext(const DeclContext* DC, unsigned Indentation);
};

void DeclContextPrinter::PrintDeclContext(const DeclContext* DC, 
                                          unsigned Indentation) {
  // Print DeclContext name.
  switch (DC->getDeclKind()) {
  case Decl::TranslationUnit:
    Out << "[translation unit] " << DC;
    break;
  case Decl::Namespace: {
    Out << "[namespace] ";
    NamespaceDecl* ND = NamespaceDecl::castFromDeclContext(DC);
    Out << ND->getNameAsString();
    break;
  }
  case Decl::Enum: {
    EnumDecl* ED = EnumDecl::castFromDeclContext(DC);
    if (ED->isDefinition())
      Out << "[enum] ";
    else
      Out << "<enum> ";
    Out << ED->getNameAsString();
    break;
  }
  case Decl::Record: {
    RecordDecl* RD = RecordDecl::castFromDeclContext(DC);
    if (RD->isDefinition())
      Out << "[struct] ";
    else
      Out << "<struct> ";
    Out << RD->getNameAsString();
    break;
  }
  case Decl::CXXRecord: {
    CXXRecordDecl* RD = CXXRecordDecl::castFromDeclContext(DC);
    if (RD->isDefinition())
      Out << "[class] ";
    else
      Out << "<class> ";
    Out << RD->getNameAsString() << " " << DC;
    break;
  }
  case Decl::ObjCMethod:
    Out << "[objc method]";
    break;
  case Decl::ObjCInterface:
    Out << "[objc interface]";
    break;
  case Decl::ObjCCategory:
    Out << "[objc category]";
    break;
  case Decl::ObjCProtocol:
    Out << "[objc protocol]";
    break;
  case Decl::ObjCImplementation:
    Out << "[objc implementation]";
    break;
  case Decl::ObjCCategoryImpl:
    Out << "[objc categoryimpl]";
    break;
  case Decl::LinkageSpec:
    Out << "[linkage spec]";
    break;
  case Decl::Block:
    Out << "[block]";
    break;
  case Decl::Function: {
    FunctionDecl* FD = FunctionDecl::castFromDeclContext(DC);
    if (FD->isThisDeclarationADefinition())
      Out << "[function] ";
    else
      Out << "<function> ";
    Out << FD->getNameAsString();
    // Print the parameters.
    Out << "(";
    bool PrintComma = false;
    for (FunctionDecl::param_const_iterator I = FD->param_begin(), 
           E = FD->param_end(); I != E; ++I) {
      if (PrintComma)
        Out << ", ";
      else
        PrintComma = true;
      Out << (*I)->getNameAsString();
    }
    Out << ")";
    break;
  }
  case Decl::CXXMethod: {
    CXXMethodDecl* D = CXXMethodDecl::castFromDeclContext(DC);
    if (D->isOutOfLineDefinition())
      Out << "[c++ method] ";
    else if (D->isImplicit())
      Out << "(c++ method) ";
    else
      Out << "<c++ method> ";
    Out << D->getNameAsString();
    // Print the parameters.
    Out << "(";
    bool PrintComma = false;
    for (FunctionDecl::param_const_iterator I = D->param_begin(), 
           E = D->param_end(); I != E; ++I) {
      if (PrintComma)
        Out << ", ";
      else
        PrintComma = true;
      Out << (*I)->getNameAsString();
    }
    Out << ")";

    // Check the semantic DeclContext.
    DeclContext* SemaDC = D->getDeclContext();
    DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";

    break;
  }
  case Decl::CXXConstructor: {
    CXXConstructorDecl* D = CXXConstructorDecl::castFromDeclContext(DC);
    if (D->isOutOfLineDefinition())
      Out << "[c++ ctor] ";
    else if (D->isImplicit())
      Out << "(c++ ctor) ";
    else
      Out << "<c++ ctor> ";
    Out << D->getNameAsString();
    // Print the parameters.
    Out << "(";
    bool PrintComma = false;
    for (FunctionDecl::param_const_iterator I = D->param_begin(), 
           E = D->param_end(); I != E; ++I) {
      if (PrintComma)
        Out << ", ";
      else
        PrintComma = true;
      Out << (*I)->getNameAsString();
    }
    Out << ")";

    // Check the semantic DC.
    DeclContext* SemaDC = D->getDeclContext();
    DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";
    break;
  }
  case Decl::CXXDestructor: {
    CXXDestructorDecl* D = CXXDestructorDecl::castFromDeclContext(DC);
    if (D->isOutOfLineDefinition())
      Out << "[c++ dtor] ";
    else if (D->isImplicit())
      Out << "(c++ dtor) ";
    else
      Out << "<c++ dtor> ";
    Out << D->getNameAsString();
    // Check the semantic DC.
    DeclContext* SemaDC = D->getDeclContext();
    DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";
    break;
  }
  case Decl::CXXConversion: {
    CXXConversionDecl* D = CXXConversionDecl::castFromDeclContext(DC);
    if (D->isOutOfLineDefinition())
      Out << "[c++ conversion] ";
    else if (D->isImplicit())
      Out << "(c++ conversion) ";
    else
      Out << "<c++ conversion> ";
    Out << D->getNameAsString();
    // Check the semantic DC.
    DeclContext* SemaDC = D->getDeclContext();
    DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";
    break;
  }

  default:
    assert(0 && "a decl that inherits DeclContext isn't handled");
  }

  Out << "\n";

  // Print decls in the DeclContext.
  for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
       I != E; ++I) {
    for (unsigned i = 0; i < Indentation; ++i)
      Out << " ";

    Decl::Kind DK = I->getKind();
    switch (DK) {
    case Decl::Namespace:
    case Decl::Enum:
    case Decl::Record:
    case Decl::CXXRecord:
    case Decl::ObjCMethod:
    case Decl::ObjCInterface:
    case Decl::ObjCCategory: 
    case Decl::ObjCProtocol:
    case Decl::ObjCImplementation:
    case Decl::ObjCCategoryImpl:
    case Decl::LinkageSpec:
    case Decl::Block:
    case Decl::Function:
    case Decl::CXXMethod:
    case Decl::CXXConstructor:
    case Decl::CXXDestructor:
    case Decl::CXXConversion:
    {
      DeclContext* DC = Decl::castToDeclContext(*I);
      PrintDeclContext(DC, Indentation+4);
      break;
    }
    case Decl::Field: {
      FieldDecl* FD = cast<FieldDecl>(*I);
      Out << "<field> " << FD->getNameAsString() << "\n";
      break;
    }
    case Decl::Typedef: {
      TypedefDecl* TD = cast<TypedefDecl>(*I);
      Out << "<typedef> " << TD->getNameAsString() << "\n";
      break;
    }
    case Decl::EnumConstant: {
      EnumConstantDecl* ECD = cast<EnumConstantDecl>(*I);
      Out << "<enum constant> " << ECD->getNameAsString() << "\n";
      break;
    }
    case Decl::Var: {
      VarDecl* VD = cast<VarDecl>(*I);
      Out << "<var> " << VD->getNameAsString() << "\n";
      break;
    }
    case Decl::ImplicitParam: {
      ImplicitParamDecl* IPD = cast<ImplicitParamDecl>(*I);
      Out << "<implicit parameter> " << IPD->getNameAsString() << "\n";
      break;
    }
    case Decl::CXXClassVar: {
      CXXClassVarDecl* CVD = cast<CXXClassVarDecl>(*I);
      Out << "<static member var> " << CVD->getNameAsString() << "\n";
      break;
    }
    case Decl::ParmVar: {
      ParmVarDecl* PVD = cast<ParmVarDecl>(*I);
      Out << "<parameter> " << PVD->getNameAsString() << "\n";
      break;
    }
    case Decl::ObjCProperty: {
      ObjCPropertyDecl* OPD = cast<ObjCPropertyDecl>(*I);
      Out << "<objc property> " << OPD->getNameAsString() << "\n";
      break;
    }
    default:
      fprintf(stderr, "DeclKind: %d\n", DK);
      assert(0 && "decl unhandled");
    }
  }
}

}

ASTConsumer *clang::CreateDeclContextPrinter() { 
  return new DeclContextPrinter(); 
}

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
        if (D->getNameAsString() == clsname) {
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
    FileID ID = SourceMgr.getMainFileID();
    assert(!ID.isInvalid() && "MainFileID not set!");
    const FileEntry* FE = SourceMgr.getFileEntryForID(ID);
    assert(FE && "No FileEntry for main file.");
    
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

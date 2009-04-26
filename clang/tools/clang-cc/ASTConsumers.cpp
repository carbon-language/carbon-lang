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
#include "clang/Frontend/PathDiagnosticClients.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "llvm/Module.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
using namespace clang;

//===----------------------------------------------------------------------===//
/// DeclPrinter - Utility class for printing top-level decls.

namespace {
  class DeclPrinter {
  public:
    llvm::raw_ostream& Out;
    unsigned Indentation;

    DeclPrinter(llvm::raw_ostream* out) : Out(out ? *out : llvm::errs()),
                                          Indentation(0) {}
    DeclPrinter() : Out(llvm::errs()), Indentation(0) {}
    virtual ~DeclPrinter();
    
    void ChangeIndent(int I) {
      Indentation += I;
    }
    
    llvm::raw_ostream& Indent() {
      for (unsigned i = 0; i < Indentation; ++i)
        Out << "  ";
      return Out;
    }

    void PrintDecl(Decl *D);
    void Print(NamedDecl *ND);
    void Print(NamespaceDecl *NS);
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

    void PrintTemplateDecl(TemplateDecl *TD);
  };
} // end anonymous namespace

DeclPrinter::~DeclPrinter() {
  Out.flush();
}

void DeclPrinter:: PrintDecl(Decl *D) {
  Indent();
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    PrintFunctionDeclStart(FD);

    // FIXME: Pass a context here so we can use getBody()
    if (FD->getBodyIfAvailable()) {
      Out << ' ';
      FD->getBodyIfAvailable()->printPretty(Out, 0, Indentation, true);
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
    for (ObjCForwardProtocolDecl::protocol_iterator I = OFPD->protocol_begin(), 
                                                    E = OFPD->protocol_end();
         I != E; ++I) {
      if (I != OFPD->protocol_begin()) Out << ", ";
      Out << (*I)->getNameAsString();
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
    for (ObjCClassDecl::iterator I = OFCD->begin(), E = OFCD->end();
         I != E; ++I) {
      if (I != OFCD->begin()) Out << ", ";
      Out << (*I)->getNameAsString();
    }
    Out << ";\n";
  } else if (EnumDecl *ED = dyn_cast<EnumDecl>(D)) {
    Out << "enum " << ED->getNameAsString() << " {\n";
    // FIXME: Shouldn't pass a NULL context
    ASTContext *Context = 0;
    for (EnumDecl::enumerator_iterator E = ED->enumerator_begin(*Context),
                                    EEnd = ED->enumerator_end(*Context);
         E != EEnd; ++E)
      Out << "  " << (*E)->getNameAsString() << ",\n";
    Out << "};\n";
  } else if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    // print a free standing tag decl (e.g. "struct x;"). 
    Out << TD->getKindName();
    Out << " ";
    if (const IdentifierInfo *II = TD->getIdentifier())
      Out << II->getName();

    Out << " {\n";
    ChangeIndent(1);
    // FIXME: Shouldn't pass a NULL context
    ASTContext *Context = 0;
    for (DeclContext::decl_iterator i = TD->decls_begin(*Context);
         i != TD->decls_end(*Context);
         ++i)
      PrintDecl(*i);    
    ChangeIndent(-1);
    Indent();
    Out << "}";

    Out << "\n";
  } else if (TemplateDecl *TempD = dyn_cast<TemplateDecl>(D)) {
    PrintTemplateDecl(TempD);
  } else if (LinkageSpecDecl *LSD = dyn_cast<LinkageSpecDecl>(D)) {
    PrintLinkageSpec(LSD);
  } else if (FileScopeAsmDecl *AD = dyn_cast<FileScopeAsmDecl>(D)) {
    Out << "asm(";
    AD->getAsmString()->printPretty(Out);
    Out << ")\n";
  } else if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    Print(ND);
  } else {
    assert(0 && "Unknown decl type!");
  }
}

void DeclPrinter::Print(NamedDecl *ND) {
  switch (ND->getKind()) {
  default:
    // FIXME: Handle the rest of the NamedDecls.
    Out << "### NamedDecl " << ND->getNameAsString() << "\n";
    break;
  case Decl::Field:
  case Decl::Var: {
    // Emit storage class for vardecls.
    if (VarDecl *V = dyn_cast<VarDecl>(ND)) {
      switch (V->getStorageClass()) {
      default: assert(0 && "Unknown storage class!");
      case VarDecl::None:          break;
      case VarDecl::Auto:          Out << "auto "; break;
      case VarDecl::Register:      Out << "register "; break;
      case VarDecl::Extern:        Out << "extern "; break;
      case VarDecl::Static:        Out << "static "; break; 
      case VarDecl::PrivateExtern: Out << "__private_extern__ "; break; 
      }
    }
    std::string Name = ND->getNameAsString();
    // This forms: "int a".
    dyn_cast<ValueDecl>(ND)->getType().getAsStringInternal(Name);
    Out << Name;
    if (VarDecl *Var = dyn_cast<VarDecl>(ND)) {
      if (Var->getInit()) {
        Out << " = ";
        Var->getInit()->printPretty(Out);
      }
    }
    Out << ";\n";
    break;
  }
  case Decl::Namespace:
    Print(dyn_cast<NamespaceDecl>(ND));
    break;
  }
}

void DeclPrinter::Print(NamespaceDecl *NS) {
  Out << "namespace " << NS->getNameAsString() << " {\n";
  ChangeIndent(1);
  // FIXME: Shouldn't pass a NULL context
  ASTContext *Context = 0;
  for (DeclContext::decl_iterator i = NS->decls_begin(*Context);
       i != NS->decls_end(*Context);
       ++i)
    PrintDecl(*i);    
  ChangeIndent(-1);
  Indent();
  Out << "}\n";
}

void DeclPrinter::PrintFunctionDeclStart(FunctionDecl *FD) {
  // FIXME: pass a context so that we can use getBody.
  bool HasBody = FD->getBodyIfAvailable();
  
  Out << '\n';

  Indent();
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

  if (const FunctionProtoType *FT = dyn_cast<FunctionProtoType>(AFT)) {
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
    assert(isa<FunctionNoProtoType>(AFT));
    Proto += "()";
  }

  AFT->getResultType().getAsStringInternal(Proto);
  Out << Proto;
  
  if (!FD->getBodyIfAvailable())
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
  if (LS->hasBraces()) {
    Out << "{\n";
    ChangeIndent(1);
  }

  // FIXME: Should not use a NULL DeclContext!
  ASTContext *Context = 0;
  for (LinkageSpecDecl::decl_iterator D = LS->decls_begin(*Context), 
                                   DEnd = LS->decls_end(*Context);
       D != DEnd; ++D)
    PrintDecl(*D);

  if (LS->hasBraces()) {
    ChangeIndent(-1);
    Indent() << "}";
  }
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
  for (ObjCMethodDecl::param_iterator PI = OMD->param_begin(),
       E = OMD->param_end(); PI != E; ++PI) {
    // FIXME: selector is missing here!    
    pos = name.find_first_of(":", lastPos);
    Out << " " << name.substr(lastPos, pos - lastPos);
    Out << ":(" << (*PI)->getType().getAsString() << ")"
        << (*PI)->getNameAsString(); 
    lastPos = pos + 1;
  }
    
  if (OMD->param_begin() == OMD->param_end())
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
  
  // FIXME: Don't use a NULL context
  ASTContext *Context = 0;
  for (ObjCImplementationDecl::instmeth_iterator 
         I = OID->instmeth_begin(*Context),
         E = OID->instmeth_end(*Context); 
       I != E; ++I) {
    ObjCMethodDecl *OMD = *I;
    PrintObjCMethodDecl(OMD);
    if (OMD->getBody()) {
      Out << ' ';
      OMD->getBody()->printPretty(Out);
      Out << '\n';
    }
  }
  
  for (ObjCImplementationDecl::classmeth_iterator 
         I = OID->classmeth_begin(*Context),
       E = OID->classmeth_end(*Context);
       I != E; ++I) {
    ObjCMethodDecl *OMD = *I;
    PrintObjCMethodDecl(OMD);
    if (OMD->getBody()) {
      Out << ' ';
      OMD->getBody()->printPretty(Out);
      Out << '\n';
    }
  }
  
  for (ObjCImplementationDecl::propimpl_iterator 
         I = OID->propimpl_begin(*Context),
         E = OID->propimpl_end(*Context); I != E; ++I)
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
  
  // FIXME: Should not use a NULL DeclContext!
  ASTContext *Context = 0;
  for (ObjCInterfaceDecl::prop_iterator I = OID->prop_begin(*Context),
       E = OID->prop_end(*Context); I != E; ++I)
    PrintObjCPropertyDecl(*I);
  bool eol_needed = false;
  for (ObjCInterfaceDecl::classmeth_iterator I = OID->classmeth_begin(*Context),
       E = OID->classmeth_end(*Context); I != E; ++I)
    eol_needed = true, PrintObjCMethodDecl(*I);
  
  for (ObjCInterfaceDecl::instmeth_iterator I = OID->instmeth_begin(*Context),
       E = OID->instmeth_end(*Context); I != E; ++I)
    eol_needed = true, PrintObjCMethodDecl(*I);
  
  Out << (eol_needed ? "\n@end\n" : "@end\n");
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCProtocolDecl(ObjCProtocolDecl *PID) {
  Out << "@protocol " << PID->getNameAsString() << '\n';
  
  // FIXME: Should not use a NULL DeclContext!
  ASTContext *Context = 0;
  for (ObjCProtocolDecl::prop_iterator I = PID->prop_begin(*Context),
       E = PID->prop_end(*Context); I != E; ++I)
    PrintObjCPropertyDecl(*I);
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCategoryImplDecl(ObjCCategoryImplDecl *PID) {
  Out << "@implementation "
      << PID->getClassInterface()->getNameAsString()
      << '(' << PID->getNameAsString() << ");\n";  

  // FIXME: Don't use a NULL context here
  ASTContext *Context = 0;
  for (ObjCCategoryImplDecl::propimpl_iterator 
         I = PID->propimpl_begin(*Context),
       E = PID->propimpl_end(*Context); I != E; ++I)
    PrintObjCPropertyImplDecl(*I);
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::PrintObjCCategoryDecl(ObjCCategoryDecl *PID) {
  // FIXME: Should not use a NULL DeclContext!
  ASTContext *Context = 0;
  Out << "@interface " 
      << PID->getClassInterface()->getNameAsString()
      << '(' << PID->getNameAsString() << ");\n";
  // Output property declarations.
  for (ObjCCategoryDecl::prop_iterator I = PID->prop_begin(*Context),
       E = PID->prop_end(*Context); I != E; ++I)
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

/// PrintTemplateParams - Print a template parameter list and recursively print
/// it's underlying top-level definition.
void DeclPrinter::PrintTemplateDecl(TemplateDecl *TD) {
  // TODO: Write template parameters.
  Out << "template <...> ";
  PrintDecl(TD->getTemplatedDecl());
}



//===----------------------------------------------------------------------===//
/// ASTPrinter - Pretty-printer of ASTs

namespace {
  class ASTPrinter : public ASTConsumer, public DeclPrinter {
  public:
    ASTPrinter(llvm::raw_ostream* o = NULL) : DeclPrinter(o) {}
    
    virtual void HandleTopLevelDecl(DeclGroupRef D) {
      for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I)
        PrintDecl(*I);
    }
  };
} // end anonymous namespace

ASTConsumer *clang::CreateASTPrinter(llvm::raw_ostream* out) {
  return new ASTPrinter(out);
}

//===----------------------------------------------------------------------===//
/// ASTDumper - Low-level dumper of ASTs

namespace {
  class ASTDumper : public ASTConsumer, public DeclPrinter {
    ASTContext *Ctx;
    bool FullDump;

  public:
    explicit ASTDumper(bool FullDump) : DeclPrinter(), FullDump(FullDump) {}
    
    void Initialize(ASTContext &Context) {
      Ctx = &Context;
    }

    virtual void HandleTopLevelDecl(DeclGroupRef D) {
      if (FullDump)
        return;
      for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I)
        HandleTopLevelSingleDecl(*I);
    }
    void HandleTopLevelSingleDecl(Decl *D);

    virtual void HandleTranslationUnit(ASTContext &Ctx) {
      if (!FullDump)
        return;

      for (DeclContext::decl_iterator 
             D = Ctx.getTranslationUnitDecl()->decls_begin(Ctx),
             DEnd = Ctx.getTranslationUnitDecl()->decls_end(Ctx);
           D != DEnd; 
           ++D)
        HandleTopLevelSingleDecl(*D);
    }
  };
} // end anonymous namespace

void ASTDumper::HandleTopLevelSingleDecl(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    PrintFunctionDeclStart(FD);
    
    if (Stmt *Body = FD->getBody(*Ctx)) {
      Out << '\n';
      // FIXME: convert dumper to use raw_ostream.
      Body->dumpAll(Ctx->getSourceManager());
      Out << '\n';
    }
  } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    PrintTypeDefDecl(TD);
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
    if (Stmt *S = MD->getBody()) {
      // FIXME: convert dumper to use raw_ostream.
      S->dumpAll(Ctx->getSourceManager());
      Out << '\n';
    }
  } else if (isa<ObjCImplementationDecl>(D)) {
    Out << "Read objc implementation decl\n";
  } else if (isa<ObjCCategoryImplDecl>(D)) {
    Out << "Read objc category implementation decl\n";
  } else if (isa<LinkageSpecDecl>(D)) {
    Out << "Read linkage spec decl\n";
  } else if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    Out << "Read top-level variable decl: '" << ND->getNameAsString()
    << "'\n";
  } else {
    assert(0 && "Unknown decl type!");
  }
}

ASTConsumer *clang::CreateASTDumper(bool FullDump) { 
  return new ASTDumper(FullDump); 
}

//===----------------------------------------------------------------------===//
/// ASTViewer - AST Visualization

namespace {
  class ASTViewer : public ASTConsumer {
    SourceManager *SM;
  public:
    void Initialize(ASTContext &Context) {
      SM = &Context.getSourceManager();
    }

    virtual void HandleTopLevelDecl(DeclGroupRef D) {
      for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I)
        HandleTopLevelSingleDecl(*I);
    }
    
    void HandleTopLevelSingleDecl(Decl *D);
  };
}

void ASTViewer::HandleTopLevelSingleDecl(Decl *D) {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    DeclPrinter().PrintFunctionDeclStart(FD);
    
    if (FD->getBodyIfAvailable()) {
      llvm::cerr << '\n';
      FD->getBodyIfAvailable()->viewAST();
      llvm::cerr << '\n';
    }
    return;
  }
  
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    DeclPrinter().PrintObjCMethodDecl(MD);
    
    if (MD->getBody()) {
      llvm::cerr << '\n';
      MD->getBody()->viewAST();
      llvm::cerr << '\n';
    }
  }
}


ASTConsumer *clang::CreateASTViewer() { return new ASTViewer(); }

//===----------------------------------------------------------------------===//
/// DeclContextPrinter - Decl and DeclContext Visualization

namespace {

class DeclContextPrinter : public ASTConsumer {
  llvm::raw_ostream& Out;
public:
  DeclContextPrinter() : Out(llvm::errs()) {}

  void HandleTranslationUnit(ASTContext &C) {
    PrintDeclContext(C.getTranslationUnitDecl(), 4);
  }

  void PrintDeclContext(const DeclContext* DC, unsigned Indentation);
};
}  // end anonymous namespace

void DeclContextPrinter::PrintDeclContext(const DeclContext* DC, 
                                          unsigned Indentation) {
  // Print DeclContext name.
  switch (DC->getDeclKind()) {
  case Decl::TranslationUnit:
    Out << "[translation unit] " << DC;
    break;
  case Decl::Namespace: {
    Out << "[namespace] ";
    const NamespaceDecl* ND = cast<NamespaceDecl>(DC);
    Out << ND->getNameAsString();
    break;
  }
  case Decl::Enum: {
    const EnumDecl* ED = cast<EnumDecl>(DC);
    if (ED->isDefinition())
      Out << "[enum] ";
    else
      Out << "<enum> ";
    Out << ED->getNameAsString();
    break;
  }
  case Decl::Record: {
    const RecordDecl* RD = cast<RecordDecl>(DC);
    if (RD->isDefinition())
      Out << "[struct] ";
    else
      Out << "<struct> ";
    Out << RD->getNameAsString();
    break;
  }
  case Decl::CXXRecord: {
    const CXXRecordDecl* RD = cast<CXXRecordDecl>(DC);
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
    const FunctionDecl* FD = cast<FunctionDecl>(DC);
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
    const CXXMethodDecl* D = cast<CXXMethodDecl>(DC);
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
    const DeclContext* SemaDC = D->getDeclContext();
    const DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";

    break;
  }
  case Decl::CXXConstructor: {
    const CXXConstructorDecl* D = cast<CXXConstructorDecl>(DC);
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
    const DeclContext* SemaDC = D->getDeclContext();
    const DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";
    break;
  }
  case Decl::CXXDestructor: {
    const CXXDestructorDecl* D = cast<CXXDestructorDecl>(DC);
    if (D->isOutOfLineDefinition())
      Out << "[c++ dtor] ";
    else if (D->isImplicit())
      Out << "(c++ dtor) ";
    else
      Out << "<c++ dtor> ";
    Out << D->getNameAsString();
    // Check the semantic DC.
    const DeclContext* SemaDC = D->getDeclContext();
    const DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";
    break;
  }
  case Decl::CXXConversion: {
    const CXXConversionDecl* D = cast<CXXConversionDecl>(DC);
    if (D->isOutOfLineDefinition())
      Out << "[c++ conversion] ";
    else if (D->isImplicit())
      Out << "(c++ conversion) ";
    else
      Out << "<c++ conversion> ";
    Out << D->getNameAsString();
    // Check the semantic DC.
    const DeclContext* SemaDC = D->getDeclContext();
    const DeclContext* LexicalDC = D->getLexicalDeclContext();
    if (SemaDC != LexicalDC)
      Out << " [[" << SemaDC << "]]";
    break;
  }

  default:
    assert(0 && "a decl that inherits DeclContext isn't handled");
  }

  Out << "\n";

  // Print decls in the DeclContext.
  // FIXME: Should not use a NULL DeclContext!
  ASTContext *Context = 0;
  for (DeclContext::decl_iterator I = DC->decls_begin(*Context), 
         E = DC->decls_end(*Context);
       I != E; ++I) {
    for (unsigned i = 0; i < Indentation; ++i)
      Out << "  ";

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
      DeclContext* DC = cast<DeclContext>(*I);
      PrintDeclContext(DC, Indentation+2);
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
    case Decl::ParmVar: {
      ParmVarDecl* PVD = cast<ParmVarDecl>(*I);
      Out << "<parameter> " << PVD->getNameAsString() << "\n";
      break;
    }
    case Decl::OriginalParmVar: {
      OriginalParmVarDecl* OPVD = cast<OriginalParmVarDecl>(*I);
      Out << "<original parameter> " << OPVD->getNameAsString() << "\n";
      break;
    }
    case Decl::ObjCProperty: {
      ObjCPropertyDecl* OPD = cast<ObjCPropertyDecl>(*I);
      Out << "<objc property> " << OPD->getNameAsString() << "\n";
      break;
    }
    default:
      fprintf(stderr, "DeclKind: %d \"%s\"\n", DK, I->getDeclKindName());
      assert(0 && "decl unhandled");
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
  
  void HandleTranslationUnit(ASTContext &C) {
    for (ASTContext::type_iterator I=C.types_begin(),E=C.types_end(); I!=E; ++I)
      if (RecordType *T = dyn_cast<RecordType>(*I)) {
        if (CXXRecordDecl *D = dyn_cast<CXXRecordDecl>(T->getDecl())) {
          // FIXME: This lookup needs to be generalized to handle namespaces and
          // (when we support them) templates.
          if (D->getNameAsString() == clsname) {
            D->viewInheritance(C);      
          }
        }
      }
  }
}; 
}

ASTConsumer *clang::CreateInheritanceViewer(const std::string& clsname) {
  return new InheritanceViewer(clsname);
}

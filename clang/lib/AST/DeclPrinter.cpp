//===--- DeclPrinter.cpp - Printing implementation for Decl ASTs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl::dump method, which pretty print the
// AST back out to C/Objective-C/C++/Objective-C++ code.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {
  class VISIBILITY_HIDDEN DeclPrinter : public DeclVisitor<DeclPrinter> {
    llvm::raw_ostream &Out;
    ASTContext &Context;
    PrintingPolicy Policy;
    unsigned Indentation;

    llvm::raw_ostream& Indent();

  public:
    DeclPrinter(llvm::raw_ostream &Out, ASTContext &Context, 
                const PrintingPolicy &Policy,
                unsigned Indentation = 0)
      : Out(Out), Context(Context), Policy(Policy), Indentation(Indentation) { }

    void VisitDeclContext(DeclContext *DC, bool Indent = true);

    void VisitTranslationUnitDecl(TranslationUnitDecl *D);
    void VisitTypedefDecl(TypedefDecl *D);
    void VisitEnumDecl(EnumDecl *D);
    void VisitRecordDecl(RecordDecl *D);
    void VisitEnumConstantDecl(EnumConstantDecl *D);
    void VisitFunctionDecl(FunctionDecl *D);
    void VisitFieldDecl(FieldDecl *D);
    void VisitVarDecl(VarDecl *D);
    void VisitParmVarDecl(ParmVarDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *D);
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    void VisitTemplateDecl(TemplateDecl *D);
    void VisitObjCClassDecl(ObjCClassDecl *D);
    void VisitObjCMethodDecl(ObjCMethodDecl *D);
    void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    void VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D);
    void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
  };
}

void Decl::print(llvm::raw_ostream &Out, ASTContext &Context, 
                 unsigned Indentation) {
  print(Out, Context, Context.PrintingPolicy, Indentation);
}

void Decl::print(llvm::raw_ostream &Out, ASTContext &Context, 
                 const PrintingPolicy &Policy, unsigned Indentation) {
  DeclPrinter Printer(Out, Context, Policy, Indentation);
  Printer.Visit(this);
}

void Decl::dump(ASTContext &Context) {
  print(llvm::errs(), Context);
}

llvm::raw_ostream& DeclPrinter::Indent() {
  for (unsigned i = 0; i < Indentation; ++i)
    Out << "  ";
  return Out;
}

//----------------------------------------------------------------------------
// Common C declarations
//----------------------------------------------------------------------------

void DeclPrinter::VisitDeclContext(DeclContext *DC, bool Indent) {
  if (Indent)
    Indentation += Policy.Indentation;

  for (DeclContext::decl_iterator D = DC->decls_begin(Context),
         DEnd = DC->decls_end(Context);
       D != DEnd; ++D) {
    this->Indent();
    Visit(*D);
    
    // FIXME: Need to be able to tell the DeclPrinter when 
    const char *Terminator = 0;
    if (isa<FunctionDecl>(*D) && 
        cast<FunctionDecl>(*D)->isThisDeclarationADefinition())
      Terminator = 0;
    else if (isa<NamespaceDecl>(*D) || isa<LinkageSpecDecl>(*D))
      Terminator = 0;
    else if (isa<EnumConstantDecl>(*D)) {
      DeclContext::decl_iterator Next = D;
      ++Next;
      if (Next != DEnd)
        Terminator = ",";
    } else
      Terminator = ";";

    if (Terminator)
      Out << Terminator;
    Out << "\n";
  }

  if (Indent)
    Indentation -= Policy.Indentation;
}

void DeclPrinter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  VisitDeclContext(D, false);
}

void DeclPrinter::VisitTypedefDecl(TypedefDecl *D) {
  std::string S = D->getNameAsString();
  D->getUnderlyingType().getAsStringInternal(S, Policy);
  Out << "typedef " << S;
}

void DeclPrinter::VisitEnumDecl(EnumDecl *D) {
  Out << "enum " << D->getNameAsString() << " {\n";
  VisitDeclContext(D);
  Indent() << "}";
}

void DeclPrinter::VisitRecordDecl(RecordDecl *D) {
  // print a free standing tag decl (e.g. "struct x;"). 
  Out << D->getKindName();
  Out << " ";
  Out << D->getNameAsString();
  
  if (D->isDefinition()) {
    Out << " {\n";
    VisitDeclContext(D);
    Indent() << "}";
  }
}

void DeclPrinter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  Out << D->getNameAsString();
  if (Expr *Init = D->getInitExpr()) {
    Out << " = ";
    Init->printPretty(Out, 0, Policy, Indentation);
  }
}

void DeclPrinter::VisitFunctionDecl(FunctionDecl *D) { 
  switch (D->getStorageClass()) {
  case FunctionDecl::None: break;
  case FunctionDecl::Extern: Out << "extern "; break;
  case FunctionDecl::Static: Out << "static "; break;
  case FunctionDecl::PrivateExtern: Out << "__private_extern__ "; break;
  }

  if (D->isInline())           Out << "inline ";
  if (D->isVirtualAsWritten()) Out << "virtual ";

  std::string Proto = D->getNameAsString();
  if (isa<FunctionType>(D->getType().getTypePtr())) {
    const FunctionType *AFT = D->getType()->getAsFunctionType();

    const FunctionProtoType *FT = 0;
    if (D->hasWrittenPrototype())
      FT = dyn_cast<FunctionProtoType>(AFT);

    Proto += "(";
    if (FT) {
      llvm::raw_string_ostream POut(Proto);
      DeclPrinter ParamPrinter(POut, Context, Policy, Indentation);
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        if (i) POut << ", ";
        ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
      }
    
      if (FT->isVariadic()) {
        if (D->getNumParams()) POut << ", ";
        POut << "...";
      }
    }

    Proto += ")";
    AFT->getResultType().getAsStringInternal(Proto, Policy);
  } else {
    D->getType().getAsStringInternal(Proto, Policy);
  }

  Out << Proto;

  if (D->isPure())
    Out << " = 0";
  else if (D->isDeleted())
    Out << " = delete";
  else if (D->isThisDeclarationADefinition()) {
    if (!D->hasPrototype() && D->getNumParams()) {
      // This is a K&R function definition, so we need to print the
      // parameters.
      Out << '\n';
      Indentation += Policy.Indentation;
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        Indent();
        VisitParmVarDecl(D->getParamDecl(i));
        Out << ";\n";
      }
      Indentation -= Policy.Indentation;
    } else
      Out << ' ';

    D->getBody(Context)->printPretty(Out, 0, Policy, Indentation);
    Out << '\n';
  }
}

void DeclPrinter::VisitFieldDecl(FieldDecl *D) {
  if (D->isMutable())
    Out << "mutable ";

  std::string Name = D->getNameAsString();
  D->getType().getAsStringInternal(Name, Policy);
  Out << Name;

  if (D->isBitField()) {
    Out << " : ";
    D->getBitWidth()->printPretty(Out, 0, Policy, Indentation);
  }
}

void DeclPrinter::VisitVarDecl(VarDecl *D) {
  if (D->getStorageClass() != VarDecl::None)
    Out << VarDecl::getStorageClassSpecifierString(D->getStorageClass()) << " ";

  if (D->isThreadSpecified())
    Out << "__thread ";

  std::string Name = D->getNameAsString();
  QualType T = D->getType();
  if (OriginalParmVarDecl *Parm = dyn_cast<OriginalParmVarDecl>(D))
    T = Parm->getOriginalType();
  T.getAsStringInternal(Name, Policy);
  Out << Name;
  if (D->getInit()) {
    if (D->hasCXXDirectInitializer())
      Out << "(";
    else
      Out << " = ";
    D->getInit()->printPretty(Out, 0, Policy, Indentation);
    if (D->hasCXXDirectInitializer())
      Out << ")";
  }
}

void DeclPrinter::VisitParmVarDecl(ParmVarDecl *D) {
  VisitVarDecl(D);
}

void DeclPrinter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
  Out << "__asm (";
  D->getAsmString()->printPretty(Out, 0, Policy, Indentation);
  Out << ")";
}

//----------------------------------------------------------------------------
// C++ declarations
//----------------------------------------------------------------------------
void DeclPrinter::VisitNamespaceDecl(NamespaceDecl *D) {
  Out << "namespace " << D->getNameAsString() << " {\n";
  VisitDeclContext(D);
  Indent() << "}";
}

void DeclPrinter::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  const char *l;
  if (D->getLanguage() == LinkageSpecDecl::lang_c)
    l = "C";
  else {
    assert(D->getLanguage() == LinkageSpecDecl::lang_cxx &&
           "unknown language in linkage specification");
    l = "C++";
  }

  Out << "extern \"" << l << "\" ";
  if (D->hasBraces()) {
    Out << "{\n";
    VisitDeclContext(D);
    Indent() << "}";
  } else
    Visit(*D->decls_begin(Context));
}

void DeclPrinter::VisitTemplateDecl(TemplateDecl *D) {
  // TODO: Write template parameters.
  Out << "template <...> ";
  Visit(D->getTemplatedDecl());
}

//----------------------------------------------------------------------------
// Objective-C declarations
//----------------------------------------------------------------------------

void DeclPrinter::VisitObjCClassDecl(ObjCClassDecl *D) {
  Out << "@class ";
  for (ObjCClassDecl::iterator I = D->begin(), E = D->end();
       I != E; ++I) {
    if (I != D->begin()) Out << ", ";
    Out << (*I)->getNameAsString();
  }
  Out << ";\n";
}

void DeclPrinter::VisitObjCMethodDecl(ObjCMethodDecl *OMD) {
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
  
  if (OMD->getBody()) {
    Out << ' ';
    OMD->getBody()->printPretty(Out, 0, Policy);
    Out << '\n';
  } else
    Out << ";";
}

void DeclPrinter::VisitObjCImplementationDecl(ObjCImplementationDecl *OID) {
  std::string I = OID->getNameAsString();
  ObjCInterfaceDecl *SID = OID->getSuperClass();

  if (SID)
    Out << "@implementation " << I << " : " << SID->getNameAsString();
  else
    Out << "@implementation " << I;
  
  VisitDeclContext(OID);
  Out << "@end\n";
}

void DeclPrinter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *OID) {
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
      Out << '\t' << (*I)->getType().getAsString(Policy)
          << ' '  << (*I)->getNameAsString() << ";\n";      
    }
    Out << "}\n";
  }
  
  VisitDeclContext(OID, false);
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {
  Out << "@protocol ";
  for (ObjCForwardProtocolDecl::protocol_iterator I = D->protocol_begin(), 
         E = D->protocol_end();
       I != E; ++I) {
    if (I != D->protocol_begin()) Out << ", ";
    Out << (*I)->getNameAsString();
  }
  Out << ";\n";
}

void DeclPrinter::VisitObjCProtocolDecl(ObjCProtocolDecl *PID) {
  Out << "@protocol " << PID->getNameAsString() << '\n';
  
  for (ObjCProtocolDecl::prop_iterator I = PID->prop_begin(Context),
       E = PID->prop_end(Context); I != E; ++I)
    VisitObjCPropertyDecl(*I);
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *PID) {
  Out << "@implementation "
      << PID->getClassInterface()->getNameAsString()
      << '(' << PID->getNameAsString() << ");\n";  

  VisitDeclContext(PID, false);
  Out << "@end\n";
  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCCategoryDecl(ObjCCategoryDecl *PID) {
  Out << "@interface " 
      << PID->getClassInterface()->getNameAsString()
      << '(' << PID->getNameAsString() << ");\n";
  VisitDeclContext(PID, false);
  Out << "@end\n";
  
  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *AID) {
  Out << "@compatibility_alias " << AID->getNameAsString() 
      << ' ' << AID->getClassInterface()->getNameAsString() << ";\n";  
}

/// PrintObjCPropertyDecl - print a property declaration.
///
void DeclPrinter::VisitObjCPropertyDecl(ObjCPropertyDecl *PDecl) {
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
  Out << ' ' << PDecl->getType().getAsString(Policy)
  << ' ' << PDecl->getNameAsString();
    
  Out << ";\n";
}

void DeclPrinter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *PID) {
  if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize)
    Out << "\n@synthesize ";
  else
    Out << "\n@dynamic ";
  Out << PID->getPropertyDecl()->getNameAsString();
  if (PID->getPropertyIvarDecl())
    Out << "=" << PID->getPropertyIvarDecl()->getNameAsString();
  Out << ";\n";
}

//===- CIndexUSR.cpp - Clang-C Source Indexing Library --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generation and use of USRs from CXEntities.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXCursor.h"
#include "clang/AST/DeclVisitor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::cxstring;

//===----------------------------------------------------------------------===//
// USR generation.
//===----------------------------------------------------------------------===//

namespace {
class USRGenerator : public DeclVisitor<USRGenerator> {
  llvm::raw_ostream &Out;
  bool IgnoreResults;
public:
  USRGenerator(llvm::raw_ostream &out) : Out(out), IgnoreResults(false) {}

  bool ignoreResults() const { return IgnoreResults; }

  // Visitation methods from generating USRs from AST elements.
  void VisitBlockDecl(BlockDecl *D);
  void VisitDeclContext(DeclContext *D);
  void VisitFieldDecl(FieldDecl *D);
  void VisitFunctionDecl(FunctionDecl *D);
  void VisitNamedDecl(NamedDecl *D);
  void VisitNamespaceDecl(NamespaceDecl *D);
  void VisitObjCContainerDecl(ObjCContainerDecl *CD);
  void VisitObjCMethodDecl(ObjCMethodDecl *MD);
  void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
  void VisitTagDecl(TagDecl *D);
  void VisitTypedefDecl(TypedefDecl *D);


  /// String generation methods used both by the visitation methods
  /// and from other clients that want to directly generate USRs.  These
  /// methods do not construct complete USRs (which incorporate the parents
  /// of an AST element), but only the fragments concerning the AST element
  /// itself.

  /// Generate a USR fragment for a named declaration.  This does
  /// not include the USR component for the parent.
  void GenNamedDecl(llvm::StringRef name);

  /// Generate a USR for an Objective-C class.
  void GenObjCClass(llvm::StringRef cls);
  /// Generate a USR for an Objective-C class category.
  void GenObjCCategory(llvm::StringRef cls, llvm::StringRef cat);
  /// Generate a USR fragment for an Objective-C instance variable.  The
  /// complete USR can be created by concatenating the USR for the
  /// encompassing class with this USR fragment.
  void GenObjCIvar(llvm::StringRef ivar);
  /// Generate a USR fragment for an Objective-C method.
  void GenObjCMethod(llvm::StringRef sel, bool isInstanceMethod);
  /// Generate a USR fragment for an Objective-C property.
  void GenObjCProperty(llvm::StringRef prop);
  /// Generate a USR for an Objective-C protocol.
  void GenObjCProtocol(llvm::StringRef prot);
};

class StringUSRGenerator {
private:
  llvm::SmallString<1024> StrBuf;
  llvm::raw_svector_ostream Out;
  USRGenerator UG;
public:
  StringUSRGenerator()
    : Out(StrBuf), UG(Out) {
    // Add the USR space prefix.
    Out << "c:";      
  }

  llvm::StringRef str() {
    return Out.str();
  }

  USRGenerator* operator->() { return &UG; }

  template <typename T>
  llvm::raw_svector_ostream &operator<<(const T &x) {
    Out << x;
    return Out;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Generating USRs from ASTS.
//===----------------------------------------------------------------------===//

void USRGenerator::VisitBlockDecl(BlockDecl *D) {
  VisitDeclContext(D->getDeclContext());
  // FIXME: Better support for anonymous blocks.
  Out << "@B^anon";
}

void USRGenerator::VisitDeclContext(DeclContext *DC) {
  if (NamedDecl *D = dyn_cast<NamedDecl>(DC))
    Visit(D);
}

void USRGenerator::VisitFieldDecl(FieldDecl *D) {
  const std::string &s = D->getNameAsString();
  if (s.empty()) {
    // Bit fields can be anonymous.
    IgnoreResults = true;
    return;
  }
  VisitDeclContext(D->getDeclContext());
  Out << "@^FI^" << s;
}

void USRGenerator::VisitFunctionDecl(FunctionDecl *D) {
  VisitDeclContext(D->getDeclContext());
  Out << "@F^" << D->getNameAsString();
}

void USRGenerator::VisitNamedDecl(NamedDecl *D) {
  VisitDeclContext(D->getDeclContext());
  const std::string &s = D->getNameAsString();
  //  assert(!s.empty());
  GenNamedDecl(s);
}

void USRGenerator::VisitNamespaceDecl(NamespaceDecl *D) {
  VisitDeclContext(D->getDeclContext());
  Out << "@N^" << D->getNameAsString();
}

void USRGenerator::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  Visit(cast<Decl>(D->getDeclContext()));
  GenObjCMethod(DeclarationName(D->getSelector()).getAsString(),
                D->isInstanceMethod());
}

void USRGenerator::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  switch (D->getKind()) {
    default:
      assert(false && "Invalid ObjC container.");
    case Decl::ObjCInterface:
    case Decl::ObjCImplementation:
      GenObjCClass(D->getName());
      break;
    case Decl::ObjCCategory: {
      ObjCCategoryDecl *CD = cast<ObjCCategoryDecl>(D);
      ObjCInterfaceDecl *ID = CD->getClassInterface();
      if (!ID) {
        // Handle invalid code where the @interface might not
        // have been specified.
        // FIXME: We should be able to generate this USR even if the
        // @interface isn't available.
        IgnoreResults = true;
        return;
      }
      GenObjCCategory(ID->getName(), CD->getName());
      break;
    }
    case Decl::ObjCCategoryImpl: {
      ObjCCategoryImplDecl *CD = cast<ObjCCategoryImplDecl>(D);
      ObjCInterfaceDecl *ID = CD->getClassInterface();
      if (!ID) {
        // Handle invalid code where the @interface might not
        // have been specified.
        // FIXME: We should be able to generate this USR even if the
        // @interface isn't available.
        IgnoreResults = true;
        return;
      }
      GenObjCCategory(ID->getName(), CD->getName());
      break;
    }
    case Decl::ObjCProtocol:
      GenObjCProtocol(cast<ObjCProtocolDecl>(D)->getName());
      break;
  }
}

void USRGenerator::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  Visit(cast<Decl>(D->getDeclContext()));
  GenObjCProperty(D->getName());
}

void USRGenerator::VisitTagDecl(TagDecl *D) {
  VisitDeclContext(D->getDeclContext());
  switch (D->getTagKind()) {
    case TagDecl::TK_struct: Out << "@S^"; break;
    case TagDecl::TK_class:  Out << "@C^"; break;
    case TagDecl::TK_union:  Out << "@U^"; break;
    case TagDecl::TK_enum:   Out << "@E^"; break;
  }

  // FIXME: Better support for anonymous structures and enums.
  const std::string &s = D->getNameAsString();
  if (s.empty()) {
    if (TypedefDecl *TD = D->getTypedefForAnonDecl())
      Out << "^anontd^" << TD->getNameAsString();
    else
      Out << "^anon";
  }
  else
    Out << s;
}

void USRGenerator::VisitTypedefDecl(TypedefDecl *D) {
  DeclContext *DC = D->getDeclContext();
  if (NamedDecl *DCN = dyn_cast<NamedDecl>(DC))
    Visit(DCN);
  Out << "typedef@" << D->getName();
}

//===----------------------------------------------------------------------===//
// General purpose USR generation methods.
//===----------------------------------------------------------------------===//

void USRGenerator::GenNamedDecl(llvm::StringRef name) {
  Out << "@^" << name;
}

void USRGenerator::GenObjCClass(llvm::StringRef cls) {
  Out << "objc(cs)" << cls;
}

void USRGenerator::GenObjCCategory(llvm::StringRef cls, llvm::StringRef cat) {
  Out << "objc(cy)" << cls << '^' << cat;
}

void USRGenerator::GenObjCIvar(llvm::StringRef ivar) {
  GenNamedDecl(ivar);
}

void USRGenerator::GenObjCMethod(llvm::StringRef meth, bool isInstanceMethod) {
  Out << (isInstanceMethod ? "(im)" : "(cm)") << meth;
}

void USRGenerator::GenObjCProperty(llvm::StringRef prop) {
  Out << "(py)" << prop;
}

void USRGenerator::GenObjCProtocol(llvm::StringRef prot) {
  Out << "objc(pl)" << prot;
}

//===----------------------------------------------------------------------===//
// API hooks.
//===----------------------------------------------------------------------===//

extern "C" {

CXString clang_getCursorUSR(CXCursor C) {
  Decl *D = cxcursor::getCursorDecl(C);
  if (!D)
    return createCXString(NULL);

  StringUSRGenerator SUG;
  SUG->Visit(static_cast<Decl*>(D));

  if (SUG->ignoreResults())
    return createCXString("");

  // Return a copy of the string that must be disposed by the caller.
  return createCXString(SUG.str(), true);
}

static inline llvm::StringRef extractUSRSuffix(llvm::StringRef s) {
  if (!(s.size() >= 2 && s[0] == 'c' && s[1] == ':'))
    return "";
  return s.substr(2);
}
  
CXString clang_constructUSR_ObjCIvar(const char *name, CXString classUSR) {
  StringUSRGenerator SUG;
  SUG << extractUSRSuffix(clang_getCString(classUSR));
  SUG->GenObjCIvar(name);
  return createCXString(SUG.str(), true);
}

CXString clang_constructUSR_ObjCMethod(const char *name,
                                       unsigned isInstanceMethod,
                                       CXString classUSR) {
  StringUSRGenerator SUG;
  SUG << extractUSRSuffix(clang_getCString(classUSR));
  SUG->GenObjCMethod(name, isInstanceMethod);
  return createCXString(SUG.str(), true);
}

CXString clang_constructUSR_ObjCClass(const char *name) {
  StringUSRGenerator SUG;
  SUG->GenObjCClass(name);
  return createCXString(SUG.str(), true);
}

CXString clang_constructUSR_ObjCProtocol(const char *name) {
  StringUSRGenerator SUG;
  SUG->GenObjCProtocol(name);
  return createCXString(SUG.str(), true);
}

CXString clang_constructUSR_ObjCCategory(const char *class_name,
                                         const char *category_name) {
  StringUSRGenerator SUG;
  SUG->GenObjCCategory(class_name, category_name);
  return createCXString(SUG.str(), true);
}

CXString clang_constructUSR_ObjCProperty(const char *property,
                                         CXString classUSR) {
  StringUSRGenerator SUG;
  SUG << extractUSRSuffix(clang_getCString(classUSR));
  SUG->GenObjCProperty(property);
  return createCXString(SUG.str(), true);
}

} // end extern "C"

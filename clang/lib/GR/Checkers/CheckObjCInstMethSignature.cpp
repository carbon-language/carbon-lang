//=- CheckObjCInstMethodRetTy.cpp - Check ObjC method signatures -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a CheckObjCInstMethSignature, a flow-insenstive check
//  that determines if an Objective-C class interface incorrectly redefines
//  the method signature in a subclass.
//
//===----------------------------------------------------------------------===//

#include "clang/GR/Checkers/LocalCheckers.h"
#include "clang/GR/BugReporter/PathDiagnostic.h"
#include "clang/GR/BugReporter/BugReporter.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Type.h"
#include "clang/AST/ASTContext.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

static bool AreTypesCompatible(QualType Derived, QualType Ancestor,
                               ASTContext& C) {

  // Right now don't compare the compatibility of pointers.  That involves
  // looking at subtyping relationships.  FIXME: Future patch.
  if (Derived->isAnyPointerType() &&  Ancestor->isAnyPointerType())
    return true;

  return C.typesAreCompatible(Derived, Ancestor);
}

static void CompareReturnTypes(const ObjCMethodDecl *MethDerived,
                               const ObjCMethodDecl *MethAncestor,
                               BugReporter &BR, ASTContext &Ctx,
                               const ObjCImplementationDecl *ID) {

  QualType ResDerived  = MethDerived->getResultType();
  QualType ResAncestor = MethAncestor->getResultType();

  if (!AreTypesCompatible(ResDerived, ResAncestor, Ctx)) {
    std::string sbuf;
    llvm::raw_string_ostream os(sbuf);

    os << "The Objective-C class '"
       << MethDerived->getClassInterface()
       << "', which is derived from class '"
       << MethAncestor->getClassInterface()
       << "', defines the instance method '"
       << MethDerived->getSelector().getAsString()
       << "' whose return type is '"
       << ResDerived.getAsString()
       << "'.  A method with the same name (same selector) is also defined in "
          "class '"
       << MethAncestor->getClassInterface()
       << "' and has a return type of '"
       << ResAncestor.getAsString()
       << "'.  These two types are incompatible, and may result in undefined "
          "behavior for clients of these classes.";

    BR.EmitBasicReport("Incompatible instance method return type",
                       os.str(), MethDerived->getLocStart());
  }
}

void ento::CheckObjCInstMethSignature(const ObjCImplementationDecl* ID,
                                    BugReporter& BR) {

  const ObjCInterfaceDecl* D = ID->getClassInterface();
  const ObjCInterfaceDecl* C = D->getSuperClass();

  if (!C)
    return;

  ASTContext& Ctx = BR.getContext();

  // Build a DenseMap of the methods for quick querying.
  typedef llvm::DenseMap<Selector,ObjCMethodDecl*> MapTy;
  MapTy IMeths;
  unsigned NumMethods = 0;

  for (ObjCImplementationDecl::instmeth_iterator I=ID->instmeth_begin(),
       E=ID->instmeth_end(); I!=E; ++I) {

    ObjCMethodDecl* M = *I;
    IMeths[M->getSelector()] = M;
    ++NumMethods;
  }

  // Now recurse the class hierarchy chain looking for methods with the
  // same signatures.
  while (C && NumMethods) {
    for (ObjCInterfaceDecl::instmeth_iterator I=C->instmeth_begin(),
         E=C->instmeth_end(); I!=E; ++I) {

      ObjCMethodDecl* M = *I;
      Selector S = M->getSelector();

      MapTy::iterator MI = IMeths.find(S);

      if (MI == IMeths.end() || MI->second == 0)
        continue;

      --NumMethods;
      ObjCMethodDecl* MethDerived = MI->second;
      MI->second = 0;

      CompareReturnTypes(MethDerived, M, BR, Ctx, ID);
    }

    C = C->getSuperClass();
  }
}

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

#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Type.h"
#include "clang/AST/ASTContext.h"

#include "llvm/ADT/DenseMap.h"
#include <sstream>

using namespace clang;

static bool AreTypesCompatible(QualType Derived, QualType Ancestor,
                               ASTContext& C) {

  // Right now don't compare the compatibility of pointers.  That involves
  // looking at subtyping relationships.  FIXME: Future patch.
  if ((Derived->isPointerType() || Derived->isObjCQualifiedIdType())  && 
      (Ancestor->isPointerType() || Ancestor->isObjCQualifiedIdType()))
    return true;

  return C.typesAreCompatible(Derived, Ancestor);
}

static void CompareReturnTypes(ObjCMethodDecl* MethDerived,
                               ObjCMethodDecl* MethAncestor,
                               BugReporter& BR, ASTContext& Ctx,
                               ObjCImplementationDecl* ID) {
    
  QualType ResDerived  = MethDerived->getResultType();
  QualType ResAncestor = MethAncestor->getResultType(); 
  
  if (!AreTypesCompatible(ResDerived, ResAncestor, Ctx)) {
    std::ostringstream os;
    
    os << "The Objective-C class '"
       << MethDerived->getClassInterface()->getName()
       << "', which is derived from class '"
       << MethAncestor->getClassInterface()->getName()
       << "', defines the instance method '"
       << MethDerived->getSelector().getName()
       << "' whose return type is '"
       << ResDerived.getAsString()
       << "'.  A method with the same name (same selector) is also defined in "
          "class '"
       << MethAncestor->getClassInterface()->getName()
       << "' and has a return type of '"
       << ResAncestor.getAsString()
       << "'.  These two types are incompatible, and may result in undefined "
          "behavior for clients of these classes.";
    
    BR.EmitBasicReport("incompatible instance method return type",
                       os.str().c_str(), MethDerived->getLocStart());
  }
}

void clang::CheckObjCInstMethSignature(ObjCImplementationDecl* ID,
                                       BugReporter& BR) {
  
  ObjCInterfaceDecl* D = ID->getClassInterface();
  ObjCInterfaceDecl* C = D->getSuperClass();

  if (!C)
    return;
  
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
  ASTContext& Ctx = BR.getContext();
  
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

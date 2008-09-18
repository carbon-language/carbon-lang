//=- CheckNSError.cpp - Coding conventions for uses of NSError ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a CheckNSError, a flow-insenstive check
//  that determines if an Objective-C class interface correctly returns
//  a non-void return type.
//
//  File under feature request PR 2600.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Type.h"
#include "clang/AST/ASTContext.h"

using namespace clang;

void clang::CheckNSError(ObjCImplementationDecl* ID, BugReporter& BR) {
  // Look at the @interface for this class.
  ObjCInterfaceDecl* D = ID->getClassInterface();
  
  // Get the ASTContext.  Useful for querying type information.
  ASTContext &Ctx = BR.getContext();
  
  // Get the IdentifierInfo* for "NSError".
  IdentifierInfo* NSErrorII = &Ctx.Idents.get("NSError");

  // Scan the methods.  See if any of them have an argument of type NSError**.  
  for (ObjCInterfaceDecl::instmeth_iterator I=D->instmeth_begin(),
        E=D->instmeth_end(); I!=E; ++I) {

    // Get the method declaration.
    ObjCMethodDecl* M = *I;
    
    // Check for a non-void return type.
    if (M->getResultType() != Ctx.VoidTy)
      continue;

    for (ObjCMethodDecl::param_iterator PI=M->param_begin(), 
         PE=M->param_end(); PI!=PE; ++PI) {
      
      const PointerType* PPT = (*PI)->getType()->getAsPointerType();
      if (!PPT) continue;
      
      const PointerType* PT = PPT->getPointeeType()->getAsPointerType();
      if (!PT) continue;
      
      const ObjCInterfaceType *IT =
        PT->getPointeeType()->getAsObjCInterfaceType();
      
      if (!IT) continue;
      
      // Check if IT is "NSError".
      if (IT->getDecl()->getIdentifier() == NSErrorII) {
        // Documentation: "Creating and Returning NSError Objects"
        BR.EmitBasicReport("Bad return type when passing NSError**",
         "Method accepting NSError** argument should have "
         "non-void return value to indicate that an error occurred.",
         M->getLocStart());
        break;
      }
    }
  }
}

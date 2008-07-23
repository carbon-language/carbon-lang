//==- CheckObjCUnusedIVars.cpp - Check for unused ivars ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a CheckObjCUnusedIvars, a checker that
//  analyzes an Objective-C class's interface/implementation to determine if it
//  has any ivars that are never accessed.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/LangOptions.h"
#include <sstream>

using namespace clang;

enum IVarState { Unused, Used };
typedef llvm::DenseMap<ObjCIvarDecl*,IVarState> IvarUsageMap;

static void Scan(IvarUsageMap& M, Stmt* S) {
  if (!S)
    return;
  
  if (ObjCIvarRefExpr* Ex = dyn_cast<ObjCIvarRefExpr>(S)) {
    ObjCIvarDecl* D = Ex->getDecl();
    IvarUsageMap::iterator I = M.find(D);
    if (I != M.end()) I->second = Used;
  }
  else
    for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I!=E;++I)
      Scan(M, *I);
}

void clang::CheckObjCUnusedIvar(ObjCImplementationDecl* D, BugReporter& BR) {

  ObjCInterfaceDecl* ID = D->getClassInterface();
  IvarUsageMap M;


  
  // Iterate over the ivars.
  for (ObjCInterfaceDecl::ivar_iterator I=ID->ivar_begin(), E=ID->ivar_end();
       I!=E; ++I) {
    
    ObjCIvarDecl* ID = *I;
    
    // Ignore ivars that aren't private.
    ObjCIvarDecl::AccessControl ac = ID->getAccessControl();
    if (!(ac == ObjCIvarDecl::None || ac == ObjCIvarDecl::Private))
      continue;
    
    if (ID->getAttr<IBOutletAttr>() == 0)
      continue;
    
    M[ID] = Unused;
  }

  if (M.empty())
    return;
  
  // Now scan the methods for accesses.
  for (ObjCImplementationDecl::instmeth_iterator I = D->instmeth_begin(),
       E = D->instmeth_end(); I!=E; ++I)
    Scan(M, (*I)->getBody());
  
  // Find ivars that are unused.
  for (IvarUsageMap::iterator I = M.begin(), E = M.end(); I!=E; ++I)
    if (I->second == Unused) {
      
      std::ostringstream os;
      os << "Private ivar '" << I->first->getName() << "' is never used.";
      
      BR.EmitBasicReport("unused ivar",
                         os.str().c_str(), I->first->getLocation());
    }
}


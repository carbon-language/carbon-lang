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

using namespace clang;

enum IVarState { Unused, Used };
typedef llvm::DenseMap<const ObjCIvarDecl*,IVarState> IvarUsageMap;

static void Scan(IvarUsageMap& M, const Stmt* S) {
  if (!S)
    return;

  if (const ObjCIvarRefExpr *Ex = dyn_cast<ObjCIvarRefExpr>(S)) {
    const ObjCIvarDecl *D = Ex->getDecl();
    IvarUsageMap::iterator I = M.find(D);
    if (I != M.end())
      I->second = Used;
    return;
  }

  // Blocks can reference an instance variable of a class.
  if (const BlockExpr *BE = dyn_cast<BlockExpr>(S)) {
    Scan(M, BE->getBody());
    return;
  }

  for (Stmt::const_child_iterator I=S->child_begin(),E=S->child_end(); I!=E;++I)
    Scan(M, *I);
}

static void Scan(IvarUsageMap& M, const ObjCPropertyImplDecl* D) {
  if (!D)
    return;

  const ObjCIvarDecl* ID = D->getPropertyIvarDecl();

  if (!ID)
    return;

  IvarUsageMap::iterator I = M.find(ID);
  if (I != M.end())
    I->second = Used;
}

void clang::CheckObjCUnusedIvar(const ObjCImplementationDecl *D,
                                BugReporter &BR) {

  const ObjCInterfaceDecl* ID = D->getClassInterface();
  IvarUsageMap M;

  // Iterate over the ivars.
  for (ObjCInterfaceDecl::ivar_iterator I=ID->ivar_begin(),
        E=ID->ivar_end(); I!=E; ++I) {

    const ObjCIvarDecl* ID = *I;

    // Ignore ivars that aren't private.
    if (ID->getAccessControl() != ObjCIvarDecl::Private)
      continue;

    // Skip IB Outlets.
    if (ID->getAttr<IBOutletAttr>())
      continue;

    M[ID] = Unused;
  }

  if (M.empty())
    return;

  // Now scan the methods for accesses.
  for (ObjCImplementationDecl::instmeth_iterator I = D->instmeth_begin(),
        E = D->instmeth_end(); I!=E; ++I)
    Scan(M, (*I)->getBody());

  // Scan for @synthesized property methods that act as setters/getters
  // to an ivar.
  for (ObjCImplementationDecl::propimpl_iterator I = D->propimpl_begin(),
       E = D->propimpl_end(); I!=E; ++I)
    Scan(M, *I);

  // Find ivars that are unused.
  for (IvarUsageMap::iterator I = M.begin(), E = M.end(); I!=E; ++I)
    if (I->second == Unused) {
      std::string sbuf;
      llvm::raw_string_ostream os(sbuf);
      os << "Instance variable '" << I->first->getNameAsString()
         << "' in class '" << ID->getNameAsString()
         << "' is never used by the methods in its @implementation "
            "(although it may be used by category methods).";

      BR.EmitBasicReport("Unused instance variable", "Optimization",
                         os.str().c_str(), I->first->getLocation());
    }
}

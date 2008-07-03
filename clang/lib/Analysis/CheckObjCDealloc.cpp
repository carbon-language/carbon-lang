//==- CheckObjCDealloc.cpp - Check ObjC -dealloc implementation --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a DeadStores, a flow-sensitive checker that looks for
//  stores to variables that are no longer live.
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

static bool scan_dealloc(Stmt* S, Selector Dealloc) {  
  
  if (ObjCMessageExpr* ME = dyn_cast<ObjCMessageExpr>(S))
    if (ME->getSelector() == Dealloc)
      if (Expr* Receiver = ME->getReceiver()->IgnoreParenCasts())
        if (PreDefinedExpr* E = dyn_cast<PreDefinedExpr>(Receiver))
          if (E->getIdentType() == PreDefinedExpr::ObjCSuper)
            return true;

  // Recurse to children.

  for (Stmt::child_iterator I = S->child_begin(), E= S->child_end(); I!=E; ++I)
    if (*I && scan_dealloc(*I, Dealloc))
      return true;
  
  return false;
}

void clang::CheckObjCDealloc(ObjCImplementationDecl* D,
                             const LangOptions& LOpts, BugReporter& BR) {

  assert (LOpts.getGCMode() != LangOptions::GCOnly);
  
  ASTContext& Ctx = BR.getContext();

  // Determine if the class subclasses NSObject.
  IdentifierInfo* NSObjectII = &Ctx.Idents.get("NSObject");
  ObjCInterfaceDecl* ID = D->getClassInterface();
  
  for ( ; ID ; ID = ID->getSuperClass())
    if (ID->getIdentifier() == NSObjectII)
      break;
  
  if (!ID)
    return;
  
  // Get the "dealloc" selector.
  IdentifierInfo* II = &Ctx.Idents.get("dealloc");
  Selector S = Ctx.Selectors.getSelector(0, &II);
  
  ObjCMethodDecl* MD = 0;
  
  // Scan the instance methods for "dealloc".
  for (ObjCImplementationDecl::instmeth_iterator I = D->instmeth_begin(),
       E = D->instmeth_end(); I!=E; ++I) {
    
    if ((*I)->getSelector() == S) {
      MD = *I;
      break;
    }    
  }
  
  if (!MD) { // No dealloc found.
    
    // FIXME: This code should be reduced to three lines if possible (Refactor).
    SimpleBugType BT(LOpts.getGCMode() == LangOptions::NonGC 
                     ? "missing -dealloc" 
                     : "missing -dealloc (Hybrid MM, non-GC)");
    
    DiagCollector C(BT);
    
    std::ostringstream os;
    os << "Objective-C class '" << D->getName()
       << "' lacks a 'dealloc' instance method";
    
    Diagnostic& Diag = BR.getDiagnostic();    
    Diag.Report(&C,
                Ctx.getFullLoc(D->getLocStart()),
                Diag.getCustomDiagID(Diagnostic::Warning, os.str().c_str()),
                NULL, 0, NULL, 0);
        
    for (DiagCollector::iterator I = C.begin(), E = C.end(); I != E; ++I)
      BR.EmitWarning(*I);
    
    return;
  }
  
  // dealloc found.  Scan for missing [super dealloc].
  if (MD->getBody() && !scan_dealloc(MD->getBody(), S)) {
    
    // FIXME: This code should be reduced to three lines if possible (Refactor).
    SimpleBugType BT(LOpts.getGCMode() == LangOptions::NonGC
                     ? "missing [super dealloc]"
                     : "missing [super dealloc] (Hybrid MM, non-GC)");
                     
    DiagCollector C(BT);
    
    std::ostringstream os;
    os << "The 'dealloc' instance method in Objective-C class '" << D->getName()
       << "' does not send a 'dealloc' message to its super class"
           " (missing [super dealloc])";
    
    Diagnostic& Diag = BR.getDiagnostic();    
    Diag.Report(&C,
                Ctx.getFullLoc(MD->getLocStart()),
                Diag.getCustomDiagID(Diagnostic::Warning, os.str().c_str()),
                NULL, 0, NULL, 0);
    
    for (DiagCollector::iterator I = C.begin(), E = C.end(); I != E; ++I)
      BR.EmitWarning(*I);
  }    
}


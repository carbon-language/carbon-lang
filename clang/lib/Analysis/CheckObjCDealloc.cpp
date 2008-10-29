//==- CheckObjCDealloc.cpp - Check ObjC -dealloc implementation --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a CheckObjCDealloc, a checker that
//  analyzes an Objective-C class's implementation to determine if it
//  correctly implements -dealloc.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

static bool scan_dealloc(Stmt* S, Selector Dealloc) {  
  
  if (ObjCMessageExpr* ME = dyn_cast<ObjCMessageExpr>(S))
    if (ME->getSelector() == Dealloc)
      if (Expr* Receiver = ME->getReceiver()->IgnoreParenCasts())
        if (PredefinedExpr* E = dyn_cast<PredefinedExpr>(Receiver))
          if (E->getIdentType() == PredefinedExpr::ObjCSuper)
            return true;

  // Recurse to children.

  for (Stmt::child_iterator I = S->child_begin(), E= S->child_end(); I!=E; ++I)
    if (*I && scan_dealloc(*I, Dealloc))
      return true;
  
  return false;
}

static bool scan_ivar_release(Stmt* S, ObjCIvarDecl* ID, Selector Release ) {  
  if (ObjCMessageExpr* ME = dyn_cast<ObjCMessageExpr>(S))
    if (ME->getSelector() == Release)
      if (Expr* Receiver = ME->getReceiver()->IgnoreParenCasts())
        if (ObjCIvarRefExpr* E = dyn_cast<ObjCIvarRefExpr>(Receiver))
          if (E->getDecl() == ID)
            return true;

  // Recurse to children.
  for (Stmt::child_iterator I = S->child_begin(), E= S->child_end(); I!=E; ++I)
    if (*I && scan_ivar_release(*I, ID, Release))
      return true;

  return false;
}

void clang::CheckObjCDealloc(ObjCImplementationDecl* D,
                             const LangOptions& LOpts, BugReporter& BR) {

  assert (LOpts.getGCMode() != LangOptions::GCOnly);
  
  ASTContext& Ctx = BR.getContext();
  ObjCInterfaceDecl* ID = D->getClassInterface();
    
  // Does the class contain any ivars that are pointers (or id<...>)?
  // If not, skip the check entirely.
  // NOTE: This is motivated by PR 2517:
  //        http://llvm.org/bugs/show_bug.cgi?id=2517
  
  bool containsPointerIvar = false;
  
  for (ObjCInterfaceDecl::ivar_iterator I=ID->ivar_begin(), E=ID->ivar_end();
       I!=E; ++I) {
    
    ObjCIvarDecl* ID = *I;
    QualType T = ID->getType();
    
    if (!Ctx.isObjCObjectPointerType(T) ||
        ID->getAttr<IBOutletAttr>()) // Skip IBOutlets.
      continue;
    
    containsPointerIvar = true;
    break;
  }
  
  if (!containsPointerIvar)
    return;
  
  // Determine if the class subclasses NSObject.
  IdentifierInfo* NSObjectII = &Ctx.Idents.get("NSObject");
  
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
    
    const char* name = LOpts.getGCMode() == LangOptions::NonGC 
                       ? "missing -dealloc" 
                       : "missing -dealloc (Hybrid MM, non-GC)";
    
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "Objective-C class '" << D->getName()
       << "' lacks a 'dealloc' instance method";
    
    BR.EmitBasicReport(name, os.str().c_str(), D->getLocStart());
    return;
  }
  
  // dealloc found.  Scan for missing [super dealloc].
  if (MD->getBody() && !scan_dealloc(MD->getBody(), S)) {
    
    const char* name = LOpts.getGCMode() == LangOptions::NonGC
                       ? "missing [super dealloc]"
                       : "missing [super dealloc] (Hybrid MM, non-GC)";
    
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "The 'dealloc' instance method in Objective-C class '" << D->getName()
       << "' does not send a 'dealloc' message to its super class"
           " (missing [super dealloc])";
    
    BR.EmitBasicReport(name, os.str().c_str(), D->getLocStart());
    return;
  }   
  
  // Get the "release" selector.
  IdentifierInfo* RII = &Ctx.Idents.get("release");
  Selector RS = Ctx.Selectors.getSelector(0, &RII);  
  
  // Scan for missing and extra releases of ivars used by implementations
  // of synthesized properties
  for (ObjCImplementationDecl::propimpl_iterator I = D->propimpl_begin(),
       E = D->propimpl_end(); I!=E; ++I) {

    // We can only check the synthesized properties
    if((*I)->getPropertyImplementation() != ObjCPropertyImplDecl::Synthesize)
      continue;
    
    ObjCIvarDecl* ID = (*I)->getPropertyIvarDecl();
    if (!ID)
      continue;
    
    QualType T = ID->getType();
    if (!Ctx.isObjCObjectPointerType(T)) // Skip non-pointer ivars
      continue;

    const ObjCPropertyDecl* PD = (*I)->getPropertyDecl();
    if(!PD)
      continue;
    
    // ivars cannot be set via read-only properties, so we'll skip them
    if(PD->isReadOnly())
       continue;
              
    // ivar must be released if and only if the kind of setter was not 'assign'
    bool requiresRelease = PD->getSetterKind() != ObjCPropertyDecl::Assign;
    if(scan_ivar_release(MD->getBody(), ID, RS) != requiresRelease) {
      const char *name;
      const char* category = "Memory (Core Foundation/Objective-C)";
      
      std::string buf;
      llvm::raw_string_ostream os(buf);

      if(requiresRelease) {
        name = LOpts.getGCMode() == LangOptions::NonGC
               ? "missing ivar release (leak)"
               : "missing ivar release (Hybrid MM, non-GC)";
        
        os << "The '" << ID->getName()
           << "' instance variable was retained by a synthesized property but "
              "wasn't released in 'dealloc'";        
      } else {
        name = LOpts.getGCMode() == LangOptions::NonGC
               ? "extra ivar release (use-after-release)"
               : "extra ivar release (Hybrid MM, non-GC)";
        
        os << "The '" << ID->getName()
           << "' instance variable was not retained by a synthesized property "
              "but was released in 'dealloc'";
      }
      
      BR.EmitBasicReport(name, category,
                         os.str().c_str(), (*I)->getLocation());
    }
  }
}


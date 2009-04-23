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
      if(ME->getReceiver())
        if (Expr* Receiver = ME->getReceiver()->IgnoreParenCasts())
          return isa<ObjCSuperExpr>(Receiver);

  // Recurse to children.

  for (Stmt::child_iterator I = S->child_begin(), E= S->child_end(); I!=E; ++I)
    if (*I && scan_dealloc(*I, Dealloc))
      return true;
  
  return false;
}

static bool scan_ivar_release(Stmt* S, ObjCIvarDecl* ID, 
                              const ObjCPropertyDecl* PD, 
                              Selector Release, 
                              IdentifierInfo* SelfII,
                              ASTContext& Ctx) {  
  
  // [mMyIvar release]
  if (ObjCMessageExpr* ME = dyn_cast<ObjCMessageExpr>(S))
    if (ME->getSelector() == Release)
      if(ME->getReceiver())
        if (Expr* Receiver = ME->getReceiver()->IgnoreParenCasts())
          if (ObjCIvarRefExpr* E = dyn_cast<ObjCIvarRefExpr>(Receiver))
            if (E->getDecl() == ID)
              return true;

  // [self setMyIvar:nil];
  if (ObjCMessageExpr* ME = dyn_cast<ObjCMessageExpr>(S))
    if(ME->getReceiver())
      if (Expr* Receiver = ME->getReceiver()->IgnoreParenCasts())
        if (DeclRefExpr* E = dyn_cast<DeclRefExpr>(Receiver))
          if (E->getDecl()->getIdentifier() == SelfII)
            if (ME->getMethodDecl() == PD->getSetterMethodDecl() &&
                ME->getNumArgs() == 1 &&
                ME->getArg(0)->isNullPointerConstant(Ctx))
              return true;
  
  // self.myIvar = nil;
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(S))
    if (BO->isAssignmentOp())
      if(ObjCPropertyRefExpr* PRE = 
         dyn_cast<ObjCPropertyRefExpr>(BO->getLHS()->IgnoreParenCasts()))
          if(PRE->getProperty() == PD)
            if(BO->getRHS()->isNullPointerConstant(Ctx)) {
              // This is only a 'release' if the property kind is not
              // 'assign'.
              return PD->getSetterKind() != ObjCPropertyDecl::Assign;;
            }
  
  // Recurse to children.
  for (Stmt::child_iterator I = S->child_begin(), E= S->child_end(); I!=E; ++I)
    if (*I && scan_ivar_release(*I, ID, PD, Release, SelfII, Ctx))
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
  IdentifierInfo* SenTestCaseII = &Ctx.Idents.get("SenTestCase");

  
  for ( ; ID ; ID = ID->getSuperClass()) {
    IdentifierInfo *II = ID->getIdentifier();

    if (II == NSObjectII)
      break;

    // FIXME: For now, ignore classes that subclass SenTestCase, as these don't
    // need to implement -dealloc.  They implement tear down in another way,
    // which we should try and catch later.
    //  http://llvm.org/bugs/show_bug.cgi?id=3187
    if (II == SenTestCaseII)
      return;
  }
    
  if (!ID)
    return;
  
  // Get the "dealloc" selector.
  IdentifierInfo* II = &Ctx.Idents.get("dealloc");
  Selector S = Ctx.Selectors.getSelector(0, &II);  
  ObjCMethodDecl* MD = 0;
  
  // Scan the instance methods for "dealloc".
  for (ObjCImplementationDecl::instmeth_iterator I = D->instmeth_begin(Ctx),
       E = D->instmeth_end(Ctx); I!=E; ++I) {
    
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
    os << "Objective-C class '" << D->getNameAsString()
       << "' lacks a 'dealloc' instance method";
    
    BR.EmitBasicReport(name, os.str().c_str(), D->getLocStart());
    return;
  }
  
  // dealloc found.  Scan for missing [super dealloc].
  if (MD->getBody(Ctx) && !scan_dealloc(MD->getBody(Ctx), S)) {
    
    const char* name = LOpts.getGCMode() == LangOptions::NonGC
                       ? "missing [super dealloc]"
                       : "missing [super dealloc] (Hybrid MM, non-GC)";
    
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "The 'dealloc' instance method in Objective-C class '"
       << D->getNameAsString()
       << "' does not send a 'dealloc' message to its super class"
           " (missing [super dealloc])";
    
    BR.EmitBasicReport(name, os.str().c_str(), D->getLocStart());
    return;
  }   
  
  // Get the "release" selector.
  IdentifierInfo* RII = &Ctx.Idents.get("release");
  Selector RS = Ctx.Selectors.getSelector(0, &RII);  
  
  // Get the "self" identifier
  IdentifierInfo* SelfII = &Ctx.Idents.get("self");
  
  // Scan for missing and extra releases of ivars used by implementations
  // of synthesized properties
  for (ObjCImplementationDecl::propimpl_iterator I = D->propimpl_begin(Ctx),
       E = D->propimpl_end(Ctx); I!=E; ++I) {

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
    if(scan_ivar_release(MD->getBody(Ctx), ID, PD, RS, SelfII, Ctx) 
       != requiresRelease) {
      const char *name;
      const char* category = "Memory (Core Foundation/Objective-C)";
      
      std::string buf;
      llvm::raw_string_ostream os(buf);

      if(requiresRelease) {
        name = LOpts.getGCMode() == LangOptions::NonGC
               ? "missing ivar release (leak)"
               : "missing ivar release (Hybrid MM, non-GC)";
        
        os << "The '" << ID->getNameAsString()
           << "' instance variable was retained by a synthesized property but "
              "wasn't released in 'dealloc'";        
      } else {
        name = LOpts.getGCMode() == LangOptions::NonGC
               ? "extra ivar release (use-after-release)"
               : "extra ivar release (Hybrid MM, non-GC)";
        
        os << "The '" << ID->getNameAsString()
           << "' instance variable was not retained by a synthesized property "
              "but was released in 'dealloc'";
      }
      
      BR.EmitBasicReport(name, category,
                         os.str().c_str(), (*I)->getLocation());
    }
  }
}


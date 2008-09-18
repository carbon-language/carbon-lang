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
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "BasicObjCFoundationChecks.h"
#include "llvm/Support/Compiler.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN NSErrorCheck : public BugTypeCacheLocation {

  void EmitGRWarnings(GRBugReporter& BR);
  
  void CheckSignature(ObjCMethodDecl& MD, QualType& ResultTy,
                      llvm::SmallVectorImpl<VarDecl*>& Params,
                      IdentifierInfo* NSErrorII);

  bool CheckArgument(QualType ArgTy, IdentifierInfo* NSErrorII);
  
public:
  void EmitWarnings(BugReporter& BR) { EmitGRWarnings(cast<GRBugReporter>(BR));}
  const char* getName() const { return "NSError** null dereference"; }
};  
  
} // end anonymous namespace

BugType* clang::CreateNSErrorCheck() {
  return new NSErrorCheck();
}

void NSErrorCheck::EmitGRWarnings(GRBugReporter& BR) {
  // Get the analysis engine and the exploded analysis graph.
  GRExprEngine& Eng = BR.getEngine();
  GRExprEngine::GraphTy& G = Eng.getGraph();
  
  // Get the declaration of the method/function that was analyzed.
  Decl& CodeDecl = G.getCodeDecl();
  
  ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(&CodeDecl);
  if (!MD)
    return;
  
  // Get the ASTContext, which is useful for querying type information.
  ASTContext &Ctx = BR.getContext();

  QualType ResultTy;
  llvm::SmallVector<VarDecl*, 5> Params;  
  CheckSignature(*MD, ResultTy, Params, &Ctx.Idents.get("NSError"));
  
  if (Params.empty())
    return;
  
  if (ResultTy == Ctx.VoidTy) {
    BR.EmitBasicReport("Bad return type when passing NSError**",
              "Method accepting NSError** argument should have "
              "non-void return value to indicate that an error occurred.",
              CodeDecl.getLocation());
    
  }
}

void NSErrorCheck::CheckSignature(ObjCMethodDecl& M, QualType& ResultTy,
                                  llvm::SmallVectorImpl<VarDecl*>& Params,
                                  IdentifierInfo* NSErrorII) {

  ResultTy = M.getResultType();
  
  for (ObjCMethodDecl::param_iterator I=M.param_begin(), 
       E=M.param_end(); I!=E; ++I) 
    if (CheckArgument((*I)->getType(), NSErrorII))
      Params.push_back(*I);
}

bool NSErrorCheck::CheckArgument(QualType ArgTy, IdentifierInfo* NSErrorII) {
  const PointerType* PPT = ArgTy->getAsPointerType();
  if (!PPT) return false;
  
  const PointerType* PT = PPT->getPointeeType()->getAsPointerType();
  if (!PT) return false;
  
  const ObjCInterfaceType *IT =
  PT->getPointeeType()->getAsObjCInterfaceType();
  
  if (!IT) return false;
  return IT->getDecl()->getIdentifier() == NSErrorII;
}

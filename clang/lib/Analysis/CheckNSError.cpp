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
  
  void CheckParamDeref(VarDecl* V, GRStateRef state, GRExprEngine& Eng,
                       GRBugReporter& BR); 
  
  const char* desc;
public:
  NSErrorCheck() : desc(0) {}
  
  void EmitWarnings(BugReporter& BR) { EmitGRWarnings(cast<GRBugReporter>(BR));}
  const char* getName() const { return "NSError** null dereference"; }
  const char* getDescription() const { return desc; }
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
  
  // Scan the NSError** parameters for an implicit null dereference.
  for (llvm::SmallVectorImpl<VarDecl*>::iterator I=Params.begin(),
        E=Params.end(); I!=E; ++I)    
    for (GRExprEngine::GraphTy::roots_iterator RI=G.roots_begin(),
         RE=G.roots_end(); RI!=RE; ++RI)
      CheckParamDeref(*I, GRStateRef((*RI)->getState(), Eng.getStateManager()),
                      Eng, BR);
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

void NSErrorCheck::CheckParamDeref(VarDecl* Param, GRStateRef rootState,
                                   GRExprEngine& Eng, GRBugReporter& BR) {
  
  RVal ParamRVal = rootState.GetRVal(lval::DeclVal(Param));

  // FIXME: For now assume that ParamRVal is symbolic.  We need to generalize
  // this later.
  lval::SymbolVal* SV = dyn_cast<lval::SymbolVal>(&ParamRVal);
  if (!SV) return;
  
  // Iterate over the implicit-null dereferences.
  for (GRExprEngine::null_deref_iterator I=Eng.implicit_null_derefs_begin(),
       E=Eng.implicit_null_derefs_end(); I!=E; ++I) {
    
    GRStateRef state = GRStateRef((*I)->getState(), Eng.getStateManager());
    const RVal* X = state.get<GRState::NullDerefTag>();    
    const lval::SymbolVal* SVX = dyn_cast_or_null<lval::SymbolVal>(X);
    if (!SVX || SVX->getSymbol() != SV->getSymbol()) continue;

    // Emit an error.
    BugReport R(*this, *I);

    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "Potential null dereference.  According to coding standards in "
          "'Creating and Returning NSError Objects' the parameter '"
        << Param->getName() << "' may be null.";    
    desc = os.str().c_str();

    BR.addNotableSymbol(SV->getSymbol());
    BR.EmitWarning(R);    
  }
}

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
                      llvm::SmallVectorImpl<VarDecl*>& NSErrorParams,
                      llvm::SmallVectorImpl<VarDecl*>& CFErrorParams,
                      IdentifierInfo* NSErrorII,
                      IdentifierInfo* CFErrorII);
  
  void CheckSignature(FunctionDecl& MD, QualType& ResultTy,
                      llvm::SmallVectorImpl<VarDecl*>& NSErrorParams,
                      llvm::SmallVectorImpl<VarDecl*>& CFErrorParams,
                      IdentifierInfo* NSErrorII,
                      IdentifierInfo* CFErrorII);

  bool CheckNSErrorArgument(QualType ArgTy, IdentifierInfo* NSErrorII);
  bool CheckCFErrorArgument(QualType ArgTy, IdentifierInfo* CFErrorII);
  
  void CheckParamDeref(VarDecl* V, GRStateRef state, GRExprEngine& Eng,
                       GRBugReporter& BR, bool isNErrorWarning); 
  
  void EmitRetTyWarning(BugReporter& BR, Decl& CodeDecl, bool isNSErrorWarning);
  
  const char* desc;
  const char* name;
public:
  NSErrorCheck() : desc(0) {}
  
  void EmitWarnings(BugReporter& BR) { EmitGRWarnings(cast<GRBugReporter>(BR));}
  const char* getName() const { return name; }
  const char* getDescription() const { return desc; }
  const char* getCategory() const { return "Coding Conventions (Apple)"; }
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
    
  // Get the ASTContext, which is useful for querying type information.
  ASTContext &Ctx = BR.getContext();

  QualType ResultTy;
  llvm::SmallVector<VarDecl*, 5> NSErrorParams;
  llvm::SmallVector<VarDecl*, 5> CFErrorParams;

  if (ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(&CodeDecl))
    CheckSignature(*MD, ResultTy, NSErrorParams, CFErrorParams,
                   &Ctx.Idents.get("NSError"), &Ctx.Idents.get("CFErrorRef"));
  else if (FunctionDecl* FD = dyn_cast<FunctionDecl>(&CodeDecl))
    CheckSignature(*FD, ResultTy, NSErrorParams, CFErrorParams,
                   &Ctx.Idents.get("NSError"), &Ctx.Idents.get("CFErrorRef"));
  else
    return;
  
  if (NSErrorParams.empty() && CFErrorParams.empty())
    return;
  
  if (ResultTy == Ctx.VoidTy) {    
    if (!NSErrorParams.empty())
      EmitRetTyWarning(BR, CodeDecl, true);
    if (!CFErrorParams.empty())
      EmitRetTyWarning(BR, CodeDecl, false);
  }
  
  for (GRExprEngine::GraphTy::roots_iterator RI=G.roots_begin(),
       RE=G.roots_end(); RI!=RE; ++RI) {

    // Scan the NSError** parameters for an implicit null dereference.
    for (llvm::SmallVectorImpl<VarDecl*>::iterator I=NSErrorParams.begin(),
          E=NSErrorParams.end(); I!=E; ++I)    
        CheckParamDeref(*I, GRStateRef((*RI)->getState(), Eng.getStateManager()),
                        Eng, BR, true);

    // Scan the CFErrorRef* parameters for an implicit null dereference.
    for (llvm::SmallVectorImpl<VarDecl*>::iterator I=CFErrorParams.begin(),
         E=CFErrorParams.end(); I!=E; ++I)    
      CheckParamDeref(*I, GRStateRef((*RI)->getState(), Eng.getStateManager()),
                      Eng, BR, false);
  }
}

void NSErrorCheck::EmitRetTyWarning(BugReporter& BR, Decl& CodeDecl,
                                    bool isNSErrorWarning) {

  std::string msg;
  llvm::raw_string_ostream os(msg);
  
  if (isa<ObjCMethodDecl>(CodeDecl))
    os << "Method";
  else
    os << "Function";      
  
  os << " accepting ";
  os << (isNSErrorWarning ? "NSError**" : "CFErrorRef*");
  os << " should have a non-void return value to indicate whether or not an "
        "error occured.";
  
  BR.EmitBasicReport(isNSErrorWarning
                     ? "Bad return type when passing NSError**"
                     : "Bad return type when passing CFError*",
                     getCategory(), os.str().c_str(), CodeDecl.getLocation());
}

void
NSErrorCheck::CheckSignature(ObjCMethodDecl& M, QualType& ResultTy,
                             llvm::SmallVectorImpl<VarDecl*>& NSErrorParams,
                             llvm::SmallVectorImpl<VarDecl*>& CFErrorParams,
                             IdentifierInfo* NSErrorII,
                             IdentifierInfo* CFErrorII) {

  ResultTy = M.getResultType();
  
  for (ObjCMethodDecl::param_iterator I=M.param_begin(), 
       E=M.param_end(); I!=E; ++I)  {

    QualType T = (*I)->getType();    

    if (CheckNSErrorArgument(T, NSErrorII))
      NSErrorParams.push_back(*I);
    else if (CheckCFErrorArgument(T, CFErrorII))
      CFErrorParams.push_back(*I);
  }
}

void
NSErrorCheck::CheckSignature(FunctionDecl& F, QualType& ResultTy,
                             llvm::SmallVectorImpl<VarDecl*>& NSErrorParams,
                             llvm::SmallVectorImpl<VarDecl*>& CFErrorParams,
                             IdentifierInfo* NSErrorII,
                             IdentifierInfo* CFErrorII) {
  
  ResultTy = F.getResultType();
  
  for (FunctionDecl::param_iterator I=F.param_begin(), 
       E=F.param_end(); I!=E; ++I)  {
    
    QualType T = (*I)->getType();    

    if (CheckNSErrorArgument(T, NSErrorII))
      NSErrorParams.push_back(*I);
    else if (CheckCFErrorArgument(T, CFErrorII))
      CFErrorParams.push_back(*I);
  }
}


bool NSErrorCheck::CheckNSErrorArgument(QualType ArgTy,
                                        IdentifierInfo* NSErrorII) {
  
  const PointerType* PPT = ArgTy->getAsPointerType();
  if (!PPT) return false;
  
  const PointerType* PT = PPT->getPointeeType()->getAsPointerType();
  if (!PT) return false;
  
  const ObjCInterfaceType *IT =
  PT->getPointeeType()->getAsObjCInterfaceType();
  
  if (!IT) return false;
  return IT->getDecl()->getIdentifier() == NSErrorII;
}

bool NSErrorCheck::CheckCFErrorArgument(QualType ArgTy,
                                        IdentifierInfo* CFErrorII) {
  
  const PointerType* PPT = ArgTy->getAsPointerType();
  if (!PPT) return false;
  
  const TypedefType* TT = PPT->getPointeeType()->getAsTypedefType();
  if (!TT) return false;

  return TT->getDecl()->getIdentifier() == CFErrorII;
}

void NSErrorCheck::CheckParamDeref(VarDecl* Param, GRStateRef rootState,
                                   GRExprEngine& Eng, GRBugReporter& BR,
                                   bool isNSErrorWarning) {
  
  RVal ParamRVal = rootState.GetLValue(Param);

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
    
    name = isNSErrorWarning ? "NSError** null dereference" 
                            : "CFErrorRef* null dereference";

    std::string msg;
    llvm::raw_string_ostream os(msg);
      os << "Potential null dereference.  According to coding standards ";
    
    if (isNSErrorWarning)
      os << "in 'Creating and Returning NSError Objects' the parameter '";
    else
      os << "documented in CoreFoundation/CFError.h the parameter '";
    
    os << Param->getName() << "' may be null.";
    desc = os.str().c_str();

    BR.addNotableSymbol(SV->getSymbol());
    BR.EmitWarning(R);
  }
}

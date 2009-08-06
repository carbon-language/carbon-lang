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
class VISIBILITY_HIDDEN NSErrorCheck : public BugType {
  const bool isNSErrorWarning;
  IdentifierInfo * const II;
  GRExprEngine &Eng;
  
  void CheckSignature(ObjCMethodDecl& MD, QualType& ResultTy,
                      llvm::SmallVectorImpl<VarDecl*>& ErrorParams);
  
  void CheckSignature(FunctionDecl& MD, QualType& ResultTy,
                      llvm::SmallVectorImpl<VarDecl*>& ErrorParams);

  bool CheckNSErrorArgument(QualType ArgTy);
  bool CheckCFErrorArgument(QualType ArgTy);
  
  void CheckParamDeref(VarDecl* V, const GRState *state, BugReporter& BR);
  
  void EmitRetTyWarning(BugReporter& BR, Decl& CodeDecl);
  
public:
  NSErrorCheck(bool isNSError, GRExprEngine& eng)
  : BugType(isNSError ? "NSError** null dereference" 
                      : "CFErrorRef* null dereference",
            "Coding Conventions (Apple)"),
    isNSErrorWarning(isNSError), 
    II(&eng.getContext().Idents.get(isNSErrorWarning ? "NSError":"CFErrorRef")),
    Eng(eng) {}
  
  void FlushReports(BugReporter& BR);
};  
  
} // end anonymous namespace

void clang::RegisterNSErrorChecks(BugReporter& BR, GRExprEngine &Eng) {
  BR.Register(new NSErrorCheck(true, Eng));
  BR.Register(new NSErrorCheck(false, Eng));
}

void NSErrorCheck::FlushReports(BugReporter& BR) {
  // Get the analysis engine and the exploded analysis graph.
  GRExprEngine::GraphTy& G = Eng.getGraph();
  
  // Get the declaration of the method/function that was analyzed.
  Decl& CodeDecl = G.getCodeDecl();
    
  // Get the ASTContext, which is useful for querying type information.
  ASTContext &Ctx = BR.getContext();

  QualType ResultTy;
  llvm::SmallVector<VarDecl*, 5> ErrorParams;

  if (ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(&CodeDecl))
    CheckSignature(*MD, ResultTy, ErrorParams);
  else if (FunctionDecl* FD = dyn_cast<FunctionDecl>(&CodeDecl))
    CheckSignature(*FD, ResultTy, ErrorParams);
  else
    return;
  
  if (ErrorParams.empty())
    return;
  
  if (ResultTy == Ctx.VoidTy) EmitRetTyWarning(BR, CodeDecl);
  
  for (GRExprEngine::GraphTy::roots_iterator RI=G.roots_begin(),
       RE=G.roots_end(); RI!=RE; ++RI) {
    // Scan the parameters for an implicit null dereference.
    for (llvm::SmallVectorImpl<VarDecl*>::iterator I=ErrorParams.begin(),
          E=ErrorParams.end(); I!=E; ++I)    
        CheckParamDeref(*I, (*RI)->getState(), BR);

  }
}

void NSErrorCheck::EmitRetTyWarning(BugReporter& BR, Decl& CodeDecl) {
  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);
  
  if (isa<ObjCMethodDecl>(CodeDecl))
    os << "Method";
  else
    os << "Function";      
  
  os << " accepting ";
  os << (isNSErrorWarning ? "NSError**" : "CFErrorRef*");
  os << " should have a non-void return value to indicate whether or not an "
        "error occurred";
  
  BR.EmitBasicReport(isNSErrorWarning
                     ? "Bad return type when passing NSError**"
                     : "Bad return type when passing CFError*",
                     getCategory().c_str(), os.str().c_str(),
                     CodeDecl.getLocation());
}

void
NSErrorCheck::CheckSignature(ObjCMethodDecl& M, QualType& ResultTy,
                             llvm::SmallVectorImpl<VarDecl*>& ErrorParams) {

  ResultTy = M.getResultType();
  
  for (ObjCMethodDecl::param_iterator I=M.param_begin(), 
       E=M.param_end(); I!=E; ++I)  {

    QualType T = (*I)->getType();    

    if (isNSErrorWarning) {
      if (CheckNSErrorArgument(T)) ErrorParams.push_back(*I);
    }
    else if (CheckCFErrorArgument(T))
      ErrorParams.push_back(*I);
  }
}

void
NSErrorCheck::CheckSignature(FunctionDecl& F, QualType& ResultTy,
                             llvm::SmallVectorImpl<VarDecl*>& ErrorParams) {
  
  ResultTy = F.getResultType();
  
  for (FunctionDecl::param_iterator I=F.param_begin(), 
       E=F.param_end(); I!=E; ++I)  {
    
    QualType T = (*I)->getType();    
    
    if (isNSErrorWarning) {
      if (CheckNSErrorArgument(T)) ErrorParams.push_back(*I);
    }
    else if (CheckCFErrorArgument(T))
      ErrorParams.push_back(*I);
  }
}


bool NSErrorCheck::CheckNSErrorArgument(QualType ArgTy) {
  
  const PointerType* PPT = ArgTy->getAs<PointerType>();
  if (!PPT)
    return false;
  
  const ObjCObjectPointerType* PT =
    PPT->getPointeeType()->getAsObjCObjectPointerType();

  if (!PT)
    return false;
  
  const ObjCInterfaceDecl *ID = PT->getInterfaceDecl();
  
  // FIXME: Can ID ever be NULL?
  if (ID)
    return II == ID->getIdentifier();
  
  return false;
}

bool NSErrorCheck::CheckCFErrorArgument(QualType ArgTy) {
  
  const PointerType* PPT = ArgTy->getAs<PointerType>();
  if (!PPT) return false;
  
  const TypedefType* TT = PPT->getPointeeType()->getAsTypedefType();
  if (!TT) return false;

  return TT->getDecl()->getIdentifier() == II;
}

void NSErrorCheck::CheckParamDeref(VarDecl* Param, const GRState *rootState,
                                   BugReporter& BR) {
  
  SVal ParamL = rootState->getLValue(Param);
  const MemRegion* ParamR = cast<loc::MemRegionVal>(ParamL).getRegionAs<VarRegion>();
  assert (ParamR && "Parameters always have VarRegions.");
  SVal ParamSVal = rootState->getSVal(ParamR);
  
  // FIXME: For now assume that ParamSVal is symbolic.  We need to generalize
  // this later.
  SymbolRef ParamSym = ParamSVal.getAsLocSymbol();
  if (!ParamSym)
    return;
  
  // Iterate over the implicit-null dereferences.
  for (GRExprEngine::null_deref_iterator I=Eng.implicit_null_derefs_begin(),
       E=Eng.implicit_null_derefs_end(); I!=E; ++I) {
    
    const GRState *state = (*I)->getState();
    const SVal* X = state->get<GRState::NullDerefTag>();    

    if (!X || X->getAsSymbol() != ParamSym)
      continue;

    // Emit an error.
    std::string sbuf;
    llvm::raw_string_ostream os(sbuf);
      os << "Potential null dereference.  According to coding standards ";
    
    if (isNSErrorWarning)
      os << "in 'Creating and Returning NSError Objects' the parameter '";
    else
      os << "documented in CoreFoundation/CFError.h the parameter '";
    
    os << Param->getNameAsString() << "' may be null.";
    
    BugReport *report = new BugReport(*this, os.str().c_str(), *I);
    // FIXME: Notable symbols are now part of the report.  We should
    //  add support for notable symbols in BugReport.
    //    BR.addNotableSymbol(SV->getSymbol());
    BR.EmitReport(report);
  }
}

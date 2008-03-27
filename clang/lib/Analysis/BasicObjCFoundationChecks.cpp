//== BasicObjCFoundationChecks.cpp - Simple Apple-Foundation checks -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BasicObjCFoundationChecks, a class that encapsulates
//  a set of simple checks to run on Objective-C code using Apple's Foundation
//  classes.
//
//===----------------------------------------------------------------------===//

#include "BasicObjCFoundationChecks.h"

#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "clang/Analysis/PathSensitive/GRSimpleAPICheck.h"
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Analysis/PathSensitive/AnnotatedPath.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Compiler.h"

#include <vector>

using namespace clang;
  
namespace {
  
class VISIBILITY_HIDDEN BasicObjCFoundationChecks : public GRSimpleAPICheck {

  ASTContext &Ctx;
  ValueStateManager* VMgr;
  
  typedef std::list<AnnotatedPath<ValueState> > ErrorsTy;
  ErrorsTy Errors;
      
  RVal GetRVal(ValueState* St, Expr* E) { return VMgr->GetRVal(St, E); }
      
  bool isNSString(ObjCInterfaceType* T, const char* suffix);
  bool AuditNSString(NodeTy* N, ObjCMessageExpr* ME);
      
  void Warn(NodeTy* N, Expr* E, const char *msg);

public:
  BasicObjCFoundationChecks(ASTContext& ctx, ValueStateManager* vmgr) 
    : Ctx(ctx), VMgr(vmgr) {}
      
  virtual ~BasicObjCFoundationChecks() {}
  
  virtual bool Audit(ExplodedNode<ValueState>* N);
  
  virtual void ReportResults(Diagnostic& D);

};
  
} // end anonymous namespace


GRSimpleAPICheck*
clang::CreateBasicObjCFoundationChecks(ASTContext& Ctx,
                                       ValueStateManager* VMgr) {
  
  return new BasicObjCFoundationChecks(Ctx, VMgr);  
}


bool BasicObjCFoundationChecks::Audit(ExplodedNode<ValueState>* N) {
  
  ObjCMessageExpr* ME =
    cast<ObjCMessageExpr>(cast<PostStmt>(N->getLocation()).getStmt());
  
  Expr* Receiver = ME->getReceiver();
  
  if (!Receiver)
    return false;
  
  assert (Receiver->getType()->isPointerType());

  const PointerType* T = Receiver->getType()->getAsPointerType();

  ObjCInterfaceType* ReceiverType =
    dyn_cast<ObjCInterfaceType>(T->getPointeeType().getTypePtr());
  
  if (!ReceiverType)
    return false;
  
  const char* name = ReceiverType->getDecl()->getIdentifier()->getName();  

  if (name[0] != 'N' || name[1] != 'S')
    return false;
      
  name += 2;
  
  // FIXME: Make all of this faster.
  
  if (isNSString(ReceiverType, name))
    return AuditNSString(N, ME);

  return false;  
}

static inline bool isNil(RVal X) {
  return isa<lval::ConcreteInt>(X);  
}

//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//


void BasicObjCFoundationChecks::Warn(NodeTy* N,
                                              Expr* E, const char *msg) {
  
  Errors.push_back(AnnotatedPath<ValueState>());
  Errors.back().push_back(N, msg, E);
}

void BasicObjCFoundationChecks::ReportResults(Diagnostic& D) {
  
  // FIXME: Expand errors into paths.  For now, just issue warnings.
  
  for (ErrorsTy::iterator I=Errors.begin(), E=Errors.end(); I!=E; ++I) {
      
    AnnotatedNode<ValueState>& AN = I->back();
    
    unsigned diag = D.getCustomDiagID(Diagnostic::Warning,
                                      AN.getString().c_str());
    
    Stmt* S = cast<PostStmt>(AN.getNode()->getLocation()).getStmt();
    FullSourceLoc L(S->getLocStart(), Ctx.getSourceManager());
    
    SourceRange R = AN.getExpr()->getSourceRange();
    
    D.Report(L, diag, &AN.getString(), 1, &R, 1);
  }
}

//===----------------------------------------------------------------------===//
// NSString checking.
//===----------------------------------------------------------------------===//

bool BasicObjCFoundationChecks::isNSString(ObjCInterfaceType* T,
                                           const char* suffix) {
  
  return !strcmp("String", suffix) || !strcmp("MutableString", suffix);
}

bool BasicObjCFoundationChecks::AuditNSString(NodeTy* N, 
                                              ObjCMessageExpr* ME) {
  
  Selector S = ME->getSelector();
  
  if (S.isUnarySelector())
    return false;

  // FIXME: This is going to be really slow doing these checks with
  //  lexical comparisons.
  
  std::string name = S.getName();
  assert (!name.empty());
  const char* cstr = &name[0];
  unsigned len = name.size();
  
  
  ValueState* St = N->getState();
  
  switch (len) {
    default:
      break;
    case 8:
      if (!strcmp(cstr, "compare:")) {
        // Check if the compared NSString is nil.
        Expr * E = ME->getArg(0);
    
        if (isNil(GetRVal(St, E))) {
          Warn(N, E, "Argument to NSString method 'compare:' cannot be nil.");
          return false;
        }
        
        break;
      }
      
      break;
  }
  
  return false;
}

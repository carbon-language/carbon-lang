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
 std::list<AnnotatedPath<ValueState> > Errors;
      
 RVal GetRVal(ValueState* St, Expr* E) { return VMgr->GetRVal(St, E); }
      
 bool isNSString(ObjCInterfaceType* T, const char* suffix);
 bool AuditNSString(NodeTy* N, ObjCMessageExpr* ME);
      
 void RegisterError(NodeTy* N, Expr* E, const char *msg);

public:
  BasicObjCFoundationChecks(ASTContext& ctx, ValueStateManager* vmgr) 
    : Ctx(ctx), VMgr(vmgr) {}
      
  virtual ~BasicObjCFoundationChecks() {}
  
  virtual bool Audit(ExplodedNode<ValueState>* N);
};
  
} // end anonymous namespace


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

//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//


void BasicObjCFoundationChecks::RegisterError(NodeTy* N,
                                              Expr* E, const char *msg) {
  
  Errors.push_back(AnnotatedPath<ValueState>());
  Errors.back().push_back(N, msg, E);
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
  ValueState* St = N->getState();
  
  if (name == "compare:") {
    // Check if the compared NSString is nil.
    Expr * E = ME->getArg(0);
    RVal X = GetRVal(St, E);
    
    if (isa<lval::ConcreteInt>(X)) {
      RegisterError(N, E,
                    "Argument to NSString method 'compare:' cannot be nil.");
    }
  }
  
  return false;
}

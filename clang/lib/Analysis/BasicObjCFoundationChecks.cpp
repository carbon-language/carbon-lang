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
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Compiler.h"

#include <vector>
#include <sstream>

using namespace clang;

static ObjCInterfaceType* GetReceiverType(ObjCMessageExpr* ME) {
  Expr* Receiver = ME->getReceiver();
  
  if (!Receiver)
    return NULL;
  
  QualType X = Receiver->getType();
  
  if (X->isPointerType()) {
    Type* TP = X.getTypePtr();
    const PointerType* T = TP->getAsPointerType();    
    return dyn_cast<ObjCInterfaceType>(T->getPointeeType().getTypePtr());
  }

  // FIXME: Support ObjCQualifiedIdType?
  return NULL;
}

static const char* GetReceiverNameType(ObjCMessageExpr* ME) {
  ObjCInterfaceType* ReceiverType = GetReceiverType(ME);
  return ReceiverType ? ReceiverType->getDecl()->getIdentifier()->getName()
                      : NULL;
}

namespace {
  
class VISIBILITY_HIDDEN NilArg : public BugTypeCacheLocation {
public:
  virtual ~NilArg() {}
  
  virtual const char* getName() const {
    return "nil argument";
  }
  
  class Report : public BugReport {
    std::string Msg;
    const char* s;
    SourceRange R;
  public:
    
    Report(NilArg& Desc, ExplodedNode<ValueState>* N, 
           ObjCMessageExpr* ME, unsigned Arg)
      : BugReport(Desc, N) {
      
      Expr* E = ME->getArg(Arg);
      R = E->getSourceRange();
      
      std::ostringstream os;
      
      os << "Argument to '" << GetReceiverNameType(ME) << "' method '"
      << ME->getSelector().getName() << "' cannot be nil.";
      
      Msg = os.str();
      s = Msg.c_str();      
    }
    
    virtual ~Report() {}
    
    virtual const char* getDescription() const { return s; }
    
    virtual void getRanges(BugReporter& BR,
                           const SourceRange*& B, const SourceRange*& E) {
      B = &R;
      E = B+1;
    }
  };  
};

  
class VISIBILITY_HIDDEN BasicObjCFoundationChecks : public GRSimpleAPICheck {
  NilArg Desc;
  ASTContext &Ctx;
  ValueStateManager* VMgr;
  
  typedef std::vector<BugReport*> ErrorsTy;
  ErrorsTy Errors;
      
  RVal GetRVal(ValueState* St, Expr* E) { return VMgr->GetRVal(St, E); }
      
  bool isNSString(ObjCInterfaceType* T, const char* suffix);
  bool AuditNSString(NodeTy* N, ObjCMessageExpr* ME);
      
  void Warn(NodeTy* N, Expr* E, const std::string& s);  
  void WarnNilArg(NodeTy* N, Expr* E);
  
  bool CheckNilArg(NodeTy* N, unsigned Arg);

public:
  BasicObjCFoundationChecks(ASTContext& ctx, ValueStateManager* vmgr) 
    : Ctx(ctx), VMgr(vmgr) {}
      
  virtual ~BasicObjCFoundationChecks() {
    for (ErrorsTy::iterator I = Errors.begin(), E = Errors.end(); I!=E; ++I)
      delete *I;    
  }
  
  virtual bool Audit(ExplodedNode<ValueState>* N);
  
  virtual void EmitWarnings(BugReporter& BR);
  
private:
  
  void AddError(BugReport* R) {
    Errors.push_back(R);
  }
  
  void WarnNilArg(NodeTy* N, ObjCMessageExpr* ME, unsigned Arg) {
    AddError(new NilArg::Report(Desc, N, ME, Arg));
  }
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

  ObjCInterfaceType* ReceiverType = GetReceiverType(ME);
  
  if (!ReceiverType)
    return false;
  
  const char* name = ReceiverType->getDecl()->getIdentifier()->getName();
  
  if (!name)
    return false;

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


void BasicObjCFoundationChecks::EmitWarnings(BugReporter& BR) {    
                 
  for (ErrorsTy::iterator I=Errors.begin(), E=Errors.end(); I!=E; ++I)    
    BR.EmitWarning(**I);
}

bool BasicObjCFoundationChecks::CheckNilArg(NodeTy* N, unsigned Arg) {
  ObjCMessageExpr* ME =
    cast<ObjCMessageExpr>(cast<PostStmt>(N->getLocation()).getStmt());
  
  Expr * E = ME->getArg(Arg);
  
  if (isNil(GetRVal(N->getState(), E))) {
    WarnNilArg(N, ME, Arg);
    return true;
  }
  
  return false;
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
      
  switch (len) {
    default:
      break;
    case 8:      
      if (!strcmp(cstr, "compare:"))
        return CheckNilArg(N, 0);
              
      break;
      
    case 15:
      // FIXME: Checking for initWithFormat: will not work in most cases
      //  yet because [NSString alloc] returns id, not NSString*.  We will
      //  need support for tracking expected-type information in the analyzer
      //  to find these errors.
      if (!strcmp(cstr, "initWithFormat:"))
        return CheckNilArg(N, 0);
      
      break;
    
    case 16:
      if (!strcmp(cstr, "compare:options:"))
        return CheckNilArg(N, 0);
      
      break;
      
    case 22:
      if (!strcmp(cstr, "compare:options:range:"))
        return CheckNilArg(N, 0);
      
      break;
      
    case 23:
      
      if (!strcmp(cstr, "caseInsensitiveCompare:"))
        return CheckNilArg(N, 0);
      
      break;

    case 29:
      if (!strcmp(cstr, "compare:options:range:locale:"))
        return CheckNilArg(N, 0);
    
      break;    
      
    case 37:
    if (!strcmp(cstr, "componentsSeparatedByCharactersInSet:"))
      return CheckNilArg(N, 0);
    
    break;    
  }
  
  return false;
}

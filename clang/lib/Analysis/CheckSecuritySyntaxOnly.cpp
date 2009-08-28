//==- CheckSecuritySyntaxOnly.cpp - Basic security checks --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of flow-insensitive security checks.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN WalkAST : public StmtVisitor<WalkAST> {
  BugReporter &BR;  
  IdentifierInfo *II_gets;
  enum { num_ids = 6 };
  IdentifierInfo *II_setid[num_ids];
  
public:
  WalkAST(BugReporter &br) : BR(br),
    II_gets(0), II_setid() {}
  
  // Statement visitor methods.
  void VisitCallExpr(CallExpr *CE);
  void VisitForStmt(ForStmt *S);
  void VisitCompoundStmt (CompoundStmt *S);
  void VisitStmt(Stmt *S) { VisitChildren(S); }

  void VisitChildren(Stmt *S);
  
  // Helpers.
  IdentifierInfo *GetIdentifier(IdentifierInfo *& II, const char *str);
    
  // Checker-specific methods.
  void CheckLoopConditionForFloat(const ForStmt *FS);
  void CheckCall_gets(const CallExpr *CE, const FunctionDecl *FD);
  void CheckUncheckedReturnValue(CallExpr *CE);
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

IdentifierInfo *WalkAST::GetIdentifier(IdentifierInfo *& II, const char *str) {
  if (!II)
    II = &BR.getContext().Idents.get(str);
  
  return II;  
}

//===----------------------------------------------------------------------===//
// AST walking.
//===----------------------------------------------------------------------===//

void WalkAST::VisitChildren(Stmt *S) {
  for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (Stmt *child = *I)
      Visit(child);
}

void WalkAST::VisitCallExpr(CallExpr *CE) {
  if (const FunctionDecl *FD = CE->getDirectCallee()) {
    CheckCall_gets(CE, FD);    
  }
  
  // Recurse and check children.
  VisitChildren(CE);
}

void WalkAST::VisitCompoundStmt(CompoundStmt *S) {
  for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (Stmt *child = *I)
      {
	if (CallExpr *CE = dyn_cast<CallExpr>(child))
	  CheckUncheckedReturnValue(CE);
	Visit(child);
      }
}

void WalkAST::VisitForStmt(ForStmt *FS) {
  CheckLoopConditionForFloat(FS);  

  // Recurse and check children.
  VisitChildren(FS);
}

//===----------------------------------------------------------------------===//
// Check: floating poing variable used as loop counter.
// Originally: <rdar://problem/6336718>
// Implements: CERT security coding advisory FLP-30.
//===----------------------------------------------------------------------===//

static const DeclRefExpr*
GetIncrementedVar(const Expr *expr, const VarDecl *x, const VarDecl *y) {
  expr = expr->IgnoreParenCasts();
  
  if (const BinaryOperator *B = dyn_cast<BinaryOperator>(expr)) {      
    if (!(B->isAssignmentOp() || B->isCompoundAssignmentOp() ||
          B->getOpcode() == BinaryOperator::Comma))
      return NULL;
      
    if (const DeclRefExpr *lhs = GetIncrementedVar(B->getLHS(), x, y))
      return lhs;
      
    if (const DeclRefExpr *rhs = GetIncrementedVar(B->getRHS(), x, y))
      return rhs;
      
    return NULL;
  }
    
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(expr)) {
    const NamedDecl *ND = DR->getDecl();
    return ND == x || ND == y ? DR : NULL;
  }
   
  if (const UnaryOperator *U = dyn_cast<UnaryOperator>(expr))
    return U->isIncrementDecrementOp()
      ? GetIncrementedVar(U->getSubExpr(), x, y) : NULL;

  return NULL;
}

/// CheckLoopConditionForFloat - This check looks for 'for' statements that
///  use a floating point variable as a loop counter.
///  CERT: FLP30-C, FLP30-CPP.
///
void WalkAST::CheckLoopConditionForFloat(const ForStmt *FS) {
  // Does the loop have a condition?
  const Expr *condition = FS->getCond();
  
  if (!condition)
    return;

  // Does the loop have an increment?
  const Expr *increment = FS->getInc();
  
  if (!increment)
    return;
    
  // Strip away '()' and casts.
  condition = condition->IgnoreParenCasts();
  increment = increment->IgnoreParenCasts();
  
  // Is the loop condition a comparison?
  const BinaryOperator *B = dyn_cast<BinaryOperator>(condition);

  if (!B)
    return;
  
  // Is this a comparison?
  if (!(B->isRelationalOp() || B->isEqualityOp()))
    return;
      
  // Are we comparing variables?
  const DeclRefExpr *drLHS = dyn_cast<DeclRefExpr>(B->getLHS()->IgnoreParens());
  const DeclRefExpr *drRHS = dyn_cast<DeclRefExpr>(B->getRHS()->IgnoreParens());
  
  // Does at least one of the variables have a floating point type?
  drLHS = drLHS && drLHS->getType()->isFloatingType() ? drLHS : NULL;
  drRHS = drRHS && drRHS->getType()->isFloatingType() ? drRHS : NULL;
  
  if (!drLHS && !drRHS)
    return;

  const VarDecl *vdLHS = drLHS ? dyn_cast<VarDecl>(drLHS->getDecl()) : NULL;
  const VarDecl *vdRHS = drRHS ? dyn_cast<VarDecl>(drRHS->getDecl()) : NULL;
  
  if (!vdLHS && !vdRHS)
    return;  
  
  // Does either variable appear in increment?
  const DeclRefExpr *drInc = GetIncrementedVar(increment, vdLHS, vdRHS);
  
  if (!drInc)
    return;
  
  // Emit the error.  First figure out which DeclRefExpr in the condition
  // referenced the compared variable.
  const DeclRefExpr *drCond = vdLHS == drInc->getDecl() ? drLHS : drRHS;

  llvm::SmallVector<SourceRange, 2> ranges;  
  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);
  
  os << "Variable '" << drCond->getDecl()->getNameAsCString()
     << "' with floating point type '" << drCond->getType().getAsString()
     << "' should not be used as a loop counter";

  ranges.push_back(drCond->getSourceRange());
  ranges.push_back(drInc->getSourceRange());
  
  const char *bugType = "Floating point variable used as loop counter";
  BR.EmitBasicReport(bugType, "Security", os.str().c_str(),
                     FS->getLocStart(), ranges.data(), ranges.size());
}

//===----------------------------------------------------------------------===//
// Check: Any use of 'gets' is insecure.
// Originally: <rdar://problem/6335715>
// Implements (part of): 300-BSI (buildsecurityin.us-cert.gov)
//===----------------------------------------------------------------------===//

void WalkAST::CheckCall_gets(const CallExpr *CE, const FunctionDecl *FD) {
  if (FD->getIdentifier() != GetIdentifier(II_gets, "gets"))
    return;
  
  const FunctionProtoType *FTP = dyn_cast<FunctionProtoType>(FD->getType());
  if (!FTP)
    return;
  
  // Verify that the function takes a single argument.
  if (FTP->getNumArgs() != 1)
    return;

  // Is the argument a 'char*'?
  const PointerType *PT = dyn_cast<PointerType>(FTP->getArgType(0));
  if (!PT)
    return;
  
  if (PT->getPointeeType().getUnqualifiedType() != BR.getContext().CharTy)
    return;
  
  // Issue a warning.
  SourceRange R = CE->getCallee()->getSourceRange();
  BR.EmitBasicReport("Potential buffer overflow in call to 'gets'",
                     "Security",
                     "Call to function 'gets' is extremely insecure as it can "
                     "always result in a buffer overflow",
                     CE->getLocStart(), &R, 1);
}

//===----------------------------------------------------------------------===//
// Check: Should check whether privileges are dropped successfully.
// Originally: <rdar://problem/6337132>
//===----------------------------------------------------------------------===//

void WalkAST::CheckUncheckedReturnValue(CallExpr *CE) {
  const FunctionDecl *FD = CE->getDirectCallee();
  if (!FD)
    return;

  if (II_setid[0] == NULL) {
    static const char * const identifiers[num_ids] = {
      "setuid", "setgid", "seteuid", "setegid",
      "setreuid", "setregid"
    };
      
    for (size_t i = 0; i < num_ids; i++)
      II_setid[i] = &BR.getContext().Idents.get(identifiers[i]);  
  }
  
  const IdentifierInfo *id = FD->getIdentifier();
  size_t identifierid;

  for (identifierid = 0; identifierid < num_ids; identifierid++)
    if (id == II_setid[identifierid])
      break;

  if (identifierid >= num_ids)
    return;
  
  const FunctionProtoType *FTP = dyn_cast<FunctionProtoType>(FD->getType());
  if (!FTP)
    return;
  
  // Verify that the function takes one or two arguments (depending on
  //   the function).
  if (FTP->getNumArgs() != (identifierid < 4 ? 1 : 2))
    return;

  // The arguments must be integers.
  for (unsigned i = 0; i < FTP->getNumArgs(); i++)
    if (! FTP->getArgType(i)->isIntegerType())
      return;

  // Issue a warning.
  std::string buf1;
  llvm::raw_string_ostream os1(buf1);
  os1 << "Return value is not checked in call to '" << FD->getNameAsString()
     << "'";

  std::string buf2;
  llvm::raw_string_ostream os2(buf2);
  os2 << "The return value from the call to '" << FD->getNameAsString()
      << "' is not checked.  If an error occurs in '"
      << FD->getNameAsString()
      << "', the following code may execute with unexpected privileges";

  SourceRange R = CE->getCallee()->getSourceRange();
  
  BR.EmitBasicReport(os1.str().c_str(), "Security", os2.str().c_str(),
                     CE->getLocStart(), &R, 1);
}

//===----------------------------------------------------------------------===//
// Entry point for check.
//===----------------------------------------------------------------------===//

void clang::CheckSecuritySyntaxOnly(const Decl *D, BugReporter &BR) {  
  WalkAST walker(BR);
  walker.Visit(D->getBody());  
}

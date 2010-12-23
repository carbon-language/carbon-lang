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

#include "clang/Basic/TargetInfo.h"
#include "clang/EntoSA/BugReporter/BugReporter.h"
#include "clang/EntoSA/Checkers/LocalCheckers.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

static bool isArc4RandomAvailable(const ASTContext &Ctx) {
  const llvm::Triple &T = Ctx.Target.getTriple();
  return T.getVendor() == llvm::Triple::Apple ||
         T.getOS() == llvm::Triple::FreeBSD;
}

namespace {
class WalkAST : public StmtVisitor<WalkAST> {
  BugReporter &BR;
  IdentifierInfo *II_gets;
  IdentifierInfo *II_getpw;
  IdentifierInfo *II_mktemp;
  enum { num_rands = 9 };
  IdentifierInfo *II_rand[num_rands];
  IdentifierInfo *II_random;
  enum { num_setids = 6 };
  IdentifierInfo *II_setid[num_setids];

  const bool CheckRand;

public:
  WalkAST(BugReporter &br) : BR(br),
                             II_gets(0), II_getpw(0), II_mktemp(0),
                             II_rand(), II_random(0), II_setid(),
                 CheckRand(isArc4RandomAvailable(BR.getContext())) {}

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
  void CheckCall_getpw(const CallExpr *CE, const FunctionDecl *FD);
  void CheckCall_mktemp(const CallExpr *CE, const FunctionDecl *FD);
  void CheckCall_rand(const CallExpr *CE, const FunctionDecl *FD);
  void CheckCall_random(const CallExpr *CE, const FunctionDecl *FD);
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
    CheckCall_getpw(CE, FD);
    CheckCall_mktemp(CE, FD);
    if (CheckRand) {
      CheckCall_rand(CE, FD);
      CheckCall_random(CE, FD);
    }
  }

  // Recurse and check children.
  VisitChildren(CE);
}

void WalkAST::VisitCompoundStmt(CompoundStmt *S) {
  for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (Stmt *child = *I) {
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
          B->getOpcode() == BO_Comma))
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
  const DeclRefExpr *drLHS =
    dyn_cast<DeclRefExpr>(B->getLHS()->IgnoreParenLValueCasts());
  const DeclRefExpr *drRHS =
    dyn_cast<DeclRefExpr>(B->getRHS()->IgnoreParenLValueCasts());

  // Does at least one of the variables have a floating point type?
  drLHS = drLHS && drLHS->getType()->isRealFloatingType() ? drLHS : NULL;
  drRHS = drRHS && drRHS->getType()->isRealFloatingType() ? drRHS : NULL;

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
  llvm::SmallString<256> sbuf;
  llvm::raw_svector_ostream os(sbuf);

  os << "Variable '" << drCond->getDecl()->getName()
     << "' with floating point type '" << drCond->getType().getAsString()
     << "' should not be used as a loop counter";

  ranges.push_back(drCond->getSourceRange());
  ranges.push_back(drInc->getSourceRange());

  const char *bugType = "Floating point variable used as loop counter";
  BR.EmitBasicReport(bugType, "Security", os.str(),
                     FS->getLocStart(), ranges.data(), ranges.size());
}

//===----------------------------------------------------------------------===//
// Check: Any use of 'gets' is insecure.
// Originally: <rdar://problem/6335715>
// Implements (part of): 300-BSI (buildsecurityin.us-cert.gov)
// CWE-242: Use of Inherently Dangerous Function
//===----------------------------------------------------------------------===//

void WalkAST::CheckCall_gets(const CallExpr *CE, const FunctionDecl *FD) {
  if (FD->getIdentifier() != GetIdentifier(II_gets, "gets"))
    return;

  const FunctionProtoType *FPT
    = dyn_cast<FunctionProtoType>(FD->getType().IgnoreParens());
  if (!FPT)
    return;

  // Verify that the function takes a single argument.
  if (FPT->getNumArgs() != 1)
    return;

  // Is the argument a 'char*'?
  const PointerType *PT = dyn_cast<PointerType>(FPT->getArgType(0));
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
// Check: Any use of 'getpwd' is insecure.
// CWE-477: Use of Obsolete Functions
//===----------------------------------------------------------------------===//

void WalkAST::CheckCall_getpw(const CallExpr *CE, const FunctionDecl *FD) {
  if (FD->getIdentifier() != GetIdentifier(II_getpw, "getpw"))
    return;

  const FunctionProtoType *FPT
    = dyn_cast<FunctionProtoType>(FD->getType().IgnoreParens());
  if (!FPT)
    return;

  // Verify that the function takes two arguments.
  if (FPT->getNumArgs() != 2)
    return;

  // Verify the first argument type is integer.
  if (!FPT->getArgType(0)->isIntegerType())
    return;

  // Verify the second argument type is char*.
  const PointerType *PT = dyn_cast<PointerType>(FPT->getArgType(1));
  if (!PT)
    return;

  if (PT->getPointeeType().getUnqualifiedType() != BR.getContext().CharTy)
    return;

  // Issue a warning.
  SourceRange R = CE->getCallee()->getSourceRange();
  BR.EmitBasicReport("Potential buffer overflow in call to 'getpw'",
                     "Security",
                     "The getpw() function is dangerous as it may overflow the "
                     "provided buffer. It is obsoleted by getpwuid().",
                     CE->getLocStart(), &R, 1);
}

//===----------------------------------------------------------------------===//
// Check: Any use of 'mktemp' is insecure.It is obsoleted by mkstemp().
// CWE-377: Insecure Temporary File
//===----------------------------------------------------------------------===//

void WalkAST::CheckCall_mktemp(const CallExpr *CE, const FunctionDecl *FD) {
  if (FD->getIdentifier() != GetIdentifier(II_mktemp, "mktemp"))
    return;

  const FunctionProtoType *FPT
    = dyn_cast<FunctionProtoType>(FD->getType().IgnoreParens());
  if(!FPT)
    return;

  // Verify that the funcion takes a single argument.
  if (FPT->getNumArgs() != 1)
    return;

  // Verify that the argument is Pointer Type.
  const PointerType *PT = dyn_cast<PointerType>(FPT->getArgType(0));
  if (!PT)
    return;

  // Verify that the argument is a 'char*'.
  if (PT->getPointeeType().getUnqualifiedType() != BR.getContext().CharTy)
    return;

  // Issue a waring.
  SourceRange R = CE->getCallee()->getSourceRange();
  BR.EmitBasicReport("Potential insecure temporary file in call 'mktemp'",
                     "Security",
                     "Call to function 'mktemp' is insecure as it always "
                     "creates or uses insecure temporary file.  Use 'mkstemp' instead",
                     CE->getLocStart(), &R, 1);
}

//===----------------------------------------------------------------------===//
// Check: Linear congruent random number generators should not be used
// Originally: <rdar://problem/63371000>
// CWE-338: Use of cryptographically weak prng
//===----------------------------------------------------------------------===//

void WalkAST::CheckCall_rand(const CallExpr *CE, const FunctionDecl *FD) {
  if (II_rand[0] == NULL) {
    // This check applies to these functions
    static const char * const identifiers[num_rands] = {
      "drand48", "erand48", "jrand48", "lrand48", "mrand48", "nrand48",
      "lcong48",
      "rand", "rand_r"
    };

    for (size_t i = 0; i < num_rands; i++)
      II_rand[i] = &BR.getContext().Idents.get(identifiers[i]);
  }

  const IdentifierInfo *id = FD->getIdentifier();
  size_t identifierid;

  for (identifierid = 0; identifierid < num_rands; identifierid++)
    if (id == II_rand[identifierid])
      break;

  if (identifierid >= num_rands)
    return;

  const FunctionProtoType *FTP
    = dyn_cast<FunctionProtoType>(FD->getType().IgnoreParens());
  if (!FTP)
    return;

  if (FTP->getNumArgs() == 1) {
    // Is the argument an 'unsigned short *'?
    // (Actually any integer type is allowed.)
    const PointerType *PT = dyn_cast<PointerType>(FTP->getArgType(0));
    if (!PT)
      return;

    if (! PT->getPointeeType()->isIntegerType())
      return;
  }
  else if (FTP->getNumArgs() != 0)
    return;

  // Issue a warning.
  llvm::SmallString<256> buf1;
  llvm::raw_svector_ostream os1(buf1);
  os1 << '\'' << FD << "' is a poor random number generator";

  llvm::SmallString<256> buf2;
  llvm::raw_svector_ostream os2(buf2);
  os2 << "Function '" << FD
      << "' is obsolete because it implements a poor random number generator."
      << "  Use 'arc4random' instead";

  SourceRange R = CE->getCallee()->getSourceRange();
  BR.EmitBasicReport(os1.str(), "Security", os2.str(),CE->getLocStart(), &R, 1);
}

//===----------------------------------------------------------------------===//
// Check: 'random' should not be used
// Originally: <rdar://problem/63371000>
//===----------------------------------------------------------------------===//

void WalkAST::CheckCall_random(const CallExpr *CE, const FunctionDecl *FD) {
  if (FD->getIdentifier() != GetIdentifier(II_random, "random"))
    return;

  const FunctionProtoType *FTP
    = dyn_cast<FunctionProtoType>(FD->getType().IgnoreParens());
  if (!FTP)
    return;

  // Verify that the function takes no argument.
  if (FTP->getNumArgs() != 0)
    return;

  // Issue a warning.
  SourceRange R = CE->getCallee()->getSourceRange();
  BR.EmitBasicReport("'random' is not a secure random number generator",
                     "Security",
                     "The 'random' function produces a sequence of values that "
                     "an adversary may be able to predict.  Use 'arc4random' "
                     "instead", CE->getLocStart(), &R, 1);
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
    static const char * const identifiers[num_setids] = {
      "setuid", "setgid", "seteuid", "setegid",
      "setreuid", "setregid"
    };

    for (size_t i = 0; i < num_setids; i++)
      II_setid[i] = &BR.getContext().Idents.get(identifiers[i]);
  }

  const IdentifierInfo *id = FD->getIdentifier();
  size_t identifierid;

  for (identifierid = 0; identifierid < num_setids; identifierid++)
    if (id == II_setid[identifierid])
      break;

  if (identifierid >= num_setids)
    return;

  const FunctionProtoType *FTP
    = dyn_cast<FunctionProtoType>(FD->getType().IgnoreParens());
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
  llvm::SmallString<256> buf1;
  llvm::raw_svector_ostream os1(buf1);
  os1 << "Return value is not checked in call to '" << FD << '\'';

  llvm::SmallString<256> buf2;
  llvm::raw_svector_ostream os2(buf2);
  os2 << "The return value from the call to '" << FD
      << "' is not checked.  If an error occurs in '" << FD
      << "', the following code may execute with unexpected privileges";

  SourceRange R = CE->getCallee()->getSourceRange();
  BR.EmitBasicReport(os1.str(), "Security", os2.str(),CE->getLocStart(), &R, 1);
}

//===----------------------------------------------------------------------===//
// Entry point for check.
//===----------------------------------------------------------------------===//

void ento::CheckSecuritySyntaxOnly(const Decl *D, BugReporter &BR) {
  WalkAST walker(BR);
  walker.Visit(D->getBody());
}

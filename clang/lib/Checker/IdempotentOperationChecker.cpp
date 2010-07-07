//==- IdempotentOperationChecker.cpp - Idempotent Operations ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a set of path-sensitive checks for idempotent and/or
// tautological operations. Each potential operation is checked along all paths
// to see if every path results in a pointless operation.
//                 +-------------------------------------------+
//                 |Table of idempotent/tautological operations|
//                 +-------------------------------------------+
//+--------------------------------------------------------------------------+
//|Operator | x op x | x op 1 | 1 op x | x op 0 | 0 op x | x op ~0 | ~0 op x |
//+--------------------------------------------------------------------------+
//  +, +=   |        |        |        |   x    |   x    |         |
//  -, -=   |        |        |        |   x    |   -x   |         |
//  *, *=   |        |   x    |   x    |   0    |   0    |         |
//  /, /=   |   1    |   x    |        |  N/A   |   0    |         |
//  &, &=   |   x    |        |        |   0    |   0    |   x     |    x
//  |, |=   |   x    |        |        |   x    |   x    |   ~0    |    ~0
//  ^, ^=   |   0    |        |        |   x    |   x    |         |
//  <<, <<= |        |        |        |   x    |   0    |         |
//  >>, >>= |        |        |        |   x    |   0    |         |
//  ||      |   1    |   1    |   1    |   x    |   x    |   1     |    1
//  &&      |   1    |   x    |   x    |   0    |   0    |   x     |    x
//  =       |   x    |        |        |        |        |         |
//  ==      |   1    |        |        |        |        |         |
//  >=      |   1    |        |        |        |        |         |
//  <=      |   1    |        |        |        |        |         |
//  >       |   0    |        |        |        |        |         |
//  <       |   0    |        |        |        |        |         |
//  !=      |   0    |        |        |        |        |         |
//===----------------------------------------------------------------------===//
//
// Ways to reduce false positives (that need to be implemented):
// - Don't flag downsizing casts
// - Improved handling of static/global variables
// - Per-block marking of incomplete analysis
// - Handling ~0 values
// - False positives involving silencing unused variable warnings
//
// Other things TODO:
// - Improved error messages
// - Handle mixed assumptions (which assumptions can belong together?)
// - Finer grained false positive control (levels)

#include "GRExprEngineExperimentalChecks.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/SVals.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

namespace {
class IdempotentOperationChecker
  : public CheckerVisitor<IdempotentOperationChecker> {
  public:
    static void *getTag();
    void PreVisitBinaryOperator(CheckerContext &C, const BinaryOperator *B);
    void VisitEndAnalysis(ExplodedGraph &G, BugReporter &B,
        bool hasWorkRemaining);

  private:
    // Our assumption about a particular operation.
    enum Assumption { Possible, Impossible, Equal, LHSis1, RHSis1, LHSis0,
        RHSis0 };

    void UpdateAssumption(Assumption &A, const Assumption &New);

    /// contains* - Useful recursive methods to see if a statement contains an
    ///   element somewhere. Used in static analysis to reduce false positives.
    static bool containsMacro(const Stmt *S);
    static bool containsEnum(const Stmt *S);
    static bool containsBuiltinOffsetOf(const Stmt *S);
    static bool containsZeroConstant(const Stmt *S);
    static bool containsOneConstant(const Stmt *S);
    template <class T> static bool containsStmt(const Stmt *S) {
      if (isa<T>(S))
          return true;

      for (Stmt::const_child_iterator I = S->child_begin(); I != S->child_end();
          ++I)
        if (const Stmt *child = *I)
          if (containsStmt<T>(child))
            return true;

        return false;
    }

    // Hash table
    typedef llvm::DenseMap<const BinaryOperator *, Assumption> AssumptionMap;
    AssumptionMap hash;
};
}

void *IdempotentOperationChecker::getTag() {
  static int x = 0;
  return &x;
}

void clang::RegisterIdempotentOperationChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new IdempotentOperationChecker());
}

void IdempotentOperationChecker::PreVisitBinaryOperator(
                                                      CheckerContext &C,
                                                      const BinaryOperator *B) {
  // Find or create an entry in the hash for this BinaryOperator instance
  AssumptionMap::iterator i = hash.find(B);
  Assumption &A = i == hash.end() ? hash[B] : i->second;

  // If we had to create an entry, initialise the value to Possible
  if (i == hash.end())
    A = Possible;

  // If we already have visited this node on a path that does not contain an
  // idempotent operation, return immediately.
  if (A == Impossible)
    return;

  // Skip binary operators containing common false positives
  if (containsMacro(B) || containsEnum(B) || containsStmt<SizeOfAlignOfExpr>(B)
      || containsZeroConstant(B) || containsOneConstant(B)
      || containsBuiltinOffsetOf(B)) {
    A = Impossible;
    return;
  }

  const Expr *LHS = B->getLHS();
  const Expr *RHS = B->getRHS();

  const GRState *state = C.getState();

  SVal LHSVal = state->getSVal(LHS);
  SVal RHSVal = state->getSVal(RHS);

  // If either value is unknown, we can't be 100% sure of all paths.
  if (LHSVal.isUnknownOrUndef() || RHSVal.isUnknownOrUndef()) {
    A = Impossible;
    return;
  }
  BinaryOperator::Opcode Op = B->getOpcode();

  // Dereference the LHS SVal if this is an assign operation
  switch (Op) {
  default:
    break;

  // Fall through intentional
  case BinaryOperator::AddAssign:
  case BinaryOperator::SubAssign:
  case BinaryOperator::MulAssign:
  case BinaryOperator::DivAssign:
  case BinaryOperator::AndAssign:
  case BinaryOperator::OrAssign:
  case BinaryOperator::XorAssign:
  case BinaryOperator::ShlAssign:
  case BinaryOperator::ShrAssign:
  case BinaryOperator::Assign:
  // Assign statements have one extra level of indirection
    if (!isa<Loc>(LHSVal)) {
      A = Impossible;
      return;
    }
    LHSVal = state->getSVal(cast<Loc>(LHSVal));
  }


  // We now check for various cases which result in an idempotent operation.

  // x op x
  switch (Op) {
  default:
    break; // We don't care about any other operators.

  // Fall through intentional
  case BinaryOperator::SubAssign:
  case BinaryOperator::DivAssign:
  case BinaryOperator::AndAssign:
  case BinaryOperator::OrAssign:
  case BinaryOperator::XorAssign:
  case BinaryOperator::Assign:
  case BinaryOperator::Sub:
  case BinaryOperator::Div:
  case BinaryOperator::And:
  case BinaryOperator::Or:
  case BinaryOperator::Xor:
  case BinaryOperator::LOr:
  case BinaryOperator::LAnd:
    if (LHSVal != RHSVal)
      break;
    UpdateAssumption(A, Equal);
    return;
  }

  // x op 1
  switch (Op) {
   default:
     break; // We don't care about any other operators.

   // Fall through intentional
   case BinaryOperator::MulAssign:
   case BinaryOperator::DivAssign:
   case BinaryOperator::Mul:
   case BinaryOperator::Div:
   case BinaryOperator::LOr:
   case BinaryOperator::LAnd:
     if (!RHSVal.isConstant(1))
       break;
     UpdateAssumption(A, RHSis1);
     return;
  }

  // 1 op x
  switch (Op) {
  default:
    break; // We don't care about any other operators.

  // Fall through intentional
  case BinaryOperator::MulAssign:
  case BinaryOperator::Mul:
  case BinaryOperator::LOr:
  case BinaryOperator::LAnd:
    if (!LHSVal.isConstant(1))
      break;
    UpdateAssumption(A, LHSis1);
    return;
  }

  // x op 0
  switch (Op) {
  default:
    break; // We don't care about any other operators.

  // Fall through intentional
  case BinaryOperator::AddAssign:
  case BinaryOperator::SubAssign:
  case BinaryOperator::MulAssign:
  case BinaryOperator::AndAssign:
  case BinaryOperator::OrAssign:
  case BinaryOperator::XorAssign:
  case BinaryOperator::Add:
  case BinaryOperator::Sub:
  case BinaryOperator::Mul:
  case BinaryOperator::And:
  case BinaryOperator::Or:
  case BinaryOperator::Xor:
  case BinaryOperator::Shl:
  case BinaryOperator::Shr:
  case BinaryOperator::LOr:
  case BinaryOperator::LAnd:
    if (!RHSVal.isConstant(0))
      break;
    UpdateAssumption(A, RHSis0);
    return;
  }

  // 0 op x
  switch (Op) {
  default:
    break; // We don't care about any other operators.

  // Fall through intentional
  //case BinaryOperator::AddAssign: // Common false positive
  case BinaryOperator::SubAssign: // Check only if unsigned
  case BinaryOperator::MulAssign:
  case BinaryOperator::DivAssign:
  case BinaryOperator::AndAssign:
  //case BinaryOperator::OrAssign: // Common false positive
  //case BinaryOperator::XorAssign: // Common false positive
  case BinaryOperator::ShlAssign:
  case BinaryOperator::ShrAssign:
  case BinaryOperator::Add:
  case BinaryOperator::Sub:
  case BinaryOperator::Mul:
  case BinaryOperator::Div:
  case BinaryOperator::And:
  case BinaryOperator::Or:
  case BinaryOperator::Xor:
  case BinaryOperator::Shl:
  case BinaryOperator::Shr:
  case BinaryOperator::LOr:
  case BinaryOperator::LAnd:
    if (!LHSVal.isConstant(0))
      break;
    UpdateAssumption(A, LHSis0);
    return;
  }

  // If we get to this point, there has been a valid use of this operation.
  A = Impossible;
}

void IdempotentOperationChecker::VisitEndAnalysis(ExplodedGraph &G,
                                                  BugReporter &B,
                                                  bool hasWorkRemaining) {
  // If there is any work remaining we cannot be 100% sure about our warnings
  if (hasWorkRemaining)
    return;

  // Iterate over the hash to see if we have any paths with definite
  // idempotent operations.
  for (AssumptionMap::const_iterator i =
      hash.begin(); i != hash.end(); ++i) {
    if (i->second != Impossible) {
      // Select the error message.
      const char *msg = 0;
      switch (i->second) {
      case Equal:
        msg = "idempotent operation; both operands are always equal in value";
        break;
      case LHSis1:
        msg = "idempotent operation; the left operand is always 1";
        break;
      case RHSis1:
        msg = "idempotent operation; the right operand is always 1";
        break;
      case LHSis0:
        msg = "idempotent operation; the left operand is always 0";
        break;
      case RHSis0:
        msg = "idempotent operation; the right operand is always 0";
        break;
      case Possible:
        llvm_unreachable("Operation was never marked with an assumption");
      case Impossible:
        llvm_unreachable(0);
      }

      // Create the SourceRange Arrays
      SourceRange S[2] = { i->first->getLHS()->getSourceRange(),
                           i->first->getRHS()->getSourceRange() };
      B.EmitBasicReport("Idempotent operation", msg, i->first->getOperatorLoc(),
          S, 2);
    }
  }
}

// Updates the current assumption given the new assumption
inline void IdempotentOperationChecker::UpdateAssumption(Assumption &A,
                                                        const Assumption &New) {
  switch (A) {
  // If we don't currently have an assumption, set it
  case Possible:
    A = New;
    return;

  // If we have determined that a valid state happened, ignore the new
  // assumption.
  case Impossible:
    return;

  // Any other case means that we had a different assumption last time. We don't
  // currently support mixing assumptions for diagnostic reasons, so we set
  // our assumption to be impossible.
  default:
    A = Impossible;
    return;
  }
}

// Recursively find any substatements containing macros
bool IdempotentOperationChecker::containsMacro(const Stmt *S) {
  if (S->getLocStart().isMacroID())
    return true;

  if (S->getLocEnd().isMacroID())
    return true;

  for (Stmt::const_child_iterator I = S->child_begin(); I != S->child_end();
      ++I)
    if (const Stmt *child = *I)
      if (containsMacro(child))
        return true;

  return false;
}

// Recursively find any substatements containing enum constants
bool IdempotentOperationChecker::containsEnum(const Stmt *S) {
  const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(S);

  if (DR && isa<EnumConstantDecl>(DR->getDecl()))
    return true;

  for (Stmt::const_child_iterator I = S->child_begin(); I != S->child_end();
      ++I)
    if (const Stmt *child = *I)
      if (containsEnum(child))
        return true;

  return false;
}

// Recursively find any substatements containing __builtin_offset_of
bool IdempotentOperationChecker::containsBuiltinOffsetOf(const Stmt *S) {
  const UnaryOperator *UO = dyn_cast<UnaryOperator>(S);

  if (UO && UO->getOpcode() == UnaryOperator::OffsetOf)
    return true;

  for (Stmt::const_child_iterator I = S->child_begin(); I != S->child_end();
      ++I)
    if (const Stmt *child = *I)
      if (containsBuiltinOffsetOf(child))
        return true;

  return false;
}

bool IdempotentOperationChecker::containsZeroConstant(const Stmt *S) {
  const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(S);
  if (IL && IL->getValue() == 0)
    return true;

  const FloatingLiteral *FL = dyn_cast<FloatingLiteral>(S);
  if (FL && FL->getValue().isZero())
    return true;

  for (Stmt::const_child_iterator I = S->child_begin(); I != S->child_end();
      ++I)
    if (const Stmt *child = *I)
      if (containsZeroConstant(child))
        return true;

  return false;
}

bool IdempotentOperationChecker::containsOneConstant(const Stmt *S) {
  const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(S);
  if (IL && IL->getValue() == 1)
    return true;

  const FloatingLiteral *FL = dyn_cast<FloatingLiteral>(S);
  const llvm::APFloat one(1.0);
  if (FL && FL->getValue().compare(one) == llvm::APFloat::cmpEqual)
    return true;

  for (Stmt::const_child_iterator I = S->child_begin(); I != S->child_end();
      ++I)
    if (const Stmt *child = *I)
      if (containsOneConstant(child))
        return true;

  return false;
}


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
// Things TODO:
// - Improved error messages
// - Handle mixed assumptions (which assumptions can belong together?)
// - Finer grained false positive control (levels)
// - Handling ~0 values

#include "GRExprEngineExperimentalChecks.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/CheckerHelpers.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/GRCoreEngine.h"
#include "clang/Checker/PathSensitive/SVals.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <deque>

using namespace clang;

namespace {
class IdempotentOperationChecker
  : public CheckerVisitor<IdempotentOperationChecker> {
  public:
    static void *getTag();
    void PreVisitBinaryOperator(CheckerContext &C, const BinaryOperator *B);
    void VisitEndAnalysis(ExplodedGraph &G, BugReporter &B, GRExprEngine &Eng);

  private:
    // Our assumption about a particular operation.
    enum Assumption { Possible = 0, Impossible, Equal, LHSis1, RHSis1, LHSis0,
        RHSis0 };

    void UpdateAssumption(Assumption &A, const Assumption &New);

    /// contains* - Useful recursive methods to see if a statement contains an
    ///   element somewhere. Used in static analysis to reduce false positives.
    static bool isParameterSelfAssign(const Expr *LHS, const Expr *RHS);
    static bool isTruncationExtensionAssignment(const Expr *LHS,
                                                const Expr *RHS);
    static bool PathWasCompletelyAnalyzed(const CFG *C,
                                          const CFGBlock *CB,
                                          const GRCoreEngine &CE);
    static bool CanVary(const Expr *Ex, ASTContext &Ctx);

    // Hash table
    typedef llvm::DenseMap<const BinaryOperator *,
                           std::pair<Assumption, AnalysisContext*> >
                           AssumptionMap;
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
  // Find or create an entry in the hash for this BinaryOperator instance.
  // If we haven't done a lookup before, it will get default initialized to
  // 'Possible'.
  std::pair<Assumption, AnalysisContext *> &Data = hash[B];
  Assumption &A = Data.first;
  Data.second = C.getCurrentAnalysisContext();

  // If we already have visited this node on a path that does not contain an
  // idempotent operation, return immediately.
  if (A == Impossible)
    return;

  // Retrieve both sides of the operator and determine if they can vary (which
  // may mean this is a false positive.
  const Expr *LHS = B->getLHS();
  const Expr *RHS = B->getRHS();
  bool LHSCanVary = CanVary(LHS, C.getASTContext());
  bool RHSCanVary = CanVary(RHS, C.getASTContext());

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
  case BinaryOperator::Assign:
    // x Assign x has a few more false positives we can check for
    if (isParameterSelfAssign(RHS, LHS)
        || isTruncationExtensionAssignment(RHS, LHS)) {
      A = Impossible;
      return;
    }

  case BinaryOperator::SubAssign:
  case BinaryOperator::DivAssign:
  case BinaryOperator::AndAssign:
  case BinaryOperator::OrAssign:
  case BinaryOperator::XorAssign:
  case BinaryOperator::Sub:
  case BinaryOperator::Div:
  case BinaryOperator::And:
  case BinaryOperator::Or:
  case BinaryOperator::Xor:
  case BinaryOperator::LOr:
  case BinaryOperator::LAnd:
    if (LHSVal != RHSVal || !LHSCanVary || !RHSCanVary)
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
     if (!RHSVal.isConstant(1) || !RHSCanVary)
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
    if (!LHSVal.isConstant(1) || !LHSCanVary)
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
    if (!RHSVal.isConstant(0) || !RHSCanVary)
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
    if (!LHSVal.isConstant(0) || !LHSCanVary)
      break;
    UpdateAssumption(A, LHSis0);
    return;
  }

  // If we get to this point, there has been a valid use of this operation.
  A = Impossible;
}

void IdempotentOperationChecker::VisitEndAnalysis(ExplodedGraph &G,
                                                  BugReporter &BR,
                                                  GRExprEngine &Eng) {
  // Iterate over the hash to see if we have any paths with definite
  // idempotent operations.
  for (AssumptionMap::const_iterator i = hash.begin(); i != hash.end(); ++i) {
    // Unpack the hash contents
    const std::pair<Assumption, AnalysisContext *> &Data = i->second;
    const Assumption &A = Data.first;
    AnalysisContext *AC = Data.second;

    const BinaryOperator *B = i->first;

    if (A == Impossible)
      continue;

    // If the analyzer did not finish, check to see if we can still emit this
    // warning
    if (Eng.hasWorkRemaining()) {
      const CFGStmtMap *CBM = CFGStmtMap::Build(AC->getCFG(),
                                                &AC->getParentMap());

      // If we can trace back
      if (!PathWasCompletelyAnalyzed(AC->getCFG(),
                                     CBM->getBlock(B),
                                     Eng.getCoreEngine()))
        continue;

      delete CBM;
    }

    // Select the error message.
    llvm::SmallString<128> buf;
    llvm::raw_svector_ostream os(buf);
    switch (A) {
    case Equal:
      if (B->getOpcode() == BinaryOperator::Assign)
        os << "Assigned value is always the same as the existing value";
      else
        os << "Both operands to '" << B->getOpcodeStr()
           << "' always have the same value";
      break;
    case LHSis1:
      os << "The left operand to '" << B->getOpcodeStr() << "' is always 1";
      break;
    case RHSis1:
      os << "The right operand to '" << B->getOpcodeStr() << "' is always 1";
      break;
    case LHSis0:
      os << "The left operand to '" << B->getOpcodeStr() << "' is always 0";
      break;
    case RHSis0:
      os << "The right operand to '" << B->getOpcodeStr() << "' is always 0";
      break;
    case Possible:
      llvm_unreachable("Operation was never marked with an assumption");
    case Impossible:
      llvm_unreachable(0);
    }

    // Create the SourceRange Arrays
    SourceRange S[2] = { i->first->getLHS()->getSourceRange(),
                         i->first->getRHS()->getSourceRange() };
    BR.EmitBasicReport("Idempotent operation", "Dead code",
                       os.str(), i->first->getOperatorLoc(), S, 2);
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

// Check for a statement were a parameter is self assigned (to avoid an unused
// variable warning)
bool IdempotentOperationChecker::isParameterSelfAssign(const Expr *LHS,
                                                       const Expr *RHS) {
  LHS = LHS->IgnoreParenCasts();
  RHS = RHS->IgnoreParenCasts();

  const DeclRefExpr *LHS_DR = dyn_cast<DeclRefExpr>(LHS);
  if (!LHS_DR)
    return false;

  const ParmVarDecl *PD = dyn_cast<ParmVarDecl>(LHS_DR->getDecl());
  if (!PD)
    return false;

  const DeclRefExpr *RHS_DR = dyn_cast<DeclRefExpr>(RHS);
  if (!RHS_DR)
    return false;

  return PD == RHS_DR->getDecl();
}

// Check for self casts truncating/extending a variable
bool IdempotentOperationChecker::isTruncationExtensionAssignment(
                                                              const Expr *LHS,
                                                              const Expr *RHS) {

  const DeclRefExpr *LHS_DR = dyn_cast<DeclRefExpr>(LHS->IgnoreParenCasts());
  if (!LHS_DR)
    return false;

  const VarDecl *VD = dyn_cast<VarDecl>(LHS_DR->getDecl());
  if (!VD)
    return false;

  const DeclRefExpr *RHS_DR = dyn_cast<DeclRefExpr>(RHS->IgnoreParenCasts());
  if (!RHS_DR)
    return false;

  if (VD != RHS_DR->getDecl())
     return false;

  return dyn_cast<DeclRefExpr>(RHS->IgnoreParens()) == NULL;
}

// Returns false if a path to this block was not completely analyzed, or true
// otherwise.
bool IdempotentOperationChecker::PathWasCompletelyAnalyzed(
                                                       const CFG *C,
                                                       const CFGBlock *CB,
                                                       const GRCoreEngine &CE) {
  std::deque<const CFGBlock *> WorkList;
  llvm::SmallSet<unsigned, 8> Aborted;
  llvm::SmallSet<unsigned, 128> Visited;

  // Create a set of all aborted blocks
  typedef GRCoreEngine::BlocksAborted::const_iterator AbortedIterator;
  for (AbortedIterator I = CE.blocks_aborted_begin(),
      E = CE.blocks_aborted_end(); I != E; ++I) {
    const BlockEdge &BE =  I->first;

    // The destination block on the BlockEdge is the first block that was not
    // analyzed.
    Aborted.insert(BE.getDst()->getBlockID());
  }

  // Save the entry block ID for early exiting
  unsigned EntryBlockID = C->getEntry().getBlockID();

  // Create initial node
  WorkList.push_back(CB);

  while (!WorkList.empty()) {
    const CFGBlock *Head = WorkList.front();
    WorkList.pop_front();
    Visited.insert(Head->getBlockID());

    // If we found the entry block, then there exists a path from the target
    // node to the entry point of this function -> the path was completely
    // analyzed.
    if (Head->getBlockID() == EntryBlockID)
      return true;

    // If any of the aborted blocks are on the path to the beginning, then all
    // paths to this block were not analyzed.
    if (Aborted.count(Head->getBlockID()))
      return false;

    // Add the predecessors to the worklist unless we have already visited them
    for (CFGBlock::const_pred_iterator I = Head->pred_begin();
        I != Head->pred_end(); ++I)
      if (!Visited.count((*I)->getBlockID()))
        WorkList.push_back(*I);
  }

  // If we get to this point, there is no connection to the entry block or an
  // aborted block. This path is unreachable and we can report the error.
  return true;
}

// Recursive function that determines whether an expression contains any element
// that varies. This could be due to a compile-time constant like sizeof. An
// expression may also involve a variable that behaves like a constant. The
// function returns true if the expression varies, and false otherwise.
bool IdempotentOperationChecker::CanVary(const Expr *Ex, ASTContext &Ctx) {
  // Parentheses and casts are irrelevant here
  Ex = Ex->IgnoreParenCasts();

  if (Ex->getLocStart().isMacroID())
    return false;

  switch (Ex->getStmtClass()) {
  // Trivially true cases
  case Stmt::ArraySubscriptExprClass:
  case Stmt::MemberExprClass:
  case Stmt::StmtExprClass:
  case Stmt::CallExprClass:
  case Stmt::VAArgExprClass:
  case Stmt::ShuffleVectorExprClass:
    return true;
  default:
    return true;

  // Trivially false cases
  case Stmt::IntegerLiteralClass:
  case Stmt::CharacterLiteralClass:
  case Stmt::FloatingLiteralClass:
  case Stmt::PredefinedExprClass:
  case Stmt::ImaginaryLiteralClass:
  case Stmt::StringLiteralClass:
  case Stmt::OffsetOfExprClass:
  case Stmt::CompoundLiteralExprClass:
  case Stmt::AddrLabelExprClass:
  case Stmt::TypesCompatibleExprClass:
  case Stmt::GNUNullExprClass:
  case Stmt::InitListExprClass:
  case Stmt::DesignatedInitExprClass:
  case Stmt::BlockExprClass:
  case Stmt::BlockDeclRefExprClass:
    return false;

  // Cases requiring custom logic
  case Stmt::SizeOfAlignOfExprClass: {
    const SizeOfAlignOfExpr *SE = cast<const SizeOfAlignOfExpr>(Ex);
    if (!SE->isSizeOf())
      return false;
    return SE->getTypeOfArgument()->isVariableArrayType();
  }
  case Stmt::DeclRefExprClass:
    //    return !IsPseudoConstant(cast<DeclRefExpr>(Ex));
    return true;

  // The next cases require recursion for subexpressions
  case Stmt::BinaryOperatorClass: {
    const BinaryOperator *B = cast<const BinaryOperator>(Ex);
    return CanVary(B->getRHS(), Ctx) || CanVary(B->getLHS(), Ctx);
   }
  case Stmt::UnaryOperatorClass: {
    const UnaryOperator *U = cast<const UnaryOperator>(Ex);
    // Handle trivial case first
    switch (U->getOpcode()) {
    case UnaryOperator::Extension:
      return false;
    default:
      return CanVary(U->getSubExpr(), Ctx);
    }
  }
  case Stmt::ChooseExprClass:
    return CanVary(cast<const ChooseExpr>(Ex)->getChosenSubExpr(Ctx), Ctx);
  case Stmt::ConditionalOperatorClass:
      return CanVary(cast<const ConditionalOperator>(Ex)->getCond(), Ctx);
  }
}


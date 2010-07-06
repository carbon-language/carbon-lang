//= CStringChecker.h - Checks calls to C string functions ----------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines CStringChecker, which is an assortment of checks on calls
// to functions in <string.h>.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineExperimentalChecks.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;

namespace {
class CStringChecker : public CheckerVisitor<CStringChecker> {
  BugType *BT_Bounds;
  BugType *BT_Overlap;
public:
  CStringChecker()
  : BT_Bounds(0), BT_Overlap(0) {}
  static void *getTag() { static int tag; return &tag; }

  bool EvalCallExpr(CheckerContext &C, const CallExpr *CE);

  typedef const GRState *(CStringChecker::*FnCheck)(CheckerContext &,
                                                    const CallExpr *);

  const GRState *EvalMemcpy(CheckerContext &C, const CallExpr *CE);
  const GRState *EvalMemmove(CheckerContext &C, const CallExpr *CE);
  const GRState *EvalBcopy(CheckerContext &C, const CallExpr *CE);

  // Utility methods
  const GRState *CheckLocation(CheckerContext &C, const GRState *state,
                               const Stmt *S, SVal l);
  const GRState *CheckBufferAccess(CheckerContext &C, const GRState *state,
                                   const Expr *Size,
                                   const Expr *FirstBuf,
                                   const Expr *SecondBuf = NULL);
  const GRState *CheckOverlap(CheckerContext &C, const GRState *state,
                              const Expr *First, const Expr *Second,
                              const Expr *Size);
  void EmitOverlapBug(CheckerContext &C, const GRState *state,
                      const Stmt *First, const Stmt *Second);
};
} //end anonymous namespace

void clang::RegisterCStringChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new CStringChecker());
}

// FIXME: This was originally copied from ArrayBoundChecker.cpp. Refactor?
const GRState *CStringChecker::CheckLocation(CheckerContext &C,
                                             const GRState *state,
                                             const Stmt *S, SVal l) {
  // Check for out of bound array element access.
  const MemRegion *R = l.getAsRegion();
  if (!R)
    return state;

  R = R->StripCasts();

  const ElementRegion *ER = dyn_cast<ElementRegion>(R);
  if (!ER)
    return state;

  assert(ER->getValueType(C.getASTContext()) == C.getASTContext().CharTy &&
    "CheckLocation should only be called with char* ElementRegions");

  // Get the size of the array.
  const SubRegion *Super = cast<SubRegion>(ER->getSuperRegion());
  ValueManager &ValMgr = C.getValueManager();
  SVal Extent = ValMgr.convertToArrayIndex(Super->getExtent(ValMgr));
  DefinedOrUnknownSVal Size = cast<DefinedOrUnknownSVal>(Extent);

  // Get the index of the accessed element.
  DefinedOrUnknownSVal &Idx = cast<DefinedOrUnknownSVal>(ER->getIndex());

  const GRState *StInBound = state->AssumeInBound(Idx, Size, true);
  const GRState *StOutBound = state->AssumeInBound(Idx, Size, false);
  if (StOutBound && !StInBound) {
    ExplodedNode *N = C.GenerateSink(StOutBound);
    if (!N)
      return NULL;

    if (!BT_Bounds)
      BT_Bounds = new BuiltinBug("Out-of-bound array access",
        "Byte string function accesses out-of-bound array element "
        "(buffer overflow)");

    // FIXME: It would be nice to eventually make this diagnostic more clear,
    // e.g., by referencing the original declaration or by saying *why* this
    // reference is outside the range.

    // Generate a report for this bug.
    BuiltinBug *BT = static_cast<BuiltinBug*>(BT_Bounds);
    RangedBugReport *report = new RangedBugReport(*BT, BT->getDescription(), N);

    report->addRange(S->getSourceRange());
    C.EmitReport(report);
    return NULL;
  }
  
  // Array bound check succeeded.  From this point forward the array bound
  // should always succeed.
  return StInBound;
}

const GRState *CStringChecker::CheckBufferAccess(CheckerContext &C,
                                                 const GRState *state,
                                                 const Expr *Size,
                                                 const Expr *FirstBuf,
                                                 const Expr *SecondBuf) {
  ValueManager &VM = C.getValueManager();
  SValuator &SV = VM.getSValuator();
  ASTContext &Ctx = C.getASTContext();

  QualType SizeTy = Ctx.getSizeType();
  QualType PtrTy = Ctx.getPointerType(Ctx.CharTy);

  // Get the access length and make sure it is known.
  SVal LengthVal = state->getSVal(Size);
  NonLoc *Length = dyn_cast<NonLoc>(&LengthVal);
  if (!Length)
    return state;

  // Compute the offset of the last element to be accessed: size-1.
  NonLoc One = cast<NonLoc>(VM.makeIntVal(1, SizeTy));
  NonLoc LastOffset = cast<NonLoc>(SV.EvalBinOpNN(state, BinaryOperator::Sub,
                                                  *Length, One, SizeTy));

  // Check that the first buffer is sufficiently long.
  SVal BufVal = state->getSVal(FirstBuf);
  Loc BufStart = cast<Loc>(SV.EvalCast(BufVal, PtrTy, FirstBuf->getType()));
  SVal BufEnd
    = SV.EvalBinOpLN(state, BinaryOperator::Add, BufStart, LastOffset, PtrTy);
  state = CheckLocation(C, state, FirstBuf, BufEnd);

  // If the buffer isn't large enough, abort.
  if (!state)
    return NULL;

  // If there's a second buffer, check it as well.
  if (SecondBuf) {
    BufVal = state->getSVal(SecondBuf);
    BufStart = cast<Loc>(SV.EvalCast(BufVal, PtrTy, SecondBuf->getType()));
    BufEnd
      = SV.EvalBinOpLN(state, BinaryOperator::Add, BufStart, LastOffset, PtrTy);
    state = CheckLocation(C, state, SecondBuf, BufEnd);
  }

  // Large enough or not, return this state!
  return state;
}

const GRState *CStringChecker::CheckOverlap(CheckerContext &C,
                                            const GRState *state,
                                            const Expr *First,
                                            const Expr *Second,
                                            const Expr *Size) {
  // Do a simple check for overlap: if the two arguments are from the same
  // buffer, see if the end of the first is greater than the start of the second
  // or vice versa.

  ValueManager &VM = state->getStateManager().getValueManager();
  SValuator &SV = VM.getSValuator();
  ASTContext &Ctx = VM.getContext();
  const GRState *stateTrue, *stateFalse;

  // Get the buffer values and make sure they're known locations.
  SVal FirstVal = state->getSVal(First);
  SVal SecondVal = state->getSVal(Second);

  Loc *FirstLoc = dyn_cast<Loc>(&FirstVal);
  if (!FirstLoc)
    return state;

  Loc *SecondLoc = dyn_cast<Loc>(&SecondVal);
  if (!SecondLoc)
    return state;

  // Are the two values the same?
  DefinedOrUnknownSVal EqualTest = SV.EvalEQ(state, *FirstLoc, *SecondLoc);
  llvm::tie(stateTrue, stateFalse) = state->Assume(EqualTest);

  if (stateTrue && !stateFalse) {
    // If the values are known to be equal, that's automatically an overlap.
    EmitOverlapBug(C, stateTrue, First, Second);
    return NULL;
  }

  // Assume the two expressions are not equal.
  assert(stateFalse);
  state = stateFalse;

  // Which value comes first?
  QualType CmpTy = Ctx.IntTy;
  SVal Reverse = SV.EvalBinOpLL(state, BinaryOperator::GT,
                                *FirstLoc, *SecondLoc, CmpTy);
  DefinedOrUnknownSVal *ReverseTest = dyn_cast<DefinedOrUnknownSVal>(&Reverse);
  if (!ReverseTest)
    return state;

  llvm::tie(stateTrue, stateFalse) = state->Assume(*ReverseTest);

  if (stateTrue) {
    if (stateFalse) {
      // If we don't know which one comes first, we can't perform this test.
      return state;
    } else {
      // Switch the values so that FirstVal is before SecondVal.
      Loc *tmpLoc = FirstLoc;
      FirstLoc = SecondLoc;
      SecondLoc = tmpLoc;

      // Switch the Exprs as well, so that they still correspond.
      const Expr *tmpExpr = First;
      First = Second;
      Second = tmpExpr;
    }
  }

  // Get the length, and make sure it too is known.
  SVal LengthVal = state->getSVal(Size);
  NonLoc *Length = dyn_cast<NonLoc>(&LengthVal);
  if (!Length)
    return state;

  // Convert the first buffer's start address to char*.
  // Bail out if the cast fails.
  QualType CharPtrTy = Ctx.getPointerType(Ctx.CharTy);
  SVal FirstStart = SV.EvalCast(*FirstLoc, CharPtrTy, First->getType());
  Loc *FirstStartLoc = dyn_cast<Loc>(&FirstStart);
  if (!FirstStartLoc)
    return state;

  // Compute the end of the first buffer. Bail out if THAT fails.
  SVal FirstEnd = SV.EvalBinOpLN(state, BinaryOperator::Add,
                                 *FirstStartLoc, *Length, CharPtrTy);
  Loc *FirstEndLoc = dyn_cast<Loc>(&FirstEnd);
  if (!FirstEndLoc)
    return state;

  // Is the end of the first buffer past the start of the second buffer?
  SVal Overlap = SV.EvalBinOpLL(state, BinaryOperator::GT,
                                *FirstEndLoc, *SecondLoc, CmpTy);
  DefinedOrUnknownSVal *OverlapTest = dyn_cast<DefinedOrUnknownSVal>(&Overlap);
  if (!OverlapTest)
    return state;

  llvm::tie(stateTrue, stateFalse) = state->Assume(*OverlapTest);

  if (stateTrue && !stateFalse) {
    // Overlap!
    EmitOverlapBug(C, stateTrue, First, Second);
    return NULL;
  }

  // Assume the two expressions don't overlap.
  assert(stateFalse);
  return stateFalse;
}

void CStringChecker::EmitOverlapBug(CheckerContext &C, const GRState *state,
                                    const Stmt *First, const Stmt *Second) {
  ExplodedNode *N = C.GenerateSink(state);
  if (!N)
    return;

  if (!BT_Overlap)
    BT_Overlap = new BugType("Unix API", "Improper arguments");

  // Generate a report for this bug.
  RangedBugReport *report = 
    new RangedBugReport(*BT_Overlap,
      "Arguments must not be overlapping buffers", N);
  report->addRange(First->getSourceRange());
  report->addRange(Second->getSourceRange());

  C.EmitReport(report);
}

const GRState *
CStringChecker::EvalMemcpy(CheckerContext &C, const CallExpr *CE) {
  // void *memcpy(void *restrict dst, const void *restrict src, size_t n);
  // memcpy() is like memmove(), but with the extra requirement that the buffers
  // not overlap.
  const GRState *state = EvalMemmove(C, CE);
  if (!state)
    return NULL;

  return CheckOverlap(C, state, CE->getArg(0), CE->getArg(1), CE->getArg(2));
}

const GRState *
CStringChecker::EvalMemmove(CheckerContext &C, const CallExpr *CE) {
  // void *memmove(void *dst, const void *src, size_t n);
  const Expr *Dest = CE->getArg(0);
  const Expr *Source = CE->getArg(1);
  const Expr *Size = CE->getArg(2);

  // Check that the accesses will stay in bounds.
  const GRState *state = C.getState();
  state = CheckBufferAccess(C, state, Size, Dest, Source);
  if (!state)
    return NULL;

  // The return value is the address of the destination buffer.
  return state->BindExpr(CE, state->getSVal(Dest));
}

const GRState *
CStringChecker::EvalBcopy(CheckerContext &C, const CallExpr *CE) {
  // void bcopy(const void *src, void *dst, size_t n);
  return CheckBufferAccess(C, C.getState(),
                           CE->getArg(2), CE->getArg(0), CE->getArg(1));
}

bool CStringChecker::EvalCallExpr(CheckerContext &C, const CallExpr *CE) {
  // Get the callee.  All the functions we care about are C functions
  // with simple identifiers.
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  const FunctionDecl *FD = state->getSVal(Callee).getAsFunctionDecl();

  if (!FD)
    return false;

  // Get the name of the callee. If it's a builtin, strip off the prefix.
  llvm::StringRef Name = FD->getName();
  if (Name.startswith("__builtin_"))
    Name = Name.substr(10);

  FnCheck EvalFunction = llvm::StringSwitch<FnCheck>(Name)
    .Case("memcpy", &CStringChecker::EvalMemcpy)
    .Case("__memcpy_chk", &CStringChecker::EvalMemcpy)
    .Case("memmove", &CStringChecker::EvalMemmove)
    .Case("__memmove_chk", &CStringChecker::EvalMemmove)
    .Case("bcopy", &CStringChecker::EvalBcopy)
    .Default(NULL);

  if (!EvalFunction)
    // The callee isn't a string function. Let another checker handle it.
    return false;

  const GRState *NewState = (this->*EvalFunction)(C, CE);

  if (NewState)
    C.addTransition(NewState);
  return true;
}

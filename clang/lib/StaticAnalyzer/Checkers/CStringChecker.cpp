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

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRStateTrait.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace ento;

namespace {
class CStringChecker : public Checker< eval::Call,
                                         check::PreStmt<DeclStmt>,
                                         check::LiveSymbols,
                                         check::DeadSymbols,
                                         check::RegionChanges
                                         > {
  mutable llvm::OwningPtr<BugType> BT_Null, BT_Bounds,
                                   BT_Overlap, BT_NotCString,
                                   BT_AdditionOverflow;
  mutable const char *CurrentFunctionDescription;

public:
  static void *getTag() { static int tag; return &tag; }

  bool evalCall(const CallExpr *CE, CheckerContext &C) const;
  void checkPreStmt(const DeclStmt *DS, CheckerContext &C) const;
  void checkLiveSymbols(const GRState *state, SymbolReaper &SR) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
  bool wantsRegionChangeUpdate(const GRState *state) const;

  const GRState *checkRegionChanges(const GRState *state,
                                    const StoreManager::InvalidatedSymbols *,
                                    const MemRegion * const *Begin,
                                    const MemRegion * const *End) const;

  typedef void (CStringChecker::*FnCheck)(CheckerContext &,
                                          const CallExpr *) const;

  void evalMemcpy(CheckerContext &C, const CallExpr *CE) const;
  void evalMempcpy(CheckerContext &C, const CallExpr *CE) const;
  void evalMemmove(CheckerContext &C, const CallExpr *CE) const;
  void evalBcopy(CheckerContext &C, const CallExpr *CE) const;
  void evalCopyCommon(CheckerContext &C, const CallExpr *CE,
                      const GRState *state,
                      const Expr *Size, const Expr *Source, const Expr *Dest,
                      bool Restricted = false,
                      bool IsMempcpy = false) const;

  void evalMemcmp(CheckerContext &C, const CallExpr *CE) const;

  void evalstrLength(CheckerContext &C, const CallExpr *CE) const;
  void evalstrnLength(CheckerContext &C, const CallExpr *CE) const;
  void evalstrLengthCommon(CheckerContext &C, const CallExpr *CE, 
                           bool IsStrnlen = false) const;

  void evalStrcpy(CheckerContext &C, const CallExpr *CE) const;
  void evalStrncpy(CheckerContext &C, const CallExpr *CE) const;
  void evalStpcpy(CheckerContext &C, const CallExpr *CE) const;
  void evalStrcpyCommon(CheckerContext &C, const CallExpr *CE, bool returnEnd,
                        bool isBounded, bool isAppending) const;

  void evalStrcat(CheckerContext &C, const CallExpr *CE) const;
  void evalStrncat(CheckerContext &C, const CallExpr *CE) const;

  void evalStrcmp(CheckerContext &C, const CallExpr *CE) const;
  void evalStrncmp(CheckerContext &C, const CallExpr *CE) const;
  void evalStrcasecmp(CheckerContext &C, const CallExpr *CE) const;
  void evalStrncasecmp(CheckerContext &C, const CallExpr *CE) const;
  void evalStrcmpCommon(CheckerContext &C, const CallExpr *CE,
                        bool isBounded = false, bool ignoreCase = false) const;

  // Utility methods
  std::pair<const GRState*, const GRState*>
  static assumeZero(CheckerContext &C,
                    const GRState *state, SVal V, QualType Ty);

  static const GRState *setCStringLength(const GRState *state,
                                         const MemRegion *MR, SVal strLength);
  static SVal getCStringLengthForRegion(CheckerContext &C,
                                        const GRState *&state,
                                        const Expr *Ex, const MemRegion *MR,
                                        bool hypothetical);
  SVal getCStringLength(CheckerContext &C, const GRState *&state,
                        const Expr *Ex, SVal Buf,
                        bool hypothetical = false) const;

  const StringLiteral *getCStringLiteral(CheckerContext &C, 
                                         const GRState *&state,
                                         const Expr *expr,  
                                         SVal val) const;

  static const GRState *InvalidateBuffer(CheckerContext &C,
                                         const GRState *state,
                                         const Expr *Ex, SVal V);

  static bool SummarizeRegion(llvm::raw_ostream& os, ASTContext& Ctx,
                              const MemRegion *MR);

  // Re-usable checks
  const GRState *checkNonNull(CheckerContext &C, const GRState *state,
                               const Expr *S, SVal l) const;
  const GRState *CheckLocation(CheckerContext &C, const GRState *state,
                               const Expr *S, SVal l,
                               const char *message = NULL) const;
  const GRState *CheckBufferAccess(CheckerContext &C, const GRState *state,
                                   const Expr *Size,
                                   const Expr *FirstBuf,
                                   const Expr *SecondBuf,
                                   const char *firstMessage = NULL,
                                   const char *secondMessage = NULL,
                                   bool WarnAboutSize = false) const;
  const GRState *CheckBufferAccess(CheckerContext &C, const GRState *state,
                                   const Expr *Size, const Expr *Buf,
                                   const char *message = NULL,
                                   bool WarnAboutSize = false) const {
    // This is a convenience override.
    return CheckBufferAccess(C, state, Size, Buf, NULL, message, NULL,
                             WarnAboutSize);
  }
  const GRState *CheckOverlap(CheckerContext &C, const GRState *state,
                              const Expr *Size, const Expr *First,
                              const Expr *Second) const;
  void emitOverlapBug(CheckerContext &C, const GRState *state,
                      const Stmt *First, const Stmt *Second) const;
  const GRState *checkAdditionOverflow(CheckerContext &C, const GRState *state,
                                       NonLoc left, NonLoc right) const;
};

class CStringLength {
public:
  typedef llvm::ImmutableMap<const MemRegion *, SVal> EntryMap;
};
} //end anonymous namespace

namespace clang {
namespace ento {
  template <>
  struct GRStateTrait<CStringLength> 
    : public GRStatePartialTrait<CStringLength::EntryMap> {
    static void *GDMIndex() { return CStringChecker::getTag(); }
  };
}
}

//===----------------------------------------------------------------------===//
// Individual checks and utility methods.
//===----------------------------------------------------------------------===//

std::pair<const GRState*, const GRState*>
CStringChecker::assumeZero(CheckerContext &C, const GRState *state, SVal V,
                           QualType Ty) {
  DefinedSVal *val = dyn_cast<DefinedSVal>(&V);
  if (!val)
    return std::pair<const GRState*, const GRState *>(state, state);

  SValBuilder &svalBuilder = C.getSValBuilder();
  DefinedOrUnknownSVal zero = svalBuilder.makeZeroVal(Ty);
  return state->assume(svalBuilder.evalEQ(state, *val, zero));
}

const GRState *CStringChecker::checkNonNull(CheckerContext &C,
                                            const GRState *state,
                                            const Expr *S, SVal l) const {
  // If a previous check has failed, propagate the failure.
  if (!state)
    return NULL;

  const GRState *stateNull, *stateNonNull;
  llvm::tie(stateNull, stateNonNull) = assumeZero(C, state, l, S->getType());

  if (stateNull && !stateNonNull) {
    ExplodedNode *N = C.generateSink(stateNull);
    if (!N)
      return NULL;

    if (!BT_Null)
      BT_Null.reset(new BuiltinBug("API",
        "Null pointer argument in call to byte string function"));

    llvm::SmallString<80> buf;
    llvm::raw_svector_ostream os(buf);
    assert(CurrentFunctionDescription);
    os << "Null pointer argument in call to " << CurrentFunctionDescription;

    // Generate a report for this bug.
    BuiltinBug *BT = static_cast<BuiltinBug*>(BT_Null.get());
    EnhancedBugReport *report = new EnhancedBugReport(*BT, os.str(), N);

    report->addRange(S->getSourceRange());
    report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, S);
    C.EmitReport(report);
    return NULL;
  }

  // From here on, assume that the value is non-null.
  assert(stateNonNull);
  return stateNonNull;
}

// FIXME: This was originally copied from ArrayBoundChecker.cpp. Refactor?
const GRState *CStringChecker::CheckLocation(CheckerContext &C,
                                             const GRState *state,
                                             const Expr *S, SVal l,
                                             const char *warningMsg) const {
  // If a previous check has failed, propagate the failure.
  if (!state)
    return NULL;

  // Check for out of bound array element access.
  const MemRegion *R = l.getAsRegion();
  if (!R)
    return state;

  const ElementRegion *ER = dyn_cast<ElementRegion>(R);
  if (!ER)
    return state;

  assert(ER->getValueType() == C.getASTContext().CharTy &&
    "CheckLocation should only be called with char* ElementRegions");

  // Get the size of the array.
  const SubRegion *superReg = cast<SubRegion>(ER->getSuperRegion());
  SValBuilder &svalBuilder = C.getSValBuilder();
  SVal Extent = 
    svalBuilder.convertToArrayIndex(superReg->getExtent(svalBuilder));
  DefinedOrUnknownSVal Size = cast<DefinedOrUnknownSVal>(Extent);

  // Get the index of the accessed element.
  DefinedOrUnknownSVal Idx = cast<DefinedOrUnknownSVal>(ER->getIndex());

  const GRState *StInBound = state->assumeInBound(Idx, Size, true);
  const GRState *StOutBound = state->assumeInBound(Idx, Size, false);
  if (StOutBound && !StInBound) {
    ExplodedNode *N = C.generateSink(StOutBound);
    if (!N)
      return NULL;

    if (!BT_Bounds) {
      BT_Bounds.reset(new BuiltinBug("Out-of-bound array access",
        "Byte string function accesses out-of-bound array element"));
    }
    BuiltinBug *BT = static_cast<BuiltinBug*>(BT_Bounds.get());

    // Generate a report for this bug.
    RangedBugReport *report;
    if (warningMsg) {
      report = new RangedBugReport(*BT, warningMsg, N);
    } else {
      assert(CurrentFunctionDescription);
      assert(CurrentFunctionDescription[0] != '\0');

      llvm::SmallString<80> buf;
      llvm::raw_svector_ostream os(buf);
      os << (char)toupper(CurrentFunctionDescription[0])
         << &CurrentFunctionDescription[1]
         << " accesses out-of-bound array element";
      report = new RangedBugReport(*BT, os.str(), N);      
    }

    // FIXME: It would be nice to eventually make this diagnostic more clear,
    // e.g., by referencing the original declaration or by saying *why* this
    // reference is outside the range.

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
                                                 const Expr *SecondBuf,
                                                 const char *firstMessage,
                                                 const char *secondMessage,
                                                 bool WarnAboutSize) const {
  // If a previous check has failed, propagate the failure.
  if (!state)
    return NULL;

  SValBuilder &svalBuilder = C.getSValBuilder();
  ASTContext &Ctx = svalBuilder.getContext();

  QualType sizeTy = Size->getType();
  QualType PtrTy = Ctx.getPointerType(Ctx.CharTy);

  // Check that the first buffer is non-null.
  SVal BufVal = state->getSVal(FirstBuf);
  state = checkNonNull(C, state, FirstBuf, BufVal);
  if (!state)
    return NULL;

  // Get the access length and make sure it is known.
  // FIXME: This assumes the caller has already checked that the access length
  // is positive. And that it's unsigned.
  SVal LengthVal = state->getSVal(Size);
  NonLoc *Length = dyn_cast<NonLoc>(&LengthVal);
  if (!Length)
    return state;

  // Compute the offset of the last element to be accessed: size-1.
  NonLoc One = cast<NonLoc>(svalBuilder.makeIntVal(1, sizeTy));
  NonLoc LastOffset = cast<NonLoc>(svalBuilder.evalBinOpNN(state, BO_Sub,
                                                    *Length, One, sizeTy));

  // Check that the first buffer is sufficiently long.
  SVal BufStart = svalBuilder.evalCast(BufVal, PtrTy, FirstBuf->getType());
  if (Loc *BufLoc = dyn_cast<Loc>(&BufStart)) {
    const Expr *warningExpr = (WarnAboutSize ? Size : FirstBuf);

    SVal BufEnd = svalBuilder.evalBinOpLN(state, BO_Add, *BufLoc,
                                          LastOffset, PtrTy);
    state = CheckLocation(C, state, warningExpr, BufEnd, firstMessage);

    // If the buffer isn't large enough, abort.
    if (!state)
      return NULL;
  }

  // If there's a second buffer, check it as well.
  if (SecondBuf) {
    BufVal = state->getSVal(SecondBuf);
    state = checkNonNull(C, state, SecondBuf, BufVal);
    if (!state)
      return NULL;

    BufStart = svalBuilder.evalCast(BufVal, PtrTy, SecondBuf->getType());
    if (Loc *BufLoc = dyn_cast<Loc>(&BufStart)) {
      const Expr *warningExpr = (WarnAboutSize ? Size : SecondBuf);

      SVal BufEnd = svalBuilder.evalBinOpLN(state, BO_Add, *BufLoc,
                                            LastOffset, PtrTy);
      state = CheckLocation(C, state, warningExpr, BufEnd, secondMessage);
    }
  }

  // Large enough or not, return this state!
  return state;
}

const GRState *CStringChecker::CheckOverlap(CheckerContext &C,
                                            const GRState *state,
                                            const Expr *Size,
                                            const Expr *First,
                                            const Expr *Second) const {
  // Do a simple check for overlap: if the two arguments are from the same
  // buffer, see if the end of the first is greater than the start of the second
  // or vice versa.

  // If a previous check has failed, propagate the failure.
  if (!state)
    return NULL;

  const GRState *stateTrue, *stateFalse;

  // Get the buffer values and make sure they're known locations.
  SVal firstVal = state->getSVal(First);
  SVal secondVal = state->getSVal(Second);

  Loc *firstLoc = dyn_cast<Loc>(&firstVal);
  if (!firstLoc)
    return state;

  Loc *secondLoc = dyn_cast<Loc>(&secondVal);
  if (!secondLoc)
    return state;

  // Are the two values the same?
  SValBuilder &svalBuilder = C.getSValBuilder();  
  llvm::tie(stateTrue, stateFalse) =
    state->assume(svalBuilder.evalEQ(state, *firstLoc, *secondLoc));

  if (stateTrue && !stateFalse) {
    // If the values are known to be equal, that's automatically an overlap.
    emitOverlapBug(C, stateTrue, First, Second);
    return NULL;
  }

  // assume the two expressions are not equal.
  assert(stateFalse);
  state = stateFalse;

  // Which value comes first?
  QualType cmpTy = svalBuilder.getConditionType();
  SVal reverse = svalBuilder.evalBinOpLL(state, BO_GT,
                                         *firstLoc, *secondLoc, cmpTy);
  DefinedOrUnknownSVal *reverseTest = dyn_cast<DefinedOrUnknownSVal>(&reverse);
  if (!reverseTest)
    return state;

  llvm::tie(stateTrue, stateFalse) = state->assume(*reverseTest);
  if (stateTrue) {
    if (stateFalse) {
      // If we don't know which one comes first, we can't perform this test.
      return state;
    } else {
      // Switch the values so that firstVal is before secondVal.
      Loc *tmpLoc = firstLoc;
      firstLoc = secondLoc;
      secondLoc = tmpLoc;

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
  ASTContext &Ctx = svalBuilder.getContext();
  QualType CharPtrTy = Ctx.getPointerType(Ctx.CharTy);
  SVal FirstStart = svalBuilder.evalCast(*firstLoc, CharPtrTy, 
                                         First->getType());
  Loc *FirstStartLoc = dyn_cast<Loc>(&FirstStart);
  if (!FirstStartLoc)
    return state;

  // Compute the end of the first buffer. Bail out if THAT fails.
  SVal FirstEnd = svalBuilder.evalBinOpLN(state, BO_Add,
                                 *FirstStartLoc, *Length, CharPtrTy);
  Loc *FirstEndLoc = dyn_cast<Loc>(&FirstEnd);
  if (!FirstEndLoc)
    return state;

  // Is the end of the first buffer past the start of the second buffer?
  SVal Overlap = svalBuilder.evalBinOpLL(state, BO_GT,
                                *FirstEndLoc, *secondLoc, cmpTy);
  DefinedOrUnknownSVal *OverlapTest = dyn_cast<DefinedOrUnknownSVal>(&Overlap);
  if (!OverlapTest)
    return state;

  llvm::tie(stateTrue, stateFalse) = state->assume(*OverlapTest);

  if (stateTrue && !stateFalse) {
    // Overlap!
    emitOverlapBug(C, stateTrue, First, Second);
    return NULL;
  }

  // assume the two expressions don't overlap.
  assert(stateFalse);
  return stateFalse;
}

void CStringChecker::emitOverlapBug(CheckerContext &C, const GRState *state,
                                  const Stmt *First, const Stmt *Second) const {
  ExplodedNode *N = C.generateSink(state);
  if (!N)
    return;

  if (!BT_Overlap)
    BT_Overlap.reset(new BugType("Unix API", "Improper arguments"));

  // Generate a report for this bug.
  RangedBugReport *report = 
    new RangedBugReport(*BT_Overlap,
      "Arguments must not be overlapping buffers", N);
  report->addRange(First->getSourceRange());
  report->addRange(Second->getSourceRange());

  C.EmitReport(report);
}

const GRState *CStringChecker::checkAdditionOverflow(CheckerContext &C,
                                                     const GRState *state,
                                                     NonLoc left,
                                                     NonLoc right) const {
  // If a previous check has failed, propagate the failure.
  if (!state)
    return NULL;

  SValBuilder &svalBuilder = C.getSValBuilder();
  BasicValueFactory &BVF = svalBuilder.getBasicValueFactory();

  QualType sizeTy = svalBuilder.getContext().getSizeType();
  const llvm::APSInt &maxValInt = BVF.getMaxValue(sizeTy);
  NonLoc maxVal = svalBuilder.makeIntVal(maxValInt);

  SVal maxMinusRight = svalBuilder.evalBinOpNN(state, BO_Sub, maxVal, right,
                                               sizeTy);

  if (maxMinusRight.isUnknownOrUndef()) {
    // Try switching the operands. (The order of these two assignments is
    // important!)
    maxMinusRight = svalBuilder.evalBinOpNN(state, BO_Sub, maxVal, left, 
                                            sizeTy);
    left = right;
  }

  if (NonLoc *maxMinusRightNL = dyn_cast<NonLoc>(&maxMinusRight)) {
    QualType cmpTy = svalBuilder.getConditionType();
    // If left > max - right, we have an overflow.
    SVal willOverflow = svalBuilder.evalBinOpNN(state, BO_GT, left,
                                                *maxMinusRightNL, cmpTy);

    const GRState *stateOverflow, *stateOkay;
    llvm::tie(stateOverflow, stateOkay) =
      state->assume(cast<DefinedOrUnknownSVal>(willOverflow));

    if (stateOverflow && !stateOkay) {
      // We have an overflow. Emit a bug report.
      ExplodedNode *N = C.generateSink(stateOverflow);
      if (!N)
        return NULL;

      if (!BT_AdditionOverflow)
        BT_AdditionOverflow.reset(new BuiltinBug("API",
          "Sum of expressions causes overflow"));

      llvm::SmallString<120> buf;
      llvm::raw_svector_ostream os(buf);
      // This isn't a great error message, but this should never occur in real
      // code anyway -- you'd have to create a buffer longer than a size_t can
      // represent, which is sort of a contradiction.
      os << "This expression will create a string whose length is too big to "
         << "be represented as a size_t";

      // Generate a report for this bug.
      BugReport *report = new BugReport(*BT_AdditionOverflow, os.str(), N);
      C.EmitReport(report);        

      return NULL;
    }

    // From now on, assume an overflow didn't occur.
    assert(stateOkay);
    state = stateOkay;
  }

  return state;
}

const GRState *CStringChecker::setCStringLength(const GRState *state,
                                                const MemRegion *MR,
                                                SVal strLength) {
  assert(!strLength.isUndef() && "Attempt to set an undefined string length");

  MR = MR->StripCasts();

  switch (MR->getKind()) {
  case MemRegion::StringRegionKind:
    // FIXME: This can happen if we strcpy() into a string region. This is
    // undefined [C99 6.4.5p6], but we should still warn about it.
    return state;

  case MemRegion::SymbolicRegionKind:
  case MemRegion::AllocaRegionKind:
  case MemRegion::VarRegionKind:
  case MemRegion::FieldRegionKind:
  case MemRegion::ObjCIvarRegionKind:
    // These are the types we can currently track string lengths for.
    break;

  case MemRegion::ElementRegionKind:
    // FIXME: Handle element regions by upper-bounding the parent region's
    // string length.
    return state;

  default:
    // Other regions (mostly non-data) can't have a reliable C string length.
    // For now, just ignore the change.
    // FIXME: These are rare but not impossible. We should output some kind of
    // warning for things like strcpy((char[]){'a', 0}, "b");
    return state;
  }

  if (strLength.isUnknown())
    return state->remove<CStringLength>(MR);

  return state->set<CStringLength>(MR, strLength);
}

SVal CStringChecker::getCStringLengthForRegion(CheckerContext &C,
                                               const GRState *&state,
                                               const Expr *Ex,
                                               const MemRegion *MR,
                                               bool hypothetical) {
  if (!hypothetical) {
    // If there's a recorded length, go ahead and return it.
    const SVal *Recorded = state->get<CStringLength>(MR);
    if (Recorded)
      return *Recorded;
  }
  
  // Otherwise, get a new symbol and update the state.
  unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
  SValBuilder &svalBuilder = C.getSValBuilder();
  QualType sizeTy = svalBuilder.getContext().getSizeType();
  SVal strLength = svalBuilder.getMetadataSymbolVal(CStringChecker::getTag(),
                                                    MR, Ex, sizeTy, Count);

  if (!hypothetical)
    state = state->set<CStringLength>(MR, strLength);

  return strLength;
}

SVal CStringChecker::getCStringLength(CheckerContext &C, const GRState *&state,
                                      const Expr *Ex, SVal Buf,
                                      bool hypothetical) const {
  const MemRegion *MR = Buf.getAsRegion();
  if (!MR) {
    // If we can't get a region, see if it's something we /know/ isn't a
    // C string. In the context of locations, the only time we can issue such
    // a warning is for labels.
    if (loc::GotoLabel *Label = dyn_cast<loc::GotoLabel>(&Buf)) {
      if (ExplodedNode *N = C.generateNode(state)) {
        if (!BT_NotCString)
          BT_NotCString.reset(new BuiltinBug("API",
            "Argument is not a null-terminated string."));

        llvm::SmallString<120> buf;
        llvm::raw_svector_ostream os(buf);
        assert(CurrentFunctionDescription);
        os << "Argument to " << CurrentFunctionDescription
           << " is the address of the label '" << Label->getLabel()->getName()
           << "', which is not a null-terminated string";

        // Generate a report for this bug.
        EnhancedBugReport *report = new EnhancedBugReport(*BT_NotCString,
                                                          os.str(), N);

        report->addRange(Ex->getSourceRange());
        C.EmitReport(report);        
      }

      return UndefinedVal();
    }

    // If it's not a region and not a label, give up.
    return UnknownVal();
  }

  // If we have a region, strip casts from it and see if we can figure out
  // its length. For anything we can't figure out, just return UnknownVal.
  MR = MR->StripCasts();

  switch (MR->getKind()) {
  case MemRegion::StringRegionKind: {
    // Modifying the contents of string regions is undefined [C99 6.4.5p6],
    // so we can assume that the byte length is the correct C string length.
    SValBuilder &svalBuilder = C.getSValBuilder();
    QualType sizeTy = svalBuilder.getContext().getSizeType();
    const StringLiteral *strLit = cast<StringRegion>(MR)->getStringLiteral();
    return svalBuilder.makeIntVal(strLit->getByteLength(), sizeTy);
  }
  case MemRegion::SymbolicRegionKind:
  case MemRegion::AllocaRegionKind:
  case MemRegion::VarRegionKind:
  case MemRegion::FieldRegionKind:
  case MemRegion::ObjCIvarRegionKind:
    return getCStringLengthForRegion(C, state, Ex, MR, hypothetical);
  case MemRegion::CompoundLiteralRegionKind:
    // FIXME: Can we track this? Is it necessary?
    return UnknownVal();
  case MemRegion::ElementRegionKind:
    // FIXME: How can we handle this? It's not good enough to subtract the
    // offset from the base string length; consider "123\x00567" and &a[5].
    return UnknownVal();
  default:
    // Other regions (mostly non-data) can't have a reliable C string length.
    // In this case, an error is emitted and UndefinedVal is returned.
    // The caller should always be prepared to handle this case.
    if (ExplodedNode *N = C.generateNode(state)) {
      if (!BT_NotCString)
        BT_NotCString.reset(new BuiltinBug("API",
          "Argument is not a null-terminated string."));

      llvm::SmallString<120> buf;
      llvm::raw_svector_ostream os(buf);

      assert(CurrentFunctionDescription);
      os << "Argument to " << CurrentFunctionDescription << " is ";

      if (SummarizeRegion(os, C.getASTContext(), MR))
        os << ", which is not a null-terminated string";
      else
        os << "not a null-terminated string";

      // Generate a report for this bug.
      EnhancedBugReport *report = new EnhancedBugReport(*BT_NotCString,
                                                        os.str(), N);

      report->addRange(Ex->getSourceRange());
      C.EmitReport(report);        
    }

    return UndefinedVal();
  }
}

const StringLiteral *CStringChecker::getCStringLiteral(CheckerContext &C,
  const GRState *&state, const Expr *expr, SVal val) const {

  // Get the memory region pointed to by the val.
  const MemRegion *bufRegion = val.getAsRegion();
  if (!bufRegion)
    return NULL; 

  // Strip casts off the memory region.
  bufRegion = bufRegion->StripCasts();

  // Cast the memory region to a string region.
  const StringRegion *strRegion= dyn_cast<StringRegion>(bufRegion);
  if (!strRegion)
    return NULL; 

  // Return the actual string in the string region.
  return strRegion->getStringLiteral();
}

const GRState *CStringChecker::InvalidateBuffer(CheckerContext &C,
                                                const GRState *state,
                                                const Expr *E, SVal V) {
  Loc *L = dyn_cast<Loc>(&V);
  if (!L)
    return state;

  // FIXME: This is a simplified version of what's in CFRefCount.cpp -- it makes
  // some assumptions about the value that CFRefCount can't. Even so, it should
  // probably be refactored.
  if (loc::MemRegionVal* MR = dyn_cast<loc::MemRegionVal>(L)) {
    const MemRegion *R = MR->getRegion()->StripCasts();

    // Are we dealing with an ElementRegion?  If so, we should be invalidating
    // the super-region.
    if (const ElementRegion *ER = dyn_cast<ElementRegion>(R)) {
      R = ER->getSuperRegion();
      // FIXME: What about layers of ElementRegions?
    }

    // Invalidate this region.
    unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
    return state->invalidateRegion(R, E, Count, NULL);
  }

  // If we have a non-region value by chance, just remove the binding.
  // FIXME: is this necessary or correct? This handles the non-Region
  //  cases.  Is it ever valid to store to these?
  return state->unbindLoc(*L);
}

bool CStringChecker::SummarizeRegion(llvm::raw_ostream& os, ASTContext& Ctx,
                                     const MemRegion *MR) {
  const TypedRegion *TR = dyn_cast<TypedRegion>(MR);
  if (!TR)
    return false;

  switch (TR->getKind()) {
  case MemRegion::FunctionTextRegionKind: {
    const FunctionDecl *FD = cast<FunctionTextRegion>(TR)->getDecl();
    if (FD)
      os << "the address of the function '" << FD << "'";
    else
      os << "the address of a function";
    return true;
  }
  case MemRegion::BlockTextRegionKind:
    os << "block text";
    return true;
  case MemRegion::BlockDataRegionKind:
    os << "a block";
    return true;
  case MemRegion::CXXThisRegionKind:
  case MemRegion::CXXTempObjectRegionKind:
    os << "a C++ temp object of type " << TR->getValueType().getAsString();
    return true;
  case MemRegion::VarRegionKind:
    os << "a variable of type" << TR->getValueType().getAsString();
    return true;
  case MemRegion::FieldRegionKind:
    os << "a field of type " << TR->getValueType().getAsString();
    return true;
  case MemRegion::ObjCIvarRegionKind:
    os << "an instance variable of type " << TR->getValueType().getAsString();
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// evaluation of individual function calls.
//===----------------------------------------------------------------------===//

void CStringChecker::evalCopyCommon(CheckerContext &C, 
                                    const CallExpr *CE,
                                    const GRState *state,
                                    const Expr *Size, const Expr *Dest,
                                    const Expr *Source, bool Restricted,
                                    bool IsMempcpy) const {
  CurrentFunctionDescription = "memory copy function";

  // See if the size argument is zero.
  SVal sizeVal = state->getSVal(Size);
  QualType sizeTy = Size->getType();

  const GRState *stateZeroSize, *stateNonZeroSize;
  llvm::tie(stateZeroSize, stateNonZeroSize) =
    assumeZero(C, state, sizeVal, sizeTy);

  // Get the value of the Dest.
  SVal destVal = state->getSVal(Dest);

  // If the size is zero, there won't be any actual memory access, so
  // just bind the return value to the destination buffer and return.
  if (stateZeroSize) {
    stateZeroSize = stateZeroSize->BindExpr(CE, destVal);
    C.addTransition(stateZeroSize);
  }

  // If the size can be nonzero, we have to check the other arguments.
  if (stateNonZeroSize) {
    state = stateNonZeroSize;

    // Ensure the destination is not null. If it is NULL there will be a
    // NULL pointer dereference.
    state = checkNonNull(C, state, Dest, destVal);
    if (!state)
      return;

    // Get the value of the Src.
    SVal srcVal = state->getSVal(Source);
    
    // Ensure the source is not null. If it is NULL there will be a
    // NULL pointer dereference.
    state = checkNonNull(C, state, Source, srcVal);
    if (!state)
      return;

    // Ensure the accesses are valid and that the buffers do not overlap.
    const char * const writeWarning =
      "Memory copy function overflows destination buffer";
    state = CheckBufferAccess(C, state, Size, Dest, Source,
                              writeWarning, /* sourceWarning = */ NULL);
    if (Restricted)
      state = CheckOverlap(C, state, Size, Dest, Source);

    if (!state)
      return;

    // If this is mempcpy, get the byte after the last byte copied and 
    // bind the expr.
    if (IsMempcpy) {
      loc::MemRegionVal *destRegVal = dyn_cast<loc::MemRegionVal>(&destVal);
      assert(destRegVal && "Destination should be a known MemRegionVal here");
      
      // Get the length to copy.
      NonLoc *lenValNonLoc = dyn_cast<NonLoc>(&sizeVal);
      
      if (lenValNonLoc) {
        // Get the byte after the last byte copied.
        SVal lastElement = C.getSValBuilder().evalBinOpLN(state, BO_Add, 
                                                          *destRegVal,
                                                          *lenValNonLoc, 
                                                          Dest->getType());
      
        // The byte after the last byte copied is the return value.
        state = state->BindExpr(CE, lastElement);
      } else {
        // If we don't know how much we copied, we can at least
        // conjure a return value for later.
        unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
        SVal result =
          C.getSValBuilder().getConjuredSymbolVal(NULL, CE, Count);
        state = state->BindExpr(CE, result);
      }

    } else {
      // All other copies return the destination buffer.
      // (Well, bcopy() has a void return type, but this won't hurt.)
      state = state->BindExpr(CE, destVal);
    }

    // Invalidate the destination.
    // FIXME: Even if we can't perfectly model the copy, we should see if we
    // can use LazyCompoundVals to copy the source values into the destination.
    // This would probably remove any existing bindings past the end of the
    // copied region, but that's still an improvement over blank invalidation.
    state = InvalidateBuffer(C, state, Dest, state->getSVal(Dest));
    C.addTransition(state);
  }
}


void CStringChecker::evalMemcpy(CheckerContext &C, const CallExpr *CE) const {
  // void *memcpy(void *restrict dst, const void *restrict src, size_t n);
  // The return value is the address of the destination buffer.
  const Expr *Dest = CE->getArg(0);
  const GRState *state = C.getState();

  evalCopyCommon(C, CE, state, CE->getArg(2), Dest, CE->getArg(1), true);
}

void CStringChecker::evalMempcpy(CheckerContext &C, const CallExpr *CE) const {
  // void *mempcpy(void *restrict dst, const void *restrict src, size_t n);
  // The return value is a pointer to the byte following the last written byte.
  const Expr *Dest = CE->getArg(0);
  const GRState *state = C.getState();
  
  evalCopyCommon(C, CE, state, CE->getArg(2), Dest, CE->getArg(1), true, true);
}

void CStringChecker::evalMemmove(CheckerContext &C, const CallExpr *CE) const {
  // void *memmove(void *dst, const void *src, size_t n);
  // The return value is the address of the destination buffer.
  const Expr *Dest = CE->getArg(0);
  const GRState *state = C.getState();

  evalCopyCommon(C, CE, state, CE->getArg(2), Dest, CE->getArg(1));
}

void CStringChecker::evalBcopy(CheckerContext &C, const CallExpr *CE) const {
  // void bcopy(const void *src, void *dst, size_t n);
  evalCopyCommon(C, CE, C.getState(), 
                 CE->getArg(2), CE->getArg(1), CE->getArg(0));
}

void CStringChecker::evalMemcmp(CheckerContext &C, const CallExpr *CE) const {
  // int memcmp(const void *s1, const void *s2, size_t n);
  CurrentFunctionDescription = "memory comparison function";

  const Expr *Left = CE->getArg(0);
  const Expr *Right = CE->getArg(1);
  const Expr *Size = CE->getArg(2);

  const GRState *state = C.getState();
  SValBuilder &svalBuilder = C.getSValBuilder();

  // See if the size argument is zero.
  SVal sizeVal = state->getSVal(Size);
  QualType sizeTy = Size->getType();

  const GRState *stateZeroSize, *stateNonZeroSize;
  llvm::tie(stateZeroSize, stateNonZeroSize) =
    assumeZero(C, state, sizeVal, sizeTy);

  // If the size can be zero, the result will be 0 in that case, and we don't
  // have to check either of the buffers.
  if (stateZeroSize) {
    state = stateZeroSize;
    state = state->BindExpr(CE, svalBuilder.makeZeroVal(CE->getType()));
    C.addTransition(state);
  }

  // If the size can be nonzero, we have to check the other arguments.
  if (stateNonZeroSize) {
    state = stateNonZeroSize;
    // If we know the two buffers are the same, we know the result is 0.
    // First, get the two buffers' addresses. Another checker will have already
    // made sure they're not undefined.
    DefinedOrUnknownSVal LV = cast<DefinedOrUnknownSVal>(state->getSVal(Left));
    DefinedOrUnknownSVal RV = cast<DefinedOrUnknownSVal>(state->getSVal(Right));

    // See if they are the same.
    DefinedOrUnknownSVal SameBuf = svalBuilder.evalEQ(state, LV, RV);
    const GRState *StSameBuf, *StNotSameBuf;
    llvm::tie(StSameBuf, StNotSameBuf) = state->assume(SameBuf);

    // If the two arguments might be the same buffer, we know the result is 0,
    // and we only need to check one size.
    if (StSameBuf) {
      state = StSameBuf;
      state = CheckBufferAccess(C, state, Size, Left);
      if (state) {
        state = StSameBuf->BindExpr(CE, svalBuilder.makeZeroVal(CE->getType()));
        C.addTransition(state); 
      }
    }

    // If the two arguments might be different buffers, we have to check the
    // size of both of them.
    if (StNotSameBuf) {
      state = StNotSameBuf;
      state = CheckBufferAccess(C, state, Size, Left, Right);
      if (state) {
        // The return value is the comparison result, which we don't know.
        unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
        SVal CmpV = svalBuilder.getConjuredSymbolVal(NULL, CE, Count);
        state = state->BindExpr(CE, CmpV);
        C.addTransition(state);
      }
    }
  }
}

void CStringChecker::evalstrLength(CheckerContext &C,
                                   const CallExpr *CE) const {
  // size_t strlen(const char *s);
  evalstrLengthCommon(C, CE, /* IsStrnlen = */ false);
}

void CStringChecker::evalstrnLength(CheckerContext &C,
                                    const CallExpr *CE) const {
  // size_t strnlen(const char *s, size_t maxlen);
  evalstrLengthCommon(C, CE, /* IsStrnlen = */ true);
}

void CStringChecker::evalstrLengthCommon(CheckerContext &C, const CallExpr *CE,
                                         bool IsStrnlen) const {
  CurrentFunctionDescription = "string length function";
  const GRState *state = C.getState();

  if (IsStrnlen) {
    const Expr *maxlenExpr = CE->getArg(1);
    SVal maxlenVal = state->getSVal(maxlenExpr);

    const GRState *stateZeroSize, *stateNonZeroSize;
    llvm::tie(stateZeroSize, stateNonZeroSize) =
      assumeZero(C, state, maxlenVal, maxlenExpr->getType());

    // If the size can be zero, the result will be 0 in that case, and we don't
    // have to check the string itself.
    if (stateZeroSize) {
      SVal zero = C.getSValBuilder().makeZeroVal(CE->getType());
      stateZeroSize = stateZeroSize->BindExpr(CE, zero);
      C.addTransition(stateZeroSize);
    }

    // If the size is GUARANTEED to be zero, we're done!
    if (!stateNonZeroSize)
      return;

    // Otherwise, record the assumption that the size is nonzero.
    state = stateNonZeroSize;
  }

  // Check that the string argument is non-null.
  const Expr *Arg = CE->getArg(0);
  SVal ArgVal = state->getSVal(Arg);

  state = checkNonNull(C, state, Arg, ArgVal);

  if (!state)
    return;

  SVal strLength = getCStringLength(C, state, Arg, ArgVal);

  // If the argument isn't a valid C string, there's no valid state to
  // transition to.
  if (strLength.isUndef())
    return;

  DefinedOrUnknownSVal result = UnknownVal();

  // If the check is for strnlen() then bind the return value to no more than
  // the maxlen value.
  if (IsStrnlen) {
    QualType cmpTy = C.getSValBuilder().getConditionType();

    // It's a little unfortunate to be getting this again,
    // but it's not that expensive...
    const Expr *maxlenExpr = CE->getArg(1);
    SVal maxlenVal = state->getSVal(maxlenExpr);

    NonLoc *strLengthNL = dyn_cast<NonLoc>(&strLength);
    NonLoc *maxlenValNL = dyn_cast<NonLoc>(&maxlenVal);

    if (strLengthNL && maxlenValNL) {
      const GRState *stateStringTooLong, *stateStringNotTooLong;

      // Check if the strLength is greater than the maxlen.
      llvm::tie(stateStringTooLong, stateStringNotTooLong) =
        state->assume(cast<DefinedOrUnknownSVal>
                      (C.getSValBuilder().evalBinOpNN(state, BO_GT, 
                                                      *strLengthNL,
                                                      *maxlenValNL,
                                                      cmpTy)));

      if (stateStringTooLong && !stateStringNotTooLong) {
        // If the string is longer than maxlen, return maxlen.
        result = *maxlenValNL;
      } else if (stateStringNotTooLong && !stateStringTooLong) {
        // If the string is shorter than maxlen, return its length.
        result = *strLengthNL;
      }
    }

    if (result.isUnknown()) {
      // If we don't have enough information for a comparison, there's
      // no guarantee the full string length will actually be returned.
      // All we know is the return value is the min of the string length
      // and the limit. This is better than nothing.
      unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
      result = C.getSValBuilder().getConjuredSymbolVal(NULL, CE, Count);
      NonLoc *resultNL = cast<NonLoc>(&result);

      if (strLengthNL) {
        state = state->assume(cast<DefinedOrUnknownSVal>
                              (C.getSValBuilder().evalBinOpNN(state, BO_LE, 
                                                              *resultNL,
                                                              *strLengthNL,
                                                              cmpTy)), true);
      }
      
      if (maxlenValNL) {
        state = state->assume(cast<DefinedOrUnknownSVal>
                              (C.getSValBuilder().evalBinOpNN(state, BO_LE, 
                                                              *resultNL,
                                                              *maxlenValNL,
                                                              cmpTy)), true);
      }
    }

  } else {
    // This is a plain strlen(), not strnlen().
    result = cast<DefinedOrUnknownSVal>(strLength);

    // If we don't know the length of the string, conjure a return
    // value, so it can be used in constraints, at least.
    if (result.isUnknown()) {
      unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
      result = C.getSValBuilder().getConjuredSymbolVal(NULL, CE, Count);
    }
  }

  // Bind the return value.
  assert(!result.isUnknown() && "Should have conjured a value by now");
  state = state->BindExpr(CE, result);
  C.addTransition(state);
}

void CStringChecker::evalStrcpy(CheckerContext &C, const CallExpr *CE) const {
  // char *strcpy(char *restrict dst, const char *restrict src);
  evalStrcpyCommon(C, CE, 
                   /* returnEnd = */ false, 
                   /* isBounded = */ false,
                   /* isAppending = */ false);
}

void CStringChecker::evalStrncpy(CheckerContext &C, const CallExpr *CE) const {
  // char *strncpy(char *restrict dst, const char *restrict src, size_t n);
  evalStrcpyCommon(C, CE, 
                   /* returnEnd = */ false, 
                   /* isBounded = */ true,
                   /* isAppending = */ false);
}

void CStringChecker::evalStpcpy(CheckerContext &C, const CallExpr *CE) const {
  // char *stpcpy(char *restrict dst, const char *restrict src);
  evalStrcpyCommon(C, CE, 
                   /* returnEnd = */ true, 
                   /* isBounded = */ false,
                   /* isAppending = */ false);
}

void CStringChecker::evalStrcat(CheckerContext &C, const CallExpr *CE) const {
  //char *strcat(char *restrict s1, const char *restrict s2);
  evalStrcpyCommon(C, CE, 
                   /* returnEnd = */ false, 
                   /* isBounded = */ false,
                   /* isAppending = */ true);
}

void CStringChecker::evalStrncat(CheckerContext &C, const CallExpr *CE) const {
  //char *strncat(char *restrict s1, const char *restrict s2, size_t n);
  evalStrcpyCommon(C, CE, 
                   /* returnEnd = */ false, 
                   /* isBounded = */ true,
                   /* isAppending = */ true);
}

void CStringChecker::evalStrcpyCommon(CheckerContext &C, const CallExpr *CE,
                                      bool returnEnd, bool isBounded,
                                      bool isAppending) const {
  CurrentFunctionDescription = "string copy function";
  const GRState *state = C.getState();

  // Check that the destination is non-null.
  const Expr *Dst = CE->getArg(0);
  SVal DstVal = state->getSVal(Dst);

  state = checkNonNull(C, state, Dst, DstVal);
  if (!state)
    return;

  // Check that the source is non-null.
  const Expr *srcExpr = CE->getArg(1);
  SVal srcVal = state->getSVal(srcExpr);
  state = checkNonNull(C, state, srcExpr, srcVal);
  if (!state)
    return;

  // Get the string length of the source.
  SVal strLength = getCStringLength(C, state, srcExpr, srcVal);

  // If the source isn't a valid C string, give up.
  if (strLength.isUndef())
    return;

  SValBuilder &svalBuilder = C.getSValBuilder();
  QualType cmpTy = svalBuilder.getConditionType();

  SVal amountCopied = UnknownVal();

  // If the function is strncpy, strncat, etc... it is bounded.
  if (isBounded) {
    // Get the max number of characters to copy.
    const Expr *lenExpr = CE->getArg(2);
    SVal lenVal = state->getSVal(lenExpr);

    // Protect against misdeclared strncpy().
    lenVal = svalBuilder.evalCast(lenVal,
                                  svalBuilder.getContext().getSizeType(),
                                  lenExpr->getType());

    NonLoc *strLengthNL = dyn_cast<NonLoc>(&strLength);
    NonLoc *lenValNL = dyn_cast<NonLoc>(&lenVal);

    // If we know both values, we might be able to figure out how much
    // we're copying.
    if (strLengthNL && lenValNL) {
      const GRState *stateSourceTooLong, *stateSourceNotTooLong;

      // Check if the max number to copy is less than the length of the src.
      llvm::tie(stateSourceTooLong, stateSourceNotTooLong) =
        state->assume(cast<DefinedOrUnknownSVal>
                      (svalBuilder.evalBinOpNN(state, BO_GT, *strLengthNL,
                                               *lenValNL, cmpTy)));

      if (stateSourceTooLong && !stateSourceNotTooLong) {
        // Max number to copy is less than the length of the src, so the actual
        // strLength copied is the max number arg.
        state = stateSourceTooLong;
        amountCopied = lenVal;

      } else if (!stateSourceTooLong && stateSourceNotTooLong) {
        // The source buffer entirely fits in the bound.
        state = stateSourceNotTooLong;
        amountCopied = strLength;
      }
    }

    // We still want to know if the bound is known to be too large.
    const char * const warningMsg =
      "Size argument is greater than the length of the destination buffer";
    state = CheckBufferAccess(C, state, lenExpr, Dst, warningMsg,
                              /* WarnAboutSize = */ true);
    // FIXME: It'd be nice for this not to be a hard error, so we can warn
    // about more than one thing, but the multiple calls to CheckLocation
    // result in the same ExplodedNode, which means we don't keep emitting
    // bug reports.
    if (!state)
      return;

    // If we couldn't pin down the copy length, at least bound it.
    if (amountCopied.isUnknown()) {
      // Try to get a "hypothetical" string length symbol, which we can later
      // set as a real value if that turns out to be the case.
      amountCopied = getCStringLength(C, state, lenExpr, srcVal, true);
      assert(!amountCopied.isUndef());

      if (NonLoc *amountCopiedNL = dyn_cast<NonLoc>(&amountCopied)) {
        if (lenValNL) {
          // amountCopied <= lenVal
          SVal copiedLessThanBound = svalBuilder.evalBinOpNN(state, BO_LE,
                                                             *amountCopiedNL,
                                                             *lenValNL,
                                                             cmpTy);
          state = state->assume(cast<DefinedOrUnknownSVal>(copiedLessThanBound),
                                true);
          if (!state)
            return;
        }

        if (strLengthNL) {
          // amountCopied <= strlen(source)
          SVal copiedLessThanSrc = svalBuilder.evalBinOpNN(state, BO_LE,
                                                           *amountCopiedNL,
                                                           *strLengthNL,
                                                           cmpTy);
          state = state->assume(cast<DefinedOrUnknownSVal>(copiedLessThanSrc),
                                true);
          if (!state)
            return;
        }
      }
    }

  } else {
    // The function isn't bounded. The amount copied should match the length
    // of the source buffer.
    amountCopied = strLength;
  }

  assert(state);

  // This represents the number of characters copied into the destination
  // buffer. (It may not actually be the strlen if the destination buffer
  // is not terminated.)
  SVal finalStrLength = UnknownVal();

  // If this is an appending function (strcat, strncat...) then set the
  // string length to strlen(src) + strlen(dst) since the buffer will
  // ultimately contain both.
  if (isAppending) {
    // Get the string length of the destination. If the destination is memory
    // that can't have a string length, we shouldn't be copying into it anyway.
    SVal dstStrLength = getCStringLength(C, state, Dst, DstVal);
    if (dstStrLength.isUndef())
      return;

    QualType sizeTy = svalBuilder.getContext().getSizeType();

    NonLoc *srcStrLengthNL = dyn_cast<NonLoc>(&amountCopied);
    NonLoc *dstStrLengthNL = dyn_cast<NonLoc>(&dstStrLength);
    
    // If we know both string lengths, we might know the final string length.
    if (srcStrLengthNL && dstStrLengthNL) {
      // Make sure the two lengths together don't overflow a size_t.
      state = checkAdditionOverflow(C, state, *srcStrLengthNL, *dstStrLengthNL);
      if (!state)
        return;

      finalStrLength = svalBuilder.evalBinOpNN(state, BO_Add, *srcStrLengthNL, 
                                               *dstStrLengthNL, sizeTy);
    }

    // If we couldn't get a single value for the final string length,
    // we can at least bound it by the individual lengths.
    if (finalStrLength.isUnknown()) {
      // Try to get a "hypothetical" string length symbol, which we can later
      // set as a real value if that turns out to be the case.
      finalStrLength = getCStringLength(C, state, CE, DstVal, true);
      assert(!finalStrLength.isUndef());

      if (NonLoc *finalStrLengthNL = dyn_cast<NonLoc>(&finalStrLength)) {
        if (srcStrLengthNL) {
          // finalStrLength >= srcStrLength
          SVal sourceInResult = svalBuilder.evalBinOpNN(state, BO_GE,
                                                        *finalStrLengthNL,
                                                        *srcStrLengthNL,
                                                        cmpTy);
          state = state->assume(cast<DefinedOrUnknownSVal>(sourceInResult),
                                true);
          if (!state)
            return;
        }

        if (dstStrLengthNL) {
          // finalStrLength >= dstStrLength
          SVal destInResult = svalBuilder.evalBinOpNN(state, BO_GE,
                                                      *finalStrLengthNL,
                                                      *dstStrLengthNL,
                                                      cmpTy);
          state = state->assume(cast<DefinedOrUnknownSVal>(destInResult),
                                true);
          if (!state)
            return;
        }
      }
    }

  } else {
    // Otherwise, this is a copy-over function (strcpy, strncpy, ...), and
    // the final string length will match the input string length.
    finalStrLength = amountCopied;
  }

  // The final result of the function will either be a pointer past the last
  // copied element, or a pointer to the start of the destination buffer.
  SVal Result = (returnEnd ? UnknownVal() : DstVal);

  assert(state);

  // If the destination is a MemRegion, try to check for a buffer overflow and
  // record the new string length.
  if (loc::MemRegionVal *dstRegVal = dyn_cast<loc::MemRegionVal>(&DstVal)) {
    // If the final length is known, we can check for an overflow.
    if (NonLoc *knownStrLength = dyn_cast<NonLoc>(&finalStrLength)) {
      SVal lastElement = svalBuilder.evalBinOpLN(state, BO_Add, *dstRegVal,
                                                 *knownStrLength,
                                                 Dst->getType());

      const char * const warningMsg =
        "String copy function overflows destination buffer";
      state = CheckLocation(C, state, Dst, lastElement, warningMsg);
      if (!state)
        return;

      // If this is a stpcpy-style copy, the last element is the return value.
      if (returnEnd)
        Result = lastElement;
    }

    // Invalidate the destination. This must happen before we set the C string
    // length because invalidation will clear the length.
    // FIXME: Even if we can't perfectly model the copy, we should see if we
    // can use LazyCompoundVals to copy the source values into the destination.
    // This would probably remove any existing bindings past the end of the
    // string, but that's still an improvement over blank invalidation.
    state = InvalidateBuffer(C, state, Dst, *dstRegVal);

    // Set the C string length of the destination, if we know it.
    if (!isBounded || (amountCopied == strLength))
      state = setCStringLength(state, dstRegVal->getRegion(), finalStrLength);
    else
      state = setCStringLength(state, dstRegVal->getRegion(), UnknownVal());
  }

  assert(state);

  // If this is a stpcpy-style copy, but we were unable to check for a buffer
  // overflow, we still need a result. Conjure a return value.
  if (returnEnd && Result.isUnknown()) {
    unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
    Result = svalBuilder.getConjuredSymbolVal(NULL, CE, Count);
  }

  // Set the return value.
  state = state->BindExpr(CE, Result);
  C.addTransition(state);
}

void CStringChecker::evalStrcmp(CheckerContext &C, const CallExpr *CE) const {
  //int strcmp(const char *s1, const char *s2);
  evalStrcmpCommon(C, CE, /* isBounded = */ false, /* ignoreCase = */ false);
}

void CStringChecker::evalStrncmp(CheckerContext &C, const CallExpr *CE) const {
  //int strncmp(const char *s1, const char *s2, size_t n);
  evalStrcmpCommon(C, CE, /* isBounded = */ true, /* ignoreCase = */ false);
}

void CStringChecker::evalStrcasecmp(CheckerContext &C, 
                                    const CallExpr *CE) const {
  //int strcasecmp(const char *s1, const char *s2);
  evalStrcmpCommon(C, CE, /* isBounded = */ false, /* ignoreCase = */ true);
}

void CStringChecker::evalStrncasecmp(CheckerContext &C, 
                                     const CallExpr *CE) const {
  //int strncasecmp(const char *s1, const char *s2, size_t n);
  evalStrcmpCommon(C, CE, /* isBounded = */ true, /* ignoreCase = */ true);
}

void CStringChecker::evalStrcmpCommon(CheckerContext &C, const CallExpr *CE,
                                      bool isBounded, bool ignoreCase) const {
  CurrentFunctionDescription = "string comparison function";
  const GRState *state = C.getState();

  // Check that the first string is non-null
  const Expr *s1 = CE->getArg(0);
  SVal s1Val = state->getSVal(s1);
  state = checkNonNull(C, state, s1, s1Val);
  if (!state)
    return;

  // Check that the second string is non-null.
  const Expr *s2 = CE->getArg(1);
  SVal s2Val = state->getSVal(s2);
  state = checkNonNull(C, state, s2, s2Val);
  if (!state)
    return;

  // Get the string length of the first string or give up.
  SVal s1Length = getCStringLength(C, state, s1, s1Val);
  if (s1Length.isUndef())
    return;

  // Get the string length of the second string or give up.
  SVal s2Length = getCStringLength(C, state, s2, s2Val);
  if (s2Length.isUndef())
    return;

  // If we know the two buffers are the same, we know the result is 0.
  // First, get the two buffers' addresses. Another checker will have already
  // made sure they're not undefined.
  DefinedOrUnknownSVal LV = cast<DefinedOrUnknownSVal>(s1Val);
  DefinedOrUnknownSVal RV = cast<DefinedOrUnknownSVal>(s2Val);

  // See if they are the same.
  SValBuilder &svalBuilder = C.getSValBuilder();
  DefinedOrUnknownSVal SameBuf = svalBuilder.evalEQ(state, LV, RV);
  const GRState *StSameBuf, *StNotSameBuf;
  llvm::tie(StSameBuf, StNotSameBuf) = state->assume(SameBuf);

  // If the two arguments might be the same buffer, we know the result is 0,
  // and we only need to check one size.
  if (StSameBuf) {
    StSameBuf = StSameBuf->BindExpr(CE, svalBuilder.makeZeroVal(CE->getType()));
    C.addTransition(StSameBuf);

    // If the two arguments are GUARANTEED to be the same, we're done!
    if (!StNotSameBuf)
      return;
  }

  assert(StNotSameBuf);
  state = StNotSameBuf;

  // At this point we can go about comparing the two buffers.
  // For now, we only do this if they're both known string literals.

  // Attempt to extract string literals from both expressions.
  const StringLiteral *s1StrLiteral = getCStringLiteral(C, state, s1, s1Val);
  const StringLiteral *s2StrLiteral = getCStringLiteral(C, state, s2, s2Val);
  bool canComputeResult = false;

  if (s1StrLiteral && s2StrLiteral) {
    llvm::StringRef s1StrRef = s1StrLiteral->getString();
    llvm::StringRef s2StrRef = s2StrLiteral->getString();

    if (isBounded) {
      // Get the max number of characters to compare.
      const Expr *lenExpr = CE->getArg(2);
      SVal lenVal = state->getSVal(lenExpr);

      // If the length is known, we can get the right substrings.
      if (const llvm::APSInt *len = svalBuilder.getKnownValue(state, lenVal)) {
        // Create substrings of each to compare the prefix.
        s1StrRef = s1StrRef.substr(0, (size_t)len->getZExtValue());
        s2StrRef = s2StrRef.substr(0, (size_t)len->getZExtValue());
        canComputeResult = true;
      }
    } else {
      // This is a normal, unbounded strcmp.
      canComputeResult = true;
    }

    if (canComputeResult) {
      // Real strcmp stops at null characters.
      size_t s1Term = s1StrRef.find('\0');
      if (s1Term != llvm::StringRef::npos)
        s1StrRef = s1StrRef.substr(0, s1Term);

      size_t s2Term = s2StrRef.find('\0');
      if (s2Term != llvm::StringRef::npos)
        s2StrRef = s2StrRef.substr(0, s2Term);

      // Use StringRef's comparison methods to compute the actual result.
      int result;

      if (ignoreCase) {
        // Compare string 1 to string 2 the same way strcasecmp() does.
        result = s1StrRef.compare_lower(s2StrRef);
      } else {
        // Compare string 1 to string 2 the same way strcmp() does.
        result = s1StrRef.compare(s2StrRef);
      }

      // Build the SVal of the comparison and bind the return value.
      SVal resultVal = svalBuilder.makeIntVal(result, CE->getType());
      state = state->BindExpr(CE, resultVal);
    }
  }

  if (!canComputeResult) {
    // Conjure a symbolic value. It's the best we can do.
    unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
    SVal resultVal = svalBuilder.getConjuredSymbolVal(NULL, CE, Count);
    state = state->BindExpr(CE, resultVal);
  }

  // Record this as a possible path.
  C.addTransition(state);
}

//===----------------------------------------------------------------------===//
// The driver method, and other Checker callbacks.
//===----------------------------------------------------------------------===//

bool CStringChecker::evalCall(const CallExpr *CE, CheckerContext &C) const {
  // Get the callee.  All the functions we care about are C functions
  // with simple identifiers.
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  const FunctionDecl *FD = state->getSVal(Callee).getAsFunctionDecl();

  if (!FD)
    return false;

  // Get the name of the callee. If it's a builtin, strip off the prefix.
  IdentifierInfo *II = FD->getIdentifier();
  if (!II)   // if no identifier, not a simple C function
    return false;
  llvm::StringRef Name = II->getName();
  if (Name.startswith("__builtin_"))
    Name = Name.substr(10);

  FnCheck evalFunction = llvm::StringSwitch<FnCheck>(Name)
    .Cases("memcpy", "__memcpy_chk", &CStringChecker::evalMemcpy)
    .Cases("mempcpy", "__mempcpy_chk", &CStringChecker::evalMempcpy)
    .Cases("memcmp", "bcmp", &CStringChecker::evalMemcmp)
    .Cases("memmove", "__memmove_chk", &CStringChecker::evalMemmove)
    .Cases("strcpy", "__strcpy_chk", &CStringChecker::evalStrcpy)
    .Cases("strncpy", "__strncpy_chk", &CStringChecker::evalStrncpy)
    .Cases("stpcpy", "__stpcpy_chk", &CStringChecker::evalStpcpy)
    .Cases("strcat", "__strcat_chk", &CStringChecker::evalStrcat)
    .Cases("strncat", "__strncat_chk", &CStringChecker::evalStrncat)
    .Case("strlen", &CStringChecker::evalstrLength)
    .Case("strnlen", &CStringChecker::evalstrnLength)
    .Case("strcmp", &CStringChecker::evalStrcmp)
    .Case("strncmp", &CStringChecker::evalStrncmp)
    .Case("strcasecmp", &CStringChecker::evalStrcasecmp)
    .Case("strncasecmp", &CStringChecker::evalStrncasecmp)
    .Case("bcopy", &CStringChecker::evalBcopy)
    .Default(NULL);

  // If the callee isn't a string function, let another checker handle it.
  if (!evalFunction)
    return false;

  // Make sure each function sets its own description.
  // (But don't bother in a release build.)
  assert(!(CurrentFunctionDescription = NULL));

  // Check and evaluate the call.
  (this->*evalFunction)(C, CE);
  return true;
}

void CStringChecker::checkPreStmt(const DeclStmt *DS, CheckerContext &C) const {
  // Record string length for char a[] = "abc";
  const GRState *state = C.getState();

  for (DeclStmt::const_decl_iterator I = DS->decl_begin(), E = DS->decl_end();
       I != E; ++I) {
    const VarDecl *D = dyn_cast<VarDecl>(*I);
    if (!D)
      continue;

    // FIXME: Handle array fields of structs.
    if (!D->getType()->isArrayType())
      continue;

    const Expr *Init = D->getInit();
    if (!Init)
      continue;
    if (!isa<StringLiteral>(Init))
      continue;

    Loc VarLoc = state->getLValue(D, C.getPredecessor()->getLocationContext());
    const MemRegion *MR = VarLoc.getAsRegion();
    if (!MR)
      continue;

    SVal StrVal = state->getSVal(Init);
    assert(StrVal.isValid() && "Initializer string is unknown or undefined");
    DefinedOrUnknownSVal strLength
      = cast<DefinedOrUnknownSVal>(getCStringLength(C, state, Init, StrVal));

    state = state->set<CStringLength>(MR, strLength);
  }

  C.addTransition(state);
}

bool CStringChecker::wantsRegionChangeUpdate(const GRState *state) const {
  CStringLength::EntryMap Entries = state->get<CStringLength>();
  return !Entries.isEmpty();
}

const GRState *
CStringChecker::checkRegionChanges(const GRState *state,
                                   const StoreManager::InvalidatedSymbols *,
                                   const MemRegion * const *Begin,
                                   const MemRegion * const *End) const {
  CStringLength::EntryMap Entries = state->get<CStringLength>();
  if (Entries.isEmpty())
    return state;

  llvm::SmallPtrSet<const MemRegion *, 8> Invalidated;
  llvm::SmallPtrSet<const MemRegion *, 32> SuperRegions;

  // First build sets for the changed regions and their super-regions.
  for ( ; Begin != End; ++Begin) {
    const MemRegion *MR = *Begin;
    Invalidated.insert(MR);

    SuperRegions.insert(MR);
    while (const SubRegion *SR = dyn_cast<SubRegion>(MR)) {
      MR = SR->getSuperRegion();
      SuperRegions.insert(MR);
    }
  }

  CStringLength::EntryMap::Factory &F = state->get_context<CStringLength>();

  // Then loop over the entries in the current state.
  for (CStringLength::EntryMap::iterator I = Entries.begin(),
       E = Entries.end(); I != E; ++I) {
    const MemRegion *MR = I.getKey();

    // Is this entry for a super-region of a changed region?
    if (SuperRegions.count(MR)) {
      Entries = F.remove(Entries, MR);
      continue;
    }

    // Is this entry for a sub-region of a changed region?
    const MemRegion *Super = MR;
    while (const SubRegion *SR = dyn_cast<SubRegion>(Super)) {
      Super = SR->getSuperRegion();
      if (Invalidated.count(Super)) {
        Entries = F.remove(Entries, MR);
        break;
      }
    }
  }

  return state->set<CStringLength>(Entries);
}

void CStringChecker::checkLiveSymbols(const GRState *state,
                                      SymbolReaper &SR) const {
  // Mark all symbols in our string length map as valid.
  CStringLength::EntryMap Entries = state->get<CStringLength>();

  for (CStringLength::EntryMap::iterator I = Entries.begin(), E = Entries.end();
       I != E; ++I) {
    SVal Len = I.getData();

    for (SVal::symbol_iterator si = Len.symbol_begin(), se = Len.symbol_end();
         si != se; ++si)
      SR.markInUse(*si);
  }
}

void CStringChecker::checkDeadSymbols(SymbolReaper &SR,
                                      CheckerContext &C) const {
  if (!SR.hasDeadSymbols())
    return;

  const GRState *state = C.getState();
  CStringLength::EntryMap Entries = state->get<CStringLength>();
  if (Entries.isEmpty())
    return;

  CStringLength::EntryMap::Factory &F = state->get_context<CStringLength>();
  for (CStringLength::EntryMap::iterator I = Entries.begin(), E = Entries.end();
       I != E; ++I) {
    SVal Len = I.getData();
    if (SymbolRef Sym = Len.getAsSymbol()) {
      if (SR.isDead(Sym))
        Entries = F.remove(Entries, I.getKey());
    }
  }

  state = state->set<CStringLength>(Entries);
  C.generateNode(state);
}

void ento::registerCStringChecker(CheckerManager &mgr) {
  mgr.registerChecker<CStringChecker>();
}

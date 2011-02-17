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
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerVisitor.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRStateTrait.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace ento;

namespace {
class CStringChecker : public CheckerVisitor<CStringChecker> {
  BugType *BT_Null, *BT_Bounds, *BT_BoundsWrite, *BT_Overlap, *BT_NotCString;
public:
  CStringChecker()
  : BT_Null(0), BT_Bounds(0), BT_BoundsWrite(0), BT_Overlap(0), BT_NotCString(0)
  {}
  static void *getTag() { static int tag; return &tag; }

  bool evalCallExpr(CheckerContext &C, const CallExpr *CE);
  void PreVisitDeclStmt(CheckerContext &C, const DeclStmt *DS);
  void MarkLiveSymbols(const GRState *state, SymbolReaper &SR);
  void evalDeadSymbols(CheckerContext &C, SymbolReaper &SR);
  bool wantsRegionChangeUpdate(const GRState *state);

  const GRState *EvalRegionChanges(const GRState *state,
                                   const MemRegion * const *Begin,
                                   const MemRegion * const *End,
                                   bool*);

  typedef void (CStringChecker::*FnCheck)(CheckerContext &, const CallExpr *);

  void evalMemcpy(CheckerContext &C, const CallExpr *CE);
  void evalMemmove(CheckerContext &C, const CallExpr *CE);
  void evalBcopy(CheckerContext &C, const CallExpr *CE);
  void evalCopyCommon(CheckerContext &C, const GRState *state,
                      const Expr *Size, const Expr *Source, const Expr *Dest,
                      bool Restricted = false);

  void evalMemcmp(CheckerContext &C, const CallExpr *CE);

  void evalstrLength(CheckerContext &C, const CallExpr *CE);

  void evalStrcpy(CheckerContext &C, const CallExpr *CE);
  void evalStpcpy(CheckerContext &C, const CallExpr *CE);
  void evalStrcpyCommon(CheckerContext &C, const CallExpr *CE, bool returnEnd);

  // Utility methods
  std::pair<const GRState*, const GRState*>
  assumeZero(CheckerContext &C, const GRState *state, SVal V, QualType Ty);

  const GRState *setCStringLength(const GRState *state, const MemRegion *MR,
                                  SVal strLength);
  SVal getCStringLengthForRegion(CheckerContext &C, const GRState *&state,
                                 const Expr *Ex, const MemRegion *MR);
  SVal getCStringLength(CheckerContext &C, const GRState *&state,
                        const Expr *Ex, SVal Buf);

  const GRState *InvalidateBuffer(CheckerContext &C, const GRState *state,
                                  const Expr *Ex, SVal V);

  bool SummarizeRegion(llvm::raw_ostream& os, ASTContext& Ctx,
                       const MemRegion *MR);

  // Re-usable checks
  const GRState *checkNonNull(CheckerContext &C, const GRState *state,
                               const Expr *S, SVal l);
  const GRState *CheckLocation(CheckerContext &C, const GRState *state,
                               const Expr *S, SVal l,
                               bool IsDestination = false);
  const GRState *CheckBufferAccess(CheckerContext &C, const GRState *state,
                                   const Expr *Size,
                                   const Expr *FirstBuf,
                                   const Expr *SecondBuf = NULL,
                                   bool FirstIsDestination = false);
  const GRState *CheckOverlap(CheckerContext &C, const GRState *state,
                              const Expr *Size, const Expr *First,
                              const Expr *Second);
  void emitOverlapBug(CheckerContext &C, const GRState *state,
                      const Stmt *First, const Stmt *Second);
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

static void RegisterCStringChecker(ExprEngine &Eng) {
  Eng.registerCheck(new CStringChecker());
}

void ento::registerCStringChecker(CheckerManager &mgr) {
  mgr.addCheckerRegisterFunction(RegisterCStringChecker);
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
                                            const Expr *S, SVal l) {
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
      BT_Null = new BuiltinBug("API",
        "Null pointer argument in call to byte string function");

    // Generate a report for this bug.
    BuiltinBug *BT = static_cast<BuiltinBug*>(BT_Null);
    EnhancedBugReport *report = new EnhancedBugReport(*BT,
                                                      BT->getDescription(), N);

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
                                             bool IsDestination) {
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
  SVal Extent = svalBuilder.convertToArrayIndex(superReg->getExtent(svalBuilder));
  DefinedOrUnknownSVal Size = cast<DefinedOrUnknownSVal>(Extent);

  // Get the index of the accessed element.
  DefinedOrUnknownSVal Idx = cast<DefinedOrUnknownSVal>(ER->getIndex());

  const GRState *StInBound = state->assumeInBound(Idx, Size, true);
  const GRState *StOutBound = state->assumeInBound(Idx, Size, false);
  if (StOutBound && !StInBound) {
    ExplodedNode *N = C.generateSink(StOutBound);
    if (!N)
      return NULL;

    BuiltinBug *BT;
    if (IsDestination) {
      if (!BT_BoundsWrite) {
        BT_BoundsWrite = new BuiltinBug("Out-of-bound array access",
          "Byte string function overflows destination buffer");
      }
      BT = static_cast<BuiltinBug*>(BT_BoundsWrite);
    } else {
      if (!BT_Bounds) {
        BT_Bounds = new BuiltinBug("Out-of-bound array access",
          "Byte string function accesses out-of-bound array element");
      }
      BT = static_cast<BuiltinBug*>(BT_Bounds);
    }

    // FIXME: It would be nice to eventually make this diagnostic more clear,
    // e.g., by referencing the original declaration or by saying *why* this
    // reference is outside the range.

    // Generate a report for this bug.
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
                                                 const Expr *SecondBuf,
                                                 bool FirstIsDestination) {
  // If a previous check has failed, propagate the failure.
  if (!state)
    return NULL;

  SValBuilder &svalBuilder = C.getSValBuilder();
  ASTContext &Ctx = C.getASTContext();

  QualType sizeTy = Size->getType();
  QualType PtrTy = Ctx.getPointerType(Ctx.CharTy);

  // Check that the first buffer is non-null.
  SVal BufVal = state->getSVal(FirstBuf);
  state = checkNonNull(C, state, FirstBuf, BufVal);
  if (!state)
    return NULL;

  // Get the access length and make sure it is known.
  SVal LengthVal = state->getSVal(Size);
  NonLoc *Length = dyn_cast<NonLoc>(&LengthVal);
  if (!Length)
    return state;

  // Compute the offset of the last element to be accessed: size-1.
  NonLoc One = cast<NonLoc>(svalBuilder.makeIntVal(1, sizeTy));
  NonLoc LastOffset = cast<NonLoc>(svalBuilder.evalBinOpNN(state, BO_Sub,
                                                    *Length, One, sizeTy));

  // Check that the first buffer is sufficently long.
  SVal BufStart = svalBuilder.evalCast(BufVal, PtrTy, FirstBuf->getType());
  if (Loc *BufLoc = dyn_cast<Loc>(&BufStart)) {
    SVal BufEnd = svalBuilder.evalBinOpLN(state, BO_Add, *BufLoc,
                                          LastOffset, PtrTy);
    state = CheckLocation(C, state, FirstBuf, BufEnd, FirstIsDestination);

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
      SVal BufEnd = svalBuilder.evalBinOpLN(state, BO_Add, *BufLoc,
                                            LastOffset, PtrTy);
      state = CheckLocation(C, state, SecondBuf, BufEnd);
    }
  }

  // Large enough or not, return this state!
  return state;
}

const GRState *CStringChecker::CheckOverlap(CheckerContext &C,
                                            const GRState *state,
                                            const Expr *Size,
                                            const Expr *First,
                                            const Expr *Second) {
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
  ASTContext &Ctx = svalBuilder.getContext();
  QualType cmpTy = Ctx.IntTy;
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
  QualType CharPtrTy = Ctx.getPointerType(Ctx.CharTy);
  SVal FirstStart = svalBuilder.evalCast(*firstLoc, CharPtrTy, First->getType());
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
                                    const Stmt *First, const Stmt *Second) {
  ExplodedNode *N = C.generateSink(state);
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

const GRState *CStringChecker::setCStringLength(const GRState *state,
                                                const MemRegion *MR,
                                                SVal strLength) {
  assert(!strLength.isUndef() && "Attempt to set an undefined string length");
  if (strLength.isUnknown())
    return state;

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
    return state->set<CStringLength>(MR, strLength);

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
}

SVal CStringChecker::getCStringLengthForRegion(CheckerContext &C,
                                               const GRState *&state,
                                               const Expr *Ex,
                                               const MemRegion *MR) {
  // If there's a recorded length, go ahead and return it.
  const SVal *Recorded = state->get<CStringLength>(MR);
  if (Recorded)
    return *Recorded;
  
  // Otherwise, get a new symbol and update the state.
  unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
  SValBuilder &svalBuilder = C.getSValBuilder();
  QualType sizeTy = svalBuilder.getContext().getSizeType();
  SVal strLength = svalBuilder.getMetadataSymbolVal(getTag(), MR, Ex, sizeTy, Count);
  state = state->set<CStringLength>(MR, strLength);
  return strLength;
}

SVal CStringChecker::getCStringLength(CheckerContext &C, const GRState *&state,
                                      const Expr *Ex, SVal Buf) {
  const MemRegion *MR = Buf.getAsRegion();
  if (!MR) {
    // If we can't get a region, see if it's something we /know/ isn't a
    // C string. In the context of locations, the only time we can issue such
    // a warning is for labels.
    if (loc::GotoLabel *Label = dyn_cast<loc::GotoLabel>(&Buf)) {
      if (ExplodedNode *N = C.generateNode(state)) {
        if (!BT_NotCString)
          BT_NotCString = new BuiltinBug("API",
            "Argument is not a null-terminated string.");

        llvm::SmallString<120> buf;
        llvm::raw_svector_ostream os(buf);
        os << "Argument to byte string function is the address of the label '"
           << Label->getLabel()->getName()
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
    return getCStringLengthForRegion(C, state, Ex, MR);
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
        BT_NotCString = new BuiltinBug("API",
          "Argument is not a null-terminated string.");

      llvm::SmallString<120> buf;
      llvm::raw_svector_ostream os(buf);

      os << "Argument to byte string function is ";

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

void CStringChecker::evalCopyCommon(CheckerContext &C, const GRState *state,
                                    const Expr *Size, const Expr *Dest,
                                    const Expr *Source, bool Restricted) {
  // See if the size argument is zero.
  SVal sizeVal = state->getSVal(Size);
  QualType sizeTy = Size->getType();

  const GRState *stateZeroSize, *stateNonZeroSize;
  llvm::tie(stateZeroSize, stateNonZeroSize) = assumeZero(C, state, sizeVal, sizeTy);

  // If the size is zero, there won't be any actual memory access.
  if (stateZeroSize)
    C.addTransition(stateZeroSize);

  // If the size can be nonzero, we have to check the other arguments.
  if (stateNonZeroSize) {
    state = stateNonZeroSize;
    state = CheckBufferAccess(C, state, Size, Dest, Source,
                              /* FirstIsDst = */ true);
    if (Restricted)
      state = CheckOverlap(C, state, Size, Dest, Source);

    if (state) {
      // Invalidate the destination.
      // FIXME: Even if we can't perfectly model the copy, we should see if we
      // can use LazyCompoundVals to copy the source values into the destination.
      // This would probably remove any existing bindings past the end of the
      // copied region, but that's still an improvement over blank invalidation.
      state = InvalidateBuffer(C, state, Dest, state->getSVal(Dest));
      C.addTransition(state);
    }
  }
}


void CStringChecker::evalMemcpy(CheckerContext &C, const CallExpr *CE) {
  // void *memcpy(void *restrict dst, const void *restrict src, size_t n);
  // The return value is the address of the destination buffer.
  const Expr *Dest = CE->getArg(0);
  const GRState *state = C.getState();
  state = state->BindExpr(CE, state->getSVal(Dest));
  evalCopyCommon(C, state, CE->getArg(2), Dest, CE->getArg(1), true);
}

void CStringChecker::evalMemmove(CheckerContext &C, const CallExpr *CE) {
  // void *memmove(void *dst, const void *src, size_t n);
  // The return value is the address of the destination buffer.
  const Expr *Dest = CE->getArg(0);
  const GRState *state = C.getState();
  state = state->BindExpr(CE, state->getSVal(Dest));
  evalCopyCommon(C, state, CE->getArg(2), Dest, CE->getArg(1));
}

void CStringChecker::evalBcopy(CheckerContext &C, const CallExpr *CE) {
  // void bcopy(const void *src, void *dst, size_t n);
  evalCopyCommon(C, C.getState(), CE->getArg(2), CE->getArg(1), CE->getArg(0));
}

void CStringChecker::evalMemcmp(CheckerContext &C, const CallExpr *CE) {
  // int memcmp(const void *s1, const void *s2, size_t n);
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

    // If the two arguments might be the same buffer, we know the result is zero,
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

void CStringChecker::evalstrLength(CheckerContext &C, const CallExpr *CE) {
  // size_t strlen(const char *s);
  const GRState *state = C.getState();
  const Expr *Arg = CE->getArg(0);
  SVal ArgVal = state->getSVal(Arg);

  // Check that the argument is non-null.
  state = checkNonNull(C, state, Arg, ArgVal);

  if (state) {
    SVal strLength = getCStringLength(C, state, Arg, ArgVal);

    // If the argument isn't a valid C string, there's no valid state to
    // transition to.
    if (strLength.isUndef())
      return;

    // If getCStringLength couldn't figure out the length, conjure a return
    // value, so it can be used in constraints, at least.
    if (strLength.isUnknown()) {
      unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
      strLength = C.getSValBuilder().getConjuredSymbolVal(NULL, CE, Count);
    }

    // Bind the return value.
    state = state->BindExpr(CE, strLength);
    C.addTransition(state);
  }
}

void CStringChecker::evalStrcpy(CheckerContext &C, const CallExpr *CE) {
  // char *strcpy(char *restrict dst, const char *restrict src);
  evalStrcpyCommon(C, CE, /* returnEnd = */ false);
}

void CStringChecker::evalStpcpy(CheckerContext &C, const CallExpr *CE) {
  // char *stpcpy(char *restrict dst, const char *restrict src);
  evalStrcpyCommon(C, CE, /* returnEnd = */ true);
}

void CStringChecker::evalStrcpyCommon(CheckerContext &C, const CallExpr *CE,
                                      bool returnEnd) {
  const GRState *state = C.getState();

  // Check that the destination is non-null
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

  SVal Result = (returnEnd ? UnknownVal() : DstVal);

  // If the destination is a MemRegion, try to check for a buffer overflow and
  // record the new string length.
  if (loc::MemRegionVal *dstRegVal = dyn_cast<loc::MemRegionVal>(&DstVal)) {
    // If the length is known, we can check for an overflow.
    if (NonLoc *knownStrLength = dyn_cast<NonLoc>(&strLength)) {
      SVal lastElement =
        C.getSValBuilder().evalBinOpLN(state, BO_Add, *dstRegVal,
                                       *knownStrLength, Dst->getType());

      state = CheckLocation(C, state, Dst, lastElement, /* IsDst = */ true);
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

    // Set the C string length of the destination.
    state = setCStringLength(state, dstRegVal->getRegion(), strLength);
  }

  // If this is a stpcpy-style copy, but we were unable to check for a buffer
  // overflow, we still need a result. Conjure a return value.
  if (returnEnd && Result.isUnknown()) {
    SValBuilder &svalBuilder = C.getSValBuilder();
    unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
    strLength = svalBuilder.getConjuredSymbolVal(NULL, CE, Count);
  }

  // Set the return value.
  state = state->BindExpr(CE, Result);
  C.addTransition(state);
}

//===----------------------------------------------------------------------===//
// The driver method, and other Checker callbacks.
//===----------------------------------------------------------------------===//

bool CStringChecker::evalCallExpr(CheckerContext &C, const CallExpr *CE) {
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
    .Cases("memcmp", "bcmp", &CStringChecker::evalMemcmp)
    .Cases("memmove", "__memmove_chk", &CStringChecker::evalMemmove)
    .Cases("strcpy", "__strcpy_chk", &CStringChecker::evalStrcpy)
    .Cases("stpcpy", "__stpcpy_chk", &CStringChecker::evalStpcpy)
    .Case("strlen", &CStringChecker::evalstrLength)
    .Case("bcopy", &CStringChecker::evalBcopy)
    .Default(NULL);

  // If the callee isn't a string function, let another checker handle it.
  if (!evalFunction)
    return false;

  // Check and evaluate the call.
  (this->*evalFunction)(C, CE);
  return true;
}

void CStringChecker::PreVisitDeclStmt(CheckerContext &C, const DeclStmt *DS) {
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

bool CStringChecker::wantsRegionChangeUpdate(const GRState *state) {
  CStringLength::EntryMap Entries = state->get<CStringLength>();
  return !Entries.isEmpty();
}

const GRState *CStringChecker::EvalRegionChanges(const GRState *state,
                                                 const MemRegion * const *Begin,
                                                 const MemRegion * const *End,
                                                 bool *) {
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

void CStringChecker::MarkLiveSymbols(const GRState *state, SymbolReaper &SR) {
  // Mark all symbols in our string length map as valid.
  CStringLength::EntryMap Entries = state->get<CStringLength>();

  for (CStringLength::EntryMap::iterator I = Entries.begin(), E = Entries.end();
       I != E; ++I) {
    SVal Len = I.getData();
    if (SymbolRef Sym = Len.getAsSymbol())
      SR.markInUse(Sym);
  }
}

void CStringChecker::evalDeadSymbols(CheckerContext &C, SymbolReaper &SR) {
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

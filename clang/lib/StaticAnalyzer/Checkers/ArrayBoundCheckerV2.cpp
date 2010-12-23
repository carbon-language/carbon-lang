//== ArrayBoundCheckerV2.cpp ------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ArrayBoundCheckerV2, which is a path-sensitive check
// which looks for an out-of-bound array element access.
//
//===----------------------------------------------------------------------===//

#include "ExprEngineInternalChecks.h"
#include "clang/StaticAnalyzer/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/PathSensitive/CheckerVisitor.h"
#include "clang/StaticAnalyzer/PathSensitive/ExprEngine.h"
#include "clang/AST/CharUnits.h"

using namespace clang;
using namespace ento;

namespace {
class ArrayBoundCheckerV2 : 
    public CheckerVisitor<ArrayBoundCheckerV2> {      
  BuiltinBug *BT;
      
  enum OOB_Kind { OOB_Precedes, OOB_Excedes };
  
  void reportOOB(CheckerContext &C, const GRState *errorState,
                 OOB_Kind kind);
      
public:
  ArrayBoundCheckerV2() : BT(0) {}
  static void *getTag() { static int x = 0; return &x; }
  void visitLocation(CheckerContext &C, const Stmt *S, SVal l);      
};

// FIXME: Eventually replace RegionRawOffset with this class.
class RegionRawOffsetV2 {
private:
  const SubRegion *baseRegion;
  SVal byteOffset;
  
  RegionRawOffsetV2()
    : baseRegion(0), byteOffset(UnknownVal()) {}

public:
  RegionRawOffsetV2(const SubRegion* base, SVal offset)
    : baseRegion(base), byteOffset(offset) {}

  NonLoc getByteOffset() const { return cast<NonLoc>(byteOffset); }
  const SubRegion *getRegion() const { return baseRegion; }
  
  static RegionRawOffsetV2 computeOffset(const GRState *state,
                                         SValBuilder &svalBuilder,
                                         SVal location);

  void dump() const;
  void dumpToStream(llvm::raw_ostream& os) const;
};
}

void ento::RegisterArrayBoundCheckerV2(ExprEngine &Eng) {
  Eng.registerCheck(new ArrayBoundCheckerV2());
}

void ArrayBoundCheckerV2::visitLocation(CheckerContext &checkerContext,
                                        const Stmt *S,
                                        SVal location) {

  // NOTE: Instead of using GRState::assumeInBound(), we are prototyping
  // some new logic here that reasons directly about memory region extents.
  // Once that logic is more mature, we can bring it back to assumeInBound()
  // for all clients to use.
  //
  // The algorithm we are using here for bounds checking is to see if the
  // memory access is within the extent of the base region.  Since we
  // have some flexibility in defining the base region, we can achieve
  // various levels of conservatism in our buffer overflow checking.
  const GRState *state = checkerContext.getState();  
  const GRState *originalState = state;

  SValBuilder &svalBuilder = checkerContext.getSValBuilder();
  const RegionRawOffsetV2 &rawOffset = 
    RegionRawOffsetV2::computeOffset(state, svalBuilder, location);

  if (!rawOffset.getRegion())
    return;

  // CHECK LOWER BOUND: Is byteOffset < 0?  If so, we are doing a load/store
  //  before the first valid offset in the memory region.

  SVal lowerBound
    = svalBuilder.evalBinOpNN(state, BO_LT, rawOffset.getByteOffset(),
                              svalBuilder.makeZeroArrayIndex(),
                              svalBuilder.getConditionType());

  NonLoc *lowerBoundToCheck = dyn_cast<NonLoc>(&lowerBound);
  if (!lowerBoundToCheck)
    return;
    
  const GRState *state_precedesLowerBound, *state_withinLowerBound;
  llvm::tie(state_precedesLowerBound, state_withinLowerBound) =
      state->assume(*lowerBoundToCheck);

  // Are we constrained enough to definitely precede the lower bound?
  if (state_precedesLowerBound && !state_withinLowerBound) {
    reportOOB(checkerContext, state_precedesLowerBound, OOB_Precedes);
    return;
  }
  
  // Otherwise, assume the constraint of the lower bound.
  assert(state_withinLowerBound);
  state = state_withinLowerBound;
  
  do {
    // CHECK UPPER BOUND: Is byteOffset >= extent(baseRegion)?  If so,
    // we are doing a load/store after the last valid offset.
    DefinedOrUnknownSVal extentVal =
      rawOffset.getRegion()->getExtent(svalBuilder);
    if (!isa<NonLoc>(extentVal))
      break;

    SVal upperbound
      = svalBuilder.evalBinOpNN(state, BO_GE, rawOffset.getByteOffset(),
                                cast<NonLoc>(extentVal),
                                svalBuilder.getConditionType());
  
    NonLoc *upperboundToCheck = dyn_cast<NonLoc>(&upperbound);
    if (!upperboundToCheck)
      break;
  
    const GRState *state_exceedsUpperBound, *state_withinUpperBound;
    llvm::tie(state_exceedsUpperBound, state_withinUpperBound) =
      state->assume(*upperboundToCheck);
  
    // Are we constrained enough to definitely exceed the upper bound?
    if (state_exceedsUpperBound && !state_withinUpperBound) {
      reportOOB(checkerContext, state_exceedsUpperBound, OOB_Excedes);
      return;
    }
  
    assert(state_withinUpperBound);
    state = state_withinUpperBound;
  }
  while (false);
  
  if (state != originalState)
    checkerContext.generateNode(state);
}

void ArrayBoundCheckerV2::reportOOB(CheckerContext &checkerContext,
                                    const GRState *errorState,
                                    OOB_Kind kind) {
  
  ExplodedNode *errorNode = checkerContext.generateSink(errorState);
  if (!errorNode)
    return;

  if (!BT)
    BT = new BuiltinBug("Out-of-bound access");

  // FIXME: This diagnostics are preliminary.  We should get far better
  // diagnostics for explaining buffer overruns.

  llvm::SmallString<256> buf;
  llvm::raw_svector_ostream os(buf);
  os << "Out of bound memory access "
     << (kind == OOB_Precedes ? "(accessed memory precedes memory block)"
                              : "(access exceeds upper limit of memory block)");

  checkerContext.EmitReport(new RangedBugReport(*BT, os.str(), errorNode));
}

void RegionRawOffsetV2::dump() const {
  dumpToStream(llvm::errs());
}

void RegionRawOffsetV2::dumpToStream(llvm::raw_ostream& os) const {
  os << "raw_offset_v2{" << getRegion() << ',' << getByteOffset() << '}';
}

// FIXME: Merge with the implementation of the same method in Store.cpp
static bool IsCompleteType(ASTContext &Ctx, QualType Ty) {
  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    const RecordDecl *D = RT->getDecl();
    if (!D->getDefinition())
      return false;
  }

  return true;
}


// Lazily computes a value to be used by 'computeOffset'.  If 'val'
// is unknown or undefined, we lazily substitute '0'.  Otherwise,
// return 'val'.
static inline SVal getValue(SVal val, SValBuilder &svalBuilder) {
  return isa<UndefinedVal>(val) ? svalBuilder.makeArrayIndex(0) : val;
}

// Scale a base value by a scaling factor, and return the scaled
// value as an SVal.  Used by 'computeOffset'.
static inline SVal scaleValue(const GRState *state,
                              NonLoc baseVal, CharUnits scaling,
                              SValBuilder &sb) {
  return sb.evalBinOpNN(state, BO_Mul, baseVal,
                        sb.makeArrayIndex(scaling.getQuantity()),
                        sb.getArrayIndexType());
}

// Add an SVal to another, treating unknown and undefined values as
// summing to UnknownVal.  Used by 'computeOffset'.
static SVal addValue(const GRState *state, SVal x, SVal y,
                     SValBuilder &svalBuilder) {
  // We treat UnknownVals and UndefinedVals the same here because we
  // only care about computing offsets.
  if (x.isUnknownOrUndef() || y.isUnknownOrUndef())
    return UnknownVal();
  
  return svalBuilder.evalBinOpNN(state, BO_Add,                                 
                                 cast<NonLoc>(x), cast<NonLoc>(y),
                                 svalBuilder.getArrayIndexType());
}

/// Compute a raw byte offset from a base region.  Used for array bounds
/// checking.
RegionRawOffsetV2 RegionRawOffsetV2::computeOffset(const GRState *state,
                                                   SValBuilder &svalBuilder,
                                                   SVal location)
{
  const MemRegion *region = location.getAsRegion();
  SVal offset = UndefinedVal();
  
  while (region) {
    switch (region->getKind()) {
      default: {
        if (const SubRegion *subReg = dyn_cast<SubRegion>(region))
          if (!offset.isUnknownOrUndef())
            return RegionRawOffsetV2(subReg, offset);
        return RegionRawOffsetV2();
      }
      case MemRegion::ElementRegionKind: {
        const ElementRegion *elemReg = cast<ElementRegion>(region);
        SVal index = elemReg->getIndex();
        if (!isa<NonLoc>(index))
          return RegionRawOffsetV2();
        QualType elemType = elemReg->getElementType();
        // If the element is an incomplete type, go no further.
        ASTContext &astContext = svalBuilder.getContext();
        if (!IsCompleteType(astContext, elemType))
          return RegionRawOffsetV2();
        
        // Update the offset.
        offset = addValue(state,
                          getValue(offset, svalBuilder),
                          scaleValue(state,
                                     cast<NonLoc>(index),
                                     astContext.getTypeSizeInChars(elemType),
                                     svalBuilder),
                          svalBuilder);

        if (offset.isUnknownOrUndef())
          return RegionRawOffsetV2();

        region = elemReg->getSuperRegion();
        continue;
      }
    }
  }
  return RegionRawOffsetV2();
}




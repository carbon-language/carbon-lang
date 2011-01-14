//== RangeConstraintManager.cpp - Manage range constraints.------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines RangeConstraintManager, a class that tracks simple
//  equality and inequality constraints on symbolic values of GRState.
//
//===----------------------------------------------------------------------===//

#include "SimpleConstraintManager.h"
#include "clang/StaticAnalyzer/PathSensitive/GRState.h"
#include "clang/StaticAnalyzer/PathSensitive/GRStateTrait.h"
#include "clang/StaticAnalyzer/PathSensitive/TransferFuncs.h"
#include "clang/StaticAnalyzer/ManagerRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace { class ConstraintRange {}; }
static int ConstraintRangeIndex = 0;

/// A Range represents the closed range [from, to].  The caller must
/// guarantee that from <= to.  Note that Range is immutable, so as not
/// to subvert RangeSet's immutability.
namespace {
class Range : public std::pair<const llvm::APSInt*,
                                                const llvm::APSInt*> {
public:
  Range(const llvm::APSInt &from, const llvm::APSInt &to)
    : std::pair<const llvm::APSInt*, const llvm::APSInt*>(&from, &to) {
    assert(from <= to);
  }
  bool Includes(const llvm::APSInt &v) const {
    return *first <= v && v <= *second;
  }
  const llvm::APSInt &From() const {
    return *first;
  }
  const llvm::APSInt &To() const {
    return *second;
  }
  const llvm::APSInt *getConcreteValue() const {
    return &From() == &To() ? &From() : NULL;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(&From());
    ID.AddPointer(&To());
  }
};


class RangeTrait : public llvm::ImutContainerInfo<Range> {
public:
  // When comparing if one Range is less than another, we should compare
  // the actual APSInt values instead of their pointers.  This keeps the order
  // consistent (instead of comparing by pointer values) and can potentially
  // be used to speed up some of the operations in RangeSet.
  static inline bool isLess(key_type_ref lhs, key_type_ref rhs) {
    return *lhs.first < *rhs.first || (!(*rhs.first < *lhs.first) &&
                                       *lhs.second < *rhs.second);
  }
};

/// RangeSet contains a set of ranges. If the set is empty, then
///  there the value of a symbol is overly constrained and there are no
///  possible values for that symbol.
class RangeSet {
  typedef llvm::ImmutableSet<Range, RangeTrait> PrimRangeSet;
  PrimRangeSet ranges; // no need to make const, since it is an
                       // ImmutableSet - this allows default operator=
                       // to work.
public:
  typedef PrimRangeSet::Factory Factory;
  typedef PrimRangeSet::iterator iterator;

  RangeSet(PrimRangeSet RS) : ranges(RS) {}

  iterator begin() const { return ranges.begin(); }
  iterator end() const { return ranges.end(); }

  bool isEmpty() const { return ranges.isEmpty(); }

  /// Construct a new RangeSet representing '{ [from, to] }'.
  RangeSet(Factory &F, const llvm::APSInt &from, const llvm::APSInt &to)
    : ranges(F.add(F.getEmptySet(), Range(from, to))) {}

  /// Profile - Generates a hash profile of this RangeSet for use
  ///  by FoldingSet.
  void Profile(llvm::FoldingSetNodeID &ID) const { ranges.Profile(ID); }

  /// getConcreteValue - If a symbol is contrained to equal a specific integer
  ///  constant then this method returns that value.  Otherwise, it returns
  ///  NULL.
  const llvm::APSInt* getConcreteValue() const {
    return ranges.isSingleton() ? ranges.begin()->getConcreteValue() : 0;
  }

private:
  void IntersectInRange(BasicValueFactory &BV, Factory &F,
                        const llvm::APSInt &Lower,
                        const llvm::APSInt &Upper,
                        PrimRangeSet &newRanges,
                        PrimRangeSet::iterator &i,
                        PrimRangeSet::iterator &e) const {
    // There are six cases for each range R in the set:
    //   1. R is entirely before the intersection range.
    //   2. R is entirely after the intersection range.
    //   3. R contains the entire intersection range.
    //   4. R starts before the intersection range and ends in the middle.
    //   5. R starts in the middle of the intersection range and ends after it.
    //   6. R is entirely contained in the intersection range.
    // These correspond to each of the conditions below.
    for (/* i = begin(), e = end() */; i != e; ++i) {
      if (i->To() < Lower) {
        continue;
      }
      if (i->From() > Upper) {
        break;
      }

      if (i->Includes(Lower)) {
        if (i->Includes(Upper)) {
          newRanges = F.add(newRanges, Range(BV.getValue(Lower),
                                             BV.getValue(Upper)));
          break;
        } else
          newRanges = F.add(newRanges, Range(BV.getValue(Lower), i->To()));
      } else {
        if (i->Includes(Upper)) {
          newRanges = F.add(newRanges, Range(i->From(), BV.getValue(Upper)));
          break;
        } else
          newRanges = F.add(newRanges, *i);
      }
    }
  }

public:
  // Returns a set containing the values in the receiving set, intersected with
  // the closed range [Lower, Upper]. Unlike the Range type, this range uses
  // modular arithmetic, corresponding to the common treatment of C integer
  // overflow. Thus, if the Lower bound is greater than the Upper bound, the
  // range is taken to wrap around. This is equivalent to taking the
  // intersection with the two ranges [Min, Upper] and [Lower, Max],
  // or, alternatively, /removing/ all integers between Upper and Lower.
  RangeSet Intersect(BasicValueFactory &BV, Factory &F,
                     const llvm::APSInt &Lower,
                     const llvm::APSInt &Upper) const {
    PrimRangeSet newRanges = F.getEmptySet();

    PrimRangeSet::iterator i = begin(), e = end();
    if (Lower <= Upper)
      IntersectInRange(BV, F, Lower, Upper, newRanges, i, e);
    else {
      // The order of the next two statements is important!
      // IntersectInRange() does not reset the iteration state for i and e.
      // Therefore, the lower range most be handled first.
      IntersectInRange(BV, F, BV.getMinValue(Upper), Upper, newRanges, i, e);
      IntersectInRange(BV, F, Lower, BV.getMaxValue(Lower), newRanges, i, e);
    }
    return newRanges;
  }

  void print(llvm::raw_ostream &os) const {
    bool isFirst = true;
    os << "{ ";
    for (iterator i = begin(), e = end(); i != e; ++i) {
      if (isFirst)
        isFirst = false;
      else
        os << ", ";

      os << '[' << i->From().toString(10) << ", " << i->To().toString(10)
         << ']';
    }
    os << " }";
  }

  bool operator==(const RangeSet &other) const {
    return ranges == other.ranges;
  }
};
} // end anonymous namespace

typedef llvm::ImmutableMap<SymbolRef,RangeSet> ConstraintRangeTy;

namespace clang {
namespace ento {
template<>
struct GRStateTrait<ConstraintRange>
  : public GRStatePartialTrait<ConstraintRangeTy> {
  static inline void* GDMIndex() { return &ConstraintRangeIndex; }
};
}
}

namespace {
class RangeConstraintManager : public SimpleConstraintManager{
  RangeSet GetRange(const GRState *state, SymbolRef sym);
public:
  RangeConstraintManager(SubEngine &subengine)
    : SimpleConstraintManager(subengine) {}

  const GRState *assumeSymNE(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment);

  const GRState *assumeSymEQ(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment);

  const GRState *assumeSymLT(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment);

  const GRState *assumeSymGT(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment);

  const GRState *assumeSymGE(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment);

  const GRState *assumeSymLE(const GRState* state, SymbolRef sym,
                             const llvm::APSInt& Int,
                             const llvm::APSInt& Adjustment);

  const llvm::APSInt* getSymVal(const GRState* St, SymbolRef sym) const;

  // FIXME: Refactor into SimpleConstraintManager?
  bool isEqual(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const {
    const llvm::APSInt *i = getSymVal(St, sym);
    return i ? *i == V : false;
  }

  const GRState* removeDeadBindings(const GRState* St, SymbolReaper& SymReaper);

  void print(const GRState* St, llvm::raw_ostream& Out,
             const char* nl, const char *sep);

private:
  RangeSet::Factory F;
};

} // end anonymous namespace

ConstraintManager* ento::CreateRangeConstraintManager(GRStateManager&,
                                                    SubEngine &subeng) {
  return new RangeConstraintManager(subeng);
}

const llvm::APSInt* RangeConstraintManager::getSymVal(const GRState* St,
                                                      SymbolRef sym) const {
  const ConstraintRangeTy::data_type *T = St->get<ConstraintRange>(sym);
  return T ? T->getConcreteValue() : NULL;
}

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
const GRState*
RangeConstraintManager::removeDeadBindings(const GRState* state,
                                           SymbolReaper& SymReaper) {

  ConstraintRangeTy CR = state->get<ConstraintRange>();
  ConstraintRangeTy::Factory& CRFactory = state->get_context<ConstraintRange>();

  for (ConstraintRangeTy::iterator I = CR.begin(), E = CR.end(); I != E; ++I) {
    SymbolRef sym = I.getKey();
    if (SymReaper.maybeDead(sym))
      CR = CRFactory.remove(CR, sym);
  }

  return state->set<ConstraintRange>(CR);
}

RangeSet
RangeConstraintManager::GetRange(const GRState *state, SymbolRef sym) {
  if (ConstraintRangeTy::data_type* V = state->get<ConstraintRange>(sym))
    return *V;

  // Lazily generate a new RangeSet representing all possible values for the
  // given symbol type.
  QualType T = state->getSymbolManager().getType(sym);
  BasicValueFactory& BV = state->getBasicVals();
  return RangeSet(F, BV.getMinValue(T), BV.getMaxValue(T));
}

//===------------------------------------------------------------------------===
// assumeSymX methods: public interface for RangeConstraintManager.
//===------------------------------------------------------------------------===/

// The syntax for ranges below is mathematical, using [x, y] for closed ranges
// and (x, y) for open ranges. These ranges are modular, corresponding with
// a common treatment of C integer overflow. This means that these methods
// do not have to worry about overflow; RangeSet::Intersect can handle such a
// "wraparound" range.
// As an example, the range [UINT_MAX-1, 3) contains five values: UINT_MAX-1,
// UINT_MAX, 0, 1, and 2.

const GRState*
RangeConstraintManager::assumeSymNE(const GRState* state, SymbolRef sym,
                                    const llvm::APSInt& Int,
                                    const llvm::APSInt& Adjustment) {
  BasicValueFactory &BV = state->getBasicVals();

  llvm::APSInt Lower = Int-Adjustment;
  llvm::APSInt Upper = Lower;
  --Lower;
  ++Upper;

  // [Int-Adjustment+1, Int-Adjustment-1]
  // Notice that the lower bound is greater than the upper bound.
  RangeSet New = GetRange(state, sym).Intersect(BV, F, Upper, Lower);
  return New.isEmpty() ? NULL : state->set<ConstraintRange>(sym, New);
}

const GRState*
RangeConstraintManager::assumeSymEQ(const GRState* state, SymbolRef sym,
                                    const llvm::APSInt& Int,
                                    const llvm::APSInt& Adjustment) {
  // [Int-Adjustment, Int-Adjustment]
  BasicValueFactory &BV = state->getBasicVals();
  llvm::APSInt AdjInt = Int-Adjustment;
  RangeSet New = GetRange(state, sym).Intersect(BV, F, AdjInt, AdjInt);
  return New.isEmpty() ? NULL : state->set<ConstraintRange>(sym, New);
}

const GRState*
RangeConstraintManager::assumeSymLT(const GRState* state, SymbolRef sym,
                                    const llvm::APSInt& Int,
                                    const llvm::APSInt& Adjustment) {
  BasicValueFactory &BV = state->getBasicVals();

  QualType T = state->getSymbolManager().getType(sym);
  const llvm::APSInt &Min = BV.getMinValue(T);

  // Special case for Int == Min. This is always false.
  if (Int == Min)
    return NULL;

  llvm::APSInt Lower = Min-Adjustment;
  llvm::APSInt Upper = Int-Adjustment;
  --Upper;

  RangeSet New = GetRange(state, sym).Intersect(BV, F, Lower, Upper);
  return New.isEmpty() ? NULL : state->set<ConstraintRange>(sym, New);
}

const GRState*
RangeConstraintManager::assumeSymGT(const GRState* state, SymbolRef sym,
                                    const llvm::APSInt& Int,
                                    const llvm::APSInt& Adjustment) {
  BasicValueFactory &BV = state->getBasicVals();

  QualType T = state->getSymbolManager().getType(sym);
  const llvm::APSInt &Max = BV.getMaxValue(T);

  // Special case for Int == Max. This is always false.
  if (Int == Max)
    return NULL;

  llvm::APSInt Lower = Int-Adjustment;
  llvm::APSInt Upper = Max-Adjustment;
  ++Lower;

  RangeSet New = GetRange(state, sym).Intersect(BV, F, Lower, Upper);
  return New.isEmpty() ? NULL : state->set<ConstraintRange>(sym, New);
}

const GRState*
RangeConstraintManager::assumeSymGE(const GRState* state, SymbolRef sym,
                                    const llvm::APSInt& Int,
                                    const llvm::APSInt& Adjustment) {
  BasicValueFactory &BV = state->getBasicVals();

  QualType T = state->getSymbolManager().getType(sym);
  const llvm::APSInt &Min = BV.getMinValue(T);

  // Special case for Int == Min. This is always feasible.
  if (Int == Min)
    return state;

  const llvm::APSInt &Max = BV.getMaxValue(T);

  llvm::APSInt Lower = Int-Adjustment;
  llvm::APSInt Upper = Max-Adjustment;

  RangeSet New = GetRange(state, sym).Intersect(BV, F, Lower, Upper);
  return New.isEmpty() ? NULL : state->set<ConstraintRange>(sym, New);
}

const GRState*
RangeConstraintManager::assumeSymLE(const GRState* state, SymbolRef sym,
                                    const llvm::APSInt& Int,
                                    const llvm::APSInt& Adjustment) {
  BasicValueFactory &BV = state->getBasicVals();

  QualType T = state->getSymbolManager().getType(sym);
  const llvm::APSInt &Max = BV.getMaxValue(T);

  // Special case for Int == Max. This is always feasible.
  if (Int == Max)
    return state;

  const llvm::APSInt &Min = BV.getMinValue(T);

  llvm::APSInt Lower = Min-Adjustment;
  llvm::APSInt Upper = Int-Adjustment;

  RangeSet New = GetRange(state, sym).Intersect(BV, F, Lower, Upper);
  return New.isEmpty() ? NULL : state->set<ConstraintRange>(sym, New);
}

//===------------------------------------------------------------------------===
// Pretty-printing.
//===------------------------------------------------------------------------===/

void RangeConstraintManager::print(const GRState* St, llvm::raw_ostream& Out,
                                   const char* nl, const char *sep) {

  ConstraintRangeTy Ranges = St->get<ConstraintRange>();

  if (Ranges.isEmpty())
    return;

  Out << nl << sep << "ranges of symbol values:";

  for (ConstraintRangeTy::iterator I=Ranges.begin(), E=Ranges.end(); I!=E; ++I){
    Out << nl << ' ' << I.getKey() << " : ";
    I.getData().print(Out);
  }
}

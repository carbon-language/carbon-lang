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
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/GRStateTrait.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "clang/Frontend/ManagerRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace { class VISIBILITY_HIDDEN ConstraintRange {}; }
static int ConstraintRangeIndex = 0;

/// A Range represents the closed range [from, to].  The caller must
/// guarantee that from <= to.  Note that Range is immutable, so as not
/// to subvert RangeSet's immutability.
namespace {
class VISIBILITY_HIDDEN Range : public std::pair<const llvm::APSInt*,
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


class VISIBILITY_HIDDEN RangeTrait : public llvm::ImutContainerInfo<Range> {
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
class VISIBILITY_HIDDEN RangeSet {
  typedef llvm::ImmutableSet<Range, RangeTrait> PrimRangeSet;
  PrimRangeSet ranges; // no need to make const, since it is an
                       // ImmutableSet - this allows default operator=
                       // to work.
public:
  typedef PrimRangeSet::Factory Factory;
  typedef PrimRangeSet::iterator iterator;

  RangeSet(PrimRangeSet RS) : ranges(RS) {}
  RangeSet(Factory& F) : ranges(F.GetEmptySet()) {}

  iterator begin() const { return ranges.begin(); }
  iterator end() const { return ranges.end(); }

  bool isEmpty() const { return ranges.isEmpty(); }

  /// Construct a new RangeSet representing '{ [from, to] }'.
  RangeSet(Factory &F, const llvm::APSInt &from, const llvm::APSInt &to)
    : ranges(F.Add(F.GetEmptySet(), Range(from, to))) {}

  /// Profile - Generates a hash profile of this RangeSet for use
  ///  by FoldingSet.
  void Profile(llvm::FoldingSetNodeID &ID) const { ranges.Profile(ID); }

  /// getConcreteValue - If a symbol is contrained to equal a specific integer
  ///  constant then this method returns that value.  Otherwise, it returns
  ///  NULL.
  const llvm::APSInt* getConcreteValue() const {
    return ranges.isSingleton() ? ranges.begin()->getConcreteValue() : 0;
  }

  /// AddEQ - Create a new RangeSet with the additional constraint that the
  ///  value be equal to V.
  RangeSet AddEQ(BasicValueFactory &BV, Factory &F, const llvm::APSInt &V) {
    // Search for a range that includes 'V'.  If so, return a new RangeSet
    // representing { [V, V] }.
    for (PrimRangeSet::iterator i = begin(), e = end(); i!=e; ++i)
      if (i->Includes(V))
        return RangeSet(F, V, V);

    return RangeSet(F);
  }

  /// AddNE - Create a new RangeSet with the additional constraint that the
  ///  value be not be equal to V.
  RangeSet AddNE(BasicValueFactory &BV, Factory &F, const llvm::APSInt &V) {
    PrimRangeSet newRanges = ranges;

    // FIXME: We can perhaps enhance ImmutableSet to do this search for us
    // in log(N) time using the sorted property of the internal AVL tree.
    for (iterator i = begin(), e = end(); i != e; ++i) {
      if (i->Includes(V)) {
        // Remove the old range.
        newRanges = F.Remove(newRanges, *i);
        // Split the old range into possibly one or two ranges.
        if (V != i->From())
          newRanges = F.Add(newRanges, Range(i->From(), BV.Sub1(V)));
        if (V != i->To())
          newRanges = F.Add(newRanges, Range(BV.Add1(V), i->To()));
        // All of the ranges are non-overlapping, so we can stop.
        break;
      }
    }

    return newRanges;
  }

  /// AddNE - Create a new RangeSet with the additional constraint that the
  ///  value be less than V.
  RangeSet AddLT(BasicValueFactory &BV, Factory &F, const llvm::APSInt &V) {
    PrimRangeSet newRanges = F.GetEmptySet();

    for (iterator i = begin(), e = end() ; i != e ; ++i) {
      if (i->Includes(V) && i->From() < V)
        newRanges = F.Add(newRanges, Range(i->From(), BV.Sub1(V)));
      else if (i->To() < V)
        newRanges = F.Add(newRanges, *i);
    }

    return newRanges;
  }

  RangeSet AddLE(BasicValueFactory &BV, Factory &F, const llvm::APSInt &V) {
    PrimRangeSet newRanges = F.GetEmptySet();

    for (iterator i = begin(), e = end(); i != e; ++i) {
      // Strictly we should test for includes *V + 1, but no harm is
      // done by this formulation
      if (i->Includes(V))
        newRanges = F.Add(newRanges, Range(i->From(), V));
      else if (i->To() <= V)
        newRanges = F.Add(newRanges, *i);
    }

    return newRanges;
  }

  RangeSet AddGT(BasicValueFactory &BV, Factory &F, const llvm::APSInt &V) {
    PrimRangeSet newRanges = F.GetEmptySet();

    for (PrimRangeSet::iterator i = begin(), e = end(); i != e; ++i) {
      if (i->Includes(V) && i->To() > V)
        newRanges = F.Add(newRanges, Range(BV.Add1(V), i->To()));
      else if (i->From() > V)
        newRanges = F.Add(newRanges, *i);
    }

    return newRanges;
  }

  RangeSet AddGE(BasicValueFactory &BV, Factory &F, const llvm::APSInt &V) {
    PrimRangeSet newRanges = F.GetEmptySet();

    for (PrimRangeSet::iterator i = begin(), e = end(); i != e; ++i) {
      // Strictly we should test for includes *V - 1, but no harm is
      // done by this formulation
      if (i->Includes(V))
        newRanges = F.Add(newRanges, Range(V, i->To()));
      else if (i->From() >= V)
        newRanges = F.Add(newRanges, *i);
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
template<>
struct GRStateTrait<ConstraintRange>
  : public GRStatePartialTrait<ConstraintRangeTy> {
  static inline void* GDMIndex() { return &ConstraintRangeIndex; }
};
}

namespace {
class VISIBILITY_HIDDEN RangeConstraintManager : public SimpleConstraintManager{
  RangeSet GetRange(const GRState *state, SymbolRef sym);
public:
  RangeConstraintManager() {}

  const GRState* AssumeSymNE(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymEQ(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymLT(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymGT(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymGE(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V);

  const GRState* AssumeSymLE(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V);

  const llvm::APSInt* getSymVal(const GRState* St, SymbolRef sym) const;

  // FIXME: Refactor into SimpleConstraintManager?
  bool isEqual(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const {
    const llvm::APSInt *i = getSymVal(St, sym);
    return i ? *i == V : false;
  }

  const GRState* RemoveDeadBindings(const GRState* St, SymbolReaper& SymReaper);

  void print(const GRState* St, llvm::raw_ostream& Out,
             const char* nl, const char *sep);

private:
  RangeSet::Factory F;
};

} // end anonymous namespace

ConstraintManager* clang::CreateRangeConstraintManager(GRStateManager&) {
  return new RangeConstraintManager();
}

const llvm::APSInt* RangeConstraintManager::getSymVal(const GRState* St,
                                                      SymbolRef sym) const {
  const ConstraintRangeTy::data_type *T = St->get<ConstraintRange>(sym);
  return T ? T->getConcreteValue() : NULL;
}

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
const GRState*
RangeConstraintManager::RemoveDeadBindings(const GRState* state,
                                           SymbolReaper& SymReaper) {

  ConstraintRangeTy CR = state->get<ConstraintRange>();
  ConstraintRangeTy::Factory& CRFactory = state->get_context<ConstraintRange>();

  for (ConstraintRangeTy::iterator I = CR.begin(), E = CR.end(); I != E; ++I) {
    SymbolRef sym = I.getKey();
    if (SymReaper.maybeDead(sym))
      CR = CRFactory.Remove(CR, sym);
  }

  return state->set<ConstraintRange>(CR);
}

//===------------------------------------------------------------------------===
// AssumeSymX methods: public interface for RangeConstraintManager.
//===------------------------------------------------------------------------===/

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
// AssumeSymX methods: public interface for RangeConstraintManager.
//===------------------------------------------------------------------------===/

#define AssumeX(OP)\
const GRState*\
RangeConstraintManager::AssumeSym ## OP(const GRState* state, SymbolRef sym,\
  const llvm::APSInt& V){\
  const RangeSet& R = GetRange(state, sym).Add##OP(state->getBasicVals(), F, V);\
  return !R.isEmpty() ? state->set<ConstraintRange>(sym, R) : NULL;\
}

AssumeX(EQ)
AssumeX(NE)
AssumeX(LT)
AssumeX(GT)
AssumeX(LE)
AssumeX(GE)

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

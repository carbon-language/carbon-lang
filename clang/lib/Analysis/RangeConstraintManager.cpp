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
#include "clang/Driver/ManagerRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace { class VISIBILITY_HIDDEN ConstRange {}; }

static int ConstRangeIndex = 0;

// A Range represents the closed range [from, to].  The caller must
// guarantee that from <= to.  Note that Range is immutable, so as not
// to subvert RangeSet's immutability.
class Range : public std::pair<llvm::APSInt, llvm::APSInt> {
public:
  Range(const llvm::APSInt &from, const llvm::APSInt &to)
    : std::pair<llvm::APSInt, llvm::APSInt>(from, to) {
    assert(from <= to);
  }
  bool Includes(const llvm::APSInt &v) const {
    return first <= v && v <= second;
  }
  const llvm::APSInt &From() const {
    return first;
  }
  const llvm::APSInt &To() const {
    return second;
  }
  const llvm::APSInt *HasConcreteValue() const {
    return From() == To() ? &From() : NULL;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    From().Profile(ID);
    To().Profile(ID);
  }
};

struct RangeCmp {
  bool operator()(const Range &r1, const Range &r2) {
    if (r1.From() < r2.From()) {
      assert(!r1.Includes(r2.From()));
      assert(!r2.Includes(r1.To()));
      return true;
    } else if (r1.From() > r2.From()) {
      assert(!r1.Includes(r2.To()));
      assert(!r2.Includes(r1.From()));
      return false;
    } else
      assert(!"Ranges should never be equal in the same set");
  }
};

typedef llvm::ImmutableSet<Range> PrimRangeSet;

class RangeSet;
std::ostream &operator<<(std::ostream &os, const RangeSet &r);


// A RangeSet contains a set of ranges. If the set is empty, then
//   noValues -> Nothing matches.
//  !noValues -> Everything (in range of the bit representation) matches.
class RangeSet {
  PrimRangeSet ranges; // no need to make const, since it is an
                       // ImmutableSet - this allows default operator=
                       // to work.
  bool noValues;  // if true, no value is possible (should never happen)

  static const llvm::APSInt Max(const llvm::APSInt &v) {
    return llvm::APSInt::getMaxValue(v.getBitWidth(), v.isUnsigned());
  }
  static const llvm::APSInt Min(const llvm::APSInt &v) {
     return llvm::APSInt::getMinValue(v.getBitWidth(), v.isUnsigned());
  }
  static const llvm::APSInt One(const llvm::APSInt &v) {
    return llvm::APSInt(llvm::APInt(v.getBitWidth(), 1), v.isUnsigned());
  }

public:
  // Create a RangeSet that allows all possible values.
  RangeSet(PrimRangeSet::Factory *factory) : ranges(factory->GetEmptySet()),
                                             noValues(false) {
  }
  // Note that if the empty set is passed, then there are no possible
  // values.  To create a RangeSet that covers all values when the
  // empty set is passed, use RangeSet(r, false).
  RangeSet(const PrimRangeSet &r) : ranges(r), noValues(r.isEmpty()) {
  }
  // Allow an empty set to be passed meaning "all values" instead of
  // "no values".
  RangeSet(const PrimRangeSet &r, bool n) : ranges(r), noValues(n) {
    assert(!n);
  }
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ranges.Profile(ID);
    ID.AddBoolean(noValues);
  }

  const llvm::APSInt *HasConcreteValue() const {
    if (!ranges.isSingleton())
      return NULL;
    return ranges.begin()->HasConcreteValue();
  }

  bool CouldBeNE(const llvm::APSInt &ne) const {
    DOUT << "CouldBeNE(" << ne.toString(10) << ") " << *this << std::endl;
    assert(!noValues);
    const llvm::APSInt *v = HasConcreteValue();
    if (v && *v == ne)
        return false;
    return true;
  }

  bool CouldBeEQ(const llvm::APSInt &eq) const {
    DOUT << "CouldBeEQ(" << eq.toString(10) << ") " << *this << std::endl;
    assert(!noValues);
    if (ranges.isEmpty())
      return true;
    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i)
      if (i->Includes(eq))
        return true;
    return false;
  }

  bool CouldBeLT(const llvm::APSInt &lt) const {
    DOUT << "CouldBeLT(" << lt.toString(10) << ") " << *this << std::endl;
    assert(!noValues);
    // FIXME: should test if lt == min -> false here, since that's
    // impossible to meet.
    if (ranges.isEmpty())
      return true;
    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i)
      if (i->From() < lt)
        return true;
    return false;
  }

  bool CouldBeLE(const llvm::APSInt &le) const {
    DOUT << "CouldBeLE(" << le.toString(10) << ") " << *this << std::endl;
    assert(!noValues);
    if (ranges.isEmpty())
      return true;
    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i)
      if (i->From() <= le)
        return true;
    return false;
  }

  bool CouldBeGT(const llvm::APSInt &gt) const {
    DOUT << "CouldBeGT(" << gt.toString(10) << ") " << *this << std::endl;
    assert(!noValues);
    // FIXME: should we test if gt == max -> false here, since that's
    // impossible to meet.
    if (ranges.isEmpty())
      return true;
    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i)
      if (i->To() > gt)
        return true;
    return false;
  }

  bool CouldBeGE(const llvm::APSInt &ge) const {
    DOUT << "CouldBeGE(" << ge.toString(10) << ") " << *this << std::endl;
    assert(!noValues);
    if (ranges.isEmpty())
      return true;
    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i)
      if (i->To() >= ge)
        return true;
    return false;
  }

  // Make all existing ranges fall within this new range
  RangeSet Restrict(PrimRangeSet::Factory *factory, const llvm::APSInt &from,
                    const llvm::APSInt &to) const {
    if (ranges.isEmpty())
      return factory->Add(ranges, Range(from, to));;

    PrimRangeSet newRanges = factory->GetEmptySet();

    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i) {
      if (i->Includes(from)) {
        if (i->Includes(to)) {
          newRanges = factory->Add(newRanges, Range(from, to));
        } else {
          newRanges = factory->Add(newRanges, Range(from, i->To()));
        }
      } else if (i->Includes(to)) {
        newRanges = factory->Add(newRanges, Range(i->From(), to));
      }
    }
    return RangeSet(newRanges);
  }

  // Create a new RangeSet with the additional constraint that the
  // range must be == eq. In other words the range becomes [eq,
  // eq]. Note that this RangeSet must have included eq in the first
  // place, or we shouldn't be here.
  RangeSet AddEQ(PrimRangeSet::Factory *factory, const llvm::APSInt &eq) {
    DOUT << "AddEQ(" << eq.toString(10) << ") " << *this << " -> ";
    assert(CouldBeEQ(eq));
    RangeSet r(factory->Add(factory->GetEmptySet(), Range(eq, eq)));
    DOUT << r << std::endl;
    return r;
  }

  RangeSet AddNE(PrimRangeSet::Factory *factory, const llvm::APSInt &ne) {
    DOUT << "AddNE(" << ne.toString(10) << ") " << *this << " -> ";

    const llvm::APSInt max = Max(ne);
    const llvm::APSInt min = Min(ne);
    const llvm::APSInt one = One(ne);

    PrimRangeSet newRanges = factory->GetEmptySet();

    if (ranges.isEmpty()) {
      if (ne != max)
        newRanges = factory->Add(newRanges, Range(ne + one, max));
      if (ne != min)
        newRanges = factory->Add(newRanges, Range(min, ne - one));
      RangeSet r(newRanges);
      DOUT << r << std::endl;
      return r;
    }

    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i) {
      if (i->Includes(ne)) {
        if (ne != i->From())
          newRanges = factory->Add(newRanges, Range(i->From(), ne - one));
        if (ne != i->To())
          newRanges = factory->Add(newRanges, Range(ne + one, i->To()));
      } else {
        newRanges = factory->Add(newRanges, *i);
      }
    }
    RangeSet r(newRanges);
    DOUT << r << std::endl;
    return r;
  }

  RangeSet AddLT(PrimRangeSet::Factory *factory, const llvm::APSInt &lt) {
    DOUT << "AddLT(" << lt.toString(10) << ") " << *this << " -> ";
    const llvm::APSInt min = Min(lt);
    const llvm::APSInt one = One(lt);

    if (ranges.isEmpty()) {
      PrimRangeSet pr = factory->GetEmptySet();
      if (lt != min)
        pr = factory->Add(pr, Range(min, lt - one));
      RangeSet r(pr, false);
      DOUT << r << std::endl;
      return r;
    }

    PrimRangeSet newRanges = factory->GetEmptySet();

    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i) {
      if (i->Includes(lt) && i->From() < lt)
        newRanges = factory->Add(newRanges, Range(i->From(), lt - one));
      else if (i->To() < lt)
        newRanges = factory->Add(newRanges, *i);
    }
    RangeSet r(newRanges);
    DOUT << r << std::endl;
    return r;
  }

  RangeSet AddLE(PrimRangeSet::Factory *factory, const llvm::APSInt &le) {
    DOUT << "AddLE(" << le.toString(10) << ") " << *this << " -> ";
    const llvm::APSInt min = Min(le);

    if (ranges.isEmpty()) {
      RangeSet r(factory->Add(ranges, Range(min, le)));
      DOUT << r << std::endl;
      return r;
    }

    PrimRangeSet newRanges = factory->GetEmptySet();

    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i) {
      // Strictly we should test for includes le + 1, but no harm is
      // done by this formulation
      if (i->Includes(le))
        newRanges = factory->Add(newRanges, Range(i->From(), le));
      else if (i->To() <= le)
        newRanges = factory->Add(newRanges, *i);
    }
    RangeSet r(newRanges);
    DOUT << r << std::endl;
    return r;
  }

  RangeSet AddGT(PrimRangeSet::Factory *factory, const llvm::APSInt &gt) {
    DOUT << "AddGT(" << gt.toString(10) << ") " << *this << " -> ";
    const llvm::APSInt max = Max(gt);
    const llvm::APSInt one = One(gt);

    if (ranges.isEmpty()) {
      RangeSet r(factory->Add(ranges, Range(gt + one, max)));
      DOUT << r << std::endl;
      return r;
    }

    PrimRangeSet newRanges = factory->GetEmptySet();

    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i) {
      if (i->Includes(gt) && i->To() > gt)
        newRanges = factory->Add(newRanges, Range(gt + one, i->To()));
      else if (i->From() > gt)
        newRanges = factory->Add(newRanges, *i);
    }
    RangeSet r(newRanges);
    DOUT << r << std::endl;
    return r;
  }

  RangeSet AddGE(PrimRangeSet::Factory *factory, const llvm::APSInt &ge) {
    DOUT << "AddGE(" << ge.toString(10) << ") " << *this << " -> ";
    const llvm::APSInt max = Max(ge);

    if (ranges.isEmpty()) {
      RangeSet r(factory->Add(ranges, Range(ge, max)));
      DOUT << r << std::endl;
      return r;
    }

    PrimRangeSet newRanges = factory->GetEmptySet();

    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i) {
      // Strictly we should test for includes ge - 1, but no harm is
      // done by this formulation
      if (i->Includes(ge))
        newRanges = factory->Add(newRanges, Range(ge, i->To()));
      else if (i->From() >= ge)
        newRanges = factory->Add(newRanges, *i);
    }

    RangeSet r(newRanges);
    DOUT << r << std::endl;
    return r;
  }

  void Print(std::ostream &os) const {
    os << "{ ";
    if (noValues) {
      os << "**no values** }";
      return;
    }
    for (PrimRangeSet::iterator i = ranges.begin() ; i != ranges.end() ; ++i) {
      if (i != ranges.begin())
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

std::ostream &operator<<(std::ostream &os, const RangeSet &r) {
  r.Print(os);
  return os;
}

typedef llvm::ImmutableMap<SymbolRef,RangeSet> ConstRangeTy;

namespace clang {
template<>
struct GRStateTrait<ConstRange> : public GRStatePartialTrait<ConstRangeTy> {
  static inline void* GDMIndex() { return &ConstRangeIndex; }  
};
}  
  
namespace {
class VISIBILITY_HIDDEN RangeConstraintManager
  : public SimpleConstraintManager {
public:
  RangeConstraintManager(GRStateManager& statemgr) 
      : SimpleConstraintManager(statemgr) {}

  const GRState* AssumeSymNE(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymEQ(const GRState* St, SymbolRef sym,
                                const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymLT(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymGT(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymGE(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AssumeSymLE(const GRState* St, SymbolRef sym,
                             const llvm::APSInt& V, bool& isFeasible);

  const GRState* AddEQ(const GRState* St, SymbolRef sym, const llvm::APSInt& V);

  const GRState* AddNE(const GRState* St, SymbolRef sym, const llvm::APSInt& V);

  const GRState* AddLT(const GRState* St, SymbolRef sym, const llvm::APSInt& V);

  const GRState* AddLE(const GRState* St, SymbolRef sym, const llvm::APSInt& V);

  const GRState* AddGT(const GRState* St, SymbolRef sym, const llvm::APSInt& V);

  const GRState* AddGE(const GRState* St, SymbolRef sym, const llvm::APSInt& V);

  // FIXME: these two are required because they are pure virtual, but
  // are they useful with ranges? Neither is used in this file.
  const llvm::APSInt* getSymVal(const GRState* St, SymbolRef sym) const;
  bool isEqual(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const;

  bool CouldBeEQ(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const;
  bool CouldBeNE(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const;

  bool CouldBeLT(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const;
  bool CouldBeLE(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const;
  bool CouldBeGT(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const;
  bool CouldBeGE(const GRState* St, SymbolRef sym, const llvm::APSInt& V) const;
  const GRState* RemoveDeadBindings(const GRState* St, SymbolReaper& SymReaper);

  void print(const GRState* St, std::ostream& Out, 
             const char* nl, const char *sep);

private:
  PrimRangeSet::Factory factory;
  BasicValueFactory& getBasicVals() { return StateMgr.getBasicVals(); }
};

} // end anonymous namespace

ConstraintManager* clang::CreateRangeConstraintManager(GRStateManager& StateMgr)
{
  return new RangeConstraintManager(StateMgr);
}

RegisterConstraintManager X(CreateRangeConstraintManager);

const GRState*
RangeConstraintManager::AssumeSymNE(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {
  isFeasible = CouldBeNE(St, sym, V);
  if (isFeasible)
    return AddNE(St, sym, V);
  return St;
}

const GRState*
RangeConstraintManager::AssumeSymEQ(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {
  isFeasible = CouldBeEQ(St, sym, V);
  if (isFeasible)
    return AddEQ(St, sym, V);
  return St;
}

const GRState*
RangeConstraintManager::AssumeSymLT(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {
  
  // Is 'V' the smallest possible value?
  if (V == llvm::APSInt::getMinValue(V.getBitWidth(), V.isUnsigned())) {
    // sym cannot be any value less than 'V'.  This path is infeasible.
    isFeasible = false;
    return St;
  }

  isFeasible = CouldBeLT(St, sym, V);
  if (isFeasible)
    return AddLT(St, sym, V);

  return St;
}

const GRState*
RangeConstraintManager::AssumeSymGT(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  // Is 'V' the largest possible value?
  if (V == llvm::APSInt::getMaxValue(V.getBitWidth(), V.isUnsigned())) {
    // sym cannot be any value greater than 'V'.  This path is infeasible.
    isFeasible = false;
    return St;
  }

  isFeasible = CouldBeGT(St, sym, V);
  if (isFeasible)
    return AddGT(St, sym, V);

  return St;
}

const GRState*
RangeConstraintManager::AssumeSymGE(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  isFeasible = CouldBeGE(St, sym, V);
  if (isFeasible)
    return AddGE(St, sym, V);

  return St;
}

const GRState*
RangeConstraintManager::AssumeSymLE(const GRState* St, SymbolRef sym,
                                    const llvm::APSInt& V, bool& isFeasible) {

  isFeasible = CouldBeLT(St, sym, V);
  if (isFeasible)
    return AddLE(St, sym, V);
    
  return St;
}

const GRState* RangeConstraintManager::AddEQ(const GRState* St, SymbolRef sym,
                                             const llvm::APSInt& V) {
  // Create a new state with the old binding replaced.
  GRStateRef state(St, StateMgr);
  RangeSet R(&factory);
  R = R.AddEQ(&factory, V);
  return state.set<ConstRange>(sym, R);
}

const GRState* RangeConstraintManager::AddNE(const GRState* St, SymbolRef sym,
                                             const llvm::APSInt& V) {
  GRStateRef state(St, StateMgr);

  ConstRangeTy::data_type* T = state.get<ConstRange>(sym);
  RangeSet R(&factory);
  if (T)
    R = *T;
  R = R.AddNE(&factory, V);
  return state.set<ConstRange>(sym, R);
}

const GRState* RangeConstraintManager::AddLT(const GRState* St, SymbolRef sym,
                                             const llvm::APSInt& V) {
  GRStateRef state(St, StateMgr);

  ConstRangeTy::data_type* T = state.get<ConstRange>(sym);
  RangeSet R(&factory);
  if (T)
    R = *T;
  R = R.AddLT(&factory, V);
  return state.set<ConstRange>(sym, R);
}

const GRState* RangeConstraintManager::AddLE(const GRState* St, SymbolRef sym,
                                             const llvm::APSInt& V) {
  GRStateRef state(St, StateMgr);

  ConstRangeTy::data_type* T = state.get<ConstRange>(sym);
  RangeSet R(&factory);
  if (T)
    R = *T;
  R = R.AddLE(&factory, V);
  return state.set<ConstRange>(sym, R);
}

const GRState* RangeConstraintManager::AddGT(const GRState* St, SymbolRef sym,
                                             const llvm::APSInt& V) {
  GRStateRef state(St, StateMgr);

  ConstRangeTy::data_type* T = state.get<ConstRange>(sym);
  RangeSet R(&factory);
  if (T)
    R = *T;
  R = R.AddGT(&factory, V);
  return state.set<ConstRange>(sym, R);
}

const GRState* RangeConstraintManager::AddGE(const GRState* St, SymbolRef sym,
                                             const llvm::APSInt& V) {
  GRStateRef state(St, StateMgr);

  ConstRangeTy::data_type* T = state.get<ConstRange>(sym);
  RangeSet R(&factory);
  if (T)
    R = *T;
  R = R.AddGE(&factory, V);
  return state.set<ConstRange>(sym, R);
}

const llvm::APSInt* RangeConstraintManager::getSymVal(const GRState* St,
                                                      SymbolRef sym) const {
  const ConstRangeTy::data_type *T = St->get<ConstRange>(sym);
  return T ? T->HasConcreteValue() : NULL;
}

bool RangeConstraintManager::CouldBeLT(const GRState* St, SymbolRef sym, 
                                       const llvm::APSInt& V) const {
  const ConstRangeTy::data_type *T = St->get<ConstRange>(sym);
  return T ? T->CouldBeLT(V) : true;
}

bool RangeConstraintManager::CouldBeLE(const GRState* St, SymbolRef sym, 
                                       const llvm::APSInt& V) const {
  const ConstRangeTy::data_type *T = St->get<ConstRange>(sym);
  return T ? T->CouldBeLE(V) : true;
}

bool RangeConstraintManager::CouldBeGT(const GRState* St, SymbolRef sym, 
                                       const llvm::APSInt& V) const {
  const ConstRangeTy::data_type *T = St->get<ConstRange>(sym);
  return T ? T->CouldBeGT(V) : true;
}

bool RangeConstraintManager::CouldBeGE(const GRState* St, SymbolRef sym, 
                                       const llvm::APSInt& V) const {
  const ConstRangeTy::data_type *T = St->get<ConstRange>(sym);
  return T ? T->CouldBeGE(V) : true;
}

bool RangeConstraintManager::CouldBeNE(const GRState* St, SymbolRef sym, 
                                        const llvm::APSInt& V) const {
  const ConstRangeTy::data_type *T = St->get<ConstRange>(sym);
  return T ? T->CouldBeNE(V) : true;
}

bool RangeConstraintManager::CouldBeEQ(const GRState* St, SymbolRef sym, 
                                        const llvm::APSInt& V) const {
  const ConstRangeTy::data_type *T = St->get<ConstRange>(sym);
  return T ? T->CouldBeEQ(V) : true;
}

bool RangeConstraintManager::isEqual(const GRState* St, SymbolRef sym,
                                     const llvm::APSInt& V) const {
  const llvm::APSInt *i = getSymVal(St, sym);
  return i ? *i == V : false;
}

/// Scan all symbols referenced by the constraints. If the symbol is not alive
/// as marked in LSymbols, mark it as dead in DSymbols.
const GRState*
RangeConstraintManager::RemoveDeadBindings(const GRState* St,
                                           SymbolReaper& SymReaper) {
  GRStateRef state(St, StateMgr);

  ConstRangeTy CR = state.get<ConstRange>();
  ConstRangeTy::Factory& CRFactory = state.get_context<ConstRange>();

  for (ConstRangeTy::iterator I = CR.begin(), E = CR.end(); I != E; ++I) {
    SymbolRef sym = I.getKey();    
    if (SymReaper.maybeDead(sym))
      CR = CRFactory.Remove(CR, sym);
  }
  
  return state.set<ConstRange>(CR);
}

void RangeConstraintManager::print(const GRState* St, std::ostream& Out, 
                                   const char* nl, const char *sep) {
#if 0
  // Print equality constraints.

  ConstEqTy CE = St->get<ConstEq>();

  if (!CE.isEmpty()) {
    Out << nl << sep << "'==' constraints:";

    for (ConstEqTy::iterator I = CE.begin(), E = CE.end(); I!=E; ++I) {
      Out << nl << " $" << I.getKey();
      llvm::raw_os_ostream OS(Out);
      OS << " : "   << *I.getData();
    }
  }

  // Print != constraints.
  
  ConstNotEqTy CNE = St->get<ConstNotEq>();
  
  if (!CNE.isEmpty()) {
    Out << nl << sep << "'!=' constraints:";
  
    for (ConstNotEqTy::iterator I = CNE.begin(), EI = CNE.end(); I!=EI; ++I) {
      Out << nl << " $" << I.getKey() << " : ";
      bool isFirst = true;
    
      GRState::IntSetTy::iterator J = I.getData().begin(), 
                                  EJ = I.getData().end();      
      
      for ( ; J != EJ; ++J) {        
        if (isFirst) isFirst = false;
        else Out << ", ";
      
        Out << (*J)->getSExtValue(); // Hack: should print to raw_ostream.
      }
    }
  }
#endif  // 0

  Out << nl << "Implement range printing";
}

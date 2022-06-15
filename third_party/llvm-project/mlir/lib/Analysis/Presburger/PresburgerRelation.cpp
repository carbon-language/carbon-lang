//===- PresburgerRelation.cpp - MLIR PresburgerRelation Class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace presburger;

PresburgerRelation::PresburgerRelation(const IntegerRelation &disjunct)
    : space(disjunct.getSpaceWithoutLocals()) {
  unionInPlace(disjunct);
}

unsigned PresburgerRelation::getNumDisjuncts() const {
  return disjuncts.size();
}

ArrayRef<IntegerRelation> PresburgerRelation::getAllDisjuncts() const {
  return disjuncts;
}

const IntegerRelation &PresburgerRelation::getDisjunct(unsigned index) const {
  assert(index < disjuncts.size() && "index out of bounds!");
  return disjuncts[index];
}

/// Mutate this set, turning it into the union of this set and the given
/// IntegerRelation.
void PresburgerRelation::unionInPlace(const IntegerRelation &disjunct) {
  assert(space.isCompatible(disjunct.getSpace()) && "Spaces should match");
  disjuncts.push_back(disjunct);
}

/// Mutate this set, turning it into the union of this set and the given set.
///
/// This is accomplished by simply adding all the disjuncts of the given set
/// to this set.
void PresburgerRelation::unionInPlace(const PresburgerRelation &set) {
  assert(space.isCompatible(set.getSpace()) && "Spaces should match");
  for (const IntegerRelation &disjunct : set.disjuncts)
    unionInPlace(disjunct);
}

/// Return the union of this set and the given set.
PresburgerRelation
PresburgerRelation::unionSet(const PresburgerRelation &set) const {
  assert(space.isCompatible(set.getSpace()) && "Spaces should match");
  PresburgerRelation result = *this;
  result.unionInPlace(set);
  return result;
}

/// A point is contained in the union iff any of the parts contain the point.
bool PresburgerRelation::containsPoint(ArrayRef<int64_t> point) const {
  return llvm::any_of(disjuncts, [&](const IntegerRelation &disjunct) {
    return (disjunct.containsPointNoLocal(point));
  });
}

PresburgerRelation
PresburgerRelation::getUniverse(const PresburgerSpace &space) {
  PresburgerRelation result(space);
  result.unionInPlace(IntegerRelation::getUniverse(space));
  return result;
}

PresburgerRelation PresburgerRelation::getEmpty(const PresburgerSpace &space) {
  return PresburgerRelation(space);
}

// Return the intersection of this set with the given set.
//
// We directly compute (S_1 or S_2 ...) and (T_1 or T_2 ...)
// as (S_1 and T_1) or (S_1 and T_2) or ...
//
// If S_i or T_j have local variables, then S_i and T_j contains the local
// variables of both.
PresburgerRelation
PresburgerRelation::intersect(const PresburgerRelation &set) const {
  assert(space.isCompatible(set.getSpace()) && "Spaces should match");

  PresburgerRelation result(getSpace());
  for (const IntegerRelation &csA : disjuncts) {
    for (const IntegerRelation &csB : set.disjuncts) {
      IntegerRelation intersection = csA.intersect(csB);
      if (!intersection.isEmpty())
        result.unionInPlace(intersection);
    }
  }
  return result;
}

/// Return the coefficients of the ineq in `rel` specified by  `idx`.
/// `idx` can refer not only to an actual inequality of `rel`, but also
/// to either of the inequalities that make up an equality in `rel`.
///
/// When 0 <= idx < rel.getNumInequalities(), this returns the coeffs of the
/// idx-th inequality of `rel`.
///
/// Otherwise, it is then considered to index into the ineqs corresponding to
/// eqs of `rel`, and it must hold that
///
/// 0 <= idx - rel.getNumInequalities() < 2*getNumEqualities().
///
/// For every eq `coeffs == 0` there are two possible ineqs to index into.
/// The first is coeffs >= 0 and the second is coeffs <= 0.
static SmallVector<int64_t, 8> getIneqCoeffsFromIdx(const IntegerRelation &rel,
                                                    unsigned idx) {
  assert(idx < rel.getNumInequalities() + 2 * rel.getNumEqualities() &&
         "idx out of bounds!");
  if (idx < rel.getNumInequalities())
    return llvm::to_vector<8>(rel.getInequality(idx));

  idx -= rel.getNumInequalities();
  ArrayRef<int64_t> eqCoeffs = rel.getEquality(idx / 2);

  if (idx % 2 == 0)
    return llvm::to_vector<8>(eqCoeffs);
  return getNegatedCoeffs(eqCoeffs);
}

/// Return the set difference b \ s.
///
/// In the following, U denotes union, /\ denotes intersection, \ denotes set
/// difference and ~ denotes complement.
///
/// Let s = (U_i s_i). We want  b \ (U_i s_i).
///
/// Let s_i = /\_j s_ij, where each s_ij is a single inequality. To compute
/// b \ s_i = b /\ ~s_i, we partition s_i based on the first violated
/// inequality: ~s_i = (~s_i1) U (s_i1 /\ ~s_i2) U (s_i1 /\ s_i2 /\ ~s_i3) U ...
/// And the required result is (b /\ ~s_i1) U (b /\ s_i1 /\ ~s_i2) U ...
/// We recurse by subtracting U_{j > i} S_j from each of these parts and
/// returning the union of the results. Each equality is handled as a
/// conjunction of two inequalities.
///
/// Note that the same approach works even if an inequality involves a floor
/// division. For example, the complement of x <= 7*floor(x/7) is still
/// x > 7*floor(x/7). Since b \ s_i contains the inequalities of both b and s_i
/// (or the complements of those inequalities), b \ s_i may contain the
/// divisions present in both b and s_i. Therefore, we need to add the local
/// division variables of both b and s_i to each part in the result. This means
/// adding the local variables of both b and s_i, as well as the corresponding
/// division inequalities to each part. Since the division inequalities are
/// added to each part, we can skip the parts where the complement of any
/// division inequality is added, as these parts will become empty anyway.
///
/// As a heuristic, we try adding all the constraints and check if simplex
/// says that the intersection is empty. If it is, then subtracting this
/// disjuncts is a no-op and we just skip it. Also, in the process we find out
/// that some constraints are redundant. These redundant constraints are
/// ignored.
///
static PresburgerRelation getSetDifference(IntegerRelation b,
                                           const PresburgerRelation &s) {
  assert(b.getSpace().isCompatible(s.getSpace()) && "Spaces should match");
  if (b.isEmptyByGCDTest())
    return PresburgerRelation::getEmpty(b.getSpaceWithoutLocals());

  // Remove duplicate divs up front here to avoid existing
  // divs disappearing in the call to mergeLocalIds below.
  b.removeDuplicateDivs();

  PresburgerRelation result =
      PresburgerRelation::getEmpty(b.getSpaceWithoutLocals());
  Simplex simplex(b);

  // This algorithm is more naturally expressed recursively, but we implement
  // it iteratively here to avoid issues with stack sizes.
  //
  // Each level of the recursion has five stack variables.
  struct Frame {
    // A snapshot of the simplex state to rollback to.
    unsigned simplexSnapshot;
    // A CountsSnapshot of `b` to rollback to.
    IntegerRelation::CountsSnapshot bCounts;
    // The IntegerRelation currently being operated on.
    IntegerRelation sI;
    // A list of indexes (see getIneqCoeffsFromIdx) of inequalities to be
    // processed.
    SmallVector<unsigned, 8> ineqsToProcess;
    // The index of the last inequality that was processed at this level.
    // This is empty when we are coming to this level for the first time.
    Optional<unsigned> lastIneqProcessed;
  };
  SmallVector<Frame, 2> frames;

  // When we "recurse", we ensure the current frame is stored in `frames` and
  // increment `level`. When we return, we decrement `level`.
  unsigned level = 1;
  while (level > 0) {
    if (level - 1 >= s.getNumDisjuncts()) {
      // No more parts to subtract; add to the result and return.
      result.unionInPlace(b);
      level = frames.size();
      continue;
    }

    if (level > frames.size()) {
      // No frame for this level yet, so we have just recursed into this level.
      IntegerRelation sI = s.getDisjunct(level - 1);
      // Remove the duplicate divs up front to avoid them possibly disappearing
      // in the call to mergeLocalIds below.
      sI.removeDuplicateDivs();

      // Below, we append some additional constraints and ids to b. We want to
      // rollback b to its initial state before returning, which we will do by
      // removing all constraints beyond the original number of inequalities
      // and equalities, so we store these counts first.
      IntegerRelation::CountsSnapshot initBCounts = b.getCounts();
      // Similarly, we also want to rollback simplex to its original state.
      unsigned initialSnapshot = simplex.getSnapshot();

      // Find out which inequalities of sI correspond to division inequalities
      // for the local variables of sI.
      std::vector<MaybeLocalRepr> repr(sI.getNumLocalIds());
      sI.getLocalReprs(repr);

      // Add sI's locals to b, after b's locals. Only those locals of sI which
      // do not already exist in b will be added. (i.e., duplicate divisions
      // will not be added.) Also add b's locals to sI, in such a way that both
      // have the same locals in the same order in the end.
      b.mergeLocalIds(sI);

      // Mark which inequalities of sI are division inequalities and add all
      // such inequalities to b.
      llvm::SmallBitVector canIgnoreIneq(sI.getNumInequalities() +
                                         2 * sI.getNumEqualities());
      for (MaybeLocalRepr &maybeRepr : repr) {
        assert(
            maybeRepr &&
            "Subtraction is not supported when a representation of the local "
            "variables of the subtrahend cannot be found!");

        if (maybeRepr.kind == ReprKind::Inequality) {
          unsigned lb = maybeRepr.repr.inequalityPair.lowerBoundIdx;
          unsigned ub = maybeRepr.repr.inequalityPair.upperBoundIdx;

          b.addInequality(sI.getInequality(lb));
          b.addInequality(sI.getInequality(ub));

          assert(lb != ub &&
                 "Upper and lower bounds must be different inequalities!");
          canIgnoreIneq[lb] = true;
          canIgnoreIneq[ub] = true;
        } else {
          assert(maybeRepr.kind == ReprKind::Equality &&
                 "ReprKind isn't inequality so should be equality");
          unsigned idx = maybeRepr.repr.equalityIdx;
          b.addEquality(sI.getEquality(idx));
          // We can ignore both inequalities corresponding to this equality.
          unsigned offset = sI.getNumInequalities();
          canIgnoreIneq[offset + 2 * idx] = true;
          canIgnoreIneq[offset + 2 * idx + 1] = true;
        }
      }

      unsigned offset = simplex.getNumConstraints();
      unsigned numLocalsAdded =
          b.getNumLocalIds() - initBCounts.getSpace().getNumLocalIds();
      simplex.appendVariable(numLocalsAdded);

      unsigned snapshotBeforeIntersect = simplex.getSnapshot();
      simplex.intersectIntegerRelation(sI);

      if (simplex.isEmpty()) {
        // b /\ s_i is empty, so b \ s_i = b. We move directly to i + 1.
        // We are ignoring level i completely, so we restore the state
        // *before* going to the next level.
        b.truncate(initBCounts);
        simplex.rollback(initialSnapshot);
        // Recurse. We haven't processed any inequalities and
        // we don't need to process anything when we return.
        //
        // TODO: consider supporting tail recursion directly if this becomes
        // relevant for performance.
        frames.push_back(Frame{initialSnapshot, initBCounts, sI,
                               /*ineqsToProcess=*/{},
                               /*lastIneqProcessed=*/{}});
        ++level;
        continue;
      }

      // Equalities are added to simplex as a pair of inequalities.
      unsigned totalNewSimplexInequalities =
          2 * sI.getNumEqualities() + sI.getNumInequalities();
      // Look for redundant constraints among the constraints of sI. We don't
      // care about redundant constraints in `b` at this point.
      //
      // When there are two copies of a constraint in `simplex`, i.e., among the
      // constraints of `b` and `sI`, only one of them can be marked redundant.
      // (Assuming no other constraint makes these redundant.)
      //
      // In a case where there is one copy in `b` and one in `sI`, we want the
      // one in `sI` to be marked, not the one in `b`. Therefore, it's not
      // enough to ignore the constraints of `b` when checking which
      // constraints `detectRedundant` has marked redundant; we explicitly tell
      // `detectRedundant` to only mark constraints from `sI` as being
      // redundant.
      simplex.detectRedundant(offset, totalNewSimplexInequalities);
      for (unsigned j = 0; j < totalNewSimplexInequalities; j++)
        canIgnoreIneq[j] = simplex.isMarkedRedundant(offset + j);
      simplex.rollback(snapshotBeforeIntersect);

      SmallVector<unsigned, 8> ineqsToProcess;
      ineqsToProcess.reserve(totalNewSimplexInequalities);
      for (unsigned i = 0; i < totalNewSimplexInequalities; ++i)
        if (!canIgnoreIneq[i])
          ineqsToProcess.push_back(i);

      if (ineqsToProcess.empty()) {
        // Nothing to process; return. (we have no frame to pop.)
        level = frames.size();
        continue;
      }

      unsigned simplexSnapshot = simplex.getSnapshot();
      IntegerRelation::CountsSnapshot bCounts = b.getCounts();
      frames.push_back(Frame{simplexSnapshot, bCounts, sI, ineqsToProcess,
                             /*lastIneqProcessed=*/llvm::None});
      // We have completed the initial setup for this level.
      // Fallthrough to the main recursive part below.
    }

    // For each inequality ineq, we first recurse with the part where ineq
    // is not satisfied, and then add ineq to b and simplex because
    // ineq must be satisfied by all later parts.
    if (level == frames.size()) {
      Frame &frame = frames.back();
      if (frame.lastIneqProcessed) {
        // Let the current value of b be b' and
        // let the initial value of b when we first came to this level be b.
        //
        // b' is equal to b /\ s_i1 /\ s_i2 /\ ... /\ s_i{j-1} /\ ~s_ij.
        // We had previously recursed with the part where s_ij was not
        // satisfied; all further parts satisfy s_ij, so we rollback to the
        // state before adding this complement constraint, and add s_ij to b.
        simplex.rollback(frame.simplexSnapshot);
        b.truncate(frame.bCounts);
        SmallVector<int64_t, 8> ineq =
            getIneqCoeffsFromIdx(frame.sI, *frame.lastIneqProcessed);
        b.addInequality(ineq);
        simplex.addInequality(ineq);
      }

      if (frame.ineqsToProcess.empty()) {
        // No ineqs left to process; pop this level's frame and return.
        frames.pop_back();
        level = frames.size();
        continue;
      }

      // "Recurse" with the part where the ineq is not satisfied.
      frame.bCounts = b.getCounts();
      frame.simplexSnapshot = simplex.getSnapshot();

      unsigned idx = frame.ineqsToProcess.back();
      SmallVector<int64_t, 8> ineq =
          getComplementIneq(getIneqCoeffsFromIdx(frame.sI, idx));
      b.addInequality(ineq);
      simplex.addInequality(ineq);

      frame.ineqsToProcess.pop_back();
      frame.lastIneqProcessed = idx;
      ++level;
      continue;
    }
  }

  return result;
}

/// Return the complement of this set.
PresburgerRelation PresburgerRelation::complement() const {
  return getSetDifference(IntegerRelation::getUniverse(getSpace()), *this);
}

/// Return the result of subtract the given set from this set, i.e.,
/// return `this \ set`.
PresburgerRelation
PresburgerRelation::subtract(const PresburgerRelation &set) const {
  assert(space.isCompatible(set.getSpace()) && "Spaces should match");
  PresburgerRelation result(getSpace());
  // We compute (U_i t_i) \ (U_i set_i) as U_i (t_i \ V_i set_i).
  for (const IntegerRelation &disjunct : disjuncts)
    result.unionInPlace(getSetDifference(disjunct, set));
  return result;
}

/// T is a subset of S iff T \ S is empty, since if T \ S contains a
/// point then this is a point that is contained in T but not S, and
/// if T contains a point that is not in S, this also lies in T \ S.
bool PresburgerRelation::isSubsetOf(const PresburgerRelation &set) const {
  return this->subtract(set).isIntegerEmpty();
}

/// Two sets are equal iff they are subsets of each other.
bool PresburgerRelation::isEqual(const PresburgerRelation &set) const {
  assert(space.isCompatible(set.getSpace()) && "Spaces should match");
  return this->isSubsetOf(set) && set.isSubsetOf(*this);
}

/// Return true if all the sets in the union are known to be integer empty,
/// false otherwise.
bool PresburgerRelation::isIntegerEmpty() const {
  // The set is empty iff all of the disjuncts are empty.
  return llvm::all_of(disjuncts, std::mem_fn(&IntegerRelation::isIntegerEmpty));
}

bool PresburgerRelation::findIntegerSample(SmallVectorImpl<int64_t> &sample) {
  // A sample exists iff any of the disjuncts contains a sample.
  for (const IntegerRelation &disjunct : disjuncts) {
    if (Optional<SmallVector<int64_t, 8>> opt = disjunct.findIntegerSample()) {
      sample = std::move(*opt);
      return true;
    }
  }
  return false;
}

Optional<uint64_t> PresburgerRelation::computeVolume() const {
  assert(getNumSymbolIds() == 0 && "Symbols are not yet supported!");
  // The sum of the volumes of the disjuncts is a valid overapproximation of the
  // volume of their union, even if they overlap.
  uint64_t result = 0;
  for (const IntegerRelation &disjunct : disjuncts) {
    Optional<uint64_t> volume = disjunct.computeVolume();
    if (!volume)
      return {};
    result += *volume;
  }
  return result;
}

/// The SetCoalescer class contains all functionality concerning the coalesce
/// heuristic. It is built from a `PresburgerRelation` and has the `coalesce()`
/// function as its main API. The coalesce heuristic simplifies the
/// representation of a PresburgerRelation. In particular, it removes all
/// disjuncts which are subsets of other disjuncts in the union and it combines
/// sets that overlap and can be combined in a convex way.
class presburger::SetCoalescer {

public:
  /// Simplifies the representation of a PresburgerSet.
  PresburgerRelation coalesce();

  /// Construct a SetCoalescer from a PresburgerSet.
  SetCoalescer(const PresburgerRelation &s);

private:
  /// The space of the set the SetCoalescer is coalescing.
  PresburgerSpace space;

  /// The current list of `IntegerRelation`s that the currently coalesced set is
  /// the union of.
  SmallVector<IntegerRelation, 2> disjuncts;
  /// The list of `Simplex`s constructed from the elements of `disjuncts`.
  SmallVector<Simplex, 2> simplices;

  /// The list of all inversed equalities during typing. This ensures that
  /// the constraints exist even after the typing function has concluded.
  SmallVector<SmallVector<int64_t, 2>, 2> negEqs;

  /// `redundantIneqsA` is the inequalities of `a` that are redundant for `b`
  /// (similarly for `cuttingIneqsA`, `redundantIneqsB`, and `cuttingIneqsB`).
  SmallVector<ArrayRef<int64_t>, 2> redundantIneqsA;
  SmallVector<ArrayRef<int64_t>, 2> cuttingIneqsA;

  SmallVector<ArrayRef<int64_t>, 2> redundantIneqsB;
  SmallVector<ArrayRef<int64_t>, 2> cuttingIneqsB;

  /// Given a Simplex `simp` and one of its inequalities `ineq`, check
  /// that the facet of `simp` where `ineq` holds as an equality is contained
  /// within `a`.
  bool isFacetContained(ArrayRef<int64_t> ineq, Simplex &simp);

  /// Removes redundant constraints from `disjunct`, adds it to `disjuncts` and
  /// removes the disjuncts at position `i` and `j`. Updates `simplices` to
  /// reflect the changes. `i` and `j` cannot be equal.
  void addCoalescedDisjunct(unsigned i, unsigned j,
                            const IntegerRelation &disjunct);

  /// Checks whether `a` and `b` can be combined in a convex sense, if there
  /// exist cutting inequalities.
  ///
  /// An example of this case:
  ///    ___________        ___________
  ///   /   /  |   /       /          /
  ///   \   \  |  /   ==>  \         /
  ///    \   \ | /          \       /
  ///     \___\|/            \_____/
  ///
  ///
  LogicalResult coalescePairCutCase(unsigned i, unsigned j);

  /// Types the inequality `ineq` according to its `IneqType` for `simp` into
  /// `redundantIneqsB` and `cuttingIneqsB`. Returns success, if no separate
  /// inequalities were encountered. Otherwise, returns failure.
  LogicalResult typeInequality(ArrayRef<int64_t> ineq, Simplex &simp);

  /// Types the equality `eq`, i.e. for `eq` == 0, types both `eq` >= 0 and
  /// -`eq` >= 0 according to their `IneqType` for `simp` into
  /// `redundantIneqsB` and `cuttingIneqsB`. Returns success, if no separate
  /// inequalities were encountered. Otherwise, returns failure.
  LogicalResult typeEquality(ArrayRef<int64_t> eq, Simplex &simp);

  /// Replaces the element at position `i` with the last element and erases
  /// the last element for both `disjuncts` and `simplices`.
  void eraseDisjunct(unsigned i);

  /// Attempts to coalesce the two IntegerRelations at position `i` and `j`
  /// in `disjuncts` in-place. Returns whether the disjuncts were
  /// successfully coalesced. The simplices in `simplices` need to be the ones
  /// constructed from `disjuncts`. At this point, there are no empty
  /// disjuncts in `disjuncts` left.
  LogicalResult coalescePair(unsigned i, unsigned j);
};

/// Constructs a `SetCoalescer` from a `PresburgerRelation`. Only adds non-empty
/// `IntegerRelation`s to the `disjuncts` vector.
SetCoalescer::SetCoalescer(const PresburgerRelation &s) : space(s.getSpace()) {

  disjuncts = s.disjuncts;

  simplices.reserve(s.getNumDisjuncts());
  // Note that disjuncts.size() changes during the loop.
  for (unsigned i = 0; i < disjuncts.size();) {
    disjuncts[i].removeRedundantConstraints();
    Simplex simp(disjuncts[i]);
    if (simp.isEmpty()) {
      disjuncts[i] = disjuncts[disjuncts.size() - 1];
      disjuncts.pop_back();
      continue;
    }
    ++i;
    simplices.push_back(simp);
  }
}

/// Simplifies the representation of a PresburgerSet.
PresburgerRelation SetCoalescer::coalesce() {
  // For all tuples of IntegerRelations, check whether they can be
  // coalesced. When coalescing is successful, the contained IntegerRelation
  // is swapped with the last element of `disjuncts` and subsequently erased
  // and similarly for simplices.
  for (unsigned i = 0; i < disjuncts.size();) {

    // TODO: This does some comparisons two times (index 0 with 1 and index 1
    // with 0).
    bool broken = false;
    for (unsigned j = 0, e = disjuncts.size(); j < e; ++j) {
      negEqs.clear();
      redundantIneqsA.clear();
      redundantIneqsB.clear();
      cuttingIneqsA.clear();
      cuttingIneqsB.clear();
      if (i == j)
        continue;
      if (coalescePair(i, j).succeeded()) {
        broken = true;
        break;
      }
    }

    // Only if the inner loop was not broken, i is incremented. This is
    // required as otherwise, if a coalescing occurs, the IntegerRelation
    // now at position i is not compared.
    if (!broken)
      ++i;
  }

  PresburgerRelation newSet = PresburgerRelation::getEmpty(space);
  for (unsigned i = 0, e = disjuncts.size(); i < e; ++i)
    newSet.unionInPlace(disjuncts[i]);

  return newSet;
}

/// Given a Simplex `simp` and one of its inequalities `ineq`, check
/// that all inequalities of `cuttingIneqsB` are redundant for the facet of
/// `simp` where `ineq` holds as an equality is contained within `a`.
bool SetCoalescer::isFacetContained(ArrayRef<int64_t> ineq, Simplex &simp) {
  SimplexRollbackScopeExit scopeExit(simp);
  simp.addEquality(ineq);
  return llvm::all_of(cuttingIneqsB, [&simp](ArrayRef<int64_t> curr) {
    return simp.isRedundantInequality(curr);
  });
}

void SetCoalescer::addCoalescedDisjunct(unsigned i, unsigned j,
                                        const IntegerRelation &disjunct) {
  assert(i != j && "The indices must refer to different disjuncts");
  unsigned n = disjuncts.size();
  if (j == n - 1) {
    // This case needs special handling since position `n` - 1 is removed
    // from the vector, hence the `IntegerRelation` at position `n` - 2 is
    // lost otherwise.
    disjuncts[i] = disjuncts[n - 2];
    disjuncts.pop_back();
    disjuncts[n - 2] = disjunct;
    disjuncts[n - 2].removeRedundantConstraints();

    simplices[i] = simplices[n - 2];
    simplices.pop_back();
    simplices[n - 2] = Simplex(disjuncts[n - 2]);

  } else {
    // Other possible edge cases are correct since for `j` or `i` == `n` -
    // 2, the `IntegerRelation` at position `n` - 2 should be lost. The
    // case `i` == `n` - 1 makes the first following statement a noop.
    // Hence, in this case the same thing is done as above, but with `j`
    // rather than `i`.
    disjuncts[i] = disjuncts[n - 1];
    disjuncts[j] = disjuncts[n - 2];
    disjuncts.pop_back();
    disjuncts[n - 2] = disjunct;
    disjuncts[n - 2].removeRedundantConstraints();

    simplices[i] = simplices[n - 1];
    simplices[j] = simplices[n - 2];
    simplices.pop_back();
    simplices[n - 2] = Simplex(disjuncts[n - 2]);
  }
}

/// Given two polyhedra `a` and `b` at positions `i` and `j` in
/// `disjuncts` and `redundantIneqsA` being the inequalities of `a` that
/// are redundant for `b` (similarly for `cuttingIneqsA`, `redundantIneqsB`,
/// and `cuttingIneqsB`), Checks whether the facets of all cutting
/// inequalites of `a` are contained in `b`. If so, a new polyhedron
/// consisting of all redundant inequalites of `a` and `b` and all
/// equalities of both is created.
///
/// An example of this case:
///    ___________        ___________
///   /   /  |   /       /          /
///   \   \  |  /   ==>  \         /
///    \   \ | /          \       /
///     \___\|/            \_____/
///
///
LogicalResult SetCoalescer::coalescePairCutCase(unsigned i, unsigned j) {
  /// All inequalities of `b` need to be redundant. We already know that the
  /// redundant ones are, so only the cutting ones remain to be checked.
  Simplex &simp = simplices[i];
  IntegerRelation &disjunct = disjuncts[i];
  if (llvm::any_of(cuttingIneqsA, [this, &simp](ArrayRef<int64_t> curr) {
        return !isFacetContained(curr, simp);
      }))
    return failure();
  IntegerRelation newSet(disjunct.getSpace());

  for (ArrayRef<int64_t> curr : redundantIneqsA)
    newSet.addInequality(curr);

  for (ArrayRef<int64_t> curr : redundantIneqsB)
    newSet.addInequality(curr);

  addCoalescedDisjunct(i, j, newSet);
  return success();
}

LogicalResult SetCoalescer::typeInequality(ArrayRef<int64_t> ineq,
                                           Simplex &simp) {
  Simplex::IneqType type = simp.findIneqType(ineq);
  if (type == Simplex::IneqType::Redundant)
    redundantIneqsB.push_back(ineq);
  else if (type == Simplex::IneqType::Cut)
    cuttingIneqsB.push_back(ineq);
  else
    return failure();
  return success();
}

LogicalResult SetCoalescer::typeEquality(ArrayRef<int64_t> eq, Simplex &simp) {
  if (typeInequality(eq, simp).failed())
    return failure();
  negEqs.push_back(getNegatedCoeffs(eq));
  ArrayRef<int64_t> inv(negEqs.back());
  if (typeInequality(inv, simp).failed())
    return failure();
  return success();
}

void SetCoalescer::eraseDisjunct(unsigned i) {
  assert(simplices.size() == disjuncts.size() &&
         "simplices and disjuncts must be equally as long");
  disjuncts[i] = disjuncts.back();
  disjuncts.pop_back();
  simplices[i] = simplices.back();
  simplices.pop_back();
}

LogicalResult SetCoalescer::coalescePair(unsigned i, unsigned j) {

  IntegerRelation &a = disjuncts[i];
  IntegerRelation &b = disjuncts[j];
  /// Handling of local ids is not yet implemented, so these cases are
  /// skipped.
  /// TODO: implement local id support.
  if (a.getNumLocalIds() != 0 || b.getNumLocalIds() != 0)
    return failure();
  Simplex &simpA = simplices[i];
  Simplex &simpB = simplices[j];

  // Organize all inequalities and equalities of `a` according to their type
  // for `b` into `redundantIneqsA` and `cuttingIneqsA` (and vice versa for
  // all inequalities of `b` according to their type in `a`). If a separate
  // inequality is encountered during typing, the two IntegerRelations
  // cannot be coalesced.
  for (int k = 0, e = a.getNumInequalities(); k < e; ++k)
    if (typeInequality(a.getInequality(k), simpB).failed())
      return failure();

  for (int k = 0, e = a.getNumEqualities(); k < e; ++k)
    if (typeEquality(a.getEquality(k), simpB).failed())
      return failure();

  std::swap(redundantIneqsA, redundantIneqsB);
  std::swap(cuttingIneqsA, cuttingIneqsB);

  for (int k = 0, e = b.getNumInequalities(); k < e; ++k)
    if (typeInequality(b.getInequality(k), simpA).failed())
      return failure();

  for (int k = 0, e = b.getNumEqualities(); k < e; ++k)
    if (typeEquality(b.getEquality(k), simpA).failed())
      return failure();

  // If there are no cutting inequalities of `a`, `b` is contained
  // within `a`.
  if (cuttingIneqsA.empty()) {
    eraseDisjunct(j);
    return success();
  }

  // Try to apply the cut case
  if (coalescePairCutCase(i, j).succeeded())
    return success();

  // Swap the vectors to compare the pair (j,i) instead of (i,j).
  std::swap(redundantIneqsA, redundantIneqsB);
  std::swap(cuttingIneqsA, cuttingIneqsB);

  // If there are no cutting inequalities of `a`, `b` is contained
  // within `a`.
  if (cuttingIneqsA.empty()) {
    eraseDisjunct(i);
    return success();
  }

  // Try to apply the cut case
  if (coalescePairCutCase(j, i).succeeded())
    return success();

  return failure();
}

PresburgerRelation PresburgerRelation::coalesce() const {
  return SetCoalescer(*this).coalesce();
}

void PresburgerRelation::print(raw_ostream &os) const {
  os << "Number of Disjuncts: " << getNumDisjuncts() << "\n";
  for (const IntegerRelation &disjunct : disjuncts) {
    disjunct.print(os);
    os << '\n';
  }
}

void PresburgerRelation::dump() const { print(llvm::errs()); }

PresburgerSet PresburgerSet::getUniverse(const PresburgerSpace &space) {
  PresburgerSet result(space);
  result.unionInPlace(IntegerPolyhedron::getUniverse(space));
  return result;
}

PresburgerSet PresburgerSet::getEmpty(const PresburgerSpace &space) {
  return PresburgerSet(space);
}

PresburgerSet::PresburgerSet(const IntegerPolyhedron &disjunct)
    : PresburgerRelation(disjunct) {}

PresburgerSet::PresburgerSet(const PresburgerRelation &set)
    : PresburgerRelation(set) {}

PresburgerSet PresburgerSet::unionSet(const PresburgerRelation &set) const {
  return PresburgerSet(PresburgerRelation::unionSet(set));
}

PresburgerSet PresburgerSet::intersect(const PresburgerRelation &set) const {
  return PresburgerSet(PresburgerRelation::intersect(set));
}

PresburgerSet PresburgerSet::complement() const {
  return PresburgerSet(PresburgerRelation::complement());
}

PresburgerSet PresburgerSet::subtract(const PresburgerRelation &set) const {
  return PresburgerSet(PresburgerRelation::subtract(set));
}

PresburgerSet PresburgerSet::coalesce() const {
  return PresburgerSet(PresburgerRelation::coalesce());
}

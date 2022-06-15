//===- PresburgerRelation.h - MLIR PresburgerRelation Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent unions of IntegerRelations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"

namespace mlir {
namespace presburger {

/// The SetCoalescer class contains all functionality concerning the coalesce
/// heuristic. It is built from a `PresburgerRelation` and has the `coalesce()`
/// function as its main API.
class SetCoalescer;

/// A PresburgerRelation represents a union of IntegerRelations that live in
/// the same PresburgerSpace with support for union, intersection, subtraction,
/// and complement operations, as well as sampling.
///
/// The IntegerRelations (disjuncts) are stored in a vector, and the set
/// represents the union of these relations. An empty list corresponds to
/// the empty set.
///
/// Note that there are no invariants guaranteed on the list of disjuncts
/// other than that they are all in the same PresburgerSpace. For example, the
/// relations may overlap with each other.
class PresburgerRelation {
public:
  /// Return a universe set of the specified type that contains all points.
  static PresburgerRelation getUniverse(const PresburgerSpace &space);

  /// Return an empty set of the specified type that contains no points.
  static PresburgerRelation getEmpty(const PresburgerSpace &space);

  explicit PresburgerRelation(const IntegerRelation &disjunct);

  unsigned getNumDomainIds() const { return space.getNumDomainIds(); }
  unsigned getNumRangeIds() const { return space.getNumRangeIds(); }
  unsigned getNumSymbolIds() const { return space.getNumSymbolIds(); }
  unsigned getNumLocalIds() const { return space.getNumLocalIds(); }
  unsigned getNumIds() const { return space.getNumIds(); }

  /// Return the number of disjuncts in the union.
  unsigned getNumDisjuncts() const;

  const PresburgerSpace &getSpace() const { return space; }

  /// Return a reference to the list of disjuncts.
  ArrayRef<IntegerRelation> getAllDisjuncts() const;

  /// Return the disjunct at the specified index.
  const IntegerRelation &getDisjunct(unsigned index) const;

  /// Mutate this set, turning it into the union of this set and the given
  /// disjunct.
  void unionInPlace(const IntegerRelation &disjunct);

  /// Mutate this set, turning it into the union of this set and the given set.
  void unionInPlace(const PresburgerRelation &set);

  /// Return the union of this set and the given set.
  PresburgerRelation unionSet(const PresburgerRelation &set) const;

  /// Return the intersection of this set and the given set.
  PresburgerRelation intersect(const PresburgerRelation &set) const;

  /// Return true if the set contains the given point, and false otherwise.
  bool containsPoint(ArrayRef<int64_t> point) const;

  /// Return the complement of this set. All local variables in the set must
  /// correspond to floor divisions.
  PresburgerRelation complement() const;

  /// Return the set difference of this set and the given set, i.e.,
  /// return `this \ set`. All local variables in `set` must correspond
  /// to floor divisions, but local variables in `this` need not correspond to
  /// divisions.
  PresburgerRelation subtract(const PresburgerRelation &set) const;

  /// Return true if this set is a subset of the given set, and false otherwise.
  bool isSubsetOf(const PresburgerRelation &set) const;

  /// Return true if this set is equal to the given set, and false otherwise.
  /// All local variables in both sets must correspond to floor divisions.
  bool isEqual(const PresburgerRelation &set) const;

  /// Return true if all the sets in the union are known to be integer empty
  /// false otherwise.
  bool isIntegerEmpty() const;

  /// Find an integer sample from the given set. This should not be called if
  /// any of the disjuncts in the union are unbounded.
  bool findIntegerSample(SmallVectorImpl<int64_t> &sample);

  /// Compute an overapproximation of the number of integer points in the
  /// disjunct. Symbol ids are currently not supported. If the computed
  /// overapproximation is infinite, an empty optional is returned.
  ///
  /// This currently just sums up the overapproximations of the volumes of the
  /// disjuncts, so the approximation might be far from the true volume in the
  /// case when there is a lot of overlap between disjuncts.
  Optional<uint64_t> computeVolume() const;

  /// Simplifies the representation of a PresburgerRelation.
  ///
  /// In particular, removes all disjuncts which are subsets of other
  /// disjuncts in the union.
  PresburgerRelation coalesce() const;

  /// Print the set's internal state.
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  /// Construct an empty PresburgerRelation with the specified number of
  /// dimension and symbols.
  explicit PresburgerRelation(const PresburgerSpace &space) : space(space) {
    assert(space.getNumLocalIds() == 0 &&
           "PresburgerRelation cannot have local ids.");
  }

  PresburgerSpace space;

  /// The list of disjuncts that this set is the union of.
  SmallVector<IntegerRelation, 2> disjuncts;

  friend class SetCoalescer;
};

class PresburgerSet : public PresburgerRelation {
public:
  /// Return a universe set of the specified type that contains all points.
  static PresburgerSet getUniverse(const PresburgerSpace &space);

  /// Return an empty set of the specified type that contains no points.
  static PresburgerSet getEmpty(const PresburgerSpace &space);

  /// Create a set from a relation.
  explicit PresburgerSet(const IntegerPolyhedron &disjunct);
  explicit PresburgerSet(const PresburgerRelation &set);

  /// These operations are the same as the ones in PresburgeRelation, they just
  /// forward the arguement and return the result as a set instead of a
  /// relation.
  PresburgerSet unionSet(const PresburgerRelation &set) const;
  PresburgerSet intersect(const PresburgerRelation &set) const;
  PresburgerSet complement() const;
  PresburgerSet subtract(const PresburgerRelation &set) const;
  PresburgerSet coalesce() const;

protected:
  /// Construct an empty PresburgerRelation with the specified number of
  /// dimension and symbols.
  explicit PresburgerSet(const PresburgerSpace &space)
      : PresburgerRelation(space) {
    assert(space.getNumDomainIds() == 0 && "Set type cannot have domain ids.");
    assert(space.getNumLocalIds() == 0 &&
           "PresburgerRelation cannot have local ids.");
  }
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H

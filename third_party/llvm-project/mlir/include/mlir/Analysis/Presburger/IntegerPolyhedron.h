//===- IntegerPolyhedron.h - MLIR IntegerPolyhedron Class -----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent an integer polyhedron.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_INTEGERPOLYHEDRON_H
#define MLIR_ANALYSIS_PRESBURGER_INTEGERPOLYHEDRON_H

#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/// An integer polyhedron is the set of solutions to a list of affine
/// constraints over n integer-valued variables/identifiers. Affine constraints
/// can be inequalities or equalities in the form:
///
/// Inequality: c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} + c_n >= 0
/// Equality  : c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} + c_n == 0
///
/// where c_0, c_1, ..., c_n are integers.
///
/// Such a set corresponds to the set of integer points lying in a convex
/// polyhedron. For example, consider the set: (x, y) : (1 <= x <= 7, x = 2y).
/// This set contains the points (2, 1), (4, 2), and (6, 3).
///
/// The integer-valued variables are distinguished into 3 types of:
///
/// Dimension: Ordinary variables over which the set is represented.
///
/// Symbol: Symbol variables correspond to fixed but unknown values.
/// Mathematically, an integer polyhedron with symbolic variables is like a
/// family of integer polyhedra indexed by the symbolic variables.
///
/// Local: Local variables correspond to existentially quantified variables. For
/// example, consider the set: (x) : (exists q : 1 <= x <= 7, x = 2q). An
/// assignment to symbolic and dimension variables is valid if there exists some
/// assignment to the local variable `q` satisfying these constraints. For this
/// example, the set is equivalent to {2, 4, 6}. Mathematically, existential
/// quantification can be thought of as the result of projection. In this
/// example, `q` is existentially quantified. This can be thought of as the
/// result of projecting out `q` from the previous example, i.e. we obtained {2,
/// 4, 6} by projecting out the second dimension from {(2, 1), (4, 2), (6, 2)}.
///
class IntegerPolyhedron {
public:
  /// All derived classes of IntegerPolyhedron.
  enum class Kind {
    FlatAffineConstraints,
    FlatAffineValueConstraints,
    IntegerPolyhedron
  };

  /// Kind of identifier (column).
  enum IdKind { Dimension, Symbol, Local };

  /// Constructs a constraint system reserving memory for the specified number
  /// of constraints and identifiers.
  IntegerPolyhedron(unsigned numReservedInequalities,
                    unsigned numReservedEqualities, unsigned numReservedCols,
                    unsigned numDims, unsigned numSymbols, unsigned numLocals)
      : numIds(numDims + numSymbols + numLocals), numDims(numDims),
        numSymbols(numSymbols),
        equalities(0, numIds + 1, numReservedEqualities, numReservedCols),
        inequalities(0, numIds + 1, numReservedInequalities, numReservedCols) {
    assert(numReservedCols >= numIds + 1);
  }

  /// Constructs a constraint system with the specified number of
  /// dimensions and symbols.
  IntegerPolyhedron(unsigned numDims = 0, unsigned numSymbols = 0,
                    unsigned numLocals = 0)
      : IntegerPolyhedron(/*numReservedInequalities=*/0,
                          /*numReservedEqualities=*/0,
                          /*numReservedCols=*/numDims + numSymbols + numLocals +
                              1,
                          numDims, numSymbols, numLocals) {}

  /// Return a system with no constraints, i.e., one which is satisfied by all
  /// points.
  static IntegerPolyhedron getUniverse(unsigned numDims = 0,
                                       unsigned numSymbols = 0) {
    return IntegerPolyhedron(numDims, numSymbols);
  }

  virtual ~IntegerPolyhedron() = default;

  /// Return the kind of this IntegerPolyhedron.
  virtual Kind getKind() const { return Kind::IntegerPolyhedron; }

  static bool classof(const IntegerPolyhedron *cst) { return true; }

  // Clones this object.
  std::unique_ptr<IntegerPolyhedron> clone() const;

  /// Clears any existing data and reserves memory for the specified
  /// constraints.
  virtual void reset(unsigned numReservedInequalities,
                     unsigned numReservedEqualities, unsigned numReservedCols,
                     unsigned numDims, unsigned numSymbols,
                     unsigned numLocals = 0);

  void reset(unsigned numDims = 0, unsigned numSymbols = 0,
             unsigned numLocals = 0);

  /// Appends constraints from `other` into `this`. This is equivalent to an
  /// intersection with no simplification of any sort attempted.
  void append(const IntegerPolyhedron &other);

  /// Returns the value at the specified equality row and column.
  inline int64_t atEq(unsigned i, unsigned j) const { return equalities(i, j); }
  inline int64_t &atEq(unsigned i, unsigned j) { return equalities(i, j); }

  /// Returns the value at the specified inequality row and column.
  inline int64_t atIneq(unsigned i, unsigned j) const {
    return inequalities(i, j);
  }
  inline int64_t &atIneq(unsigned i, unsigned j) { return inequalities(i, j); }

  unsigned getNumConstraints() const {
    return getNumInequalities() + getNumEqualities();
  }
  inline unsigned getNumIds() const { return numIds; }
  inline unsigned getNumDimIds() const { return numDims; }
  inline unsigned getNumSymbolIds() const { return numSymbols; }
  inline unsigned getNumDimAndSymbolIds() const { return numDims + numSymbols; }
  inline unsigned getNumLocalIds() const {
    return numIds - numDims - numSymbols;
  }

  /// Returns the number of columns in the constraint system.
  inline unsigned getNumCols() const { return numIds + 1; }

  inline unsigned getNumEqualities() const { return equalities.getNumRows(); }

  inline unsigned getNumInequalities() const {
    return inequalities.getNumRows();
  }

  inline unsigned getNumReservedEqualities() const {
    return equalities.getNumReservedRows();
  }

  inline unsigned getNumReservedInequalities() const {
    return inequalities.getNumReservedRows();
  }

  inline ArrayRef<int64_t> getEquality(unsigned idx) const {
    return equalities.getRow(idx);
  }

  inline ArrayRef<int64_t> getInequality(unsigned idx) const {
    return inequalities.getRow(idx);
  }

  /// Insert `num` identifiers of the specified kind at position `pos`.
  /// Positions are relative to the kind of identifier. The coefficient columns
  /// corresponding to the added identifiers are initialized to zero. Return the
  /// absolute column position (i.e., not relative to the kind of identifier)
  /// of the first added identifier.
  unsigned insertDimId(unsigned pos, unsigned num = 1);
  unsigned insertSymbolId(unsigned pos, unsigned num = 1);
  unsigned insertLocalId(unsigned pos, unsigned num = 1);
  virtual unsigned insertId(IdKind kind, unsigned pos, unsigned num = 1);

  /// Append `num` identifiers of the specified kind after the last identifier.
  /// of that kind. Return the position of the first appended column. The
  /// coefficient columns corresponding to the added identifiers are initialized
  /// to zero.
  unsigned appendDimId(unsigned num = 1);
  unsigned appendSymbolId(unsigned num = 1);
  unsigned appendLocalId(unsigned num = 1);

  /// Adds an inequality (>= 0) from the coefficients specified in `inEq`.
  void addInequality(ArrayRef<int64_t> inEq);
  /// Adds an equality from the coefficients specified in `eq`.
  void addEquality(ArrayRef<int64_t> eq);

  /// Removes identifiers of the specified kind with the specified pos (or
  /// within the specified range) from the system. The specified location is
  /// relative to the first identifier of the specified kind.
  void removeId(IdKind kind, unsigned pos);
  void removeIdRange(IdKind kind, unsigned idStart, unsigned idLimit);

  /// Removes the specified identifier from the system.
  void removeId(unsigned pos);

  void removeEquality(unsigned pos);
  void removeInequality(unsigned pos);

  /// Remove the (in)equalities at positions [start, end).
  void removeEqualityRange(unsigned start, unsigned end);
  void removeInequalityRange(unsigned start, unsigned end);

  /// Swap the posA^th identifier with the posB^th identifier.
  virtual void swapId(unsigned posA, unsigned posB);

  /// Removes all equalities and inequalities.
  void clearConstraints();

  /// Sets the `values.size()` identifiers starting at `po`s to the specified
  /// values and removes them.
  void setAndEliminate(unsigned pos, ArrayRef<int64_t> values);

  /// Replaces the contents of this IntegerPolyhedron with `other`.
  virtual void clearAndCopyFrom(const IntegerPolyhedron &other);

  /// Gather positions of all lower and upper bounds of the identifier at `pos`,
  /// and optionally any equalities on it. In addition, the bounds are to be
  /// independent of identifiers in position range [`offset`, `offset` + `num`).
  void
  getLowerAndUpperBoundIndices(unsigned pos,
                               SmallVectorImpl<unsigned> *lbIndices,
                               SmallVectorImpl<unsigned> *ubIndices,
                               SmallVectorImpl<unsigned> *eqIndices = nullptr,
                               unsigned offset = 0, unsigned num = 0) const;

  /// Checks for emptiness by performing variable elimination on all
  /// identifiers, running the GCD test on each equality constraint, and
  /// checking for invalid constraints. Returns true if the GCD test fails for
  /// any equality, or if any invalid constraints are discovered on any row.
  /// Returns false otherwise.
  bool isEmpty() const;

  /// Runs the GCD test on all equality constraints. Returns true if this test
  /// fails on any equality. Returns false otherwise.
  /// This test can be used to disprove the existence of a solution. If it
  /// returns true, no integer solution to the equality constraints can exist.
  bool isEmptyByGCDTest() const;

  /// Returns true if the set of constraints is found to have no solution,
  /// false if a solution exists. Uses the same algorithm as
  /// `findIntegerSample`.
  bool isIntegerEmpty() const;

  /// Returns a matrix where each row is a vector along which the polytope is
  /// bounded. The span of the returned vectors is guaranteed to contain all
  /// such vectors. The returned vectors are NOT guaranteed to be linearly
  /// independent. This function should not be called on empty sets.
  Matrix getBoundedDirections() const;

  /// Find an integer sample point satisfying the constraints using a
  /// branch and bound algorithm with generalized basis reduction, with some
  /// additional processing using Simplex for unbounded sets.
  ///
  /// Returns an integer sample point if one exists, or an empty Optional
  /// otherwise.
  Optional<SmallVector<int64_t, 8>> findIntegerSample() const;

  /// Returns true if the given point satisfies the constraints, or false
  /// otherwise.
  bool containsPoint(ArrayRef<int64_t> point) const;

  /// Find pairs of inequalities identified by their position indices, using
  /// which an explicit representation for each local variable can be computed.
  /// The pairs are stored as indices of upperbound, lowerbound inequalities. If
  /// no such pair can be found, it is stored as llvm::None.
  ///
  /// The dividends of the explicit representations are stored in `dividends`
  /// and the denominators in `denominators`. If no explicit representation
  /// could be found for the `i^th` local identifier, `denominators[i]` is set
  /// to 0.
  void getLocalReprs(
      std::vector<SmallVector<int64_t, 8>> &dividends,
      SmallVector<unsigned, 4> &denominators,
      std::vector<llvm::Optional<std::pair<unsigned, unsigned>>> &repr) const;
  void getLocalReprs(
      std::vector<llvm::Optional<std::pair<unsigned, unsigned>>> &repr) const;
  void getLocalReprs(std::vector<SmallVector<int64_t, 8>> &dividends,
                     SmallVector<unsigned, 4> &denominators) const;

  /// The type of bound: equal, lower bound or upper bound.
  enum BoundType { EQ, LB, UB };

  /// Adds a constant bound for the specified identifier.
  void addBound(BoundType type, unsigned pos, int64_t value);

  /// Adds a constant bound for the specified expression.
  void addBound(BoundType type, ArrayRef<int64_t> expr, int64_t value);

  /// Adds a new local identifier as the floordiv of an affine function of other
  /// identifiers, the coefficients of which are provided in `dividend` and with
  /// respect to a positive constant `divisor`. Two constraints are added to the
  /// system to capture equivalence with the floordiv:
  /// q = dividend floordiv c    <=>   c*q <= dividend <= c*q + c - 1.
  void addLocalFloorDiv(ArrayRef<int64_t> dividend, int64_t divisor);

  /// Projects out (aka eliminates) `num` identifiers starting at position
  /// `pos`. The resulting constraint system is the shadow along the dimensions
  /// that still exist. This method may not always be integer exact.
  // TODO: deal with integer exactness when necessary - can return a value to
  // mark exactness for example.
  void projectOut(unsigned pos, unsigned num);
  inline void projectOut(unsigned pos) { return projectOut(pos, 1); }

  /// Changes the partition between dimensions and symbols. Depending on the new
  /// symbol count, either a chunk of trailing dimensional identifiers becomes
  /// symbols, or some of the leading symbols become dimensions.
  void setDimSymbolSeparation(unsigned newSymbolCount);

  /// Tries to fold the specified identifier to a constant using a trivial
  /// equality detection; if successful, the constant is substituted for the
  /// identifier everywhere in the constraint system and then removed from the
  /// system.
  LogicalResult constantFoldId(unsigned pos);

  /// This method calls `constantFoldId` for the specified range of identifiers,
  /// `num` identifiers starting at position `pos`.
  void constantFoldIdRange(unsigned pos, unsigned num);

  /// Updates the constraints to be the smallest bounding (enclosing) box that
  /// contains the points of `this` set and that of `other`, with the symbols
  /// being treated specially. For each of the dimensions, the min of the lower
  /// bounds (symbolic) and the max of the upper bounds (symbolic) is computed
  /// to determine such a bounding box. `other` is expected to have the same
  /// dimensional identifiers as this constraint system (in the same order).
  ///
  /// E.g.:
  /// 1) this   = {0 <= d0 <= 127},
  ///    other  = {16 <= d0 <= 192},
  ///    output = {0 <= d0 <= 192}
  /// 2) this   = {s0 + 5 <= d0 <= s0 + 20},
  ///    other  = {s0 + 1 <= d0 <= s0 + 9},
  ///    output = {s0 + 1 <= d0 <= s0 + 20}
  /// 3) this   = {0 <= d0 <= 5, 1 <= d1 <= 9}
  ///    other  = {2 <= d0 <= 6, 5 <= d1 <= 15},
  ///    output = {0 <= d0 <= 6, 1 <= d1 <= 15}
  LogicalResult unionBoundingBox(const IntegerPolyhedron &other);

  /// Returns the smallest known constant bound for the extent of the specified
  /// identifier (pos^th), i.e., the smallest known constant that is greater
  /// than or equal to 'exclusive upper bound' - 'lower bound' of the
  /// identifier. This constant bound is guaranteed to be non-negative. Returns
  /// None if it's not a constant. This method employs trivial (low complexity /
  /// cost) checks and detection. Symbolic identifiers are treated specially,
  /// i.e., it looks for constant differences between affine expressions
  /// involving only the symbolic identifiers. `lb` and `ub` (along with the
  /// `boundFloorDivisor`) are set to represent the lower and upper bound
  /// associated with the constant difference: `lb`, `ub` have the coefficients,
  /// and `boundFloorDivisor`, their divisor. `minLbPos` and `minUbPos` if
  /// non-null are set to the position of the constant lower bound and upper
  /// bound respectively (to the same if they are from an equality). Ex: if the
  /// lower bound is [(s0 + s2 - 1) floordiv 32] for a system with three
  /// symbolic identifiers, *lb = [1, 0, 1], lbDivisor = 32. See comments at
  /// function definition for examples.
  Optional<int64_t> getConstantBoundOnDimSize(
      unsigned pos, SmallVectorImpl<int64_t> *lb = nullptr,
      int64_t *boundFloorDivisor = nullptr,
      SmallVectorImpl<int64_t> *ub = nullptr, unsigned *minLbPos = nullptr,
      unsigned *minUbPos = nullptr) const;

  /// Returns the constant bound for the pos^th identifier if there is one;
  /// None otherwise.
  // TODO: Support EQ bounds.
  Optional<int64_t> getConstantBound(BoundType type, unsigned pos) const;

  /// Removes constraints that are independent of (i.e., do not have a
  /// coefficient) identifiers in the range [pos, pos + num).
  void removeIndependentConstraints(unsigned pos, unsigned num);

  /// Returns true if the set can be trivially detected as being
  /// hyper-rectangular on the specified contiguous set of identifiers.
  bool isHyperRectangular(unsigned pos, unsigned num) const;

  /// Removes duplicate constraints, trivially true constraints, and constraints
  /// that can be detected as redundant as a result of differing only in their
  /// constant term part. A constraint of the form <non-negative constant> >= 0
  /// is considered trivially true. This method is a linear time method on the
  /// constraints, does a single scan, and updates in place. It also normalizes
  /// constraints by their GCD and performs GCD tightening on inequalities.
  void removeTrivialRedundancy();

  /// A more expensive check than `removeTrivialRedundancy` to detect redundant
  /// inequalities.
  void removeRedundantInequalities();

  /// Removes redundant constraints using Simplex. Although the algorithm can
  /// theoretically take exponential time in the worst case (rare), it is known
  /// to perform much better in the average case. If V is the number of vertices
  /// in the polytope and C is the number of constraints, the algorithm takes
  /// O(VC) time.
  void removeRedundantConstraints();

  /// Converts identifiers in the column range [idStart, idLimit) to local
  /// variables.
  void convertDimToLocal(unsigned dimStart, unsigned dimLimit);

  /// Adds additional local ids to the sets such that they both have the union
  /// of the local ids in each set, without changing the set of points that
  /// lie in `this` and `other`. The ordering of the local ids in the
  /// sets may also be changed. After merging, if the `i^th` local variable in
  /// one set has a known division representation, then the `i^th` local
  /// variable in the other set either has the same division representation or
  /// no known division representation.
  ///
  /// The number of dimensions and symbol ids in `this` and `other` should
  /// match.
  void mergeLocalIds(IntegerPolyhedron &other);

  void print(raw_ostream &os) const;
  void dump() const;

protected:
  /// Checks all rows of equality/inequality constraints for trivial
  /// contradictions (for example: 1 == 0, 0 >= 1), which may have surfaced
  /// after elimination. Returns true if an invalid constraint is found;
  /// false otherwise.
  bool hasInvalidConstraint() const;

  /// Returns the constant lower bound bound if isLower is true, and the upper
  /// bound if isLower is false.
  template <bool isLower>
  Optional<int64_t> computeConstantLowerOrUpperBound(unsigned pos);

  /// Eliminates a single identifier at `position` from equality and inequality
  /// constraints. Returns `success` if the identifier was eliminated, and
  /// `failure` otherwise.
  inline LogicalResult gaussianEliminateId(unsigned position) {
    return success(gaussianEliminateIds(position, position + 1) == 1);
  }

  /// Removes local variables using equalities. Each equality is checked if it
  /// can be reduced to the form: `e = affine-expr`, where `e` is a local
  /// variable and `affine-expr` is an affine expression not containing `e`.
  /// If an equality satisfies this form, the local variable is replaced in
  /// each constraint and then removed. The equality used to replace this local
  /// variable is also removed.
  void removeRedundantLocalVars();

  /// Eliminates identifiers from equality and inequality constraints
  /// in column range [posStart, posLimit).
  /// Returns the number of variables eliminated.
  unsigned gaussianEliminateIds(unsigned posStart, unsigned posLimit);

  /// Eliminates the identifier at the specified position using Fourier-Motzkin
  /// variable elimination, but uses Gaussian elimination if there is an
  /// equality involving that identifier. If the result of the elimination is
  /// integer exact, `*isResultIntegerExact` is set to true. If `darkShadow` is
  /// set to true, a potential under approximation (subset) of the rational
  /// shadow / exact integer shadow is computed.
  // See implementation comments for more details.
  virtual void fourierMotzkinEliminate(unsigned pos, bool darkShadow = false,
                                       bool *isResultIntegerExact = nullptr);

  /// Tightens inequalities given that we are dealing with integer spaces. This
  /// is similar to the GCD test but applied to inequalities. The constant term
  /// can be reduced to the preceding multiple of the GCD of the coefficients,
  /// i.e.,
  ///  64*i - 100 >= 0  =>  64*i - 128 >= 0 (since 'i' is an integer). This is a
  /// fast method (linear in the number of coefficients).
  void gcdTightenInequalities();

  /// Normalized each constraints by the GCD of its coefficients.
  void normalizeConstraintsByGCD();

  /// Searches for a constraint with a non-zero coefficient at `colIdx` in
  /// equality (isEq=true) or inequality (isEq=false) constraints.
  /// Returns true and sets row found in search in `rowIdx`, false otherwise.
  bool findConstraintWithNonZeroAt(unsigned colIdx, bool isEq,
                                   unsigned *rowIdx) const;

  /// Returns true if the pos^th column is all zero for both inequalities and
  /// equalities.
  bool isColZero(unsigned pos) const;

  /// Returns false if the fields corresponding to various identifier counts, or
  /// equality/inequality buffer sizes aren't consistent; true otherwise. This
  /// is meant to be used within an assert internally.
  virtual bool hasConsistentState() const;

  /// Prints the number of constraints, dimensions, symbols and locals in the
  /// IntegerPolyhedron.
  virtual void printSpace(raw_ostream &os) const;

  /// Return the index at which the specified kind of id starts.
  unsigned getIdKindOffset(IdKind kind) const;

  /// Get the number of ids of the specified kind.
  unsigned getNumIdKind(IdKind kind) const;

  /// Removes identifiers in the column range [idStart, idLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  virtual void removeIdRange(unsigned idStart, unsigned idLimit);

  /// A parameter that controls detection of an unrealistic number of
  /// constraints. If the number of constraints is this many times the number of
  /// variables, we consider such a system out of line with the intended use
  /// case of IntegerPolyhedron.
  // The rationale for 32 is that in the typical simplest of cases, an
  // identifier is expected to have one lower bound and one upper bound
  // constraint. With a level of tiling or a connection to another identifier
  // through a div or mod, an extra pair of bounds gets added. As a limit, we
  // don't expect an identifier to have more than 32 lower/upper/equality
  // constraints. This is conservatively set low and can be raised if needed.
  constexpr static unsigned kExplosionFactor = 32;

  /// Total number of identifiers.
  unsigned numIds;

  /// Number of identifiers corresponding to real dimensions.
  unsigned numDims;

  /// Number of identifiers corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;

  /// Coefficients of affine equalities (in == 0 form).
  Matrix equalities;

  /// Coefficients of affine inequalities (in >= 0 form).
  Matrix inequalities;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_INTEGERPOLYHEDRON_H

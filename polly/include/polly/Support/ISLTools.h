//===------ ISLTools.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tools, utilities, helpers and extensions useful in conjunction with the
// Integer Set Library (isl).
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_ISLTOOLS_H
#define POLLY_ISLTOOLS_H

#include "polly/Support/GICHelper.h"

namespace polly {

/// Return the range elements that are lexicographically smaller.
///
/// @param Map    { Space[] -> Scatter[] }
/// @param Strict True for strictly lexicographically smaller elements (exclude
///               same timepoints from the result).
///
/// @return { Space[] -> Scatter[] }
///         A map to all timepoints that happen before the timepoints the input
///         mapped to.
IslPtr<isl_map> beforeScatter(IslPtr<isl_map> Map, bool Strict);

/// Piecewise beforeScatter(IslPtr<isl_map>,bool).
IslPtr<isl_union_map> beforeScatter(IslPtr<isl_union_map> UMap, bool Strict);

/// Return the range elements that are lexicographically larger.
///
/// @param Map    { Space[] -> Scatter[] }
/// @param Strict True for strictly lexicographically larger elements (exclude
///               same timepoints from the result).
///
/// @return { Space[] -> Scatter[] }
///         A map to all timepoints that happen after the timepoints the input
///         map originally mapped to.
IslPtr<isl_map> afterScatter(IslPtr<isl_map> Map, bool Strict);

/// Piecewise afterScatter(IslPtr<isl_map>,bool).
IslPtr<isl_union_map> afterScatter(NonowningIslPtr<isl_union_map> UMap,
                                   bool Strict);

/// Construct a range of timepoints between two timepoints.
///
/// Example:
/// From := { A[] -> [0]; B[] -> [0] }
/// To   := {             B[] -> [10]; C[] -> [20] }
///
/// Result:
/// { B[] -> [i] : 0 < i < 10 }
///
/// Note that A[] and C[] are not in the result because they do not have a start
/// or end timepoint. If a start (or end) timepoint is not unique, the first
/// (respectively last) is chosen.
///
/// @param From     { Space[] -> Scatter[] }
///                 Map to start timepoints.
/// @param To       { Space[] -> Scatter[] }
///                 Map to end timepoints.
/// @param InclFrom Whether to include the start timepoints in the result. In
///                 the example, this would add { B[] -> [0] }
/// @param InclTo   Whether to include the end timepoints in the result. In this
///                 example, this would add { B[] -> [10] }
///
/// @return { Space[] -> Scatter[] }
///         A map for each domain element of timepoints between two extreme
///         points, or nullptr if @p From or @p To is nullptr, or the isl max
///         operations is exceeded.
IslPtr<isl_map> betweenScatter(IslPtr<isl_map> From, IslPtr<isl_map> To,
                               bool InclFrom, bool InclTo);

/// Piecewise betweenScatter(IslPtr<isl_map>,IslPtr<isl_map>,bool,bool).
IslPtr<isl_union_map> betweenScatter(IslPtr<isl_union_map> From,
                                     IslPtr<isl_union_map> To, bool InclFrom,
                                     bool InclTo);

/// If by construction a union map is known to contain only a single map, return
/// it.
///
/// This function combines isl_map_from_union_map() and
/// isl_union_map_extract_map(). isl_map_from_union_map() fails if the map is
/// empty because it does not know which space it would be in.
/// isl_union_map_extract_map() on the other hand does not check whether there
/// is (at most) one isl_map in the union, i.e. how it has been constructed is
/// probably wrong.
IslPtr<isl_map> singleton(IslPtr<isl_union_map> UMap,
                          IslPtr<isl_space> ExpectedSpace);

/// If by construction an isl_union_set is known to contain only a single
/// isl_set, return it.
///
/// This function combines isl_set_from_union_set() and
/// isl_union_set_extract_set(). isl_map_from_union_set() fails if the set is
/// empty because it does not know which space it would be in.
/// isl_union_set_extract_set() on the other hand does not check whether there
/// is (at most) one isl_set in the union, i.e. how it has been constructed is
/// probably wrong.
IslPtr<isl_set> singleton(IslPtr<isl_union_set> USet,
                          IslPtr<isl_space> ExpectedSpace);

/// Determine how many dimensions the scatter space of @p Schedule has.
///
/// The schedule must not be empty and have equal number of dimensions of any
/// subspace it contains.
///
/// The implementation currently returns the maximum number of dimensions it
/// encounters, if different, and 0 if none is encountered. However, most other
/// code will most likely fail if one of these happen.
unsigned getNumScatterDims(NonowningIslPtr<isl_union_map> Schedule);

/// Return the scatter space of a @p Schedule.
///
/// This is basically the range space of the schedule map, but harder to
/// determine because it is an isl_union_map.
IslPtr<isl_space> getScatterSpace(NonowningIslPtr<isl_union_map> Schedule);

/// Construct an identity map for the given domain values.
///
/// There is no type resembling isl_union_space, hence we have to pass an
/// isl_union_set as the map's domain and range space.
///
/// @param USet           { Space[] }
///                       The returned map's domain and range.
/// @param RestrictDomain If true, the returned map only maps elements contained
///                       in @p USet and no other. If false, it returns an
///                       overapproximation with the identity maps of any space
///                       in @p USet, not just the elements in it.
///
/// @return { Space[] -> Space[] }
///         A map that maps each value of @p USet to itself.
IslPtr<isl_union_map> makeIdentityMap(NonowningIslPtr<isl_union_set> USet,
                                      bool RestrictDomain);

/// Reverse the nested map tuple in @p Map's domain.
///
/// @param Map { [Space1[] -> Space2[]] -> Space3[] }
///
/// @return { [Space2[] -> Space1[]] -> Space3[] }
IslPtr<isl_map> reverseDomain(IslPtr<isl_map> Map);

/// Piecewise reverseDomain(IslPtr<isl_map>).
IslPtr<isl_union_map> reverseDomain(NonowningIslPtr<isl_union_map> UMap);

/// Add a constant to one dimension of a set.
///
/// @param Map    The set to shift a dimension in.
/// @param Pos    The dimension to shift. If negative, the dimensions are
///               counted from the end instead from the beginning. E.g. -1 is
///               the last dimension in the tuple.
/// @param Amount The offset to add to the specified dimension.
///
/// @return The modified set.
IslPtr<isl_set> shiftDim(IslPtr<isl_set> Set, int Pos, int Amount);

/// Piecewise shiftDim(IslPtr<isl_set>,int,int).
IslPtr<isl_union_set> shiftDim(IslPtr<isl_union_set> USet, int Pos, int Amount);

/// Simplify a set inplace.
void simplify(IslPtr<isl_set> &Set);

/// Simplify a union set inplace.
void simplify(IslPtr<isl_union_set> &USet);

/// Simplify a map inplace.
void simplify(IslPtr<isl_map> &Map);

/// Simplify a union map inplace.
void simplify(IslPtr<isl_union_map> &UMap);

/// Compute the reaching definition statement or the next overwrite for each
/// definition of an array element.
///
/// The reaching definition of an array element at a specific timepoint is the
/// statement instance that has written the current element's content.
/// Alternatively, this function determines for each timepoint and element which
/// write is going to overwrite an element at a future timepoint. This can be
/// seen as "reaching definition in reverse" where definitions are found in the
/// past.
///
/// For example:
///
/// Schedule := { Write[] -> [0]; Overwrite[] -> [10] }
/// Defs := { Write[] -> A[5]; Overwrite[] -> A[5] }
///
/// If index 5 of array A is written at timepoint 0 and 10, the resulting
/// reaching definitions are:
///
/// { [A[5] -> [i]] -> Write[] : 0 < i < 10;
///   [A[5] -> [i]] -> Overwrite[] : 10 < i }
///
/// Between timepoint 0 (Write[]) and timepoint 10 (Overwrite[]), the
/// content of A[5] is written by statement instance Write[] and after
/// timepoint 10 by Overwrite[]. Values not defined in the map have no known
/// definition. This includes the statement instance timepoints themselves,
/// because reads at those timepoints could either read the old or the new
/// value, defined only by the statement itself. But this can be changed by @p
/// InclPrevDef and @p InclNextDef. InclPrevDef=false and InclNextDef=true
/// returns a zone. Unless @p InclPrevDef and @p InclNextDef are both true,
/// there is only one unique definition per element and timepoint.
///
/// @param Schedule    { DomainWrite[] -> Scatter[] }
///                    Schedule of (at least) all array writes. Instances not in
///                    @p Writes are ignored.
/// @param Writes      { DomainWrite[] -> Element[] }
///                    Elements written to by the statement instances.
/// @param Reverse     If true, look for definitions in the future. That is,
///                    find the write that is overwrites the current value.
/// @param InclPrevDef Include the definition's timepoint to the set of
///                    well-defined elements (any load at that timepoint happen
///                    at the writes). In the example, enabling this option adds
///                    {[A[5] -> [0]] -> Write[]; [A[5] -> [10]] -> Overwrite[]}
///                    to the result.
/// @param InclNextDef Whether to assume that at the timepoint where an element
///                    is overwritten, it still contains the old value (any load
///                    at that timepoint would happen before the overwrite). In
///                    this example, enabling this adds
///                    { [A[] -> [10]] -> Write[] } to the result.
///
/// @return { [Element[] -> Scatter[]] -> DomainWrite[] }
///         The reaching definitions or future overwrite as described above, or
///         nullptr if either @p Schedule or @p Writes is nullptr, or the isl
///         max operations count has exceeded.
IslPtr<isl_union_map> computeReachingWrite(IslPtr<isl_union_map> Schedule,
                                           IslPtr<isl_union_map> Writes,
                                           bool Reverse, bool InclPrevDef,
                                           bool InclNextDef);

} // namespace polly

#endif /* POLLY_ISLTOOLS_H */

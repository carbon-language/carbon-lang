// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_TYPE_STRUCTURE_H_
#define CARBON_EXPLORER_INTERPRETER_TYPE_STRUCTURE_H_

#include <vector>

#include "common/ostream.h"
#include "explorer/base/nonnull.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Value;

// A type structure sort key represents the information needed to order `impl`
// declarations by their type structures.
//
// The type structure for an `impl` declaration is the `type as interface`
// portion, with all references to the enclosing generic parameters replaced by
// `?`s. Two type structures that match the same type and interface are ordered
// based on the lexical position of the first `?` that is in one but not the
// other. The type structure with the earlier `?` has the worse match.
//
// This class extracts the relevant information needed to order type
// structures, and provides a weak ordering over type structures in which type
// structures that provide a better match compare earlier.
class TypeStructureSortKey : public Printable<TypeStructureSortKey> {
 public:
  // Compute the sort key for `impl type as interface`.
  static auto ForImpl(Nonnull<const Value*> type,
                      Nonnull<const Value*> interface) -> TypeStructureSortKey;

  // Order by sort key. Smaller keys represent better matches.
  friend bool operator<(const TypeStructureSortKey& a,
                        const TypeStructureSortKey& b) {
    return a.holes_ > b.holes_;
  }
  friend bool operator<=(const TypeStructureSortKey& a,
                         const TypeStructureSortKey& b) {
    return a.holes_ >= b.holes_;
  }
  friend bool operator>(const TypeStructureSortKey& a,
                        const TypeStructureSortKey& b) {
    return a.holes_ < b.holes_;
  }
  friend bool operator>=(const TypeStructureSortKey& a,
                         const TypeStructureSortKey& b) {
    return a.holes_ <= b.holes_;
  }

  // Determine whether two sort keys are in the same equivalence class. If so,
  // the sort keys represent type structures with `?`s in the same positions.
  // This does not imply that the remainder of the type structure is the same.
  // For example, the sort key for `Optional(?) as Hash` is the same as the
  // sort key for `Vector(?) as Ordered`.
  friend bool operator==(const TypeStructureSortKey& a,
                         const TypeStructureSortKey& b) {
    return a.holes_ == b.holes_;
  }
  friend bool operator!=(const TypeStructureSortKey& a,
                         const TypeStructureSortKey& b) {
    return a.holes_ != b.holes_;
  }

  void Print(llvm::raw_ostream& out) const;

  LLVM_DUMP_METHOD void Dump() const;

 private:
  // Positions of holes (`?`s) in the structure. Each hole is described as a
  // path of indexes from the root of the type tree to the position of the `?`.
  // Holes are listed in appearance order, separated by -1s, and the vector is
  // terminated by std::numeric_limits<int>::max().
  //
  // This representation is chosen so that better matches are lexicographically
  // larger than worse matches.
  std::vector<int> holes_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_TYPE_STRUCTURE_H_

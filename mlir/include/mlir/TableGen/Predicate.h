//===- Predicate.h - Predicate class ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper around predicates defined in TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PREDICATE_H_
#define MLIR_TABLEGEN_PREDICATE_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"

#include <string>
#include <vector>

namespace llvm {
class Init;
class ListInit;
class Record;
class SMLoc;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// A logical predicate.  This class must closely follow the definition of
// TableGen class 'Pred'.
class Pred {
public:
  // Constructs the null Predicate (e.g., always true).
  explicit Pred() : def(nullptr) {}
  // Construct a Predicate from a record.
  explicit Pred(const llvm::Record *record);
  // Construct a Predicate from an initializer.
  explicit Pred(const llvm::Init *init);

  // Check if the predicate is defined.  Callers may use this to interpret the
  // missing predicate as either true (e.g. in filters) or false (e.g. in
  // precondition verification).
  bool isNull() const { return def == nullptr; }

  // Get the predicate condition.  This may dispatch to getConditionImpl() of
  // the underlying predicate type.
  std::string getCondition() const;

  // Whether the predicate is a combination of other predicates, i.e. an
  // record of type CombinedPred.
  bool isCombined() const;

  // Records are pointer-comparable.
  bool operator==(const Pred &other) const { return def == other.def; }

  // Get the location of the predicate.
  ArrayRef<llvm::SMLoc> getLoc() const;

protected:
  friend llvm::DenseMapInfo<Pred>;

  // The TableGen definition of this predicate.
  const llvm::Record *def;
};

// A logical predicate wrapping a C expression.  This class must closely follow
// the definition of TableGen class 'CPred'.
class CPred : public Pred {
public:
  // Construct a CPred from a record.
  explicit CPred(const llvm::Record *record);
  // Construct a CPred an initializer.
  explicit CPred(const llvm::Init *init);

  // Get the predicate condition.
  std::string getConditionImpl() const;
};

// A logical predicate that is a combination of other predicates.  This class
// must closely follow the definition of TableGen class 'CombinedPred'.
class CombinedPred : public Pred {
public:
  // Construct a CombinedPred from a record.
  explicit CombinedPred(const llvm::Record *record);
  // Construct a CombinedPred from an initializer.
  explicit CombinedPred(const llvm::Init *init);

  // Get the predicate condition.
  std::string getConditionImpl() const;

  // Get the definition of the combiner used in this predicate.
  const llvm::Record *getCombinerDef() const;

  // Get the predicates that are combined by this predicate.
  const std::vector<llvm::Record *> getChildren() const;
};

// A combined predicate that requires all child predicates of 'CPred' type to
// have their expression rewritten with a simple string substitution rule.
class SubstLeavesPred : public CombinedPred {
public:
  // Get the replacement pattern.
  StringRef getPattern() const;
  // Get the string used to replace the pattern.
  StringRef getReplacement() const;
};

// A combined predicate that prepends a prefix and appends a suffix to the
// predicate string composed from a child predicate.
class ConcatPred : public CombinedPred {
public:
  StringRef getPrefix() const;
  StringRef getSuffix() const;
};

} // end namespace tblgen
} // end namespace mlir

namespace llvm {
template <>
struct DenseMapInfo<mlir::tblgen::Pred> {
  static mlir::tblgen::Pred getEmptyKey() { return mlir::tblgen::Pred(); }
  static mlir::tblgen::Pred getTombstoneKey() { return mlir::tblgen::Pred(); }
  static unsigned getHashValue(mlir::tblgen::Pred pred) {
    return llvm::hash_value(pred.def);
  }
  static bool isEqual(mlir::tblgen::Pred lhs, mlir::tblgen::Pred rhs) {
    return lhs == rhs;
  }
};
} // end namespace llvm

#endif // MLIR_TABLEGEN_PREDICATE_H_

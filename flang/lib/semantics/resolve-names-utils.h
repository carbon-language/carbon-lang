// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_SEMANTICS_RESOLVE_NAMES_UTILS_H_
#define FORTRAN_SEMANTICS_RESOLVE_NAMES_UTILS_H_

// Utility functions and class for use in resolve-names.cc.

#include "scope.h"
#include "symbol.h"
#include "type.h"
#include "../parser/message.h"

namespace Fortran::parser {
class CharBlock;
struct ArraySpec;
struct CoarraySpec;
struct ComponentArraySpec;
struct DataRef;
struct DefinedOpName;
struct Designator;
struct Expr;
struct GenericSpec;
struct Name;
}

namespace Fortran::semantics {

using SourceName = parser::CharBlock;
class SemanticsContext;

// Record that a Name has been resolved to a Symbol
Symbol &Resolve(const parser::Name &, Symbol &);
Symbol *Resolve(const parser::Name &, Symbol *);

// Create a copy of msg with a new isFatal value.
parser::MessageFixedText WithIsFatal(
    const parser::MessageFixedText &msg, bool isFatal);

// Is this the name of a defined operator, e.g. ".foo."
bool IsDefinedOperator(const SourceName &);
bool IsInstrinsicOperator(const SemanticsContext &, const SourceName &);
bool IsLogicalConstant(const SemanticsContext &, const SourceName &);

// Analyze a generic-spec and generate a symbol name and GenericKind for it.
class GenericSpecInfo {
public:
  GenericSpecInfo(const parser::DefinedOpName &x) { Analyze(x); }
  GenericSpecInfo(const parser::GenericSpec &x) { Analyze(x); }

  const SourceName &symbolName() const { return *symbolName_; }
  // Set the GenericKind in this symbol and resolve the corresponding
  // name if there is one
  void Resolve(Symbol *);

private:
  GenericKind kind_;
  const parser::Name *parseName_{nullptr};
  const SourceName *symbolName_{nullptr};

  void Analyze(const parser::DefinedOpName &);
  void Analyze(const parser::GenericSpec &);
};

// Analyze a parser::ArraySpec or parser::CoarraySpec
ArraySpec AnalyzeArraySpec(SemanticsContext &, const parser::ArraySpec &);
ArraySpec AnalyzeArraySpec(
    SemanticsContext &, const parser::ComponentArraySpec &);
ArraySpec AnalyzeCoarraySpec(
    SemanticsContext &context, const parser::CoarraySpec &);

// Perform consistency checks on equivalence sets
class EquivalenceSets {
public:
  EquivalenceSets(SemanticsContext &context) : context_{context} {}
  std::vector<EquivalenceSet> &sets() { return sets_; };
  // Resolve this designator and add to the current equivalence set
  void AddToSet(const parser::Designator &);
  // Finish the current equivalence set: determine if it overlaps
  // with any of the others and perform necessary merges if it does.
  void FinishSet(const parser::CharBlock &);

private:
  bool CheckCanEquivalence(
      const parser::CharBlock &, const Symbol &, const Symbol &);
  void MergeInto(const parser::CharBlock &, EquivalenceSet &, std::size_t);
  const EquivalenceObject *Find(const EquivalenceSet &, const Symbol &);
  bool CheckDesignator(const parser::Designator &);
  bool CheckDataRef(const parser::CharBlock &, const parser::DataRef &);
  bool CheckObject(const parser::Name &);
  bool CheckBound(const parser::Expr &, bool isSubstring = false);
  bool IsCharacterSequenceType(const DeclTypeSpec *);
  bool IsDefaultKindNumericType(const IntrinsicTypeSpec &);
  bool IsNumericSequenceType(const DeclTypeSpec *);
  bool IsSequenceType(
      const DeclTypeSpec *, std::function<bool(const IntrinsicTypeSpec &)>);

  SemanticsContext &context_;
  std::vector<EquivalenceSet> sets_;  // all equivalence sets in this scope
  // Map object to index of set it is in
  std::map<EquivalenceObject, std::size_t> objectToSet_;
  EquivalenceSet currSet_;  // equivalence set currently being constructed
  struct {
    Symbol *symbol{nullptr};
    std::vector<ConstantSubscript> subscripts;
  } currObject_;  // equivalence object currently being constructed
};

}
#endif  // FORTRAN_SEMANTICS_RESOLVE_NAMES_H_

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

#include "symbol.h"
#include "type.h"
#include "../parser/message.h"

namespace Fortran::parser {
class CharBlock;
struct ArraySpec;
struct CoarraySpec;
struct DefinedOpName;
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

// Analyze a parser::ArraySpec or parser::CoarraySpec into the provide ArraySpec
void AnalyzeArraySpec(
    ArraySpec &, SemanticsContext &, const parser::ArraySpec &);
void AnalyzeCoarraySpec(
    ArraySpec &, SemanticsContext &, const parser::CoarraySpec &);

}

#endif  // FORTRAN_SEMANTICS_RESOLVE_NAMES_H_

// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "semantics.h"
#include "assignment.h"
#include "canonicalize-do.h"
#include "check-do-concurrent.h"
#include "expression.h"
#include "mod-file.h"
#include "resolve-labels.h"
#include "resolve-names.h"
#include "rewrite-parse-tree.h"
#include "scope.h"
#include "symbol.h"
#include "../common/default-kinds.h"
#include "../parser/parse-tree-visitor.h"
#include <ostream>

namespace Fortran::semantics {

static void DoDumpSymbols(std::ostream &, const Scope &, int indent = 0);
static void PutIndent(std::ostream &, int indent);

// A parse tree visitor that calls Enter/Leave functions from each checker
// class C supplied as template parameters. Enter is called before the node's
// children are visited, Leave is called after. No two checkers may have the
// same Enter or Leave function. Each checker must be constructible from
// SemanticsContext and have BaseChecker as a virtual base class.
template<typename... C> struct SemanticsVisitor : public virtual C... {
  using C::Enter...;
  using C::Leave...;
  using BaseChecker::Enter;
  using BaseChecker::Leave;
  SemanticsVisitor(SemanticsContext &context) : C{context}... {}
  template<typename N> bool Pre(const N &node) {
    Enter(node);
    return true;
  }
  template<typename N> void Post(const N &node) { Leave(node); }
};

using StatementSemantics =
    SemanticsVisitor<ExprChecker, AssignmentChecker, DoConcurrentChecker>;

SemanticsContext::SemanticsContext(
    const common::IntrinsicTypeDefaultKinds &defaultKinds)
  : defaultKinds_{defaultKinds},
    intrinsics_{evaluate::IntrinsicProcTable::Configure(defaultKinds)},
    foldingContext_{evaluate::FoldingContext{
        parser::ContextualMessages{parser::CharBlock{}, &messages_}}} {}

const DeclTypeSpec &SemanticsContext::MakeNumericType(
    TypeCategory category, int kind) {
  if (kind == 0) {
    kind = defaultKinds_.GetDefaultKind(category);
  }
  return globalScope_.MakeNumericType(category, KindExpr{kind});
}
const DeclTypeSpec &SemanticsContext::MakeLogicalType(int kind) {
  if (kind == 0) {
    kind = defaultKinds_.GetDefaultKind(TypeCategory::Logical);
  }
  return globalScope_.MakeLogicalType(KindExpr{kind});
}

bool SemanticsContext::AnyFatalError() const {
  return !messages_.empty() &&
      (warningsAreErrors_ || messages_.AnyFatalError());
}

const Scope &SemanticsContext::FindScope(
    const parser::CharBlock &source) const {
  if (const auto *scope{globalScope_.FindScope(source)}) {
    return *scope;
  } else {
    common::die("invalid source location");
  }
}

bool Semantics::Perform() {
  ValidateLabels(context_.messages(), program_);
  if (AnyFatalError()) {
    return false;
  }
  parser::CanonicalizeDo(program_);
  ResolveNames(context_, program_);
  if (AnyFatalError()) {
    return false;
  }
  RewriteParseTree(context_, program_);
  if (AnyFatalError()) {
    return false;
  }
  StatementSemantics visitor{context_};
  parser::Walk(program_, visitor);
  if (AnyFatalError()) {
    return false;
  }
  ModFileWriter writer{context_};
  writer.WriteAll();
  return !AnyFatalError();
}

void Semantics::EmitMessages(std::ostream &os) const {
  context_.messages().Emit(os, cooked_);
}

void Semantics::DumpSymbols(std::ostream &os) {
  DoDumpSymbols(os, context_.globalScope());
}

void DoDumpSymbols(std::ostream &os, const Scope &scope, int indent) {
  PutIndent(os, indent);
  os << Scope::EnumToString(scope.kind()) << " scope:";
  if (const auto *symbol{scope.symbol()}) {
    os << ' ' << symbol->name().ToString();
  }
  os << '\n';
  ++indent;
  for (const auto &pair : scope) {
    const auto &symbol{*pair.second};
    PutIndent(os, indent);
    os << symbol << '\n';
    if (const auto *details{symbol.detailsIf<GenericDetails>()}) {
      if (const auto &type{details->derivedType()}) {
        PutIndent(os, indent);
        os << *type << '\n';
      }
    }
  }
  for (const auto &pair : scope.commonBlocks()) {
    const auto &symbol{*pair.second};
    PutIndent(os, indent);
    os << symbol << '\n';
  }
  for (const auto &child : scope.children()) {
    DoDumpSymbols(os, child, indent);
  }
  --indent;
}

static void PutIndent(std::ostream &os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << "  ";
  }
}

}

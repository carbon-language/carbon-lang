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

#include "resolve-names-utils.h"
#include "expression.h"
#include "semantics.h"
#include "../common/idioms.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../evaluate/type.h"
#include "../parser/char-block.h"
#include "../parser/features.h"
#include "../parser/parse-tree.h"
#include <ostream>
#include <variant>

namespace Fortran::semantics {

using IntrinsicOperator = parser::DefinedOperator::IntrinsicOperator;

static GenericKind MapIntrinsicOperator(IntrinsicOperator);

Symbol *Resolve(const parser::Name &name, Symbol *symbol) {
  if (symbol && !name.symbol) {
    name.symbol = symbol;
  }
  return symbol;
}
Symbol &Resolve(const parser::Name &name, Symbol &symbol) {
  return *Resolve(name, &symbol);
}

parser::MessageFixedText WithIsFatal(
    const parser::MessageFixedText &msg, bool isFatal) {
  return parser::MessageFixedText{
      msg.text().begin(), msg.text().size(), isFatal};
}

bool IsDefinedOperator(const SourceName &name) {
  const char *begin{name.begin()};
  const char *end{name.end()};
  return begin != end && begin[0] == '.' && end[-1] == '.';
}

bool IsInstrinsicOperator(
    const SemanticsContext &context, const SourceName &name) {
  std::string str{name.ToString()};
  std::set<std::string> intrinsics{".and.", ".eq.", ".eqv.", ".ge.", ".gt.",
      ".le.", ".lt.", ".ne.", ".neqv.", ".not.", ".or."};
  if (intrinsics.count(str) > 0) {
    return true;
  }
  if (context.IsEnabled(parser::LanguageFeature::XOROperator) &&
      str == ".xor.") {
    return true;
  }
  if (context.IsEnabled(parser::LanguageFeature::LogicalAbbreviations) &&
      (str == ".n." || str == ".a" || str == ".o." || str == ".x.")) {
    return true;
  }
  return false;
}

bool IsLogicalConstant(
    const SemanticsContext &context, const SourceName &name) {
  std::string str{name.ToString()};
  return str == ".true." || str == ".false." ||
      (context.IsEnabled(parser::LanguageFeature::LogicalAbbreviations) &&
          (str == ".t" || str == ".f."));
}

void GenericSpecInfo::Resolve(Symbol *symbol) {
  if (symbol) {
    if (auto *details{symbol->detailsIf<GenericDetails>()}) {
      details->set_kind(kind_);
    } else if (auto *details{symbol->detailsIf<GenericBindingDetails>()}) {
      details->set_kind(kind_);
    }
    if (parseName_) {
      semantics::Resolve(*parseName_, symbol);
    }
  }
}

void GenericSpecInfo::Analyze(const parser::DefinedOpName &name) {
  kind_ = GenericKind::DefinedOp;
  parseName_ = &name.v;
  symbolName_ = &name.v.source;
}

void GenericSpecInfo::Analyze(const parser::GenericSpec &x) {
  symbolName_ = &x.source;
  kind_ = std::visit(
      common::visitors{
          [&](const parser::Name &y) {
            parseName_ = &y;
            symbolName_ = &y.source;
            return GenericKind::Name;
          },
          [&](const parser::DefinedOperator &y) {
            return std::visit(
                common::visitors{
                    [&](const parser::DefinedOpName &z) {
                      Analyze(z);
                      return GenericKind::DefinedOp;
                    },
                    [&](const IntrinsicOperator &z) {
                      return MapIntrinsicOperator(z);
                    },
                },
                y.u);
          },
          [&](const parser::GenericSpec::Assignment &y) {
            return GenericKind::Assignment;
          },
          [&](const parser::GenericSpec::ReadFormatted &y) {
            return GenericKind::ReadFormatted;
          },
          [&](const parser::GenericSpec::ReadUnformatted &y) {
            return GenericKind::ReadUnformatted;
          },
          [&](const parser::GenericSpec::WriteFormatted &y) {
            return GenericKind::WriteFormatted;
          },
          [&](const parser::GenericSpec::WriteUnformatted &y) {
            return GenericKind::WriteUnformatted;
          },
      },
      x.u);
}

// parser::DefinedOperator::IntrinsicOperator -> GenericKind
static GenericKind MapIntrinsicOperator(IntrinsicOperator op) {
  switch (op) {
  case IntrinsicOperator::Power: return GenericKind::OpPower;
  case IntrinsicOperator::Multiply: return GenericKind::OpMultiply;
  case IntrinsicOperator::Divide: return GenericKind::OpDivide;
  case IntrinsicOperator::Add: return GenericKind::OpAdd;
  case IntrinsicOperator::Subtract: return GenericKind::OpSubtract;
  case IntrinsicOperator::Concat: return GenericKind::OpConcat;
  case IntrinsicOperator::LT: return GenericKind::OpLT;
  case IntrinsicOperator::LE: return GenericKind::OpLE;
  case IntrinsicOperator::EQ: return GenericKind::OpEQ;
  case IntrinsicOperator::NE: return GenericKind::OpNE;
  case IntrinsicOperator::GE: return GenericKind::OpGE;
  case IntrinsicOperator::GT: return GenericKind::OpGT;
  case IntrinsicOperator::NOT: return GenericKind::OpNOT;
  case IntrinsicOperator::AND: return GenericKind::OpAND;
  case IntrinsicOperator::OR: return GenericKind::OpOR;
  case IntrinsicOperator::XOR: return GenericKind::OpXOR;
  case IntrinsicOperator::EQV: return GenericKind::OpEQV;
  case IntrinsicOperator::NEQV: return GenericKind::OpNEQV;
  default: CRASH_NO_CASE;
  }
}

class ArraySpecAnalyzer {
public:
  ArraySpecAnalyzer(SemanticsContext &context) : context_{context} {}
  ArraySpec Analyze(const parser::ArraySpec &);
  ArraySpec Analyze(const parser::ComponentArraySpec &);
  ArraySpec Analyze(const parser::CoarraySpec &);

private:
  SemanticsContext &context_;
  ArraySpec arraySpec_;

  template<typename T> void Analyze(const std::list<T> &list) {
    for (const auto &elem : list) {
      Analyze(elem);
    }
  }
  void Analyze(const parser::AssumedShapeSpec &);
  void Analyze(const parser::ExplicitShapeSpec &);
  void Analyze(const parser::AssumedImpliedSpec &);
  void Analyze(const parser::DeferredShapeSpecList &);
  void Analyze(const parser::AssumedRankSpec &);
  void MakeExplicit(const std::optional<parser::SpecificationExpr> &,
      const parser::SpecificationExpr &);
  void MakeImplied(const std::optional<parser::SpecificationExpr> &);
  void MakeDeferred(int);
  Bound GetBound(const std::optional<parser::SpecificationExpr> &);
  Bound GetBound(const parser::SpecificationExpr &);
};

ArraySpec AnalyzeArraySpec(
    SemanticsContext &context, const parser::ArraySpec &arraySpec) {
  return ArraySpecAnalyzer{context}.Analyze(arraySpec);
}
ArraySpec AnalyzeArraySpec(
    SemanticsContext &context, const parser::ComponentArraySpec &arraySpec) {
  return ArraySpecAnalyzer{context}.Analyze(arraySpec);
}
ArraySpec AnalyzeCoarraySpec(
    SemanticsContext &context, const parser::CoarraySpec &coarraySpec) {
  return ArraySpecAnalyzer{context}.Analyze(coarraySpec);
}

ArraySpec ArraySpecAnalyzer::Analyze(const parser::ComponentArraySpec &x) {
  std::visit([this](const auto &y) { Analyze(y); }, x.u);
  return arraySpec_;
}
ArraySpec ArraySpecAnalyzer::Analyze(const parser::ArraySpec &x) {
  std::visit(
      common::visitors{
          [&](const parser::AssumedSizeSpec &y) {
            Analyze(std::get<std::list<parser::ExplicitShapeSpec>>(y.t));
            Analyze(std::get<parser::AssumedImpliedSpec>(y.t));
          },
          [&](const parser::ImpliedShapeSpec &y) { Analyze(y.v); },
          [&](const auto &y) { Analyze(y); },
      },
      x.u);
  return arraySpec_;
}
ArraySpec ArraySpecAnalyzer::Analyze(const parser::CoarraySpec &x) {
  std::visit(
      common::visitors{
          [&](const parser::DeferredCoshapeSpecList &y) { MakeDeferred(y.v); },
          [&](const parser::ExplicitCoshapeSpec &y) {
            Analyze(std::get<std::list<parser::ExplicitShapeSpec>>(y.t));
            MakeImplied(
                std::get<std::optional<parser::SpecificationExpr>>(y.t));
          },
      },
      x.u);
  return arraySpec_;
}

void ArraySpecAnalyzer::Analyze(const parser::AssumedShapeSpec &x) {
  arraySpec_.push_back(ShapeSpec::MakeAssumed(GetBound(x.v)));
}
void ArraySpecAnalyzer::Analyze(const parser::ExplicitShapeSpec &x) {
  MakeExplicit(std::get<std::optional<parser::SpecificationExpr>>(x.t),
      std::get<parser::SpecificationExpr>(x.t));
}
void ArraySpecAnalyzer::Analyze(const parser::AssumedImpliedSpec &x) {
  MakeImplied(x.v);
}
void ArraySpecAnalyzer::Analyze(const parser::DeferredShapeSpecList &x) {
  MakeDeferred(x.v);
}
void ArraySpecAnalyzer::Analyze(const parser::AssumedRankSpec &) {
  arraySpec_.push_back(ShapeSpec::MakeAssumedRank());
}

void ArraySpecAnalyzer::MakeExplicit(
    const std::optional<parser::SpecificationExpr> &lb,
    const parser::SpecificationExpr &ub) {
  arraySpec_.push_back(ShapeSpec::MakeExplicit(GetBound(lb), GetBound(ub)));
}
void ArraySpecAnalyzer::MakeImplied(
    const std::optional<parser::SpecificationExpr> &lb) {
  arraySpec_.push_back(ShapeSpec::MakeImplied(GetBound(lb)));
}
void ArraySpecAnalyzer::MakeDeferred(int n) {
  for (int i = 0; i < n; ++i) {
    arraySpec_.push_back(ShapeSpec::MakeDeferred());
  }
}

Bound ArraySpecAnalyzer::GetBound(
    const std::optional<parser::SpecificationExpr> &x) {
  return x ? GetBound(*x) : Bound{1};
}
Bound ArraySpecAnalyzer::GetBound(const parser::SpecificationExpr &x) {
  MaybeSubscriptIntExpr expr;
  if (MaybeExpr maybeExpr{AnalyzeExpr(context_, x.v)}) {
    if (auto *intExpr{evaluate::UnwrapExpr<SomeIntExpr>(*maybeExpr)}) {
      expr = evaluate::Fold(context_.foldingContext(),
          evaluate::ConvertToType<evaluate::SubscriptInteger>(
              std::move(*intExpr)));
    }
  }
  return Bound{std::move(expr)};
}

template<typename T>
static ProgramTree BuildSubprogramTree(const parser::Name &name, const T &x) {
  const auto &spec{std::get<parser::SpecificationPart>(x.t)};
  const auto &exec{std::get<parser::ExecutionPart>(x.t)};
  const auto &subps{
      std::get<std::optional<parser::InternalSubprogramPart>>(x.t)};
  ProgramTree node{name, spec, &exec};
  if (subps) {
    for (const auto &subp :
        std::get<std::list<parser::InternalSubprogram>>(subps->t)) {
      std::visit(
          [&](const auto &y) { node.AddChild(ProgramTree::Build(y.value())); },
          subp.u);
    }
  }
  return node;
}

template<typename T>
static ProgramTree BuildModuleTree(const parser::Name &name, const T &x) {
  const auto &spec{std::get<parser::SpecificationPart>(x.t)};
  const auto &subps{std::get<std::optional<parser::ModuleSubprogramPart>>(x.t)};
  ProgramTree node{name, spec};
  if (subps) {
    for (const auto &subp :
        std::get<std::list<parser::ModuleSubprogram>>(subps->t)) {
      std::visit(
          [&](const auto &y) { node.AddChild(ProgramTree::Build(y.value())); },
          subp.u);
    }
  }
  return node;
}

ProgramTree ProgramTree::Build(const parser::ProgramUnit &x) {
  return std::visit([](const auto &y) { return Build(y.value()); }, x.u);
}
ProgramTree ProgramTree::Build(const parser::MainProgram &x) {
  const auto &stmt{
      std::get<std::optional<parser::Statement<parser::ProgramStmt>>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndProgramStmt>>(x.t)};
  static parser::Name emptyName;
  const auto &name{stmt ? stmt->statement.v : emptyName};
  return BuildSubprogramTree(name, x).set_stmt(*stmt).set_endStmt(end);
}
ProgramTree ProgramTree::Build(const parser::FunctionSubprogram &x) {
  const auto &stmt{std::get<parser::Statement<parser::FunctionStmt>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndFunctionStmt>>(x.t)};
  const auto &name{std::get<parser::Name>(stmt.statement.t)};
  return BuildSubprogramTree(name, x).set_stmt(stmt).set_endStmt(end);
}
ProgramTree ProgramTree::Build(const parser::SubroutineSubprogram &x) {
  const auto &stmt{std::get<parser::Statement<parser::SubroutineStmt>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndSubroutineStmt>>(x.t)};
  const auto &name{std::get<parser::Name>(stmt.statement.t)};
  return BuildSubprogramTree(name, x).set_stmt(stmt).set_endStmt(end);
}
ProgramTree ProgramTree::Build(const parser::SeparateModuleSubprogram &x) {
  const auto &stmt{std::get<parser::Statement<parser::MpSubprogramStmt>>(x.t)};
  const auto &end{
      std::get<parser::Statement<parser::EndMpSubprogramStmt>>(x.t)};
  const auto &name{stmt.statement.v};
  return BuildSubprogramTree(name, x).set_stmt(stmt).set_endStmt(end);
}
ProgramTree ProgramTree::Build(const parser::Module &x) {
  const auto &stmt{std::get<parser::Statement<parser::ModuleStmt>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndModuleStmt>>(x.t)};
  const auto &name{stmt.statement.v};
  return BuildModuleTree(name, x).set_stmt(stmt).set_endStmt(end);
}
ProgramTree ProgramTree::Build(const parser::Submodule &x) {
  const auto &stmt{std::get<parser::Statement<parser::SubmoduleStmt>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndSubmoduleStmt>>(x.t)};
  const auto &name{std::get<parser::Name>(stmt.statement.t)};
  return BuildModuleTree(name, x).set_stmt(stmt).set_endStmt(end);
}
ProgramTree ProgramTree::Build(const parser::BlockData &x) {
  DIE("BlockData not yet implemented");
}

const parser::ParentIdentifier &ProgramTree::GetParentId() const {
  const auto *stmt{
      std::get<const parser::Statement<parser::SubmoduleStmt> *>(stmt_)};
  return std::get<parser::ParentIdentifier>(stmt->statement.t);
}

bool ProgramTree::IsModule() const {
  auto kind{GetKind()};
  return kind == Kind::Module || kind == Kind::Submodule;
}

Symbol::Flag ProgramTree::GetSubpFlag() const {
  return GetKind() == Kind::Function ? Symbol::Flag::Function
                                     : Symbol::Flag::Subroutine;
}

bool ProgramTree::HasModulePrefix() const {
  using ListType = std::list<parser::PrefixSpec>;
  const auto *prefixes{std::visit(
      common::visitors{
          [](const parser::Statement<parser::FunctionStmt> *x) {
            return &std::get<ListType>(x->statement.t);
          },
          [](const parser::Statement<parser::SubroutineStmt> *x) {
            return &std::get<ListType>(x->statement.t);
          },
          [](const auto *) -> const ListType * { return nullptr; },
      },
      stmt_)};
  if (prefixes) {
    for (const auto &prefix : *prefixes) {
      if (std::holds_alternative<parser::PrefixSpec::Module>(prefix.u)) {
        return true;
      }
    }
  }
  return false;
}

ProgramTree::Kind ProgramTree::GetKind() const {
  return std::visit(
      common::visitors{
          [](const parser::Statement<parser::ProgramStmt> *) {
            return Kind::Program;
          },
          [](const parser::Statement<parser::FunctionStmt> *) {
            return Kind::Function;
          },
          [](const parser::Statement<parser::SubroutineStmt> *) {
            return Kind::Subroutine;
          },
          [](const parser::Statement<parser::MpSubprogramStmt> *) {
            return Kind::MpSubprogram;
          },
          [](const parser::Statement<parser::ModuleStmt> *) {
            return Kind::Module;
          },
          [](const parser::Statement<parser::SubmoduleStmt> *) {
            return Kind::Submodule;
          },
      },
      stmt_);
}

void ProgramTree::set_scope(Scope &scope) {
  scope_ = &scope;
  CHECK(endStmt_);
  scope.AddSourceRange(*endStmt_);
}

void ProgramTree::AddChild(ProgramTree &&child) {
  children_.emplace_back(std::move(child));
}

}

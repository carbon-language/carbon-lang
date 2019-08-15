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

#include "program-tree.h"
#include "scope.h"
#include "../common/idioms.h"
#include "../parser/char-block.h"

namespace Fortran::semantics {

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

ProgramTree ProgramTree::Build(const parser::BlockData &) {
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

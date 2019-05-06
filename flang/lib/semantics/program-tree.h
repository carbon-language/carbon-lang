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

#ifndef FORTRAN_SEMANTICS_PROGRAM_TREE_H_
#define FORTRAN_SEMANTICS_PROGRAM_TREE_H_

#include "symbol.h"
#include "../parser/parse-tree.h"
#include <variant>

// A ProgramTree represents a tree of program units and their contained
// subprograms. The root nodes represent: main program, function, subroutine,
// module subprogram, module, or submodule.
// Each node of the tree consists of:
//   - the statement that introduces the program unit
//   - the specification part
//   - the execution part if applicable (not for module or submodule)
//   - a child node for each contained subprogram

namespace Fortran::semantics {

class Scope;

class ProgramTree {
public:
  // Build the ProgramTree rooted at one of these program units.
  static ProgramTree Build(const parser::ProgramUnit &);
  static ProgramTree Build(const parser::MainProgram &);
  static ProgramTree Build(const parser::FunctionSubprogram &);
  static ProgramTree Build(const parser::SubroutineSubprogram &);
  static ProgramTree Build(const parser::SeparateModuleSubprogram &);
  static ProgramTree Build(const parser::Module &);
  static ProgramTree Build(const parser::Submodule &);
  static ProgramTree Build(const parser::BlockData &);

  ENUM_CLASS(Kind,  // kind of node
      Program, Function, Subroutine, MpSubprogram, Module, Submodule)
  using Stmt = std::variant<  // the statement that introduces the program unit
      const parser::Statement<parser::ProgramStmt> *,
      const parser::Statement<parser::FunctionStmt> *,
      const parser::Statement<parser::SubroutineStmt> *,
      const parser::Statement<parser::MpSubprogramStmt> *,
      const parser::Statement<parser::ModuleStmt> *,
      const parser::Statement<parser::SubmoduleStmt> *>;

  ProgramTree(const parser::Name &name, const parser::SpecificationPart &spec,
      const parser::ExecutionPart *exec = nullptr)
    : name_{name}, spec_{spec}, exec_{exec} {}

  const parser::Name &name() const { return name_; }
  Kind GetKind() const;
  const Stmt &stmt() const { return stmt_; }
  const parser::ParentIdentifier &GetParentId() const;  // only for Submodule
  const parser::SpecificationPart &spec() const { return spec_; }
  const parser::ExecutionPart *exec() const { return exec_; }
  std::vector<ProgramTree> &children() { return children_; }
  const std::vector<ProgramTree> &children() const { return children_; }
  Symbol::Flag GetSubpFlag() const;
  bool IsModule() const;  // Module or Submodule
  bool HasModulePrefix() const;  // in function or subroutine stmt
  Scope *scope() const { return scope_; }
  void set_scope(Scope &);
  void AddChild(ProgramTree &&);

  template<typename T> ProgramTree &set_stmt(const parser::Statement<T> &stmt) {
    stmt_ = &stmt;
    return *this;
  }
  template<typename T>
  ProgramTree &set_endStmt(const parser::Statement<T> &stmt) {
    endStmt_ = &stmt.source;
    return *this;
  }

private:
  const parser::Name &name_;
  Stmt stmt_{
      static_cast<const parser::Statement<parser::ProgramStmt> *>(nullptr)};
  const parser::SpecificationPart &spec_;
  const parser::ExecutionPart *exec_{nullptr};
  std::vector<ProgramTree> children_;
  Scope *scope_{nullptr};
  const parser::CharBlock *endStmt_{nullptr};
};

}
#endif  // FORTRAN_SEMANTICS_PROGRAM_TREE_H_

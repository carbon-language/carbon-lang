//===-- lib/Semantics/program-tree.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_PROGRAM_TREE_H_
#define FORTRAN_SEMANTICS_PROGRAM_TREE_H_

#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/symbol.h"
#include <list>
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
  using EntryStmtList = std::list<common::Reference<const parser::EntryStmt>>;

  // Build the ProgramTree rooted at one of these program units.
  static ProgramTree Build(const parser::ProgramUnit &);
  static ProgramTree Build(const parser::MainProgram &);
  static ProgramTree Build(const parser::FunctionSubprogram &);
  static ProgramTree Build(const parser::SubroutineSubprogram &);
  static ProgramTree Build(const parser::SeparateModuleSubprogram &);
  static ProgramTree Build(const parser::Module &);
  static ProgramTree Build(const parser::Submodule &);
  static ProgramTree Build(const parser::BlockData &);
  static ProgramTree Build(const parser::CompilerDirective &);

  ENUM_CLASS(Kind, // kind of node
      Program, Function, Subroutine, MpSubprogram, Module, Submodule, BlockData)
  using Stmt = std::variant< // the statement that introduces the program unit
      const parser::Statement<parser::ProgramStmt> *,
      const parser::Statement<parser::FunctionStmt> *,
      const parser::Statement<parser::SubroutineStmt> *,
      const parser::Statement<parser::MpSubprogramStmt> *,
      const parser::Statement<parser::ModuleStmt> *,
      const parser::Statement<parser::SubmoduleStmt> *,
      const parser::Statement<parser::BlockDataStmt> *>;

  ProgramTree(const parser::Name &name, const parser::SpecificationPart &spec,
      const parser::ExecutionPart *exec = nullptr)
      : name_{name}, spec_{spec}, exec_{exec} {}

  const parser::Name &name() const { return name_; }
  Kind GetKind() const;
  const Stmt &stmt() const { return stmt_; }
  bool isSpecificationPartResolved() const {
    return isSpecificationPartResolved_;
  }
  void set_isSpecificationPartResolved(bool yes = true) {
    isSpecificationPartResolved_ = yes;
  }
  const parser::ParentIdentifier &GetParentId() const; // only for Submodule
  const parser::SpecificationPart &spec() const { return spec_; }
  const parser::ExecutionPart *exec() const { return exec_; }
  std::list<ProgramTree> &children() { return children_; }
  const std::list<ProgramTree> &children() const { return children_; }
  const std::list<common::Reference<const parser::EntryStmt>> &
  entryStmts() const {
    return entryStmts_;
  }
  Symbol::Flag GetSubpFlag() const;
  bool IsModule() const; // Module or Submodule
  bool HasModulePrefix() const; // in function or subroutine stmt
  Scope *scope() const { return scope_; }
  void set_scope(Scope &);
  void AddChild(ProgramTree &&);
  void AddEntry(const parser::EntryStmt &);

  template <typename T>
  ProgramTree &set_stmt(const parser::Statement<T> &stmt) {
    stmt_ = &stmt;
    return *this;
  }
  template <typename T>
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
  std::list<ProgramTree> children_;
  EntryStmtList entryStmts_;
  Scope *scope_{nullptr};
  const parser::CharBlock *endStmt_{nullptr};
  bool isSpecificationPartResolved_{false};
};

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_PROGRAM_TREE_H_

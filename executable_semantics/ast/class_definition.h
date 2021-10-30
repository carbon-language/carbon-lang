// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_CLASS_DEFINITION_H_
#define EXECUTABLE_SEMANTICS_AST_CLASS_DEFINITION_H_

#include <string>
#include <vector>

#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/source_location.h"

namespace Carbon {

class ScopedNames;

class ClassDefinition {
 public:
  ClassDefinition(SourceLocation source_loc, std::string name,
                  std::vector<Nonnull<Member*>> members)
      : source_loc_(source_loc),
        name_(std::move(name)),
        members_(std::move(members)) {}

  auto source_loc() const -> SourceLocation { return source_loc_; }
  auto name() const -> const std::string& { return name_; }
  auto members() const -> llvm::ArrayRef<Nonnull<Member*>> { return members_; }

  // Contains class members.
  // scoped_names_ should only be accessed after set_scoped_names is called.
  auto scoped_names() const -> const ScopedNames& { return **scoped_names_; }
  auto scoped_names() -> ScopedNames& { return **scoped_names_; }

  // scoped_names_ should only be set once during name resolution.
  auto set_scoped_names(Nonnull<ScopedNames*> scoped_names) {
    CHECK(!scoped_names_.has_value());
    scoped_names_ = scoped_names;
  }

 private:
  SourceLocation source_loc_;
  std::string name_;
  std::vector<Nonnull<Member*>> members_;
  std::optional<Nonnull<ScopedNames*>> scoped_names_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_CLASS_DEFINITION_H_

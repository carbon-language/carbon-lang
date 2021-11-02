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

class StaticScope;

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
  // static_scope_ should only be accessed after set_static_scope is called.
  auto static_scope() const -> const StaticScope& { return **static_scope_; }
  auto static_scope() -> StaticScope& { return **static_scope_; }

  // static_scope_ should only be set once during name resolution.
  void set_static_scope(Nonnull<StaticScope*> static_scope) {
    CHECK(!static_scope_.has_value());
    static_scope_ = static_scope;
  }

 private:
  SourceLocation source_loc_;
  std::string name_;
  std::vector<Nonnull<Member*>> members_;
  std::optional<Nonnull<StaticScope*>> static_scope_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_CLASS_DEFINITION_H_

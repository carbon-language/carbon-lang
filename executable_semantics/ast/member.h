// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_MEMBER_H_
#define EXECUTABLE_SEMANTICS_AST_MEMBER_H_

#include <optional>
#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/return_term.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/common/nonnull.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// Abstract base class of all AST nodes representing patterns.
//
// Member and its derived classes support LLVM-style RTTI, including
// llvm::isa, llvm::cast, and llvm::dyn_cast. To support this, every
// class derived from Member must provide a `classof` operation, and
// every concrete derived class must have a corresponding enumerator
// in `Kind`; see https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html for
// details.
class Member : public AstNode {
 public:
  ~Member() override = 0;

  Member(const Member&) = delete;
  auto operator=(const Member&) -> Member& = delete;

  void Print(llvm::raw_ostream& out) const override;

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromMember(node->kind());
  }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> MemberKind {
    return static_cast<MemberKind>(root_kind());
  }

 protected:
  Member(AstNodeKind kind, SourceLocation source_loc)
      : AstNode(kind, source_loc) {}
};

class FieldMember : public Member {
 public:
  FieldMember(SourceLocation source_loc, Nonnull<BindingPattern*> binding)
      : Member(AstNodeKind::FieldMember, source_loc), binding_(binding) {
    CHECK(binding->name() != AnonymousName);
  }

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromFieldMember(node->kind());
  }

  auto binding() const -> const BindingPattern& { return *binding_; }
  auto binding() -> BindingPattern& { return *binding_; }

 private:
  Nonnull<BindingPattern*> binding_;
};

class ClassFunctionMember : public Member {
 public:
  using ImplementsCarbonReturnTarget = void;

  ClassFunctionMember(SourceLocation source_loc, std::string name,
                      Nonnull<TuplePattern*> param_pattern,
                      ReturnTerm return_term,
                      std::optional<Nonnull<Block*>> body)
      : Member(AstNodeKind::ClassFunctionMember, source_loc),
        name_(std::move(name)),
        param_pattern_(param_pattern),
        return_term_(return_term),
        body_(body) {}

  auto name() const -> const std::string& { return name_; }
  auto param_pattern() const -> const TuplePattern& { return *param_pattern_; }
  auto param_pattern() -> TuplePattern& { return *param_pattern_; }
  auto return_term() const -> const ReturnTerm& { return return_term_; }
  auto return_term() -> ReturnTerm& { return return_term_; }
  auto body() const -> std::optional<Nonnull<const Block*>> { return body_; }
  auto body() -> std::optional<Nonnull<Block*>> { return body_; }

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromClassFunctionMember(node->kind());
  }

 private:
  std::string name_;
  Nonnull<TuplePattern*> param_pattern_;
  ReturnTerm return_term_;
  std::optional<Nonnull<Block*>> body_;
};

class MethodMember : public Member {
 public:
  using ImplementsCarbonReturnTarget = void;

  MethodMember(SourceLocation source_loc, std::string name,
               Nonnull<BindingPattern*> me_pattern,
               Nonnull<TuplePattern*> param_pattern, ReturnTerm return_term,
               std::optional<Nonnull<Block*>> body)
      : Member(AstNodeKind::MethodMember, source_loc),
        name_(std::move(name)),
        me_pattern_(me_pattern),
        param_pattern_(param_pattern),
        return_term_(return_term),
        body_(body) {}

  auto name() const -> const std::string& { return name_; }
  auto me_pattern() const -> const BindingPattern& { return *me_pattern_; }
  auto me_pattern() -> BindingPattern& { return *me_pattern_; }
  auto param_pattern() const -> const TuplePattern& { return *param_pattern_; }
  auto param_pattern() -> TuplePattern& { return *param_pattern_; }
  auto return_term() const -> const ReturnTerm& { return return_term_; }
  auto return_term() -> ReturnTerm& { return return_term_; }
  auto body() const -> std::optional<Nonnull<const Block*>> { return body_; }
  auto body() -> std::optional<Nonnull<Block*>> { return body_; }

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromMethodMember(node->kind());
  }

 private:
  std::string name_;
  Nonnull<BindingPattern*> me_pattern_;
  Nonnull<TuplePattern*> param_pattern_;
  ReturnTerm return_term_;
  std::optional<Nonnull<Block*>> body_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_MEMBER_H_

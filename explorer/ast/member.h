// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_MEMBER_H_
#define CARBON_EXPLORER_AST_MEMBER_H_

#include <optional>
#include <string>
#include <string_view>

#include "explorer/ast/ast_rtti.h"
#include "explorer/common/nonnull.h"
#include "llvm/ADT/PointerUnion.h"

namespace Carbon {

class Declaration;
class Value;

// A NamedValue represents a value with a name, such as a single struct field.
struct NamedValue {
  // The field name.
  std::string name;

  // The field's value.
  Nonnull<const Value*> value;
};

// A generic member of a type.
//
// This is can be a named, positional or other type of member.
class Member {
 protected:
  explicit Member(MemberKind kind) : kind_(kind) {}
  explicit Member(MemberKind kind, std::string_view name)
      : name_(name), kind_(kind) {}

 public:
  virtual ~Member() = default;

  // Prints the Member
  virtual void Print(llvm::raw_ostream& out) const = 0;

  // Return whether the member's name matches `name`.
  auto IsNamed(std::string_view name) const -> bool {
    return name_ && name_.value() == name;
  }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> MemberKind { return kind_; }

  // The declared type of the member, which might include type variables.
  virtual auto type() const -> const Value& = 0;

 protected:
  const std::optional<const std::string_view> name_;

 private:
  const MemberKind kind_;
};

// A named member of a type.
//
// This is either a declared member of a class, interface, or similar, or a
// member of a struct with no declaration.
class NominalMember : public Member {
 public:
  explicit NominalMember(Nonnull<const Declaration*> declaration);
  explicit NominalMember(Nonnull<const NamedValue*> struct_member);
  NominalMember(const NominalMember& other);
  NominalMember(NominalMember&& other) noexcept;
  ~NominalMember() override = default;

  // Prints the Member
  void Print(llvm::raw_ostream& out) const override;

  static auto classof(const Member* member) -> bool {
    return InheritsFromNominalMember(member->kind());
  }

  auto type() const -> const Value& override;
  // The name of the member.
  auto name() const -> std::string_view;
  // A declaration of the member, if any exists.
  auto declaration() const -> std::optional<Nonnull<const Declaration*>>;

 private:
  const llvm::PointerUnion<Nonnull<const Declaration*>,
                           Nonnull<const NamedValue*>>
      member_;
};

// A positional member of a type.
//
// This is a member of a tuple, or other index-based value.
class PositionalMember : public Member {
 public:
  explicit PositionalMember(size_t index, Nonnull<const Value*> type)
      : Member(MemberKind::PositionalMember), index_(index), type_(type) {}

  // Prints the Member
  void Print(llvm::raw_ostream& out) const override;

  static auto classof(const Member* member) -> bool {
    return InheritsFromPositionalMember(member->kind());
  }

  auto index() const -> size_t { return index_; }
  auto type() const -> const Value& override { return *type_; }

 private:
  const size_t index_;
  const Nonnull<const Value*> type_;
};

// A positional member of a type.
//
// This is a member of a tuple, or other index-based value.
class BaseClass : public Member {
 public:
  explicit BaseClass(Nonnull<const Value*> type)
      : Member(MemberKind::BaseClass), type_(type) {}

  // Prints the Member
  void Print(llvm::raw_ostream& out) const override;

  static auto classof(const Member* member) -> bool {
    return InheritsFromBaseClass(member->kind());
  }

  auto type() const -> const Value& override { return *type_; }

 private:
  const Nonnull<const Value*> type_;
};
}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_MEMBER_H_

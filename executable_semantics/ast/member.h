// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_MEMBER_H_
#define EXECUTABLE_SEMANTICS_AST_MEMBER_H_

#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/source_location.h"
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
class Member {
 public:
  enum class Kind { FieldMember };

  Member(const Member&) = delete;
  auto operator=(const Member&) -> Member& = delete;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

  auto source_loc() const -> SourceLocation { return source_loc_; }

 protected:
  // Constructs a Member representing syntax at the given line number.
  // `kind` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Member(Kind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {}

 private:
  const Kind kind_;
  SourceLocation source_loc_;
};

class FieldMember : public Member {
 public:
  FieldMember(SourceLocation source_loc, Nonnull<const BindingPattern*> binding)
      : Member(Kind::FieldMember, source_loc), binding_(binding) {}

  static auto classof(const Member* member) -> bool {
    return member->kind() == Kind::FieldMember;
  }

  auto binding() const -> const BindingPattern& { return *binding_; }

 private:
  // TODO: split this into a non-optional name and a type, initialized by
  // a constructor that takes a BindingPattern and handles errors like a
  // missing name.
  Nonnull<const BindingPattern*> binding_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_MEMBER_H_

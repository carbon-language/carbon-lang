// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_MEMBER_H_
#define EXECUTABLE_SEMANTICS_AST_MEMBER_H_

#include <string>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
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
  Member& operator=(const Member&) = delete;

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto Tag() const -> Kind { return tag; }

  auto LineNumber() const -> int { return line_num; }

  void Print(llvm::raw_ostream& out) const;

 protected:
  // Constructs a Member representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Member(Kind tag, int line_num) : tag(tag), line_num(line_num) {}

 private:
  const Kind tag;
  int line_num;
};

class FieldMember : public Member {
 public:
  FieldMember(int line_num, const BindingPattern* binding)
      : Member(Kind::FieldMember, line_num), binding(binding) {}

  static auto classof(const Member* member) -> bool {
    return member->Tag() == Kind::FieldMember;
  }

  auto Binding() const -> const BindingPattern* { return binding; }

 private:
  // TODO: split this into a non-optional name and a type, initialized by
  // a constructor that takes a BindingPattern and handles errors like a
  // missing name.
  const BindingPattern* binding;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_MEMBER_H_

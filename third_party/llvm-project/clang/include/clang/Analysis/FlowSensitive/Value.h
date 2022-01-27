//===-- Value.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for values computed by abstract interpretation
// during dataflow analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_VALUE_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_VALUE_H

#include "clang/AST/Decl.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "llvm/ADT/DenseMap.h"
#include <cassert>
#include <utility>

namespace clang {
namespace dataflow {

/// Base class for all values computed by abstract interpretation.
class Value {
public:
  enum class Kind { Integer, Reference, Pointer, Struct };

  explicit Value(Kind ValKind) : ValKind(ValKind) {}

  virtual ~Value() = default;

  Kind getKind() const { return ValKind; }

private:
  Kind ValKind;
};

/// Models an integer.
class IntegerValue : public Value {
public:
  explicit IntegerValue() : Value(Kind::Integer) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Integer;
  }
};

/// Base class for values that refer to storage locations.
class IndirectionValue : public Value {
public:
  /// Constructs a value that refers to `PointeeLoc`.
  explicit IndirectionValue(Kind ValueKind, StorageLocation &PointeeLoc)
      : Value(ValueKind), PointeeLoc(PointeeLoc) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Reference || Val->getKind() == Kind::Pointer;
  }

  StorageLocation &getPointeeLoc() const { return PointeeLoc; }

private:
  StorageLocation &PointeeLoc;
};

/// Models a dereferenced pointer. For example, a reference in C++ or an lvalue
/// in C.
class ReferenceValue final : public IndirectionValue {
public:
  explicit ReferenceValue(StorageLocation &PointeeLoc)
      : IndirectionValue(Kind::Reference, PointeeLoc) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Reference;
  }
};

/// Models a symbolic pointer. Specifically, any value of type `T*`.
class PointerValue final : public IndirectionValue {
public:
  explicit PointerValue(StorageLocation &PointeeLoc)
      : IndirectionValue(Kind::Pointer, PointeeLoc) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Pointer;
  }
};

/// Models a value of `struct` or `class` type.
class StructValue final : public Value {
public:
  StructValue() : StructValue(llvm::DenseMap<const ValueDecl *, Value *>()) {}

  explicit StructValue(llvm::DenseMap<const ValueDecl *, Value *> Children)
      : Value(Kind::Struct), Children(std::move(Children)) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Struct;
  }

  /// Returns the child value for `D`.
  Value &getChild(const ValueDecl &D) const {
    auto It = Children.find(&D);
    assert(It != Children.end());
    return *It->second;
  }

private:
  const llvm::DenseMap<const ValueDecl *, Value *> Children;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_VALUE_H

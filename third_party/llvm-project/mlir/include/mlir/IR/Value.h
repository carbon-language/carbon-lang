//===- Value.h - Base of the SSA Value hierarchy ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines generic Value type and manipulation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_VALUE_H
#define MLIR_IR_VALUE_H

#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
class AsmState;
class Block;
class BlockArgument;
class Operation;
class OpOperand;
class OpResult;
class Region;
class Value;

//===----------------------------------------------------------------------===//
// Value
//===----------------------------------------------------------------------===//

namespace detail {

/// The base class for all derived Value classes. It contains all of the
/// components that are shared across Value classes.
class alignas(8) ValueImpl : public IRObjectWithUseList<OpOperand> {
public:
  /// The enumeration represents the various different kinds of values the
  /// internal representation may take. We use all of the bits from Type that we
  /// can to store indices inline.
  enum class Kind {
    /// The first N kinds are all inline operation results. An inline operation
    /// result means that the kind represents the result number. This removes
    /// the need to store an additional index value. The derived class here is
    /// an `OpResultImpl`.
    InlineOpResult = 0,

    /// The next kind represents a 'out-of-line' operation result. This is for
    /// results with numbers larger than we can represent inline. The derived
    /// class here is an `OpResultImpl`.
    OutOfLineOpResult = 6,

    /// The last kind represents a block argument. The derived class here is an
    /// `BlockArgumentImpl`.
    BlockArgument = 7
  };

  /// Return the type of this value.
  Type getType() const { return typeAndKind.getPointer(); }

  /// Set the type of this value.
  void setType(Type type) { return typeAndKind.setPointer(type); }

  /// Return the kind of this value.
  Kind getKind() const { return typeAndKind.getInt(); }

protected:
  ValueImpl(Type type, Kind kind) : typeAndKind(type, kind) {}

  /// The type of this result and the kind.
  llvm::PointerIntPair<Type, 3, Kind> typeAndKind;
};
} // namespace detail

/// This class represents an instance of an SSA value in the MLIR system,
/// representing a computable value that has a type and a set of users. An SSA
/// value is either a BlockArgument or the result of an operation. Note: This
/// class has value-type semantics and is just a simple wrapper around a
/// ValueImpl that is either owner by a block(in the case of a BlockArgument) or
/// an Operation(in the case of an OpResult).
class Value {
public:
  constexpr Value(detail::ValueImpl *impl = nullptr) : impl(impl) {}

  template <typename U>
  bool isa() const {
    assert(*this && "isa<> used on a null type.");
    return U::classof(*this);
  }

  template <typename First, typename Second, typename... Rest>
  bool isa() const {
    return isa<First>() || isa<Second, Rest...>();
  }
  template <typename U>
  U dyn_cast() const {
    return isa<U>() ? U(impl) : U(nullptr);
  }
  template <typename U>
  U dyn_cast_or_null() const {
    return (*this && isa<U>()) ? U(impl) : U(nullptr);
  }
  template <typename U>
  U cast() const {
    assert(isa<U>());
    return U(impl);
  }

  explicit operator bool() const { return impl; }
  bool operator==(const Value &other) const { return impl == other.impl; }
  bool operator!=(const Value &other) const { return !(*this == other); }

  /// Return the type of this value.
  Type getType() const { return impl->getType(); }

  /// Utility to get the associated MLIRContext that this value is defined in.
  MLIRContext *getContext() const { return getType().getContext(); }

  /// Mutate the type of this Value to be of the specified type.
  ///
  /// Note that this is an extremely dangerous operation which can create
  /// completely invalid IR very easily.  It is strongly recommended that you
  /// recreate IR objects with the right types instead of mutating them in
  /// place.
  void setType(Type newType) { impl->setType(newType); }

  /// If this value is the result of an operation, return the operation that
  /// defines it.
  Operation *getDefiningOp() const;

  /// If this value is the result of an operation of type OpTy, return the
  /// operation that defines it.
  template <typename OpTy>
  OpTy getDefiningOp() const {
    return llvm::dyn_cast_or_null<OpTy>(getDefiningOp());
  }

  /// Return the location of this value.
  Location getLoc() const;
  void setLoc(Location loc);

  /// Return the Region in which this Value is defined.
  Region *getParentRegion();

  /// Return the Block in which this Value is defined.
  Block *getParentBlock();

  //===--------------------------------------------------------------------===//
  // UseLists
  //===--------------------------------------------------------------------===//

  /// Drop all uses of this object from their respective owners.
  void dropAllUses() const { return impl->dropAllUses(); }

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(Value newValue) const {
    impl->replaceAllUsesWith(newValue);
  }

  /// Replace all uses of 'this' value with 'newValue', updating anything in the
  /// IR that uses 'this' to use the other value instead except if the user is
  /// listed in 'exceptions' .
  void
  replaceAllUsesExcept(Value newValue,
                       const SmallPtrSetImpl<Operation *> &exceptions) const;

  /// Replace all uses of 'this' value with 'newValue', updating anything in the
  /// IR that uses 'this' to use the other value instead except if the user is
  /// 'exceptedUser'.
  void replaceAllUsesExcept(Value newValue, Operation *exceptedUser) const;

  /// Replace all uses of 'this' value with 'newValue' if the given callback
  /// returns true.
  void replaceUsesWithIf(Value newValue,
                         function_ref<bool(OpOperand &)> shouldReplace);

  /// Returns true if the value is used outside of the given block.
  bool isUsedOutsideOfBlock(Block *block);

  //===--------------------------------------------------------------------===//
  // Uses

  /// This class implements an iterator over the uses of a value.
  using use_iterator = ValueUseIterator<OpOperand>;
  using use_range = iterator_range<use_iterator>;

  use_iterator use_begin() const { return impl->use_begin(); }
  use_iterator use_end() const { return use_iterator(); }

  /// Returns a range of all uses, which is useful for iterating over all uses.
  use_range getUses() const { return {use_begin(), use_end()}; }

  /// Returns true if this value has exactly one use.
  bool hasOneUse() const { return impl->hasOneUse(); }

  /// Returns true if this value has no uses.
  bool use_empty() const { return impl->use_empty(); }

  //===--------------------------------------------------------------------===//
  // Users

  using user_iterator = ValueUserIterator<use_iterator, OpOperand>;
  using user_range = iterator_range<user_iterator>;

  user_iterator user_begin() const { return use_begin(); }
  user_iterator user_end() const { return use_end(); }
  user_range getUsers() const { return {user_begin(), user_end()}; }

  //===--------------------------------------------------------------------===//
  // Utilities

  void print(raw_ostream &os);
  void print(raw_ostream &os, AsmState &state);
  void dump();

  /// Print this value as if it were an operand.
  void printAsOperand(raw_ostream &os, AsmState &state);

  /// Methods for supporting PointerLikeTypeTraits.
  void *getAsOpaquePointer() const { return impl; }
  static Value getFromOpaquePointer(const void *pointer) {
    return reinterpret_cast<detail::ValueImpl *>(const_cast<void *>(pointer));
  }
  detail::ValueImpl *getImpl() const { return impl; }

  friend ::llvm::hash_code hash_value(Value arg);

protected:
  /// A pointer to the internal implementation of the value.
  detail::ValueImpl *impl;
};

inline raw_ostream &operator<<(raw_ostream &os, Value value) {
  value.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// OpOperand
//===----------------------------------------------------------------------===//

/// This class represents an operand of an operation. Instances of this class
/// contain a reference to a specific `Value`.
class OpOperand : public IROperand<OpOperand, Value> {
public:
  /// Provide the use list that is attached to the given value.
  static IRObjectWithUseList<OpOperand> *getUseList(Value value) {
    return value.getImpl();
  }

  /// Return which operand this is in the OpOperand list of the Operation.
  unsigned getOperandNumber();

private:
  /// Keep the constructor private and accessible to the OperandStorage class
  /// only to avoid hard-to-debug typo/programming mistakes.
  friend class OperandStorage;
  using IROperand<OpOperand, Value>::IROperand;
};

//===----------------------------------------------------------------------===//
// BlockArgument
//===----------------------------------------------------------------------===//

namespace detail {
/// The internal implementation of a BlockArgument.
class BlockArgumentImpl : public ValueImpl {
public:
  static bool classof(const ValueImpl *value) {
    return value->getKind() == ValueImpl::Kind::BlockArgument;
  }

private:
  BlockArgumentImpl(Type type, Block *owner, int64_t index, Location loc)
      : ValueImpl(type, Kind::BlockArgument), owner(owner), index(index),
        loc(loc) {}

  /// The owner of this argument.
  Block *owner;

  /// The position in the argument list.
  int64_t index;

  /// The source location of this argument.
  Location loc;

  /// Allow access to owner and constructor.
  friend BlockArgument;
};
} // namespace detail

/// This class represents an argument of a Block.
class BlockArgument : public Value {
public:
  using Value::Value;

  static bool classof(Value value) {
    return llvm::isa<detail::BlockArgumentImpl>(value.getImpl());
  }

  /// Returns the block that owns this argument.
  Block *getOwner() const { return getImpl()->owner; }

  /// Returns the number of this argument.
  unsigned getArgNumber() const { return getImpl()->index; }

  /// Return the location for this argument.
  Location getLoc() const { return getImpl()->loc; }
  void setLoc(Location loc) { getImpl()->loc = loc; }

private:
  /// Allocate a new argument with the given type and owner.
  static BlockArgument create(Type type, Block *owner, int64_t index,
                              Location loc) {
    return new detail::BlockArgumentImpl(type, owner, index, loc);
  }

  /// Destroy and deallocate this argument.
  void destroy() { delete getImpl(); }

  /// Get a raw pointer to the internal implementation.
  detail::BlockArgumentImpl *getImpl() const {
    return reinterpret_cast<detail::BlockArgumentImpl *>(impl);
  }

  /// Cache the position in the block argument list.
  void setArgNumber(int64_t index) { getImpl()->index = index; }

  /// Allow access to `create`, `destroy` and `setArgNumber`.
  friend Block;

  /// Allow access to 'getImpl'.
  friend Value;
};

//===----------------------------------------------------------------------===//
// OpResult
//===----------------------------------------------------------------------===//

namespace detail {
/// This class provides the implementation for an operation result.
class alignas(8) OpResultImpl : public ValueImpl {
public:
  using ValueImpl::ValueImpl;

  static bool classof(const ValueImpl *value) {
    return value->getKind() != ValueImpl::Kind::BlockArgument;
  }

  /// Returns the parent operation of this result.
  Operation *getOwner() const;

  /// Returns the result number of this op result.
  unsigned getResultNumber() const;

  /// Returns the next operation result at `offset` after this result. This
  /// method is useful when indexing the result storage of an operation, given
  /// that there is more than one kind of operation result (with the different
  /// kinds having different sizes) and that operations are stored in reverse
  /// order.
  OpResultImpl *getNextResultAtOffset(intptr_t offset);

  /// Returns the maximum number of results that can be stored inline.
  static unsigned getMaxInlineResults() {
    return static_cast<unsigned>(Kind::OutOfLineOpResult);
  }
};

/// This class provides the implementation for an operation result whose index
/// can be represented "inline" in the underlying ValueImpl.
struct InlineOpResult : public OpResultImpl {
public:
  InlineOpResult(Type type, unsigned resultNo)
      : OpResultImpl(type, static_cast<ValueImpl::Kind>(resultNo)) {
    assert(resultNo < getMaxInlineResults());
  }

  /// Return the result number of this op result.
  unsigned getResultNumber() const { return static_cast<unsigned>(getKind()); }

  static bool classof(const OpResultImpl *value) {
    return value->getKind() != ValueImpl::Kind::OutOfLineOpResult;
  }
};

/// This class provides the implementation for an operation result whose index
/// cannot be represented "inline", and thus requires an additional index field.
class OutOfLineOpResult : public OpResultImpl {
public:
  OutOfLineOpResult(Type type, uint64_t outOfLineIndex)
      : OpResultImpl(type, Kind::OutOfLineOpResult),
        outOfLineIndex(outOfLineIndex) {}

  static bool classof(const OpResultImpl *value) {
    return value->getKind() == ValueImpl::Kind::OutOfLineOpResult;
  }

  /// Return the result number of this op result.
  unsigned getResultNumber() const {
    return outOfLineIndex + getMaxInlineResults();
  }

  /// The trailing result number, or the offset from the beginning of the
  /// `OutOfLineOpResult` array.
  uint64_t outOfLineIndex;
};

/// Return the result number of this op result.
inline unsigned OpResultImpl::getResultNumber() const {
  if (const auto *outOfLineResult = dyn_cast<OutOfLineOpResult>(this))
    return outOfLineResult->getResultNumber();
  return cast<InlineOpResult>(this)->getResultNumber();
}

} // namespace detail

/// This is a value defined by a result of an operation.
class OpResult : public Value {
public:
  using Value::Value;

  static bool classof(Value value) {
    return llvm::isa<detail::OpResultImpl>(value.getImpl());
  }

  /// Returns the operation that owns this result.
  Operation *getOwner() const { return getImpl()->getOwner(); }

  /// Returns the number of this result.
  unsigned getResultNumber() const { return getImpl()->getResultNumber(); }

private:
  /// Get a raw pointer to the internal implementation.
  detail::OpResultImpl *getImpl() const {
    return reinterpret_cast<detail::OpResultImpl *>(impl);
  }

  /// Given a number of operation results, returns the number that need to be
  /// stored inline.
  static unsigned getNumInline(unsigned numResults);

  /// Given a number of operation results, returns the number that need to be
  /// stored as trailing.
  static unsigned getNumTrailing(unsigned numResults);

  /// Allow access to constructor.
  friend Operation;
};

/// Make Value hashable.
inline ::llvm::hash_code hash_value(Value arg) {
  return ::llvm::hash_value(arg.getImpl());
}

} // namespace mlir

namespace llvm {

template <>
struct DenseMapInfo<mlir::Value> {
  static mlir::Value getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Value::getFromOpaquePointer(pointer);
  }
  static mlir::Value getTombstoneKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Value::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::Value val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Value lhs, mlir::Value rhs) { return lhs == rhs; }
};
template <>
struct DenseMapInfo<mlir::BlockArgument> : public DenseMapInfo<mlir::Value> {
  static mlir::BlockArgument getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return reinterpret_cast<mlir::detail::BlockArgumentImpl *>(pointer);
  }
  static mlir::BlockArgument getTombstoneKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return reinterpret_cast<mlir::detail::BlockArgumentImpl *>(pointer);
  }
};
template <>
struct DenseMapInfo<mlir::OpResult> : public DenseMapInfo<mlir::Value> {
  static mlir::OpResult getEmptyKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return reinterpret_cast<mlir::detail::OpResultImpl *>(pointer);
  }
  static mlir::OpResult getTombstoneKey() {
    void *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return reinterpret_cast<mlir::detail::OpResultImpl *>(pointer);
  }
};

/// Allow stealing the low bits of a value.
template <>
struct PointerLikeTypeTraits<mlir::Value> {
public:
  static inline void *getAsVoidPointer(mlir::Value value) {
    return const_cast<void *>(value.getAsOpaquePointer());
  }
  static inline mlir::Value getFromVoidPointer(void *pointer) {
    return mlir::Value::getFromOpaquePointer(pointer);
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::detail::ValueImpl *>::NumLowBitsAvailable
  };
};
template <>
struct PointerLikeTypeTraits<mlir::BlockArgument>
    : public PointerLikeTypeTraits<mlir::Value> {
public:
  static inline mlir::BlockArgument getFromVoidPointer(void *pointer) {
    return reinterpret_cast<mlir::detail::BlockArgumentImpl *>(pointer);
  }
};
template <>
struct PointerLikeTypeTraits<mlir::OpResult>
    : public PointerLikeTypeTraits<mlir::Value> {
public:
  static inline mlir::OpResult getFromVoidPointer(void *pointer) {
    return reinterpret_cast<mlir::detail::OpResultImpl *>(pointer);
  }
};

} // namespace llvm

#endif

//===- AffineExpr.h - MLIR Affine Expr Class --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An affine expression is an affine combination of dimension identifiers and
// symbols, including ceildiv/floordiv/mod by a constant integer.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_AFFINEEXPR_H
#define MLIR_IR_AFFINEEXPR_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Casting.h"
#include <functional>
#include <type_traits>

namespace mlir {

class MLIRContext;
class AffineMap;
class IntegerSet;

namespace detail {

struct AffineExprStorage;
struct AffineBinaryOpExprStorage;
struct AffineDimExprStorage;
struct AffineSymbolExprStorage;
struct AffineConstantExprStorage;

} // namespace detail

enum class AffineExprKind {
  Add,
  /// RHS of mul is always a constant or a symbolic expression.
  Mul,
  /// RHS of mod is always a constant or a symbolic expression with a positive
  /// value.
  Mod,
  /// RHS of floordiv is always a constant or a symbolic expression.
  FloorDiv,
  /// RHS of ceildiv is always a constant or a symbolic expression.
  CeilDiv,

  /// This is a marker for the last affine binary op. The range of binary
  /// op's is expected to be this element and earlier.
  LAST_AFFINE_BINARY_OP = CeilDiv,

  /// Constant integer.
  Constant,
  /// Dimensional identifier.
  DimId,
  /// Symbolic identifier.
  SymbolId,
};

/// Base type for affine expression.
/// AffineExpr's are immutable value types with intuitive operators to
/// operate on chainable, lightweight compositions.
/// An AffineExpr is an interface to the underlying storage type pointer.
class AffineExpr {
public:
  using ImplType = detail::AffineExprStorage;

  constexpr AffineExpr() {}
  /* implicit */ AffineExpr(const ImplType *expr)
      : expr(const_cast<ImplType *>(expr)) {}

  bool operator==(AffineExpr other) const { return expr == other.expr; }
  bool operator!=(AffineExpr other) const { return !(*this == other); }
  bool operator==(int64_t v) const;
  bool operator!=(int64_t v) const { return !(*this == v); }
  explicit operator bool() const { return expr; }

  bool operator!() const { return expr == nullptr; }

  template <typename U>
  bool isa() const;
  template <typename U>
  U dyn_cast() const;
  template <typename U>
  U dyn_cast_or_null() const;
  template <typename U>
  U cast() const;

  MLIRContext *getContext() const;

  /// Return the classification for this type.
  AffineExprKind getKind() const;

  void print(raw_ostream &os) const;
  void dump() const;

  /// Returns true if this expression is made out of only symbols and
  /// constants, i.e., it does not involve dimensional identifiers.
  bool isSymbolicOrConstant() const;

  /// Returns true if this is a pure affine expression, i.e., multiplication,
  /// floordiv, ceildiv, and mod is only allowed w.r.t constants.
  bool isPureAffine() const;

  /// Returns the greatest known integral divisor of this affine expression. The
  /// result is always positive.
  int64_t getLargestKnownDivisor() const;

  /// Return true if the affine expression is a multiple of 'factor'.
  bool isMultipleOf(int64_t factor) const;

  /// Return true if the affine expression involves AffineDimExpr `position`.
  bool isFunctionOfDim(unsigned position) const;

  /// Return true if the affine expression involves AffineSymbolExpr `position`.
  bool isFunctionOfSymbol(unsigned position) const;

  /// Walk all of the AffineExpr's in this expression in postorder.
  void walk(std::function<void(AffineExpr)> callback) const;

  /// This method substitutes any uses of dimensions and symbols (e.g.
  /// dim#0 with dimReplacements[0]) and returns the modified expression tree.
  /// This is a dense replacement method: a replacement must be specified for
  /// every single dim and symbol.
  AffineExpr replaceDimsAndSymbols(ArrayRef<AffineExpr> dimReplacements,
                                   ArrayRef<AffineExpr> symReplacements) const;

  /// Dim-only version of replaceDimsAndSymbols.
  AffineExpr replaceDims(ArrayRef<AffineExpr> dimReplacements) const;

  /// Symbol-only version of replaceDimsAndSymbols.
  AffineExpr replaceSymbols(ArrayRef<AffineExpr> symReplacements) const;

  /// Sparse replace method. Replace `expr` by `replacement` and return the
  /// modified expression tree.
  AffineExpr replace(AffineExpr expr, AffineExpr replacement) const;

  /// Sparse replace method. If `*this` appears in `map` replaces it by
  /// `map[*this]` and return the modified expression tree. Otherwise traverse
  /// `*this` and apply replace with `map` on its subexpressions.
  AffineExpr replace(const DenseMap<AffineExpr, AffineExpr> &map) const;

  /// Replace dims[offset ... numDims)
  /// by dims[offset + shift ... shift + numDims).
  AffineExpr shiftDims(unsigned numDims, unsigned shift,
                       unsigned offset = 0) const;

  /// Replace symbols[offset ... numSymbols)
  /// by symbols[offset + shift ... shift + numSymbols).
  AffineExpr shiftSymbols(unsigned numSymbols, unsigned shift,
                          unsigned offset = 0) const;

  AffineExpr operator+(int64_t v) const;
  AffineExpr operator+(AffineExpr other) const;
  AffineExpr operator-() const;
  AffineExpr operator-(int64_t v) const;
  AffineExpr operator-(AffineExpr other) const;
  AffineExpr operator*(int64_t v) const;
  AffineExpr operator*(AffineExpr other) const;
  AffineExpr floorDiv(uint64_t v) const;
  AffineExpr floorDiv(AffineExpr other) const;
  AffineExpr ceilDiv(uint64_t v) const;
  AffineExpr ceilDiv(AffineExpr other) const;
  AffineExpr operator%(uint64_t v) const;
  AffineExpr operator%(AffineExpr other) const;

  /// Compose with an AffineMap.
  /// Returns the composition of this AffineExpr with `map`.
  ///
  /// Prerequisites:
  /// `this` and `map` are composable, i.e. that the number of AffineDimExpr of
  /// `this` is smaller than the number of results of `map`. If a result of a
  /// map does not have a corresponding AffineDimExpr, that result simply does
  /// not appear in the produced AffineExpr.
  ///
  /// Example:
  ///   expr: `d0 + d2`
  ///   map:  `(d0, d1, d2)[s0, s1] -> (d0 + s1, d1 + s0, d0 + d1 + d2)`
  ///   returned expr: `d0 * 2 + d1 + d2 + s1`
  AffineExpr compose(AffineMap map) const;

  friend ::llvm::hash_code hash_value(AffineExpr arg);

  /// Methods supporting C API.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(expr);
  }
  static AffineExpr getFromOpaquePointer(const void *pointer) {
    return AffineExpr(
        reinterpret_cast<ImplType *>(const_cast<void *>(pointer)));
  }

protected:
  ImplType *expr{nullptr};
};

/// Affine binary operation expression. An affine binary operation could be an
/// add, mul, floordiv, ceildiv, or a modulo operation. (Subtraction is
/// represented through a multiply by -1 and add.) These expressions are always
/// constructed in a simplified form. For eg., the LHS and RHS operands can't
/// both be constants. There are additional canonicalizing rules depending on
/// the op type: see checks in the constructor.
class AffineBinaryOpExpr : public AffineExpr {
public:
  using ImplType = detail::AffineBinaryOpExprStorage;
  /* implicit */ AffineBinaryOpExpr(AffineExpr::ImplType *ptr);
  AffineExpr getLHS() const;
  AffineExpr getRHS() const;
};

/// A dimensional identifier appearing in an affine expression.
class AffineDimExpr : public AffineExpr {
public:
  using ImplType = detail::AffineDimExprStorage;
  /* implicit */ AffineDimExpr(AffineExpr::ImplType *ptr);
  unsigned getPosition() const;
};

/// A symbolic identifier appearing in an affine expression.
class AffineSymbolExpr : public AffineExpr {
public:
  using ImplType = detail::AffineDimExprStorage;
  /* implicit */ AffineSymbolExpr(AffineExpr::ImplType *ptr);
  unsigned getPosition() const;
};

/// An integer constant appearing in affine expression.
class AffineConstantExpr : public AffineExpr {
public:
  using ImplType = detail::AffineConstantExprStorage;
  /* implicit */ AffineConstantExpr(AffineExpr::ImplType *ptr = nullptr);
  int64_t getValue() const;
};

/// Make AffineExpr hashable.
inline ::llvm::hash_code hash_value(AffineExpr arg) {
  return ::llvm::hash_value(arg.expr);
}

inline AffineExpr operator+(int64_t val, AffineExpr expr) { return expr + val; }
inline AffineExpr operator*(int64_t val, AffineExpr expr) { return expr * val; }
inline AffineExpr operator-(int64_t val, AffineExpr expr) {
  return expr * (-1) + val;
}

/// These free functions allow clients of the API to not use classes in detail.
AffineExpr getAffineDimExpr(unsigned position, MLIRContext *context);
AffineExpr getAffineSymbolExpr(unsigned position, MLIRContext *context);
AffineExpr getAffineConstantExpr(int64_t constant, MLIRContext *context);
AffineExpr getAffineBinaryOpExpr(AffineExprKind kind, AffineExpr lhs,
                                 AffineExpr rhs);

/// Constructs an affine expression from a flat ArrayRef. If there are local
/// identifiers (neither dimensional nor symbolic) that appear in the sum of
/// products expression, 'localExprs' is expected to have the AffineExpr
/// for it, and is substituted into. The ArrayRef 'eq' is expected to be in the
/// format [dims, symbols, locals, constant term].
AffineExpr getAffineExprFromFlatForm(ArrayRef<int64_t> flatExprs,
                                     unsigned numDims, unsigned numSymbols,
                                     ArrayRef<AffineExpr> localExprs,
                                     MLIRContext *context);

raw_ostream &operator<<(raw_ostream &os, AffineExpr expr);

template <typename U>
bool AffineExpr::isa() const {
  if (std::is_same<U, AffineBinaryOpExpr>::value)
    return getKind() <= AffineExprKind::LAST_AFFINE_BINARY_OP;
  if (std::is_same<U, AffineDimExpr>::value)
    return getKind() == AffineExprKind::DimId;
  if (std::is_same<U, AffineSymbolExpr>::value)
    return getKind() == AffineExprKind::SymbolId;
  if (std::is_same<U, AffineConstantExpr>::value)
    return getKind() == AffineExprKind::Constant;
}
template <typename U>
U AffineExpr::dyn_cast() const {
  if (isa<U>())
    return U(expr);
  return U(nullptr);
}
template <typename U>
U AffineExpr::dyn_cast_or_null() const {
  return (!*this || !isa<U>()) ? U(nullptr) : U(expr);
}
template <typename U>
U AffineExpr::cast() const {
  assert(isa<U>());
  return U(expr);
}

/// Simplify an affine expression by flattening and some amount of simple
/// analysis. This has complexity linear in the number of nodes in 'expr'.
/// Returns the simplified expression, which is the same as the input expression
/// if it can't be simplified. When `expr` is semi-affine, a simplified
/// semi-affine expression is constructed in the sorted order of dimension and
/// symbol positions.
AffineExpr simplifyAffineExpr(AffineExpr expr, unsigned numDims,
                              unsigned numSymbols);

namespace detail {
template <int N>
void bindDims(MLIRContext *ctx) {}

template <int N, typename AffineExprTy, typename... AffineExprTy2>
void bindDims(MLIRContext *ctx, AffineExprTy &e, AffineExprTy2 &...exprs) {
  e = getAffineDimExpr(N, ctx);
  bindDims<N + 1, AffineExprTy2 &...>(ctx, exprs...);
}

template <int N>
void bindSymbols(MLIRContext *ctx) {}

template <int N, typename AffineExprTy, typename... AffineExprTy2>
void bindSymbols(MLIRContext *ctx, AffineExprTy &e, AffineExprTy2 &...exprs) {
  e = getAffineSymbolExpr(N, ctx);
  bindSymbols<N + 1, AffineExprTy2 &...>(ctx, exprs...);
}
} // namespace detail

/// Bind a list of AffineExpr references to DimExpr at positions:
///   [0 .. sizeof...(exprs)]
template <typename... AffineExprTy>
void bindDims(MLIRContext *ctx, AffineExprTy &...exprs) {
  detail::bindDims<0>(ctx, exprs...);
}

/// Bind a list of AffineExpr references to SymbolExpr at positions:
///   [0 .. sizeof...(exprs)]
template <typename... AffineExprTy>
void bindSymbols(MLIRContext *ctx, AffineExprTy &...exprs) {
  detail::bindSymbols<0>(ctx, exprs...);
}

} // namespace mlir

namespace llvm {

// AffineExpr hash just like pointers
template <>
struct DenseMapInfo<mlir::AffineExpr> {
  static mlir::AffineExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::AffineExpr(static_cast<mlir::AffineExpr::ImplType *>(pointer));
  }
  static mlir::AffineExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::AffineExpr(static_cast<mlir::AffineExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::AffineExpr val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::AffineExpr LHS, mlir::AffineExpr RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // MLIR_IR_AFFINEEXPR_H

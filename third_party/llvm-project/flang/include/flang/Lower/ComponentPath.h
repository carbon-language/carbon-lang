//===-- ComponentPath.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_COMPONENTPATH_H
#define FORTRAN_LOWER_COMPONENTPATH_H

#include "flang/Lower/IterationSpace.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
class ArrayLoadOp;
}
namespace Fortran::evaluate {
class ArrayRef;
}

namespace Fortran::lower {

namespace details {
class ImplicitSubscripts {};
} // namespace details

using PathComponent =
    std::variant<const evaluate::ArrayRef *, const evaluate::Component *,
                 const evaluate::ComplexPart *, details::ImplicitSubscripts>;

/// Collection of components.
///
/// This class is used both to collect front-end post-order functional Expr
/// trees and their translations to Values to be used in a pre-order list of
/// arguments.
class ComponentPath {
public:
  using ExtendRefFunc = std::function<mlir::Value(const mlir::Value &)>;

  ComponentPath(bool isImplicit) { setPC(isImplicit); }
  ComponentPath(bool isImplicit, const evaluate::Substring *ss)
      : substring(ss) {
    setPC(isImplicit);
  }
  ComponentPath() = delete;

  bool isSlice() const { return !trips.empty() || hasComponents(); }
  bool hasComponents() const { return !suffixComponents.empty(); }
  void clear();

  bool hasExtendCoorRef() const { return extendCoorRef.hasValue(); }
  ExtendRefFunc getExtendCoorRef() const;
  void resetExtendCoorRef() { extendCoorRef = llvm::None; }
  void resetPC();

  llvm::SmallVector<PathComponent> reversePath;
  const evaluate::Substring *substring = nullptr;
  bool applied = false;

  llvm::SmallVector<mlir::Value> prefixComponents;
  llvm::SmallVector<mlir::Value> trips;
  llvm::SmallVector<mlir::Value> suffixComponents;
  std::function<IterationSpace(const IterationSpace &)> pc;

  /// In the case where a path of components involves members that are POINTER
  /// or ALLOCATABLE, a dereference is required in FIR for semantic correctness.
  /// This optional continuation allows the generation of those dereferences.
  /// These accesses are always on Fortran entities of record types, which are
  /// implicitly in-memory objects.
  llvm::Optional<ExtendRefFunc> extendCoorRef = llvm::None;

private:
  void setPC(bool isImplicit);
};

/// Examine each subscript expression of \p x and return true if and only if any
/// of the subscripts is a vector or has a rank greater than 0.
bool isRankedArrayAccess(const Fortran::evaluate::ArrayRef &x);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_COMPONENTPATH_H

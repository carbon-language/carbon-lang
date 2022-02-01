//===-- Lower/ConvertType.h -- lowering of types ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//
///
/// Conversion of front-end TYPE, KIND, ATTRIBUTE (TKA) information to FIR/MLIR.
/// This is meant to be the single point of truth (SPOT) for all type
/// conversions when lowering to FIR.  This implements all lowering of parse
/// tree TKA to the FIR type system. If one is converting front-end types and
/// not using one of the routines provided here, it's being done wrong.
///
/// [Coding style](https://llvm.org/docs/CodingStandards.html)
///
//----------------------------------------------------------------------------//

#ifndef FORTRAN_LOWER_CONVERT_TYPE_H
#define FORTRAN_LOWER_CONVERT_TYPE_H

#include "flang/Common/Fortran.h"
#include "mlir/IR/Types.h"

namespace mlir {
class Location;
class MLIRContext;
class Type;
} // namespace mlir

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
template <typename>
class Reference;
} // namespace common

namespace evaluate {
struct DataRef;
template <typename>
class Designator;
template <typename>
class Expr;
template <common::TypeCategory>
struct SomeKind;
struct SomeType;
template <common::TypeCategory, int>
class Type;
} // namespace evaluate

namespace semantics {
class Symbol;
} // namespace semantics

namespace lower {
namespace pft {
struct Variable;
}

using SomeExpr = evaluate::Expr<evaluate::SomeType>;
using SymbolRef = common::Reference<const semantics::Symbol>;

/// Get a FIR type based on a category and kind.
mlir::Type getFIRType(mlir::MLIRContext *ctxt,
                      common::IntrinsicTypeDefaultKinds const &defaults,
                      common::TypeCategory tc, int kind);

/// Get a FIR type based on a category.
mlir::Type getFIRType(mlir::MLIRContext *ctxt,
                      common::IntrinsicTypeDefaultKinds const &defaults,
                      common::TypeCategory tc);

/// Translate a Fortran::evaluate::DataRef to an mlir::Type.
mlir::Type
translateDataRefToFIRType(mlir::MLIRContext *ctxt,
                          common::IntrinsicTypeDefaultKinds const &defaults,
                          const evaluate::DataRef &dataRef);

/// Translate a Fortran::evaluate::Designator<> to an mlir::Type.
template <common::TypeCategory TC, int KIND>
inline mlir::Type translateDesignatorToFIRType(
    mlir::MLIRContext *ctxt, common::IntrinsicTypeDefaultKinds const &defaults,
    const evaluate::Designator<evaluate::Type<TC, KIND>> &) {
  return getFIRType(ctxt, defaults, TC, KIND);
}

/// Translate a Fortran::evaluate::Designator<> to an mlir::Type.
template <common::TypeCategory TC>
inline mlir::Type translateDesignatorToFIRType(
    mlir::MLIRContext *ctxt, common::IntrinsicTypeDefaultKinds const &defaults,
    const evaluate::Designator<evaluate::SomeKind<TC>> &) {
  return getFIRType(ctxt, defaults, TC);
}

/// Translate a SomeExpr to an mlir::Type.
mlir::Type
translateSomeExprToFIRType(mlir::MLIRContext *ctxt,
                           common::IntrinsicTypeDefaultKinds const &defaults,
                           const SomeExpr *expr);

/// Translate a Fortran::semantics::Symbol to an mlir::Type.
mlir::Type
translateSymbolToFIRType(mlir::MLIRContext *ctxt,
                         common::IntrinsicTypeDefaultKinds const &defaults,
                         const SymbolRef symbol);

/// Translate a Fortran::lower::pft::Variable to an mlir::Type.
mlir::Type
translateVariableToFIRType(mlir::MLIRContext *ctxt,
                           common::IntrinsicTypeDefaultKinds const &defaults,
                           const pft::Variable &variable);

/// Translate a REAL of KIND to the mlir::Type.
mlir::Type convertReal(mlir::MLIRContext *ctxt, int KIND);

// Given a ReferenceType of a base type, returns the ReferenceType to
// the SequenceType of this base type.
// The created SequenceType has one dimension of unknown extent.
// This is useful to do pointer arithmetic using fir::CoordinateOp that requires
// a memory reference to a sequence type.
mlir::Type getSequenceRefType(mlir::Type referenceType);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERT_TYPE_H

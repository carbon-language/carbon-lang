//===-- Optimizer/Dialect/FIROpsSupport.h -- FIR op support -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FIROPSSUPPORT_H
#define FORTRAN_OPTIMIZER_DIALECT_FIROPSSUPPORT_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace fir {

/// Return true iff the Operation is a non-volatile LoadOp or ArrayLoadOp.
inline bool nonVolatileLoad(mlir::Operation *op) {
  if (auto load = mlir::dyn_cast<fir::LoadOp>(op))
    return !load->getAttr("volatile");
  if (auto arrLoad = mlir::dyn_cast<fir::ArrayLoadOp>(op))
    return !arrLoad->getAttr("volatile");
  return false;
}

/// Return true iff the Operation is a call.
inline bool isaCall(mlir::Operation *op) {
  return mlir::isa<fir::CallOp>(op) || mlir::isa<fir::DispatchOp>(op) ||
         mlir::isa<mlir::func::CallOp>(op) ||
         mlir::isa<mlir::func::CallIndirectOp>(op);
}

/// return true iff the Operation is a fir::CallOp, fir::DispatchOp,
/// mlir::CallOp, or mlir::CallIndirectOp and not pure
/// NB: this is not the same as `!pureCall(op)`
inline bool impureCall(mlir::Operation *op) {
  // Should we also auto-detect that the called function is pure if its
  // arguments are not references?  For now, rely on a "pure" attribute.
  return op && isaCall(op) && !op->getAttr("pure");
}

/// return true iff the Operation is a fir::CallOp, fir::DispatchOp,
/// mlir::CallOp, or mlir::CallIndirectOp and is also pure.
/// NB: this is not the same as `!impureCall(op)`
inline bool pureCall(mlir::Operation *op) {
  // Should we also auto-detect that the called function is pure if its
  // arguments are not references?  For now, rely on a "pure" attribute.
  return op && isaCall(op) && op->getAttr("pure");
}

/// Get or create a FuncOp in a module.
///
/// If `module` already contains FuncOp `name`, it is returned. Otherwise, a new
/// FuncOp is created, and that new FuncOp is returned.
mlir::func::FuncOp
createFuncOp(mlir::Location loc, mlir::ModuleOp module, llvm::StringRef name,
             mlir::FunctionType type,
             llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

/// Get or create a GlobalOp in a module.
fir::GlobalOp createGlobalOp(mlir::Location loc, mlir::ModuleOp module,
                             llvm::StringRef name, mlir::Type type,
                             llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

/// Attribute to mark Fortran entities with the CONTIGUOUS attribute.
constexpr llvm::StringRef getContiguousAttrName() { return "fir.contiguous"; }

/// Attribute to mark Fortran entities with the OPTIONAL attribute.
constexpr llvm::StringRef getOptionalAttrName() { return "fir.optional"; }

/// Attribute to mark Fortran entities with the TARGET attribute.
static constexpr llvm::StringRef getTargetAttrName() { return "fir.target"; }

/// Attribute to mark that a function argument is a character dummy procedure.
/// Character dummy procedure have special ABI constraints.
static constexpr llvm::StringRef getCharacterProcedureDummyAttrName() {
  return "fir.char_proc";
}

/// Attribute to keep track of Fortran scoping information for a symbol.
static constexpr llvm::StringRef getSymbolAttrName() { return "fir.sym_name"; }

/// Attribute to mark a function that takes a host associations argument.
static constexpr llvm::StringRef getHostAssocAttrName() {
  return "fir.host_assoc";
}

/// Does the function, \p func, have a host-associations tuple argument?
/// Some internal procedures may have access to host procedure variables.
bool hasHostAssociationArgument(mlir::func::FuncOp func);

/// Tell if \p value is:
///   - a function argument that has attribute \p attributeName
///   - or, the result of fir.alloca/fir.allocamem op that has attribute \p
///     attributeName
///   - or, the result of a fir.address_of of a fir.global that has attribute \p
///     attributeName
///   - or, a fir.box loaded from a fir.ref<fir.box> that matches one of the
///     previous cases.
bool valueHasFirAttribute(mlir::Value value, llvm::StringRef attributeName);

/// Scan the arguments of a FuncOp to determine if any arguments have the
/// attribute `attr` placed on them. This can be used to determine if the
/// function has any host associations, for example.
bool anyFuncArgsHaveAttr(mlir::func::FuncOp func, llvm::StringRef attr);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIROPSSUPPORT_H

//===- BuiltinDialect.cpp - MLIR Builtin Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Builtin dialect that contains all of the attributes,
// operations, and types that are necessary for the validity of the IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Builtin Dialect
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.cpp.inc"

namespace {
struct BuiltinOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (attr.isa<AffineMapAttr>()) {
      os << "map";
      return AliasResult::OverridableAlias;
    }
    if (attr.isa<IntegerSetAttr>()) {
      os << "set";
      return AliasResult::OverridableAlias;
    }
    if (attr.isa<LocationAttr>()) {
      os << "loc";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto tupleType = type.dyn_cast<TupleType>()) {
      if (tupleType.size() > 16) {
        os << "tuple";
        return AliasResult::OverridableAlias;
      }
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void BuiltinDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerLocationAttributes();
  addOperations<
#define GET_OP_LIST
#include "mlir/IR/BuiltinOps.cpp.inc"
      >();
  addInterfaces<BuiltinOpAsmDialectInterface>();
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

void ModuleOp::build(OpBuilder &builder, OperationState &state,
                     Optional<StringRef> name) {
  state.addRegion()->emplaceBlock();
  if (name) {
    state.attributes.push_back(builder.getNamedAttr(
        mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(*name)));
  }
}

/// Construct a module from the given context.
ModuleOp ModuleOp::create(Location loc, Optional<StringRef> name) {
  OpBuilder builder(loc->getContext());
  return builder.create<ModuleOp>(loc, name);
}

DataLayoutSpecInterface ModuleOp::getDataLayoutSpec() {
  // Take the first and only (if present) attribute that implements the
  // interface. This needs a linear search, but is called only once per data
  // layout object construction that is used for repeated queries.
  for (NamedAttribute attr : getOperation()->getAttrs())
    if (auto spec = attr.getValue().dyn_cast<DataLayoutSpecInterface>())
      return spec;
  return {};
}

LogicalResult ModuleOp::verify() {
  // Check that none of the attributes are non-dialect attributes, except for
  // the symbol related attributes.
  for (auto attr : (*this)->getAttrs()) {
    if (!attr.getName().strref().contains('.') &&
        !llvm::is_contained(
            ArrayRef<StringRef>{mlir::SymbolTable::getSymbolAttrName(),
                                mlir::SymbolTable::getVisibilityAttrName()},
            attr.getName().strref()))
      return emitOpError() << "can only contain attributes with "
                              "dialect-prefixed names, found: '"
                           << attr.getName().getValue() << "'";
  }

  // Check that there is at most one data layout spec attribute.
  StringRef layoutSpecAttrName;
  DataLayoutSpecInterface layoutSpec;
  for (const NamedAttribute &na : (*this)->getAttrs()) {
    if (auto spec = na.getValue().dyn_cast<DataLayoutSpecInterface>()) {
      if (layoutSpec) {
        InFlightDiagnostic diag =
            emitOpError() << "expects at most one data layout attribute";
        diag.attachNote() << "'" << layoutSpecAttrName
                          << "' is a data layout attribute";
        diag.attachNote() << "'" << na.getName().getValue()
                          << "' is a data layout attribute";
      }
      layoutSpecAttrName = na.getName().strref();
      layoutSpec = spec;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnrealizedConversionCastOp
//===----------------------------------------------------------------------===//

LogicalResult
UnrealizedConversionCastOp::fold(ArrayRef<Attribute> attrOperands,
                                 SmallVectorImpl<OpFoldResult> &foldResults) {
  OperandRange operands = getInputs();
  ResultRange results = getOutputs();

  if (operands.getType() == results.getType()) {
    foldResults.append(operands.begin(), operands.end());
    return success();
  }

  if (operands.empty())
    return failure();

  // Check that the input is a cast with results that all feed into this
  // operation, and operand types that directly match the result types of this
  // operation.
  Value firstInput = operands.front();
  auto inputOp = firstInput.getDefiningOp<UnrealizedConversionCastOp>();
  if (!inputOp || inputOp.getResults() != operands ||
      inputOp.getOperandTypes() != results.getTypes())
    return failure();

  // If everything matches up, we can fold the passthrough.
  foldResults.append(inputOp->operand_begin(), inputOp->operand_end());
  return success();
}

bool UnrealizedConversionCastOp::areCastCompatible(TypeRange inputs,
                                                   TypeRange outputs) {
  // `UnrealizedConversionCastOp` is agnostic of the input/output types.
  return true;
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/IR/BuiltinOps.cpp.inc"

//===- ComplexOps.cpp - MLIR Complex Operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::complex;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cst");
}

bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  if (auto arrAttr = value.dyn_cast<ArrayAttr>()) {
    auto complexTy = type.dyn_cast<ComplexType>();
    if (!complexTy)
      return false;
    auto complexEltTy = complexTy.getElementType();
    return arrAttr.size() == 2 && arrAttr[0].getType() == complexEltTy &&
           arrAttr[1].getType() == complexEltTy;
  }
  return false;
}

LogicalResult ConstantOp::verify() {
  ArrayAttr arrayAttr = getValue();
  if (arrayAttr.size() != 2) {
    return emitOpError(
        "requires 'value' to be a complex constant, represented as array of "
        "two values");
  }

  auto complexEltTy = getType().getElementType();
  if (complexEltTy != arrayAttr[0].getType() ||
      complexEltTy != arrayAttr[1].getType()) {
    return emitOpError()
           << "requires attribute's element types (" << arrayAttr[0].getType()
           << ", " << arrayAttr[1].getType()
           << ") to match the element type of the op's return type ("
           << complexEltTy << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CreateOp
//===----------------------------------------------------------------------===//

OpFoldResult CreateOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary op takes two operands");
  // Fold complex.create(complex.re(op), complex.im(op)).
  if (auto reOp = getOperand(0).getDefiningOp<ReOp>()) {
    if (auto imOp = getOperand(1).getDefiningOp<ImOp>()) {
      if (reOp.getOperand() == imOp.getOperand()) {
        return reOp.getOperand();
      }
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ImOp
//===----------------------------------------------------------------------===//

OpFoldResult ImOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");
  ArrayAttr arrayAttr = operands[0].dyn_cast_or_null<ArrayAttr>();
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[1];
  if (auto createOp = getOperand().getDefiningOp<CreateOp>())
    return createOp.getOperand(1);
  return {};
}

//===----------------------------------------------------------------------===//
// ReOp
//===----------------------------------------------------------------------===//

OpFoldResult ReOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary op takes 1 operand");
  ArrayAttr arrayAttr = operands[0].dyn_cast_or_null<ArrayAttr>();
  if (arrayAttr && arrayAttr.size() == 2)
    return arrayAttr[0];
  if (auto createOp = getOperand().getDefiningOp<CreateOp>())
    return createOp.getOperand(0);
  return {};
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Complex/IR/ComplexOps.cpp.inc"

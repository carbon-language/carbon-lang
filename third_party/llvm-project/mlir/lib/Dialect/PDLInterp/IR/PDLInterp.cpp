//===- PDLInterp.cpp - PDL Interpreter Dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::pdl_interp;

#include "mlir/Dialect/PDLInterp/IR/PDLInterpOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PDLInterp Dialect
//===----------------------------------------------------------------------===//

void PDLInterpDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateOperationOp
//===----------------------------------------------------------------------===//

static ParseResult parseCreateOperationOpAttributes(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::OperandType> &attrOperands,
    ArrayAttr &attrNamesAttr) {
  Builder &builder = p.getBuilder();
  SmallVector<Attribute, 4> attrNames;
  if (succeeded(p.parseOptionalLBrace())) {
    do {
      StringAttr nameAttr;
      OpAsmParser::OperandType operand;
      if (p.parseAttribute(nameAttr) || p.parseEqual() ||
          p.parseOperand(operand))
        return failure();
      attrNames.push_back(nameAttr);
      attrOperands.push_back(operand);
    } while (succeeded(p.parseOptionalComma()));
    if (p.parseRBrace())
      return failure();
  }
  attrNamesAttr = builder.getArrayAttr(attrNames);
  return success();
}

static void printCreateOperationOpAttributes(OpAsmPrinter &p,
                                             CreateOperationOp op,
                                             OperandRange attrArgs,
                                             ArrayAttr attrNames) {
  if (attrNames.empty())
    return;
  p << " {";
  interleaveComma(llvm::seq<int>(0, attrNames.size()), p,
                  [&](int i) { p << attrNames[i] << " = " << attrArgs[i]; });
  p << '}';
}

//===----------------------------------------------------------------------===//
// pdl_interp::ForEachOp
//===----------------------------------------------------------------------===//

void ForEachOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                      Value range, Block *successor, bool initLoop) {
  build(builder, state, range, successor);
  if (initLoop) {
    // Create the block and the loop variable.
    auto rangeType = range.getType().cast<pdl::RangeType>();
    state.regions.front()->emplaceBlock();
    state.regions.front()->addArgument(rangeType.getElementType());
  }
}

static ParseResult parseForEachOp(OpAsmParser &parser, OperationState &result) {
  // Parse the loop variable followed by type.
  OpAsmParser::OperandType loopVariable;
  Type loopVariableType;
  if (parser.parseRegionArgument(loopVariable) ||
      parser.parseColonType(loopVariableType))
    return failure();

  // Parse the "in" keyword.
  if (parser.parseKeyword("in", " after loop variable"))
    return failure();

  // Parse the operand (value range).
  OpAsmParser::OperandType operandInfo;
  if (parser.parseOperand(operandInfo))
    return failure();

  // Resolve the operand.
  Type rangeType = pdl::RangeType::get(loopVariableType);
  if (parser.resolveOperand(operandInfo, rangeType, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {loopVariable}, {loopVariableType}))
    return failure();

  // Parse the attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the successor.
  Block *successor;
  if (parser.parseArrow() || parser.parseSuccessor(successor))
    return failure();
  result.addSuccessors(successor);

  return success();
}

static void print(OpAsmPrinter &p, ForEachOp op) {
  BlockArgument arg = op.getLoopVariable();
  p << ' ' << arg << " : " << arg.getType() << " in " << op.values();
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(op->getAttrs());
  p << " -> ";
  p.printSuccessor(op.successor());
}

static LogicalResult verify(ForEachOp op) {
  // Verify that the operation has exactly one argument.
  if (op.region().getNumArguments() != 1)
    return op.emitOpError("requires exactly one argument");

  // Verify that the loop variable and the operand (value range)
  // have compatible types.
  BlockArgument arg = op.getLoopVariable();
  Type rangeType = pdl::RangeType::get(arg.getType());
  if (rangeType != op.values().getType())
    return op.emitOpError("operand must be a range of loop variable type");

  return success();
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetValueTypeOp
//===----------------------------------------------------------------------===//

/// Given the result type of a `GetValueTypeOp`, return the expected input type.
static Type getGetValueTypeOpValueType(Type type) {
  Type valueTy = pdl::ValueType::get(type.getContext());
  return type.isa<pdl::RangeType>() ? pdl::RangeType::get(valueTy) : valueTy;
}

//===----------------------------------------------------------------------===//
// TableGen Auto-Generated Op and Interface Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.cpp.inc"

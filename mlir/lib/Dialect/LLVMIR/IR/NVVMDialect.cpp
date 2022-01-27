//===- NVVMDialect.cpp - NVVM IR Ops and Dialect registration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the NVVM IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The NVVM dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace NVVM;

#include "mlir/Dialect/LLVMIR/NVVMOpsDialect.cpp.inc"
#include "mlir/Dialect/LLVMIR/NVVMOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Printing/parsing for NVVM ops
//===----------------------------------------------------------------------===//

static void printNVVMIntrinsicOp(OpAsmPrinter &p, Operation *op) {
  p << " " << op->getOperands();
  if (op->getNumResults() > 0)
    p << " : " << op->getResultTypes();
}

// <operation> ::= `llvm.nvvm.vote.ballot.sync %mask, %pred` : result_type
static ParseResult parseNVVMVoteBallotOp(OpAsmParser &parser,
                                         OperationState &result) {
  MLIRContext *context = parser.getContext();
  auto int32Ty = IntegerType::get(context, 32);
  auto int1Ty = IntegerType::get(context, 1);

  SmallVector<OpAsmParser::OperandType, 8> ops;
  Type type;
  return failure(parser.parseOperandList(ops) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.addTypeToList(type, result.types) ||
                 parser.resolveOperands(ops, {int32Ty, int1Ty},
                                        parser.getNameLoc(), result.operands));
}

static LogicalResult verify(MmaOp op) {
  MLIRContext *context = op.getContext();
  auto f16Ty = Float16Type::get(context);
  auto f16x2Ty = LLVM::getFixedVectorType(f16Ty, 2);
  auto f32Ty = Float32Type::get(context);
  auto f16x2x4StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});
  auto f32x8StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty});

  SmallVector<Type, 12> operandTypes(op.getOperandTypes().begin(),
                                     op.getOperandTypes().end());
  if (operandTypes != SmallVector<Type, 8>(8, f16x2Ty) &&
      operandTypes != SmallVector<Type, 12>{f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty,
                                            f32Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                                            f32Ty, f32Ty, f32Ty}) {
    return op.emitOpError(
        "expected operands to be 4 <halfx2>s followed by either "
        "4 <halfx2>s or 8 floats");
  }
  if (op.getType() != f32x8StructTy && op.getType() != f16x2x4StructTy) {
    return op.emitOpError("expected result type to be a struct of either 4 "
                          "<halfx2>s or 8 floats");
  }

  auto alayout = op->getAttrOfType<StringAttr>("alayout");
  auto blayout = op->getAttrOfType<StringAttr>("blayout");

  if (!(alayout && blayout) ||
      !(alayout.getValue() == "row" || alayout.getValue() == "col") ||
      !(blayout.getValue() == "row" || blayout.getValue() == "col")) {
    return op.emitOpError(
        "alayout and blayout attributes must be set to either "
        "\"row\" or \"col\"");
  }

  if (operandTypes == SmallVector<Type, 12>{f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty,
                                            f32Ty, f32Ty, f32Ty, f32Ty, f32Ty,
                                            f32Ty, f32Ty, f32Ty} &&
      op.getType() == f32x8StructTy && alayout.getValue() == "row" &&
      blayout.getValue() == "col") {
    return success();
  }
  return op.emitOpError("unimplemented mma.sync variant");
}

std::pair<mlir::Type, unsigned>
inferMMAType(NVVM::MMATypes type, NVVM::MMAFrag frag, MLIRContext *context) {
  unsigned numberElements = 0;
  Type elementType;
  OpBuilder builder(context);
  Type f16x2 = VectorType::get(2, builder.getF16Type());
  if (type == NVVM::MMATypes::f16) {
    elementType = f16x2;
    if (frag == NVVM::MMAFrag::a || frag == NVVM::MMAFrag::b)
      numberElements = 8;
    else
      numberElements = 4;
  } else if (type == NVVM::MMATypes::f32) {
    elementType = builder.getF32Type();
    numberElements = 8;
  } else if (type == NVVM::MMATypes::tf32) {
    elementType = builder.getI32Type();
    numberElements = 4;
  }
  assert(numberElements != 0 && elementType != nullptr);
  return std::make_pair(elementType, numberElements);
}

static LogicalResult verify(NVVM::WMMALoadOp op) {
  unsigned addressSpace =
      op.ptr().getType().cast<LLVM::LLVMPointerType>().getAddressSpace();
  if (addressSpace != 0 && addressSpace != 1 && addressSpace != 3)
    return op.emitOpError("expected source pointer in memory "
                          "space 0, 1, 3");

  if (NVVM::WMMALoadOp::getIntrinsicID(op.m(), op.n(), op.k(), op.layout(),
                                       op.eltype(), op.frag()) == 0)
    return op.emitOpError() << "invalid attribute combination";
  std::pair<Type, unsigned> typeInfo =
      inferMMAType(op.eltype(), op.frag(), op.getContext());
  Type dstType = LLVM::LLVMStructType::getLiteral(
      op.getContext(), SmallVector<Type, 8>(typeInfo.second, typeInfo.first));
  if (op.getType() != dstType)
    return op.emitOpError("expected destination type is a structure of ")
           << typeInfo.second << " elements of type " << typeInfo.first;
  return success();
}

static LogicalResult verify(NVVM::WMMAStoreOp op) {
  unsigned addressSpace =
      op.ptr().getType().cast<LLVM::LLVMPointerType>().getAddressSpace();
  if (addressSpace != 0 && addressSpace != 1 && addressSpace != 3)
    return op.emitOpError("expected operands to be a source pointer in memory "
                          "space 0, 1, 3");

  if (NVVM::WMMAStoreOp::getIntrinsicID(op.m(), op.n(), op.k(), op.layout(),
                                        op.eltype()) == 0)
    return op.emitOpError() << "invalid attribute combination";
  std::pair<Type, unsigned> typeInfo =
      inferMMAType(op.eltype(), NVVM::MMAFrag::c, op.getContext());
  if (op.args().size() != typeInfo.second)
    return op.emitOpError()
           << "expected " << typeInfo.second << " data operands";
  if (llvm::any_of(op.args(), [&typeInfo](Value operands) {
        return operands.getType() != typeInfo.first;
      }))
    return op.emitOpError()
           << "expected data operands of type " << typeInfo.first;
  return success();
}

static LogicalResult verify(NVVM::WMMAMmaOp op) {
  if (NVVM::WMMAMmaOp::getIntrinsicID(op.m(), op.n(), op.k(), op.layoutA(),
                                      op.layoutB(), op.eltypeA(),
                                      op.eltypeB()) == 0)
    return op.emitOpError() << "invalid attribute combination";
  std::pair<Type, unsigned> typeInfoA =
      inferMMAType(op.eltypeA(), NVVM::MMAFrag::a, op.getContext());
  std::pair<Type, unsigned> typeInfoB =
      inferMMAType(op.eltypeA(), NVVM::MMAFrag::b, op.getContext());
  std::pair<Type, unsigned> typeInfoC =
      inferMMAType(op.eltypeB(), NVVM::MMAFrag::c, op.getContext());
  SmallVector<Type, 32> arguments;
  arguments.append(typeInfoA.second, typeInfoA.first);
  arguments.append(typeInfoB.second, typeInfoB.first);
  arguments.append(typeInfoC.second, typeInfoC.first);
  unsigned numArgs = arguments.size();
  if (op.args().size() != numArgs)
    return op.emitOpError() << "expected " << numArgs << " arguments";
  for (unsigned i = 0; i < numArgs; i++) {
    if (op.args()[i].getType() != arguments[i])
      return op.emitOpError()
             << "expected argument " << i << " to be of type " << arguments[i];
  }
  Type dstType = LLVM::LLVMStructType::getLiteral(
      op.getContext(), SmallVector<Type, 8>(typeInfoC.second, typeInfoC.first));
  if (op.getType() != dstType)
    return op.emitOpError("expected destination type is a structure of ")
           << typeInfoC.second << " elements of type " << typeInfoC.first;
  return success();
}

//===----------------------------------------------------------------------===//
// NVVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

// TODO: This should be the llvm.nvvm dialect once this is supported.
void NVVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/NVVMOpsAttributes.cpp.inc"
      >();

  // Support unknown operations because not all NVVM operations are
  // registered.
  allowUnknownOperations();
}

LogicalResult NVVMDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // Kernel function attribute should be attached to functions.
  if (attr.getName() == NVVMDialect::getKernelFuncAttrName()) {
    if (!isa<LLVM::LLVMFuncOp>(op)) {
      return op->emitError() << "'" << NVVMDialect::getKernelFuncAttrName()
                             << "' attribute attached to unexpected op";
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOpsAttributes.cpp.inc"

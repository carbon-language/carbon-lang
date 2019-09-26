// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FIR_FIROPS_H
#define FIR_FIROPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using llvm::ArrayRef;
using llvm::StringRef;

namespace fir {

class FirEndOp;

/// `fir.global` is a typed symbol with an optional list of initializers.
class GlobalOp
    : public mlir::Op<
          GlobalOp, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult,
          mlir::OpTrait::IsIsolatedFromAbove,
          mlir::OpTrait::SingleBlockImplicitTerminator<FirEndOp>::Impl> {
public:
  using Op::Op;
  using Op::print;

  static llvm::StringRef getOperationName() { return "fir.global"; }
  static llvm::StringRef getTypeAttrName() { return "type"; }

  static void build(mlir::Builder *builder, mlir::OperationState *result,
                    llvm::StringRef name, mlir::Type type,
                    llvm::ArrayRef<mlir::NamedAttribute> attrs);

  /// Operation hooks.
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &p);
  mlir::LogicalResult verify();

  mlir::Type getType() {
    return getAttrOfType<TypeAttr>(getTypeAttrName()).getValue();
  }

  void appendInitialValue(mlir::Operation *op);

private:
  mlir::Region &front();
};

/// `fir.dispatch_table` is an untyped symbol that is a list of associations
/// between method identifiers and a FuncOp symbol.
class DispatchTableOp
    : public mlir::Op<
          DispatchTableOp, mlir::OpTrait::ZeroOperands,
          mlir::OpTrait::ZeroResult, mlir::OpTrait::IsIsolatedFromAbove,
          mlir::OpTrait::SingleBlockImplicitTerminator<FirEndOp>::Impl> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.dispatch_table"; }

  static void build(mlir::Builder *builder, mlir::OperationState *result,
                    llvm::StringRef name, mlir::Type type,
                    llvm::ArrayRef<mlir::NamedAttribute> attrs);

  /// Operation hooks.
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &p);
  mlir::LogicalResult verify();

  void appendTableEntry(mlir::Operation *op);

private:
  mlir::Region &front();
};

mlir::ParseResult isValidCaseAttr(mlir::Attribute attr);
unsigned getCaseArgumentOffset(llvm::ArrayRef<mlir::Attribute> cases,
                               unsigned dest);
mlir::ParseResult parseSelector(mlir::OpAsmParser *parser,
                                mlir::OperationState *result,
                                mlir::OpAsmParser::OperandType &selector,
                                mlir::Type &type);

#define GET_OP_CLASSES
#include "fir/FIROps.h.inc"

LoopOp getForInductionVarOwner(mlir::Value *val);

} // namespace fir

#endif // FIR_FIROPS_H

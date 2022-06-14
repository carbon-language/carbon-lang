//===- TestTransformDialectExtension.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension of the MLIR Transform dialect for testing
// purposes.
//
//===----------------------------------------------------------------------===//

#include "TestTransformDialectExtension.h"
#include "TestTransformStateExtension.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

namespace {
/// Simple transform op defined outside of the dialect. Just emits a remark when
/// applied. This op is defined in C++ to test that C++ definitions also work
/// for op injection into the Transform dialect.
class TestTransformOp
    : public Op<TestTransformOp, transform::TransformOpInterface::Trait,
                MemoryEffectOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTransformOp)

  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("transform.test_transform_op");
  }

  DiagnosedSilencableFailure apply(transform::TransformResults &results,
                                   transform::TransformState &state) {
    InFlightDiagnostic remark = emitRemark() << "applying transformation";
    if (Attribute message = getMessage())
      remark << " " << message;

    return DiagnosedSilencableFailure::success();
  }

  Attribute getMessage() { return getOperation()->getAttr("message"); }

  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    StringAttr message;
    OptionalParseResult result = parser.parseOptionalAttribute(message);
    if (!result.hasValue())
      return success();

    if (result.getValue().succeeded())
      state.addAttribute("message", message);
    return result.getValue();
  }

  void print(OpAsmPrinter &printer) {
    if (getMessage())
      printer << " " << getMessage();
  }

  // No side effects.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}
};

/// A test op to exercise the verifier of the PossibleTopLevelTransformOpTrait
/// in cases where it is attached to ops that do not comply with the trait
/// requirements. This op cannot be defined in ODS because ODS generates strict
/// verifiers that overalp with those in the trait and run earlier.
class TestTransformUnrestrictedOpNoInterface
    : public Op<TestTransformUnrestrictedOpNoInterface,
                transform::PossibleTopLevelTransformOpTrait,
                transform::TransformOpInterface::Trait,
                MemoryEffectOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestTransformUnrestrictedOpNoInterface)

  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral(
        "transform.test_transform_unrestricted_op_no_interface");
  }

  DiagnosedSilencableFailure apply(transform::TransformResults &results,
                                   transform::TransformState &state) {
    return DiagnosedSilencableFailure::success();
  }

  // No side effects.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}
};
} // namespace

DiagnosedSilencableFailure
mlir::test::TestProduceParamOrForwardOperandOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  if (getOperation()->getNumOperands() != 0) {
    results.set(getResult().cast<OpResult>(),
                getOperation()->getOperand(0).getDefiningOp());
  } else {
    results.set(getResult().cast<OpResult>(),
                reinterpret_cast<Operation *>(*getParameter()));
  }
  return DiagnosedSilencableFailure::success();
}

LogicalResult mlir::test::TestProduceParamOrForwardOperandOp::verify() {
  if (getParameter().hasValue() ^ (getNumOperands() != 1))
    return emitOpError() << "expects either a parameter or an operand";
  return success();
}

DiagnosedSilencableFailure
mlir::test::TestConsumeOperand::apply(transform::TransformResults &results,
                                      transform::TransformState &state) {
  return DiagnosedSilencableFailure::success();
}

DiagnosedSilencableFailure
mlir::test::TestConsumeOperandIfMatchesParamOrFail::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  ArrayRef<Operation *> payload = state.getPayloadOps(getOperand());
  assert(payload.size() == 1 && "expected a single target op");
  auto value = reinterpret_cast<intptr_t>(payload[0]);
  if (static_cast<uint64_t>(value) != getParameter()) {
    return emitSilencableError()
           << "op expected the operand to be associated with " << getParameter()
           << " got " << value;
  }

  emitRemark() << "succeeded";
  return DiagnosedSilencableFailure::success();
}

DiagnosedSilencableFailure mlir::test::TestPrintRemarkAtOperandOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  ArrayRef<Operation *> payload = state.getPayloadOps(getOperand());
  for (Operation *op : payload)
    op->emitRemark() << getMessage();

  return DiagnosedSilencableFailure::success();
}

DiagnosedSilencableFailure
mlir::test::TestAddTestExtensionOp::apply(transform::TransformResults &results,
                                          transform::TransformState &state) {
  state.addExtension<TestTransformStateExtension>(getMessageAttr());
  return DiagnosedSilencableFailure::success();
}

DiagnosedSilencableFailure mlir::test::TestCheckIfTestExtensionPresentOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  auto *extension = state.getExtension<TestTransformStateExtension>();
  if (!extension) {
    emitRemark() << "extension absent";
    return DiagnosedSilencableFailure::success();
  }

  InFlightDiagnostic diag = emitRemark()
                            << "extension present, " << extension->getMessage();
  for (Operation *payload : state.getPayloadOps(getOperand())) {
    diag.attachNote(payload->getLoc()) << "associated payload op";
    assert(state.getHandleForPayloadOp(payload) == getOperand() &&
           "inconsistent mapping between transform IR handles and payload IR "
           "operations");
  }

  return DiagnosedSilencableFailure::success();
}

DiagnosedSilencableFailure mlir::test::TestRemapOperandPayloadToSelfOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  auto *extension = state.getExtension<TestTransformStateExtension>();
  if (!extension) {
    emitError() << "TestTransformStateExtension missing";
    return DiagnosedSilencableFailure::definiteFailure();
  }

  if (failed(extension->updateMapping(state.getPayloadOps(getOperand()).front(),
                                      getOperation())))
    return DiagnosedSilencableFailure::definiteFailure();
  return DiagnosedSilencableFailure::success();
}

DiagnosedSilencableFailure mlir::test::TestRemoveTestExtensionOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  state.removeExtension<TestTransformStateExtension>();
  return DiagnosedSilencableFailure::success();
}
DiagnosedSilencableFailure mlir::test::TestTransformOpWithRegions::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  return DiagnosedSilencableFailure::success();
}

void mlir::test::TestTransformOpWithRegions::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}

DiagnosedSilencableFailure
mlir::test::TestBranchingTransformOpTerminator::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  return DiagnosedSilencableFailure::success();
}

void mlir::test::TestBranchingTransformOpTerminator::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}

DiagnosedSilencableFailure mlir::test::TestEmitRemarkAndEraseOperandOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  emitRemark() << getRemark();
  for (Operation *op : state.getPayloadOps(getTarget()))
    op->erase();

  if (getFailAfterErase())
    return emitSilencableError() << "silencable error";
  return DiagnosedSilencableFailure::success();
}

namespace {
/// Test extension of the Transform dialect. Registers additional ops and
/// declares PDL as dependent dialect since the additional ops are using PDL
/// types for operands and results.
class TestTransformDialectExtension
    : public transform::TransformDialectExtension<
          TestTransformDialectExtension> {
public:
  TestTransformDialectExtension() {
    declareDependentDialect<pdl::PDLDialect>();
    registerTransformOps<TestTransformOp,
                         TestTransformUnrestrictedOpNoInterface,
#define GET_OP_LIST
#include "TestTransformDialectExtension.cpp.inc"
                         >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "TestTransformDialectExtension.cpp.inc"

void ::test::registerTestTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<TestTransformDialectExtension>();
}

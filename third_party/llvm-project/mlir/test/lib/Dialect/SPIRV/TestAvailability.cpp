//===- TestAvailability.cpp - Pass to test SPIR-V op availability ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Printing op availability pass
//===----------------------------------------------------------------------===//

namespace {
/// A pass for testing SPIR-V op availability.
struct PrintOpAvailability
    : public PassWrapper<PrintOpAvailability, OperationPass<FuncOp>> {
  void runOnOperation() override;
  StringRef getArgument() const final { return "test-spirv-op-availability"; }
  StringRef getDescription() const final {
    return "Test SPIR-V op availability";
  }
};
} // namespace

void PrintOpAvailability::runOnOperation() {
  auto f = getOperation();
  llvm::outs() << f.getName() << "\n";

  Dialect *spvDialect = getContext().getLoadedDialect("spv");

  f->walk([&](Operation *op) {
    if (op->getDialect() != spvDialect)
      return WalkResult::advance();

    auto opName = op->getName();
    auto &os = llvm::outs();

    if (auto minVersionIfx = dyn_cast<spirv::QueryMinVersionInterface>(op)) {
      Optional<spirv::Version> minVersion = minVersionIfx.getMinVersion();
      os << opName << " min version: ";
      if (minVersion)
        os << spirv::stringifyVersion(*minVersion) << "\n";
      else
        os << "None\n";
    }

    if (auto maxVersionIfx = dyn_cast<spirv::QueryMaxVersionInterface>(op)) {
      Optional<spirv::Version> maxVersion = maxVersionIfx.getMaxVersion();
      os << opName << " max version: ";
      if (maxVersion)
        os << spirv::stringifyVersion(*maxVersion) << "\n";
      else
        os << "None\n";
    }

    if (auto extension = dyn_cast<spirv::QueryExtensionInterface>(op)) {
      os << opName << " extensions: [";
      for (const auto &exts : extension.getExtensions()) {
        os << " [";
        llvm::interleaveComma(exts, os, [&](spirv::Extension ext) {
          os << spirv::stringifyExtension(ext);
        });
        os << "]";
      }
      os << " ]\n";
    }

    if (auto capability = dyn_cast<spirv::QueryCapabilityInterface>(op)) {
      os << opName << " capabilities: [";
      for (const auto &caps : capability.getCapabilities()) {
        os << " [";
        llvm::interleaveComma(caps, os, [&](spirv::Capability cap) {
          os << spirv::stringifyCapability(cap);
        });
        os << "]";
      }
      os << " ]\n";
    }
    os.flush();

    return WalkResult::advance();
  });
}

namespace mlir {
void registerPrintSpirvAvailabilityPass() {
  PassRegistration<PrintOpAvailability>();
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// Converting target environment pass
//===----------------------------------------------------------------------===//

namespace {
/// A pass for testing SPIR-V op availability.
struct ConvertToTargetEnv
    : public PassWrapper<ConvertToTargetEnv, OperationPass<FuncOp>> {
  StringRef getArgument() const override { return "test-spirv-target-env"; }
  StringRef getDescription() const override {
    return "Test SPIR-V target environment";
  }
  void runOnOperation() override;
};

struct ConvertToAtomCmpExchangeWeak : public RewritePattern {
  ConvertToAtomCmpExchangeWeak(MLIRContext *context);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

struct ConvertToBitReverse : public RewritePattern {
  ConvertToBitReverse(MLIRContext *context);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

struct ConvertToGroupNonUniformBallot : public RewritePattern {
  ConvertToGroupNonUniformBallot(MLIRContext *context);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

struct ConvertToModule : public RewritePattern {
  ConvertToModule(MLIRContext *context);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

struct ConvertToSubgroupBallot : public RewritePattern {
  ConvertToSubgroupBallot(MLIRContext *context);
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

void ConvertToTargetEnv::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp fn = getOperation();

  auto targetEnv = fn.getOperation()
                       ->getAttr(spirv::getTargetEnvAttrName())
                       .cast<spirv::TargetEnvAttr>();
  if (!targetEnv) {
    fn.emitError("missing 'spv.target_env' attribute");
    return signalPassFailure();
  }

  auto target = SPIRVConversionTarget::get(targetEnv);

  RewritePatternSet patterns(context);
  patterns.add<ConvertToAtomCmpExchangeWeak, ConvertToBitReverse,
               ConvertToGroupNonUniformBallot, ConvertToModule,
               ConvertToSubgroupBallot>(context);

  if (failed(applyPartialConversion(fn, *target, std::move(patterns))))
    return signalPassFailure();
}

ConvertToAtomCmpExchangeWeak::ConvertToAtomCmpExchangeWeak(MLIRContext *context)
    : RewritePattern("test.convert_to_atomic_compare_exchange_weak_op", 1,
                     context, {"spv.AtomicCompareExchangeWeak"}) {}

LogicalResult
ConvertToAtomCmpExchangeWeak::matchAndRewrite(Operation *op,
                                              PatternRewriter &rewriter) const {
  Value ptr = op->getOperand(0);
  Value value = op->getOperand(1);
  Value comparator = op->getOperand(2);

  // Create a spv.AtomicCompareExchangeWeak op with AtomicCounterMemory bits in
  // memory semantics to additionally require AtomicStorage capability.
  rewriter.replaceOpWithNewOp<spirv::AtomicCompareExchangeWeakOp>(
      op, value.getType(), ptr, spirv::Scope::Workgroup,
      spirv::MemorySemantics::AcquireRelease |
          spirv::MemorySemantics::AtomicCounterMemory,
      spirv::MemorySemantics::Acquire, value, comparator);
  return success();
}

ConvertToBitReverse::ConvertToBitReverse(MLIRContext *context)
    : RewritePattern("test.convert_to_bit_reverse_op", 1, context,
                     {"spv.BitReverse"}) {}

LogicalResult
ConvertToBitReverse::matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const {
  Value predicate = op->getOperand(0);

  rewriter.replaceOpWithNewOp<spirv::BitReverseOp>(
      op, op->getResult(0).getType(), predicate);
  return success();
}

ConvertToGroupNonUniformBallot::ConvertToGroupNonUniformBallot(
    MLIRContext *context)
    : RewritePattern("test.convert_to_group_non_uniform_ballot_op", 1, context,
                     {"spv.GroupNonUniformBallot"}) {}

LogicalResult ConvertToGroupNonUniformBallot::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  Value predicate = op->getOperand(0);

  rewriter.replaceOpWithNewOp<spirv::GroupNonUniformBallotOp>(
      op, op->getResult(0).getType(), spirv::Scope::Workgroup, predicate);
  return success();
}

ConvertToModule::ConvertToModule(MLIRContext *context)
    : RewritePattern("test.convert_to_module_op", 1, context, {"spv.module"}) {}

LogicalResult
ConvertToModule::matchAndRewrite(Operation *op,
                                 PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<spirv::ModuleOp>(
      op, spirv::AddressingModel::PhysicalStorageBuffer64,
      spirv::MemoryModel::Vulkan);
  return success();
}

ConvertToSubgroupBallot::ConvertToSubgroupBallot(MLIRContext *context)
    : RewritePattern("test.convert_to_subgroup_ballot_op", 1, context,
                     {"spv.SubgroupBallotKHR"}) {}

LogicalResult
ConvertToSubgroupBallot::matchAndRewrite(Operation *op,
                                         PatternRewriter &rewriter) const {
  Value predicate = op->getOperand(0);

  rewriter.replaceOpWithNewOp<spirv::SubgroupBallotKHROp>(
      op, op->getResult(0).getType(), predicate);
  return success();
}

namespace mlir {
void registerConvertToTargetEnvPass() {
  PassRegistration<ConvertToTargetEnv>();
}
} // namespace mlir

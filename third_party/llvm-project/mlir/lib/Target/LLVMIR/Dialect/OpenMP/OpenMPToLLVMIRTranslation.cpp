//===- OpenMPToLLVMIRTranslation.cpp - Translate OpenMP dialect to LLVM IR-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR OpenMP dialect and LLVM
// IR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/IRBuilder.h"

using namespace mlir;

namespace {
/// ModuleTranslation stack frame for OpenMP operations. This keeps track of the
/// insertion points for allocas.
class OpenMPAllocaStackFrame
    : public LLVM::ModuleTranslation::StackFrameBase<OpenMPAllocaStackFrame> {
public:
  explicit OpenMPAllocaStackFrame(llvm::OpenMPIRBuilder::InsertPointTy allocaIP)
      : allocaInsertPoint(allocaIP) {}
  llvm::OpenMPIRBuilder::InsertPointTy allocaInsertPoint;
};
} // namespace

/// Find the insertion point for allocas given the current insertion point for
/// normal operations in the builder.
static llvm::OpenMPIRBuilder::InsertPointTy
findAllocaInsertPoint(llvm::IRBuilderBase &builder,
                      const LLVM::ModuleTranslation &moduleTranslation) {
  // If there is an alloca insertion point on stack, i.e. we are in a nested
  // operation and a specific point was provided by some surrounding operation,
  // use it.
  llvm::OpenMPIRBuilder::InsertPointTy allocaInsertPoint;
  WalkResult walkResult = moduleTranslation.stackWalk<OpenMPAllocaStackFrame>(
      [&](const OpenMPAllocaStackFrame &frame) {
        allocaInsertPoint = frame.allocaInsertPoint;
        return WalkResult::interrupt();
      });
  if (walkResult.wasInterrupted())
    return allocaInsertPoint;

  // Otherwise, insert to the entry block of the surrounding function.
  llvm::BasicBlock &funcEntryBlock =
      builder.GetInsertBlock()->getParent()->getEntryBlock();
  return llvm::OpenMPIRBuilder::InsertPointTy(
      &funcEntryBlock, funcEntryBlock.getFirstInsertionPt());
}

/// Converts the given region that appears within an OpenMP dialect operation to
/// LLVM IR, creating a branch from the `sourceBlock` to the entry block of the
/// region, and a branch from any block with an successor-less OpenMP terminator
/// to `continuationBlock`.
static void convertOmpOpRegions(Region &region, StringRef blockName,
                                llvm::BasicBlock &sourceBlock,
                                llvm::BasicBlock &continuationBlock,
                                llvm::IRBuilderBase &builder,
                                LLVM::ModuleTranslation &moduleTranslation,
                                LogicalResult &bodyGenStatus) {
  llvm::LLVMContext &llvmContext = builder.getContext();
  for (Block &bb : region) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        llvmContext, blockName, builder.GetInsertBlock()->getParent());
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  llvm::Instruction *sourceTerminator = sourceBlock.getTerminator();

  // Convert blocks one by one in topological order to ensure
  // defs are converted before uses.
  SetVector<Block *> blocks =
      LLVM::detail::getTopologicallySortedBlocks(region);
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
    // Retarget the branch of the entry block to the entry block of the
    // converted region (regions are single-entry).
    if (bb->isEntryBlock()) {
      assert(sourceTerminator->getNumSuccessors() == 1 &&
             "provided entry block has multiple successors");
      assert(sourceTerminator->getSuccessor(0) == &continuationBlock &&
             "ContinuationBlock is not the successor of the entry block");
      sourceTerminator->setSuccessor(0, llvmBB);
    }

    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    if (failed(
            moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder))) {
      bodyGenStatus = failure();
      return;
    }

    // Special handling for `omp.yield` and `omp.terminator` (we may have more
    // than one): they return the control to the parent OpenMP dialect operation
    // so replace them with the branch to the continuation block. We handle this
    // here to avoid relying inter-function communication through the
    // ModuleTranslation class to set up the correct insertion point. This is
    // also consistent with MLIR's idiom of handling special region terminators
    // in the same code that handles the region-owning operation.
    if (isa<omp::TerminatorOp, omp::YieldOp>(bb->getTerminator()))
      builder.CreateBr(&continuationBlock);
  }
  // Finally, after all blocks have been traversed and values mapped,
  // connect the PHI nodes to the results of preceding blocks.
  LLVM::detail::connectPHINodes(region, moduleTranslation);
}

/// Converts the OpenMP parallel operation to LLVM IR.
static LogicalResult
convertOmpParallel(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                       llvm::BasicBlock &continuationBlock) {
    // Save the alloca insertion point on ModuleTranslation stack for use in
    // nested regions.
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);

    // ParallelOp has only one region associated with it.
    auto &region = cast<omp::ParallelOp>(opInst).getRegion();
    convertOmpOpRegions(region, "omp.par.region", *codeGenIP.getBlock(),
                        continuationBlock, builder, moduleTranslation,
                        bodyGenStatus);
  };

  // TODO: Perform appropriate actions according to the data-sharing
  // attribute (shared, private, firstprivate, ...) of variables.
  // Currently defaults to shared.
  auto privCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                    llvm::Value &, llvm::Value &vPtr,
                    llvm::Value *&replacementValue) -> InsertPointTy {
    replacementValue = &vPtr;

    return codeGenIP;
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::Value *ifCond = nullptr;
  if (auto ifExprVar = cast<omp::ParallelOp>(opInst).if_expr_var())
    ifCond = moduleTranslation.lookupValue(ifExprVar);
  llvm::Value *numThreads = nullptr;
  if (auto numThreadsVar = cast<omp::ParallelOp>(opInst).num_threads_var())
    numThreads = moduleTranslation.lookupValue(numThreadsVar);
  llvm::omp::ProcBindKind pbKind = llvm::omp::OMP_PROC_BIND_default;
  if (auto bind = cast<omp::ParallelOp>(opInst).proc_bind_val())
    pbKind = llvm::omp::getProcBindKind(bind.getValue());
  // TODO: Is the Parallel construct cancellable?
  bool isCancellable = false;

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createParallel(
      ompLoc, findAllocaInsertPoint(builder, moduleTranslation), bodyGenCB,
      privCB, finiCB, ifCond, numThreads, pbKind, isCancellable));

  return bodyGenStatus;
}

/// Converts an OpenMP 'master' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpMaster(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                       llvm::BasicBlock &continuationBlock) {
    // MasterOp has only one region associated with it.
    auto &region = cast<omp::MasterOp>(opInst).getRegion();
    convertOmpOpRegions(region, "omp.master.region", *codeGenIP.getBlock(),
                        continuationBlock, builder, moduleTranslation,
                        bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createMaster(
      ompLoc, bodyGenCB, finiCB));
  return success();
}

/// Converts an OpenMP 'critical' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpCritical(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto criticalOp = cast<omp::CriticalOp>(opInst);
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                       llvm::BasicBlock &continuationBlock) {
    // CriticalOp has only one region associated with it.
    auto &region = cast<omp::CriticalOp>(opInst).getRegion();
    convertOmpOpRegions(region, "omp.critical.region", *codeGenIP.getBlock(),
                        continuationBlock, builder, moduleTranslation,
                        bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
  llvm::Constant *hint = nullptr;
  if (criticalOp.hint().hasValue()) {
    hint =
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext),
                               static_cast<int>(criticalOp.hint().getValue()));
  } else {
    hint = llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext), 0);
  }
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createCritical(
      ompLoc, bodyGenCB, finiCB, criticalOp.name().getValueOr(""), hint));
  return success();
}

/// Converts an OpenMP workshare loop into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpWsLoop(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  auto loop = cast<omp::WsLoopOp>(opInst);
  // TODO: this should be in the op verifier instead.
  if (loop.lowerBound().empty())
    return failure();

  // Static is the default.
  omp::ClauseScheduleKind schedule = omp::ClauseScheduleKind::Static;
  if (loop.schedule_val().hasValue())
    schedule =
        *omp::symbolizeClauseScheduleKind(loop.schedule_val().getValue());

  // Set up the source location value for OpenMP runtime.
  llvm::DISubprogram *subprogram =
      builder.GetInsertBlock()->getParent()->getSubprogram();
  const llvm::DILocation *diLoc =
      moduleTranslation.translateLoc(opInst.getLoc(), subprogram);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder.saveIP(),
                                                    llvm::DebugLoc(diLoc));

  // Generator of the canonical loop body.
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  SmallVector<llvm::CanonicalLoopInfo *> loopInfos;
  SmallVector<llvm::OpenMPIRBuilder::InsertPointTy> bodyInsertPoints;
  LogicalResult bodyGenStatus = success();
  auto bodyGen = [&](llvm::OpenMPIRBuilder::InsertPointTy ip, llvm::Value *iv) {
    // Make sure further conversions know about the induction variable.
    moduleTranslation.mapValue(
        loop.getRegion().front().getArgument(loopInfos.size()), iv);

    // Capture the body insertion point for use in nested loops. BodyIP of the
    // CanonicalLoopInfo always points to the beginning of the entry block of
    // the body.
    bodyInsertPoints.push_back(ip);

    if (loopInfos.size() != loop.getNumLoops() - 1)
      return;

    // Convert the body of the loop.
    llvm::BasicBlock *entryBlock = ip.getBlock();
    llvm::BasicBlock *exitBlock =
        entryBlock->splitBasicBlock(ip.getPoint(), "omp.wsloop.exit");
    convertOmpOpRegions(loop.region(), "omp.wsloop.region", *entryBlock,
                        *exitBlock, builder, moduleTranslation, bodyGenStatus);
  };

  // Delegate actual loop construction to the OpenMP IRBuilder.
  // TODO: this currently assumes WsLoop is semantically similar to SCF loop,
  // i.e. it has a positive step, uses signed integer semantics. Reconsider
  // this code when WsLoop clearly supports more cases.
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  for (unsigned i = 0, e = loop.getNumLoops(); i < e; ++i) {
    llvm::Value *lowerBound =
        moduleTranslation.lookupValue(loop.lowerBound()[i]);
    llvm::Value *upperBound =
        moduleTranslation.lookupValue(loop.upperBound()[i]);
    llvm::Value *step = moduleTranslation.lookupValue(loop.step()[i]);

    // Make sure loop trip count are emitted in the preheader of the outermost
    // loop at the latest so that they are all available for the new collapsed
    // loop will be created below.
    llvm::OpenMPIRBuilder::LocationDescription loc = ompLoc;
    llvm::OpenMPIRBuilder::InsertPointTy computeIP = ompLoc.IP;
    if (i != 0) {
      loc = llvm::OpenMPIRBuilder::LocationDescription(bodyInsertPoints.back(),
                                                       llvm::DebugLoc(diLoc));
      computeIP = loopInfos.front()->getPreheaderIP();
    }
    loopInfos.push_back(ompBuilder->createCanonicalLoop(
        loc, bodyGen, lowerBound, upperBound, step,
        /*IsSigned=*/true, loop.inclusive(), computeIP));

    if (failed(bodyGenStatus))
      return failure();
  }

  // Collapse loops. Store the insertion point because LoopInfos may get
  // invalidated.
  llvm::IRBuilderBase::InsertPoint afterIP = loopInfos.front()->getAfterIP();
  llvm::CanonicalLoopInfo *loopInfo =
      ompBuilder->collapseLoops(diLoc, loopInfos, {});

  // Find the loop configuration.
  llvm::Type *ivType = loopInfo->getIndVar()->getType();
  llvm::Value *chunk =
      loop.schedule_chunk_var()
          ? moduleTranslation.lookupValue(loop.schedule_chunk_var())
          : llvm::ConstantInt::get(ivType, 1);
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  if (schedule == omp::ClauseScheduleKind::Static) {
    ompBuilder->applyStaticWorkshareLoop(ompLoc.DL, loopInfo, allocaIP,
                                         !loop.nowait(), chunk);
  } else {
    llvm::omp::OMPScheduleType schedType;
    switch (schedule) {
    case omp::ClauseScheduleKind::Dynamic:
      schedType = llvm::omp::OMPScheduleType::DynamicChunked;
      break;
    case omp::ClauseScheduleKind::Guided:
      schedType = llvm::omp::OMPScheduleType::GuidedChunked;
      break;
    case omp::ClauseScheduleKind::Auto:
      schedType = llvm::omp::OMPScheduleType::Auto;
      break;
    case omp::ClauseScheduleKind::Runtime:
      schedType = llvm::omp::OMPScheduleType::Runtime;
      break;
    default:
      llvm_unreachable("Unknown schedule value");
      break;
    }

    ompBuilder->applyDynamicWorkshareLoop(ompLoc.DL, loopInfo, allocaIP,
                                          schedType, !loop.nowait(), chunk);
  }

  // Continue building IR after the loop. Note that the LoopInfo returned by
  // `collapseLoops` points inside the outermost loop and is intended for
  // potential further loop transformations. Use the insertion point stored
  // before collapsing loops instead.
  builder.restoreIP(afterIP);
  return success();
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the OpenMP dialect to LLVM IR.
class OpenMPDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // end namespace

/// Given an OpenMP MLIR operation, create the corresponding LLVM IR
/// (including OpenMP runtime calls).
LogicalResult OpenMPDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](omp::BarrierOp) {
        ompBuilder->createBarrier(builder.saveIP(), llvm::omp::OMPD_barrier);
        return success();
      })
      .Case([&](omp::TaskwaitOp) {
        ompBuilder->createTaskwait(builder.saveIP());
        return success();
      })
      .Case([&](omp::TaskyieldOp) {
        ompBuilder->createTaskyield(builder.saveIP());
        return success();
      })
      .Case([&](omp::FlushOp) {
        // No support in Openmp runtime function (__kmpc_flush) to accept
        // the argument list.
        // OpenMP standard states the following:
        //  "An implementation may implement a flush with a list by ignoring
        //   the list, and treating it the same as a flush without a list."
        //
        // The argument list is discarded so that, flush with a list is treated
        // same as a flush without a list.
        ompBuilder->createFlush(builder.saveIP());
        return success();
      })
      .Case([&](omp::ParallelOp) {
        return convertOmpParallel(*op, builder, moduleTranslation);
      })
      .Case([&](omp::MasterOp) {
        return convertOmpMaster(*op, builder, moduleTranslation);
      })
      .Case([&](omp::CriticalOp) {
        return convertOmpCritical(*op, builder, moduleTranslation);
      })
      .Case([&](omp::WsLoopOp) {
        return convertOmpWsLoop(*op, builder, moduleTranslation);
      })
      .Case<omp::YieldOp, omp::TerminatorOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure was
        // created in the function that handles their parent operation.
        assert(op->getNumOperands() == 0 &&
               "unexpected OpenMP terminator with operands");
        return success();
      })
      .Default([&](Operation *inst) {
        return inst->emitError("unsupported OpenMP operation: ")
               << inst->getName();
      });
}

void mlir::registerOpenMPDialectTranslation(DialectRegistry &registry) {
  registry.insert<omp::OpenMPDialect>();
  registry.addDialectInterface<omp::OpenMPDialect,
                               OpenMPDialectLLVMIRTranslationInterface>();
}

void mlir::registerOpenMPDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerOpenMPDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}

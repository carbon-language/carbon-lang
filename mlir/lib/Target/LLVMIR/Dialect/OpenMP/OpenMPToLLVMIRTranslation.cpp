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
#include "mlir/IR/BlockAndValueMapping.h"
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

/// ModuleTranslation stack frame containing the partial mapping between MLIR
/// values and their LLVM IR equivalents.
class OpenMPVarMappingStackFrame
    : public LLVM::ModuleTranslation::StackFrameBase<
          OpenMPVarMappingStackFrame> {
public:
  explicit OpenMPVarMappingStackFrame(
      const DenseMap<Value, llvm::Value *> &mapping)
      : mapping(mapping) {}

  DenseMap<Value, llvm::Value *> mapping;
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
/// to `continuationBlock`. Populates `continuationBlockPHIs` with the PHI nodes
/// of the continuation block if provided.
static void convertOmpOpRegions(
    Region &region, StringRef blockName, llvm::BasicBlock &sourceBlock,
    llvm::BasicBlock &continuationBlock, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation, LogicalResult &bodyGenStatus,
    SmallVectorImpl<llvm::PHINode *> *continuationBlockPHIs = nullptr) {
  llvm::LLVMContext &llvmContext = builder.getContext();
  for (Block &bb : region) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        llvmContext, blockName, builder.GetInsertBlock()->getParent(),
        builder.GetInsertBlock()->getNextNode());
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  llvm::Instruction *sourceTerminator = sourceBlock.getTerminator();

  // Terminators (namely YieldOp) may be forwarding values to the region that
  // need to be available in the continuation block. Collect the types of these
  // operands in preparation of creating PHI nodes.
  SmallVector<llvm::Type *> continuationBlockPHITypes;
  bool operandsProcessed = false;
  unsigned numYields = 0;
  for (Block &bb : region.getBlocks()) {
    if (omp::YieldOp yield = dyn_cast<omp::YieldOp>(bb.getTerminator())) {
      if (!operandsProcessed) {
        for (unsigned i = 0, e = yield->getNumOperands(); i < e; ++i) {
          continuationBlockPHITypes.push_back(
              moduleTranslation.convertType(yield->getOperand(i).getType()));
        }
        operandsProcessed = true;
      } else {
        assert(continuationBlockPHITypes.size() == yield->getNumOperands() &&
               "mismatching number of values yielded from the region");
        for (unsigned i = 0, e = yield->getNumOperands(); i < e; ++i) {
          llvm::Type *operandType =
              moduleTranslation.convertType(yield->getOperand(i).getType());
          (void)operandType;
          assert(continuationBlockPHITypes[i] == operandType &&
                 "values of mismatching types yielded from the region");
        }
      }
      numYields++;
    }
  }

  // Insert PHI nodes in the continuation block for any values forwarded by the
  // terminators in this region.
  if (!continuationBlockPHITypes.empty())
    assert(
        continuationBlockPHIs &&
        "expected continuation block PHIs if converted regions yield values");
  if (continuationBlockPHIs) {
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    continuationBlockPHIs->reserve(continuationBlockPHITypes.size());
    builder.SetInsertPoint(&continuationBlock, continuationBlock.begin());
    for (llvm::Type *ty : continuationBlockPHITypes)
      continuationBlockPHIs->push_back(builder.CreatePHI(ty, numYields));
  }

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
    Operation *terminator = bb->getTerminator();
    if (isa<omp::TerminatorOp, omp::YieldOp>(terminator)) {
      builder.CreateBr(&continuationBlock);

      for (unsigned i = 0, e = terminator->getNumOperands(); i < e; ++i)
        (*continuationBlockPHIs)[i]->addIncoming(
            moduleTranslation.lookupValue(terminator->getOperand(i)), llvmBB);
    }
  }
  // After all blocks have been traversed and values mapped, connect the PHI
  // nodes to the results of preceding blocks.
  LLVM::detail::connectPHINodes(region, moduleTranslation);

  // Remove the blocks and values defined in this region from the mapping since
  // they are not visible outside of this region. This allows the same region to
  // be converted several times, that is cloned, without clashes, and slightly
  // speeds up the lookups.
  moduleTranslation.forgetMapping(region);
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

  // If it has a name, it probably has a hint too.
  if (criticalOp.nameAttr()) {
    // The verifiers in OpenMP Dialect guarentee that all the pointers are
    // non-null
    auto symbolRef = criticalOp.nameAttr().cast<SymbolRefAttr>();
    auto criticalDeclareOp =
        SymbolTable::lookupNearestSymbolFrom<omp::CriticalDeclareOp>(criticalOp,
                                                                     symbolRef);
    hint = llvm::ConstantInt::get(llvm::Type::getInt32Ty(llvmContext),
                                  static_cast<int>(criticalDeclareOp.hint()));
  }
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createCritical(
      ompLoc, bodyGenCB, finiCB, criticalOp.name().getValueOr(""), hint));
  return success();
}

/// Returns a reduction declaration that corresponds to the given reduction
/// operation in the given container. Currently only supports reductions inside
/// WsLoopOp but can be easily extended.
static omp::ReductionDeclareOp findReductionDecl(omp::WsLoopOp container,
                                                 omp::ReductionOp reduction) {
  SymbolRefAttr reductionSymbol;
  for (unsigned i = 0, e = container.getNumReductionVars(); i < e; ++i) {
    if (container.reduction_vars()[i] != reduction.accumulator())
      continue;
    reductionSymbol = (*container.reductions())[i].cast<SymbolRefAttr>();
    break;
  }
  assert(reductionSymbol &&
         "reduction operation must be associated with a declaration");

  return SymbolTable::lookupNearestSymbolFrom<omp::ReductionDeclareOp>(
      container, reductionSymbol);
}

/// Populates `reductions` with reduction declarations used in the given loop.
static void
collectReductionDecls(omp::WsLoopOp loop,
                      SmallVectorImpl<omp::ReductionDeclareOp> &reductions) {
  Optional<ArrayAttr> attr = loop.reductions();
  if (!attr)
    return;

  reductions.reserve(reductions.size() + loop.getNumReductionVars());
  for (auto symbolRef : attr->getAsRange<SymbolRefAttr>()) {
    reductions.push_back(
        SymbolTable::lookupNearestSymbolFrom<omp::ReductionDeclareOp>(
            loop, symbolRef));
  }
}

/// Translates the blocks contained in the given region and appends them to at
/// the current insertion point of `builder`. The operations of the entry block
/// are appended to the current insertion block, which is not expected to have a
/// terminator. If set, `continuationBlockArgs` is populated with translated
/// values that correspond to the values omp.yield'ed from the region.
static LogicalResult inlineConvertOmpRegions(
    Region &region, StringRef blockName, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    SmallVectorImpl<llvm::Value *> *continuationBlockArgs = nullptr) {
  if (region.empty())
    return success();

  // Special case for single-block regions that don't create additional blocks:
  // insert operations without creating additional blocks.
  if (llvm::hasSingleElement(region)) {
    moduleTranslation.mapBlock(&region.front(), builder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(
            region.front(), /*ignoreArguments=*/true, builder)))
      return failure();

    // The continuation arguments are simply the translated terminator operands.
    if (continuationBlockArgs)
      llvm::append_range(
          *continuationBlockArgs,
          moduleTranslation.lookupValues(region.front().back().getOperands()));

    // Drop the mapping that is no longer necessary so that the same region can
    // be processed multiple times.
    moduleTranslation.forgetMapping(region);
    return success();
  }

  // Create the continuation block manually instead of calling splitBlock
  // because the current insertion block may not have a terminator.
  llvm::BasicBlock *continuationBlock =
      llvm::BasicBlock::Create(builder.getContext(), blockName + ".cont",
                               builder.GetInsertBlock()->getParent(),
                               builder.GetInsertBlock()->getNextNode());
  builder.CreateBr(continuationBlock);

  LogicalResult bodyGenStatus = success();
  SmallVector<llvm::PHINode *> phis;
  convertOmpOpRegions(region, blockName, *builder.GetInsertBlock(),
                      *continuationBlock, builder, moduleTranslation,
                      bodyGenStatus, &phis);
  if (failed(bodyGenStatus))
    return failure();
  if (continuationBlockArgs)
    llvm::append_range(*continuationBlockArgs, phis);
  builder.SetInsertPoint(continuationBlock,
                         continuationBlock->getFirstInsertionPt());
  return success();
}

namespace {
/// Owning equivalents of OpenMPIRBuilder::(Atomic)ReductionGen that are used to
/// store lambdas with capture.
using OwningReductionGen = std::function<llvm::OpenMPIRBuilder::InsertPointTy(
    llvm::OpenMPIRBuilder::InsertPointTy, llvm::Value *, llvm::Value *,
    llvm::Value *&)>;
using OwningAtomicReductionGen =
    std::function<llvm::OpenMPIRBuilder::InsertPointTy(
        llvm::OpenMPIRBuilder::InsertPointTy, llvm::Type *, llvm::Value *,
        llvm::Value *)>;
} // namespace

/// Create an OpenMPIRBuilder-compatible reduction generator for the given
/// reduction declaration. The generator uses `builder` but ignores its
/// insertion point.
static OwningReductionGen
makeReductionGen(omp::ReductionDeclareOp decl, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  // The lambda is mutable because we need access to non-const methods of decl
  // (which aren't actually mutating it), and we must capture decl by-value to
  // avoid the dangling reference after the parent function returns.
  OwningReductionGen gen =
      [&, decl](llvm::OpenMPIRBuilder::InsertPointTy insertPoint,
                llvm::Value *lhs, llvm::Value *rhs,
                llvm::Value *&result) mutable {
        Region &reductionRegion = decl.reductionRegion();
        moduleTranslation.mapValue(reductionRegion.front().getArgument(0), lhs);
        moduleTranslation.mapValue(reductionRegion.front().getArgument(1), rhs);
        builder.restoreIP(insertPoint);
        SmallVector<llvm::Value *> phis;
        if (failed(inlineConvertOmpRegions(reductionRegion,
                                           "omp.reduction.nonatomic.body",
                                           builder, moduleTranslation, &phis)))
          return llvm::OpenMPIRBuilder::InsertPointTy();
        assert(phis.size() == 1);
        result = phis[0];
        return builder.saveIP();
      };
  return gen;
}

/// Create an OpenMPIRBuilder-compatible atomic reduction generator for the
/// given reduction declaration. The generator uses `builder` but ignores its
/// insertion point. Returns null if there is no atomic region available in the
/// reduction declaration.
static OwningAtomicReductionGen
makeAtomicReductionGen(omp::ReductionDeclareOp decl,
                       llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  if (decl.atomicReductionRegion().empty())
    return OwningAtomicReductionGen();

  // The lambda is mutable because we need access to non-const methods of decl
  // (which aren't actually mutating it), and we must capture decl by-value to
  // avoid the dangling reference after the parent function returns.
  OwningAtomicReductionGen atomicGen =
      [&, decl](llvm::OpenMPIRBuilder::InsertPointTy insertPoint, llvm::Type *,
                llvm::Value *lhs, llvm::Value *rhs) mutable {
        Region &atomicRegion = decl.atomicReductionRegion();
        moduleTranslation.mapValue(atomicRegion.front().getArgument(0), lhs);
        moduleTranslation.mapValue(atomicRegion.front().getArgument(1), rhs);
        builder.restoreIP(insertPoint);
        SmallVector<llvm::Value *> phis;
        if (failed(inlineConvertOmpRegions(atomicRegion,
                                           "omp.reduction.atomic.body", builder,
                                           moduleTranslation, &phis)))
          return llvm::OpenMPIRBuilder::InsertPointTy();
        assert(phis.empty());
        return builder.saveIP();
      };
  return atomicGen;
}

/// Converts an OpenMP 'ordered' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpOrdered(Operation &opInst, llvm::IRBuilderBase &builder,
                  LLVM::ModuleTranslation &moduleTranslation) {
  auto orderedOp = cast<omp::OrderedOp>(opInst);

  omp::ClauseDepend dependType =
      *omp::symbolizeClauseDepend(orderedOp.depend_type_valAttr().getValue());
  bool isDependSource = dependType == omp::ClauseDepend::dependsource;
  unsigned numLoops = orderedOp.num_loops_val().getValue();
  SmallVector<llvm::Value *> vecValues =
      moduleTranslation.lookupValues(orderedOp.depend_vec_vars());

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  size_t indexVecValues = 0;
  while (indexVecValues < vecValues.size()) {
    SmallVector<llvm::Value *> storeValues;
    storeValues.reserve(numLoops);
    for (unsigned i = 0; i < numLoops; i++) {
      storeValues.push_back(vecValues[indexVecValues]);
      indexVecValues++;
    }
    builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createOrderedDepend(
        ompLoc, findAllocaInsertPoint(builder, moduleTranslation), numLoops,
        storeValues, ".cnt.addr", isDependSource));
  }
  return success();
}

/// Converts an OpenMP 'ordered_region' operation into LLVM IR using
/// OpenMPIRBuilder.
static LogicalResult
convertOmpOrderedRegion(Operation &opInst, llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto orderedRegionOp = cast<omp::OrderedRegionOp>(opInst);

  // TODO: The code generation for ordered simd directive is not supported yet.
  if (orderedRegionOp.simd())
    return failure();

  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                       llvm::BasicBlock &continuationBlock) {
    // OrderedOp has only one region associated with it.
    auto &region = cast<omp::OrderedRegionOp>(opInst).getRegion();
    convertOmpOpRegions(region, "omp.ordered.region", *codeGenIP.getBlock(),
                        continuationBlock, builder, moduleTranslation,
                        bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(
      moduleTranslation.getOpenMPBuilder()->createOrderedThreadsSimd(
          ompLoc, bodyGenCB, finiCB, !orderedRegionOp.simd()));
  return bodyGenStatus;
}

static LogicalResult
convertOmpSections(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  using StorableBodyGenCallbackTy =
      llvm::OpenMPIRBuilder::StorableBodyGenCallbackTy;

  auto sectionsOp = cast<omp::SectionsOp>(opInst);

  // TODO: Support the following clauses: private, firstprivate, lastprivate,
  // reduction, allocate
  if (!sectionsOp.private_vars().empty() ||
      !sectionsOp.firstprivate_vars().empty() ||
      !sectionsOp.lastprivate_vars().empty() ||
      !sectionsOp.reduction_vars().empty() || sectionsOp.reductions() ||
      !sectionsOp.allocate_vars().empty() ||
      !sectionsOp.allocators_vars().empty())
    return emitError(sectionsOp.getLoc())
           << "private, firstprivate, lastprivate, reduction and allocate "
              "clauses are not supported for sections construct";

  LogicalResult bodyGenStatus = success();
  SmallVector<StorableBodyGenCallbackTy> sectionCBs;

  for (Operation &op : *sectionsOp.region().begin()) {
    auto sectionOp = dyn_cast<omp::SectionOp>(op);
    if (!sectionOp) // omp.terminator
      continue;

    Region &region = sectionOp.region();
    auto sectionCB = [&region, &builder, &moduleTranslation, &bodyGenStatus](
                         InsertPointTy allocaIP, InsertPointTy codeGenIP,
                         llvm::BasicBlock &finiBB) {
      builder.restoreIP(codeGenIP);
      builder.CreateBr(&finiBB);
      convertOmpOpRegions(region, "omp.section.region", *codeGenIP.getBlock(),
                          finiBB, builder, moduleTranslation, bodyGenStatus);
    };
    sectionCBs.push_back(sectionCB);
  }

  // No sections within omp.sections operation - skip generation. This situation
  // is only possible if there is only a terminator operation inside the
  // sections operation
  if (sectionCBs.empty())
    return success();

  assert(isa<omp::SectionOp>(*sectionsOp.region().op_begin()));

  // TODO: Perform appropriate actions according to the data-sharing
  // attribute (shared, private, firstprivate, ...) of variables.
  // Currently defaults to shared.
  auto privCB = [&](InsertPointTy, InsertPointTy codeGenIP, llvm::Value &,
                    llvm::Value &vPtr,
                    llvm::Value *&replacementValue) -> InsertPointTy {
    replacementValue = &vPtr;
    return codeGenIP;
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createSections(
      ompLoc, findAllocaInsertPoint(builder, moduleTranslation), sectionCBs,
      privCB, finiCB, false, sectionsOp.nowait()));
  return bodyGenStatus;
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

  // Find the loop configuration.
  llvm::Value *step = moduleTranslation.lookupValue(loop.step()[0]);
  llvm::Type *ivType = step->getType();
  llvm::Value *chunk =
      loop.schedule_chunk_var()
          ? moduleTranslation.lookupValue(loop.schedule_chunk_var())
          : llvm::ConstantInt::get(ivType, 1);

  SmallVector<omp::ReductionDeclareOp> reductionDecls;
  collectReductionDecls(loop, reductionDecls);
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  // Allocate space for privatized reduction variables.
  SmallVector<llvm::Value *> privateReductionVariables;
  DenseMap<Value, llvm::Value *> reductionVariableMap;
  unsigned numReductions = loop.getNumReductionVars();
  privateReductionVariables.reserve(numReductions);
  if (numReductions != 0) {
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    builder.restoreIP(allocaIP);
    for (unsigned i = 0; i < numReductions; ++i) {
      auto reductionType =
          loop.reduction_vars()[i].getType().cast<LLVM::LLVMPointerType>();
      llvm::Value *var = builder.CreateAlloca(
          moduleTranslation.convertType(reductionType.getElementType()));
      privateReductionVariables.push_back(var);
      reductionVariableMap.try_emplace(loop.reduction_vars()[i], var);
    }
  }

  // Store the mapping between reduction variables and their private copies on
  // ModuleTranslation stack. It can be then recovered when translating
  // omp.reduce operations in a separate call.
  LLVM::ModuleTranslation::SaveStack<OpenMPVarMappingStackFrame> mappingGuard(
      moduleTranslation, reductionVariableMap);

  // Before the loop, store the initial values of reductions into reduction
  // variables. Although this could be done after allocas, we don't want to mess
  // up with the alloca insertion point.
  for (unsigned i = 0; i < numReductions; ++i) {
    SmallVector<llvm::Value *> phis;
    if (failed(inlineConvertOmpRegions(reductionDecls[i].initializerRegion(),
                                       "omp.reduction.neutral", builder,
                                       moduleTranslation, &phis)))
      return failure();
    assert(phis.size() == 1 && "expected one value to be yielded from the "
                               "reduction neutral element declaration region");
    builder.CreateStore(phis[0], privateReductionVariables[i]);
  }

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

  allocaIP = findAllocaInsertPoint(builder, moduleTranslation);

  bool isSimd = loop.simd_modifier();

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
      if (isSimd)
        schedType = llvm::omp::OMPScheduleType::GuidedSimd;
      else
        schedType = llvm::omp::OMPScheduleType::GuidedChunked;
      break;
    case omp::ClauseScheduleKind::Auto:
      schedType = llvm::omp::OMPScheduleType::Auto;
      break;
    case omp::ClauseScheduleKind::Runtime:
      if (isSimd)
        schedType = llvm::omp::OMPScheduleType::RuntimeSimd;
      else
        schedType = llvm::omp::OMPScheduleType::Runtime;
      break;
    default:
      llvm_unreachable("Unknown schedule value");
      break;
    }

    if (loop.schedule_modifier().hasValue()) {
      omp::ScheduleModifier modifier =
          *omp::symbolizeScheduleModifier(loop.schedule_modifier().getValue());
      switch (modifier) {
      case omp::ScheduleModifier::monotonic:
        schedType |= llvm::omp::OMPScheduleType::ModifierMonotonic;
        break;
      case omp::ScheduleModifier::nonmonotonic:
        schedType |= llvm::omp::OMPScheduleType::ModifierNonmonotonic;
        break;
      default:
        // Nothing to do here.
        break;
      }
    }
    afterIP = ompBuilder->applyDynamicWorkshareLoop(
        ompLoc.DL, loopInfo, allocaIP, schedType, !loop.nowait(), chunk);
  }

  // Continue building IR after the loop. Note that the LoopInfo returned by
  // `collapseLoops` points inside the outermost loop and is intended for
  // potential further loop transformations. Use the insertion point stored
  // before collapsing loops instead.
  builder.restoreIP(afterIP);

  // Process the reductions if required.
  if (numReductions == 0)
    return success();

  // Create the reduction generators. We need to own them here because
  // ReductionInfo only accepts references to the generators.
  SmallVector<OwningReductionGen> owningReductionGens;
  SmallVector<OwningAtomicReductionGen> owningAtomicReductionGens;
  for (unsigned i = 0; i < numReductions; ++i) {
    owningReductionGens.push_back(
        makeReductionGen(reductionDecls[i], builder, moduleTranslation));
    owningAtomicReductionGens.push_back(
        makeAtomicReductionGen(reductionDecls[i], builder, moduleTranslation));
  }

  // Collect the reduction information.
  SmallVector<llvm::OpenMPIRBuilder::ReductionInfo> reductionInfos;
  reductionInfos.reserve(numReductions);
  for (unsigned i = 0; i < numReductions; ++i) {
    llvm::OpenMPIRBuilder::AtomicReductionGenTy atomicGen = nullptr;
    if (owningAtomicReductionGens[i])
      atomicGen = owningAtomicReductionGens[i];
    llvm::Value *variable =
        moduleTranslation.lookupValue(loop.reduction_vars()[i]);
    reductionInfos.push_back({variable->getType()->getPointerElementType(),
                              variable, privateReductionVariables[i],
                              owningReductionGens[i], atomicGen});
  }

  // The call to createReductions below expects the block to have a
  // terminator. Create an unreachable instruction to serve as terminator
  // and remove it later.
  llvm::UnreachableInst *tempTerminator = builder.CreateUnreachable();
  builder.SetInsertPoint(tempTerminator);
  llvm::OpenMPIRBuilder::InsertPointTy contInsertPoint =
      ompBuilder->createReductions(builder.saveIP(), allocaIP, reductionInfos,
                                   loop.nowait());
  if (!contInsertPoint.getBlock())
    return loop->emitOpError() << "failed to convert reductions";
  auto nextInsertionPoint =
      ompBuilder->createBarrier(contInsertPoint, llvm::omp::OMPD_for);
  tempTerminator->eraseFromParent();
  builder.restoreIP(nextInsertionPoint);

  return success();
}

// Convert an Atomic Ordering attribute to llvm::AtomicOrdering.
llvm::AtomicOrdering convertAtomicOrdering(Optional<StringRef> aoAttr) {
  if (!aoAttr.hasValue())
    return llvm::AtomicOrdering::Monotonic; // Default Memory Ordering

  return StringSwitch<llvm::AtomicOrdering>(aoAttr.getValue())
      .Case("seq_cst", llvm::AtomicOrdering::SequentiallyConsistent)
      .Case("acq_rel", llvm::AtomicOrdering::AcquireRelease)
      .Case("acquire", llvm::AtomicOrdering::Acquire)
      .Case("release", llvm::AtomicOrdering::Release)
      .Case("relaxed", llvm::AtomicOrdering::Monotonic)
      .Default(llvm::AtomicOrdering::Monotonic);
}

// Convert omp.atomic.read operation to LLVM IR.
static LogicalResult
convertOmpAtomicRead(Operation &opInst, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {

  auto readOp = cast<omp::AtomicReadOp>(opInst);
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // Set up the source location value for OpenMP runtime.
  llvm::DISubprogram *subprogram =
      builder.GetInsertBlock()->getParent()->getSubprogram();
  const llvm::DILocation *diLoc =
      moduleTranslation.translateLoc(opInst.getLoc(), subprogram);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder.saveIP(),
                                                    llvm::DebugLoc(diLoc));
  llvm::AtomicOrdering AO = convertAtomicOrdering(readOp.memory_order());
  llvm::Value *x = moduleTranslation.lookupValue(readOp.x());
  llvm::Value *v = moduleTranslation.lookupValue(readOp.v());
  llvm::OpenMPIRBuilder::AtomicOpValue V = {v, false, false};
  llvm::OpenMPIRBuilder::AtomicOpValue X = {x, false, false};
  builder.restoreIP(ompBuilder->createAtomicRead(ompLoc, X, V, AO));
  return success();
}

/// Converts an omp.atomic.write operation to LLVM IR.
static LogicalResult
convertOmpAtomicWrite(Operation &opInst, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  auto writeOp = cast<omp::AtomicWriteOp>(opInst);
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // Set up the source location value for OpenMP runtime.
  llvm::DISubprogram *subprogram =
      builder.GetInsertBlock()->getParent()->getSubprogram();
  const llvm::DILocation *diLoc =
      moduleTranslation.translateLoc(opInst.getLoc(), subprogram);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder.saveIP(),
                                                    llvm::DebugLoc(diLoc));
  llvm::AtomicOrdering ao = convertAtomicOrdering(writeOp.memory_order());
  llvm::Value *expr = moduleTranslation.lookupValue(writeOp.value());
  llvm::Value *dest = moduleTranslation.lookupValue(writeOp.address());
  llvm::OpenMPIRBuilder::AtomicOpValue x = {dest, /*isSigned=*/false,
                                            /*isVolatile=*/false};
  builder.restoreIP(ompBuilder->createAtomicWrite(ompLoc, x, expr, ao));
  return success();
}

/// Converts an OpenMP reduction operation using OpenMPIRBuilder. Expects the
/// mapping between reduction variables and their private equivalents to have
/// been stored on the ModuleTranslation stack. Currently only supports
/// reduction within WsLoopOp, but can be easily extended.
static LogicalResult
convertOmpReductionOp(omp::ReductionOp reductionOp,
                      llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  // Find the declaration that corresponds to the reduction op.
  auto reductionContainer = reductionOp->getParentOfType<omp::WsLoopOp>();
  omp::ReductionDeclareOp declaration =
      findReductionDecl(reductionContainer, reductionOp);
  assert(declaration && "could not find reduction declaration");

  // Retrieve the mapping between reduction variables and their private
  // equivalents.
  const DenseMap<Value, llvm::Value *> *reductionVariableMap = nullptr;
  moduleTranslation.stackWalk<OpenMPVarMappingStackFrame>(
      [&](const OpenMPVarMappingStackFrame &frame) {
        reductionVariableMap = &frame.mapping;
        return WalkResult::interrupt();
      });
  assert(reductionVariableMap && "couldn't find private reduction variables");

  // Translate the reduction operation by emitting the body of the corresponding
  // reduction declaration.
  Region &reductionRegion = declaration.reductionRegion();
  llvm::Value *privateReductionVar =
      reductionVariableMap->lookup(reductionOp.accumulator());
  llvm::Value *reductionVal = builder.CreateLoad(
      moduleTranslation.convertType(reductionOp.operand().getType()),
      privateReductionVar);

  moduleTranslation.mapValue(reductionRegion.front().getArgument(0),
                             reductionVal);
  moduleTranslation.mapValue(
      reductionRegion.front().getArgument(1),
      moduleTranslation.lookupValue(reductionOp.operand()));

  SmallVector<llvm::Value *> phis;
  if (failed(inlineConvertOmpRegions(reductionRegion, "omp.reduction.body",
                                     builder, moduleTranslation, &phis)))
    return failure();
  assert(phis.size() == 1 && "expected one value to be yielded from "
                             "the reduction body declaration region");
  builder.CreateStore(phis[0], privateReductionVar);
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

} // namespace

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
      .Case([&](omp::ReductionOp reductionOp) {
        return convertOmpReductionOp(reductionOp, builder, moduleTranslation);
      })
      .Case([&](omp::MasterOp) {
        return convertOmpMaster(*op, builder, moduleTranslation);
      })
      .Case([&](omp::CriticalOp) {
        return convertOmpCritical(*op, builder, moduleTranslation);
      })
      .Case([&](omp::OrderedRegionOp) {
        return convertOmpOrderedRegion(*op, builder, moduleTranslation);
      })
      .Case([&](omp::OrderedOp) {
        return convertOmpOrdered(*op, builder, moduleTranslation);
      })
      .Case([&](omp::WsLoopOp) {
        return convertOmpWsLoop(*op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicReadOp) {
        return convertOmpAtomicRead(*op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicWriteOp) {
        return convertOmpAtomicWrite(*op, builder, moduleTranslation);
      })
      .Case([&](omp::SectionsOp) {
        return convertOmpSections(*op, builder, moduleTranslation);
      })
      .Case<omp::YieldOp, omp::TerminatorOp, omp::ReductionDeclareOp,
            omp::CriticalDeclareOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure
        // was created in the region that handles their parent operation.
        // `reduction.declare` will be used by reductions and is not
        // converted directly, skip it.
        // `critical.declare` is only used to declare names of critical
        // sections which will be used by `critical` ops and hence can be
        // ignored for lowering. The OpenMP IRBuilder will create unique
        // name for critical section names.
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

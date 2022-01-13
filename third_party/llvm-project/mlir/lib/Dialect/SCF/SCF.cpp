//===- SCF.cpp - Structured Control Flow Operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/InliningUtils.h"
using namespace mlir;
using namespace mlir::scf;

#include "mlir/Dialect/SCF/SCFOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SCFDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct SCFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  // Operations in scf dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    auto retValOp = dyn_cast<scf::YieldOp>(op);
    if (!retValOp)
      return;

    for (auto retValue : llvm::zip(valuesToRepl, retValOp.getOperands())) {
      std::get<0>(retValue).replaceAllUsesWith(std::get<1>(retValue));
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SCFDialect
//===----------------------------------------------------------------------===//

void SCFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SCF/SCFOps.cpp.inc"
      >();
  addInterfaces<SCFInlinerInterface>();
}

/// Default callback for IfOp builders. Inserts a yield without arguments.
void mlir::scf::buildTerminatedBody(OpBuilder &builder, Location loc) {
  builder.create<scf::YieldOp>(loc);
}

//===----------------------------------------------------------------------===//
// ExecuteRegionOp
//===----------------------------------------------------------------------===//

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

///
/// (ssa-id `=`)? `execute_region` `->` function-result-type `{`
///    block+
/// `}`
///
/// Example:
///   scf.execute_region -> i32 {
///     %idx = load %rI[%i] : memref<128xi32>
///     return %idx : i32
///   }
///
static ParseResult parseExecuteRegionOp(OpAsmParser &parser,
                                        OperationState &result) {
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Introduce the body region and parse it.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, ExecuteRegionOp op) {
  p.printOptionalArrowTypeList(op.getResultTypes());

  p.printRegion(op.getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  p.printOptionalAttrDict(op->getAttrs());
}

static LogicalResult verify(ExecuteRegionOp op) {
  if (op.getRegion().empty())
    return op.emitOpError("region needs to have at least one block");
  if (op.getRegion().front().getNumArguments() > 0)
    return op.emitOpError("region cannot have any arguments");
  return success();
}

// Inline an ExecuteRegionOp if it only contains one block.
//     "test.foo"() : () -> ()
//      %v = scf.execute_region -> i64 {
//        %x = "test.val"() : () -> i64
//        scf.yield %x : i64
//      }
//      "test.bar"(%v) : (i64) -> ()
//
//  becomes
//
//     "test.foo"() : () -> ()
//     %x = "test.val"() : () -> i64
//     "test.bar"(%x) : (i64) -> ()
//
struct SingleBlockExecuteInliner : public OpRewritePattern<ExecuteRegionOp> {
  using OpRewritePattern<ExecuteRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::hasSingleElement(op.getRegion()))
      return failure();
    replaceOpWithRegion(rewriter, op, op.getRegion());
    return success();
  }
};

// Inline an ExecuteRegionOp if its parent can contain multiple blocks.
// TODO generalize the conditions for operations which can be inlined into.
// func @func_execute_region_elim() {
//     "test.foo"() : () -> ()
//     %v = scf.execute_region -> i64 {
//       %c = "test.cmp"() : () -> i1
//       cond_br %c, ^bb2, ^bb3
//     ^bb2:
//       %x = "test.val1"() : () -> i64
//       br ^bb4(%x : i64)
//     ^bb3:
//       %y = "test.val2"() : () -> i64
//       br ^bb4(%y : i64)
//     ^bb4(%z : i64):
//       scf.yield %z : i64
//     }
//     "test.bar"(%v) : (i64) -> ()
//   return
// }
//
//  becomes
//
// func @func_execute_region_elim() {
//    "test.foo"() : () -> ()
//    %c = "test.cmp"() : () -> i1
//    cond_br %c, ^bb1, ^bb2
//  ^bb1:  // pred: ^bb0
//    %x = "test.val1"() : () -> i64
//    br ^bb3(%x : i64)
//  ^bb2:  // pred: ^bb0
//    %y = "test.val2"() : () -> i64
//    br ^bb3(%y : i64)
//  ^bb3(%z: i64):  // 2 preds: ^bb1, ^bb2
//    "test.bar"(%z) : (i64) -> ()
//    return
//  }
//
struct MultiBlockExecuteInliner : public OpRewritePattern<ExecuteRegionOp> {
  using OpRewritePattern<ExecuteRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<FuncOp, ExecuteRegionOp>(op->getParentOp()))
      return failure();

    Block *prevBlock = op->getBlock();
    Block *postBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);

    rewriter.create<BranchOp>(op.getLoc(), &op.getRegion().front());

    for (Block &blk : op.getRegion()) {
      if (YieldOp yieldOp = dyn_cast<YieldOp>(blk.getTerminator())) {
        rewriter.setInsertionPoint(yieldOp);
        rewriter.create<BranchOp>(yieldOp.getLoc(), postBlock,
                                  yieldOp.getResults());
        rewriter.eraseOp(yieldOp);
      }
    }

    rewriter.inlineRegionBefore(op.getRegion(), postBlock);
    SmallVector<Value> blockArgs;

    for (auto res : op.getResults())
      blockArgs.push_back(postBlock->addArgument(res.getType()));

    rewriter.replaceOp(op, blockArgs);
    return success();
  }
};

void ExecuteRegionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<SingleBlockExecuteInliner, MultiBlockExecuteInliner>(context);
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

MutableOperandRange
ConditionOp::getMutableSuccessorOperands(Optional<unsigned> index) {
  // Pass all operands except the condition to the successor region.
  return getArgsMutable();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(OpBuilder &builder, OperationState &result, Value lb,
                  Value ub, Value step, ValueRange iterArgs,
                  BodyBuilderFn bodyBuilder) {
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);
  for (Value v : iterArgs)
    result.addTypes(v.getType());
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(builder.getIndexType());
  for (Value v : iterArgs)
    bodyBlock.addArgument(v.getType());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (iterArgs.empty() && !bodyBuilder) {
    ForOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArgument(0),
                bodyBlock.getArguments().drop_front());
  }
}

static LogicalResult verify(ForOp op) {
  if (auto cst = op.getStep().getDefiningOp<arith::ConstantIndexOp>())
    if (cst.value() <= 0)
      return op.emitOpError("constant step operand must be positive");

  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (!body->getArgument(0).getType().isIndex())
    return op.emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = op.getNumResults();
  if (opNumResults == 0)
    return success();
  // If ForOp defines values, check that the number and types of
  // the defined values match ForOp initial iter operands and backedge
  // basic block arguments.
  if (op.getNumIterOperands() != opNumResults)
    return op.emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (op.getNumRegionIterArgs() != opNumResults)
    return op.emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = op.getIterOperands();
  auto iterArgs = op.getRegionIterArgs();
  auto opResults = op.getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter region arg and defined value";

    i++;
  }

  return RegionBranchOpInterface::verifyTypes(op);
}

/// Prints the initialization list in the form of
///   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
/// where 'inner' values are assumed to be region arguments and 'outer' values
/// are regular SSA values.
static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

static void print(OpAsmPrinter &p, ForOp op) {
  p << " " << op.getInductionVar() << " = " << op.getLowerBound() << " to "
    << op.getUpperBound() << " step " << op.getStep();

  printInitializationList(p, op.getRegionIterArgs(), op.getIterOperands(),
                          " iter_args");
  if (!op.getIterOperands().empty())
    p << " -> (" << op.getIterOperands().getTypes() << ')';
  p.printRegion(op.getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/op.hasIterOperands());
  p.printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseForOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return failure();

  // Parse loop bounds.
  Type indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::OperandType, 4> regionArgs, operands;
  SmallVector<Type, 4> argTypes;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    // Resolve input operands.
    for (auto operandType : llvm::zip(operands, result.types))
      if (parser.resolveOperand(std::get<0>(operandType),
                                std::get<1>(operandType), result.operands))
        return failure();
  }
  // Induction variable.
  argTypes.push_back(indexType);
  // Loop carried variables
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  Region *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs, argTypes))
    return failure();

  ForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

Region &ForOp::getLoopBody() { return getRegion(); }

bool ForOp::isDefinedOutsideOfLoop(Value value) {
  return !getRegion().isAncestor(value.getParentRegion());
}

LogicalResult ForOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(*this);
  return success();
}

ForOp mlir::scf::getForInductionVarOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return ForOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<ForOp>(containingOp);
}

/// Return operands used when entering the region at 'index'. These operands
/// correspond to the loop iterator operands, i.e., those excluding the
/// induction variable. LoopOp only has one region, so 0 is the only valid value
/// for `index`.
OperandRange ForOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0 && "invalid region index");

  // The initial operands map to the loop arguments after the induction
  // variable.
  return getInitArgs();
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ForOp::getSuccessorRegions(Optional<unsigned> index,
                                ArrayRef<Attribute> operands,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the ForOp, branch into the body using the iterator
  // arguments.
  if (!index.hasValue()) {
    regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
    return;
  }

  // Otherwise, the loop may branch back to itself or the parent operation.
  assert(index.getValue() == 0 && "expected loop region");
  regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
  regions.push_back(RegionSuccessor(getResults()));
}

void ForOp::getNumRegionInvocations(ArrayRef<Attribute> operands,
                                    SmallVectorImpl<int64_t> &countPerRegion) {
  assert(countPerRegion.empty());
  countPerRegion.resize(1);

  auto lb = operands[0].dyn_cast_or_null<IntegerAttr>();
  auto ub = operands[1].dyn_cast_or_null<IntegerAttr>();
  auto step = operands[2].dyn_cast_or_null<IntegerAttr>();

  // Loop bounds are not known statically.
  if (!lb || !ub || !step || step.getValue().getSExtValue() == 0) {
    countPerRegion[0] = kUnknownNumRegionInvocations;
    return;
  }

  countPerRegion[0] =
      ceilDiv(ub.getValue().getSExtValue() - lb.getValue().getSExtValue(),
              step.getValue().getSExtValue());
}

LoopNest mlir::scf::buildLoopNest(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps, ValueRange iterArgs,
    function_ref<ValueVector(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilder) {
  assert(lbs.size() == ubs.size() &&
         "expected the same number of lower and upper bounds");
  assert(lbs.size() == steps.size() &&
         "expected the same number of lower bounds and steps");

  // If there are no bounds, call the body-building function and return early.
  if (lbs.empty()) {
    ValueVector results =
        bodyBuilder ? bodyBuilder(builder, loc, ValueRange(), iterArgs)
                    : ValueVector();
    assert(results.size() == iterArgs.size() &&
           "loop nest body must return as many values as loop has iteration "
           "arguments");
    return LoopNest();
  }

  // First, create the loop structure iteratively using the body-builder
  // callback of `ForOp::build`. Do not create `YieldOp`s yet.
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<scf::ForOp, 4> loops;
  SmallVector<Value, 4> ivs;
  loops.reserve(lbs.size());
  ivs.reserve(lbs.size());
  ValueRange currentIterArgs = iterArgs;
  Location currentLoc = loc;
  for (unsigned i = 0, e = lbs.size(); i < e; ++i) {
    auto loop = builder.create<scf::ForOp>(
        currentLoc, lbs[i], ubs[i], steps[i], currentIterArgs,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange args) {
          ivs.push_back(iv);
          // It is safe to store ValueRange args because it points to block
          // arguments of a loop operation that we also own.
          currentIterArgs = args;
          currentLoc = nestedLoc;
        });
    // Set the builder to point to the body of the newly created loop. We don't
    // do this in the callback because the builder is reset when the callback
    // returns.
    builder.setInsertionPointToStart(loop.getBody());
    loops.push_back(loop);
  }

  // For all loops but the innermost, yield the results of the nested loop.
  for (unsigned i = 0, e = loops.size() - 1; i < e; ++i) {
    builder.setInsertionPointToEnd(loops[i].getBody());
    builder.create<scf::YieldOp>(loc, loops[i + 1].getResults());
  }

  // In the body of the innermost loop, call the body building function if any
  // and yield its results.
  builder.setInsertionPointToStart(loops.back().getBody());
  ValueVector results = bodyBuilder
                            ? bodyBuilder(builder, currentLoc, ivs,
                                          loops.back().getRegionIterArgs())
                            : ValueVector();
  assert(results.size() == iterArgs.size() &&
         "loop nest body must return as many values as loop has iteration "
         "arguments");
  builder.setInsertionPointToEnd(loops.back().getBody());
  builder.create<scf::YieldOp>(loc, results);

  // Return the loops.
  LoopNest res;
  res.loops.assign(loops.begin(), loops.end());
  return res;
}

LoopNest mlir::scf::buildLoopNest(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  // Delegate to the main function by wrapping the body builder.
  return buildLoopNest(builder, loc, lbs, ubs, steps, llvm::None,
                       [&bodyBuilder](OpBuilder &nestedBuilder,
                                      Location nestedLoc, ValueRange ivs,
                                      ValueRange) -> ValueVector {
                         if (bodyBuilder)
                           bodyBuilder(nestedBuilder, nestedLoc, ivs);
                         return {};
                       });
}

namespace {
// Fold away ForOp iter arguments when:
// 1) The op yields the iter arguments.
// 2) The iter arguments have no use and the corresponding outer region
// iterators (inputs) are yielded.
// 3) The iter arguments have no use and the corresponding (operation) results
// have no use.
//
// These arguments must be defined outside of
// the ForOp region and can just be forwarded after simplifying the op inits,
// yields and returns.
//
// The implementation uses `mergeBlockBefore` to steal the content of the
// original ForOp and avoid cloning.
struct ForOpIterArgsFolder : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    bool canonicalize = false;
    Block &block = forOp.getRegion().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

    // An internal flat vector of block transfer
    // arguments `newBlockTransferArgs` keeps the 1-1 mapping of original to
    // transformed block argument mappings. This plays the role of a
    // BlockAndValueMapping for the particular use case of calling into
    // `mergeBlockBefore`.
    SmallVector<bool, 4> keepMask;
    keepMask.reserve(yieldOp.getNumOperands());
    SmallVector<Value, 4> newBlockTransferArgs, newIterArgs, newYieldValues,
        newResultValues;
    newBlockTransferArgs.reserve(1 + forOp.getNumIterOperands());
    newBlockTransferArgs.push_back(Value()); // iv placeholder with null value
    newIterArgs.reserve(forOp.getNumIterOperands());
    newYieldValues.reserve(yieldOp.getNumOperands());
    newResultValues.reserve(forOp.getNumResults());
    for (auto it : llvm::zip(forOp.getIterOperands(),   // iter from outside
                             forOp.getRegionIterArgs(), // iter inside region
                             forOp.getResults(),        // op results
                             yieldOp.getOperands()      // iter yield
                             )) {
      // Forwarded is `true` when:
      // 1) The region `iter` argument is yielded.
      // 2) The region `iter` argument has no use, and the corresponding iter
      // operand (input) is yielded.
      // 3) The region `iter` argument has no use, and the corresponding op
      // result has no use.
      bool forwarded = ((std::get<1>(it) == std::get<3>(it)) ||
                        (std::get<1>(it).use_empty() &&
                         (std::get<0>(it) == std::get<3>(it) ||
                          std::get<2>(it).use_empty())));
      keepMask.push_back(!forwarded);
      canonicalize |= forwarded;
      if (forwarded) {
        newBlockTransferArgs.push_back(std::get<0>(it));
        newResultValues.push_back(std::get<0>(it));
        continue;
      }
      newIterArgs.push_back(std::get<0>(it));
      newYieldValues.push_back(std::get<3>(it));
      newBlockTransferArgs.push_back(Value()); // placeholder with null value
      newResultValues.push_back(Value());      // placeholder with null value
    }

    if (!canonicalize)
      return failure();

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newIterArgs);
    Block &newBlock = newForOp.getRegion().front();

    // Replace the null placeholders with newly constructed values.
    newBlockTransferArgs[0] = newBlock.getArgument(0); // iv
    for (unsigned idx = 0, collapsedIdx = 0, e = newResultValues.size();
         idx != e; ++idx) {
      Value &blockTransferArg = newBlockTransferArgs[1 + idx];
      Value &newResultVal = newResultValues[idx];
      assert((blockTransferArg && newResultVal) ||
             (!blockTransferArg && !newResultVal));
      if (!blockTransferArg) {
        blockTransferArg = newForOp.getRegionIterArgs()[collapsedIdx];
        newResultVal = newForOp.getResult(collapsedIdx++);
      }
    }

    Block &oldBlock = forOp.getRegion().front();
    assert(oldBlock.getNumArguments() == newBlockTransferArgs.size() &&
           "unexpected argument size mismatch");

    // No results case: the scf::ForOp builder already created a zero
    // result terminator. Merge before this terminator and just get rid of the
    // original terminator that has been merged in.
    if (newIterArgs.empty()) {
      auto newYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
      rewriter.mergeBlockBefore(&oldBlock, newYieldOp, newBlockTransferArgs);
      rewriter.eraseOp(newBlock.getTerminator()->getPrevNode());
      rewriter.replaceOp(forOp, newResultValues);
      return success();
    }

    // No terminator case: merge and rewrite the merged terminator.
    auto cloneFilteredTerminator = [&](scf::YieldOp mergedTerminator) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(mergedTerminator);
      SmallVector<Value, 4> filteredOperands;
      filteredOperands.reserve(newResultValues.size());
      for (unsigned idx = 0, e = keepMask.size(); idx < e; ++idx)
        if (keepMask[idx])
          filteredOperands.push_back(mergedTerminator.getOperand(idx));
      rewriter.create<scf::YieldOp>(mergedTerminator.getLoc(),
                                    filteredOperands);
    };

    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);
    auto mergedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
    cloneFilteredTerminator(mergedYieldOp);
    rewriter.eraseOp(mergedYieldOp);
    rewriter.replaceOp(forOp, newResultValues);
    return success();
  }
};

/// Rewriting pattern that erases loops that are known not to iterate and
/// replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    // If the upper bound is the same as the lower bound, the loop does not
    // iterate, just remove it.
    if (op.getLowerBound() == op.getUpperBound()) {
      rewriter.replaceOp(op, op.getIterOperands());
      return success();
    }

    auto lb = op.getLowerBound().getDefiningOp<arith::ConstantOp>();
    auto ub = op.getUpperBound().getDefiningOp<arith::ConstantOp>();
    if (!lb || !ub)
      return failure();

    // If the loop is known to have 0 iterations, remove it.
    llvm::APInt lbValue = lb.getValue().cast<IntegerAttr>().getValue();
    llvm::APInt ubValue = ub.getValue().cast<IntegerAttr>().getValue();
    if (lbValue.sge(ubValue)) {
      rewriter.replaceOp(op, op.getIterOperands());
      return success();
    }

    auto step = op.getStep().getDefiningOp<arith::ConstantOp>();
    if (!step)
      return failure();

    // If the loop is known to have 1 iteration, inline its body and remove the
    // loop.
    llvm::APInt stepValue = step.getValue().cast<IntegerAttr>().getValue();
    if ((lbValue + stepValue).sge(ubValue)) {
      SmallVector<Value, 4> blockArgs;
      blockArgs.reserve(op.getNumIterOperands() + 1);
      blockArgs.push_back(op.getLowerBound());
      llvm::append_range(blockArgs, op.getIterOperands());
      replaceOpWithRegion(rewriter, op, op.getLoopBody(), blockArgs);
      return success();
    }

    return failure();
  }
};

/// Perform a replacement of one iter OpOperand of an scf.for to the
/// `replacement` value which is expected to be the source of a tensor.cast.
/// tensor.cast ops are inserted inside the block to account for the type cast.
static ForOp replaceTensorCastForOpIterArg(PatternRewriter &rewriter,
                                           OpOperand &operand,
                                           Value replacement) {
  Type oldType = operand.get().getType(), newType = replacement.getType();
  assert(oldType.isa<RankedTensorType>() && newType.isa<RankedTensorType>() &&
         "expected ranked tensor types");

  // 1. Create new iter operands, exactly 1 is replaced.
  ForOp forOp = cast<ForOp>(operand.getOwner());
  assert(operand.getOperandNumber() >= forOp.getNumControlOperands() &&
         "expected an iter OpOperand");
  if (operand.get().getType() == replacement.getType())
    return forOp;
  SmallVector<Value> newIterOperands;
  for (OpOperand &opOperand : forOp.getIterOpOperands()) {
    if (opOperand.getOperandNumber() == operand.getOperandNumber()) {
      newIterOperands.push_back(replacement);
      continue;
    }
    newIterOperands.push_back(opOperand.get());
  }

  // 2. Create the new forOp shell.
  scf::ForOp newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newIterOperands);
  Block &newBlock = newForOp.getRegion().front();
  SmallVector<Value, 4> newBlockTransferArgs(newBlock.getArguments().begin(),
                                             newBlock.getArguments().end());

  // 3. Inject an incoming cast op at the beginning of the block for the bbArg
  // corresponding to the `replacement` value.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(&newBlock, newBlock.begin());
  BlockArgument newRegionIterArg = newForOp.getRegionIterArgForOpOperand(
      newForOp->getOpOperand(operand.getOperandNumber()));
  Value castIn = rewriter.create<tensor::CastOp>(newForOp.getLoc(), oldType,
                                                 newRegionIterArg);
  newBlockTransferArgs[newRegionIterArg.getArgNumber()] = castIn;

  // 4. Steal the old block ops, mapping to the newBlockTransferArgs.
  Block &oldBlock = forOp.getRegion().front();
  rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);

  // 5. Inject an outgoing cast op at the end of the block and yield it instead.
  auto clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
  rewriter.setInsertionPoint(clonedYieldOp);
  unsigned yieldIdx =
      newRegionIterArg.getArgNumber() - forOp.getNumInductionVars();
  Value castOut = rewriter.create<tensor::CastOp>(
      newForOp.getLoc(), newType, clonedYieldOp.getOperand(yieldIdx));
  SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
  newYieldOperands[yieldIdx] = castOut;
  rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
  rewriter.eraseOp(clonedYieldOp);

  // 6. Inject an outgoing cast op after the forOp.
  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> newResults = newForOp.getResults();
  newResults[yieldIdx] = rewriter.create<tensor::CastOp>(
      newForOp.getLoc(), oldType, newResults[yieldIdx]);

  return newForOp;
}

/// Fold scf.for iter_arg/result pairs that go through incoming/ougoing
/// a tensor.cast op pair so as to pull the tensor.cast inside the scf.for:
///
/// ```
///   %0 = tensor.cast %t0 : tensor<32x1024xf32> to tensor<?x?xf32>
///   %1 = scf.for %i = %c0 to %c1024 step %c32 iter_args(%iter_t0 = %0)
///      -> (tensor<?x?xf32>) {
///     %2 = call @do(%iter_t0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
///     scf.yield %2 : tensor<?x?xf32>
///   }
///   %2 = tensor.cast %1 : tensor<?x?xf32> to tensor<32x1024xf32>
///   use_of(%2)
/// ```
///
/// folds into:
///
/// ```
///   %0 = scf.for %arg2 = %c0 to %c1024 step %c32 iter_args(%arg3 = %arg0)
///       -> (tensor<32x1024xf32>) {
///     %2 = tensor.cast %arg3 : tensor<32x1024xf32> to tensor<?x?xf32>
///     %3 = call @do(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
///     %4 = tensor.cast %3 : tensor<?x?xf32> to tensor<32x1024xf32>
///     scf.yield %4 : tensor<32x1024xf32>
///   }
///   use_of(%0)
/// ```
struct ForOpTensorCastFolder : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    for (auto it : llvm::zip(op.getIterOpOperands(), op.getResults())) {
      OpOperand &iterOpOperand = std::get<0>(it);
      auto incomingCast = iterOpOperand.get().getDefiningOp<tensor::CastOp>();
      if (!incomingCast)
        continue;
      if (!std::get<1>(it).hasOneUse())
        continue;
      auto outgoingCastOp =
          dyn_cast<tensor::CastOp>(*std::get<1>(it).user_begin());
      if (!outgoingCastOp)
        continue;

      // Must be a tensor.cast op pair with matching types.
      if (outgoingCastOp.getResult().getType() !=
          incomingCast.source().getType())
        continue;

      // Create a new ForOp with that iter operand replaced.
      auto newForOp = replaceTensorCastForOpIterArg(rewriter, iterOpOperand,
                                                    incomingCast.source());

      // Insert outgoing cast and use it to replace the corresponding result.
      rewriter.setInsertionPointAfter(newForOp);
      SmallVector<Value> replacements = newForOp.getResults();
      unsigned returnIdx =
          iterOpOperand.getOperandNumber() - op.getNumControlOperands();
      replacements[returnIdx] = rewriter.create<tensor::CastOp>(
          op.getLoc(), incomingCast.dest().getType(), replacements[returnIdx]);
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};

/// Canonicalize the iter_args of an scf::ForOp that involve a
/// `bufferization.to_tensor` and for which only the last loop iteration is
/// actually visible outside of the loop. The canonicalization looks for a
/// pattern such as:
/// ```
///    %t0 = ... : tensor_type
///    %0 = scf.for ... iter_args(%bb0 : %t0) -> (tensor_type) {
///      ...
///      // %m is either buffer_cast(%bb00) or defined above the loop
///      %m... : memref_type
///      ... // uses of %m with potential inplace updates
///      %new_tensor = bufferization.to_tensor %m : memref_type
///      ...
///      scf.yield %new_tensor : tensor_type
///    }
/// ```
///
/// `%bb0` may have either 0 or 1 use. If it has 1 use it must be exactly a
/// `%m = buffer_cast %bb0` op that feeds into the yielded
/// `bufferization.to_tensor` op.
///
/// If no aliasing write to the memref `%m`, from which `%new_tensor`is loaded,
/// occurs between `bufferization.to_tensor and yield then the value %0
/// visible outside of the loop is the last `bufferization.to_tensor`
/// produced in the loop.
///
/// For now, we approximate the absence of aliasing by only supporting the case
/// when the bufferization.to_tensor is the operation immediately preceding
/// the yield.
//
/// The canonicalization rewrites the pattern as:
/// ```
///    // %m is either a buffer_cast or defined above
///    %m... : memref_type
///    scf.for ... iter_args(%bb0 : %t0) -> (tensor_type) {
///      ... // uses of %m with potential inplace updates
///      scf.yield %bb0: tensor_type
///    }
///    %0 = bufferization.to_tensor %m : memref_type
/// ```
///
/// A later bbArg canonicalization will further rewrite as:
/// ```
///    // %m is either a buffer_cast or defined above
///    %m... : memref_type
///    scf.for ... { // no iter_args
///      ... // uses of %m with potential inplace updates
///    }
///    %0 = bufferization.to_tensor %m : memref_type
/// ```
struct LastTensorLoadCanonicalization : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    assert(std::next(forOp.getRegion().begin()) == forOp.getRegion().end() &&
           "unexpected multiple blocks");

    Location loc = forOp.getLoc();
    DenseMap<Value, Value> replacements;
    for (BlockArgument bbArg : forOp.getRegionIterArgs()) {
      unsigned idx = bbArg.getArgNumber() - /*numIv=*/1;
      auto yieldOp =
          cast<scf::YieldOp>(forOp.getRegion().front().getTerminator());
      Value yieldVal = yieldOp->getOperand(idx);
      auto tensorLoadOp = yieldVal.getDefiningOp<bufferization::ToTensorOp>();
      bool isTensor = bbArg.getType().isa<TensorType>();

      bufferization::ToMemrefOp tensorToMemref;
      // Either bbArg has no use or it has a single buffer_cast use.
      if (bbArg.hasOneUse())
        tensorToMemref =
            dyn_cast<bufferization::ToMemrefOp>(*bbArg.getUsers().begin());
      if (!isTensor || !tensorLoadOp || (!bbArg.use_empty() && !tensorToMemref))
        continue;
      // If tensorToMemref is present, it must feed into the `ToTensorOp`.
      if (tensorToMemref && tensorLoadOp.memref() != tensorToMemref)
        continue;
      // TODO: Any aliasing write of tensorLoadOp.memref() nested under `forOp`
      // must be before `ToTensorOp` in the block so that the lastWrite
      // property is not subject to additional side-effects.
      // For now, we only support the case when ToTensorOp appears
      // immediately before the terminator.
      if (tensorLoadOp->getNextNode() != yieldOp)
        continue;

      // Clone the optional tensorToMemref before forOp.
      if (tensorToMemref) {
        rewriter.setInsertionPoint(forOp);
        rewriter.replaceOpWithNewOp<bufferization::ToMemrefOp>(
            tensorToMemref, tensorToMemref.memref().getType(),
            tensorToMemref.tensor());
      }

      // Clone the tensorLoad after forOp.
      rewriter.setInsertionPointAfter(forOp);
      Value newTensorLoad = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorLoadOp.memref());
      Value forOpResult = forOp.getResult(bbArg.getArgNumber() - /*iv=*/1);
      replacements.insert(std::make_pair(forOpResult, newTensorLoad));

      // Make the terminator just yield the bbArg, the old tensorLoadOp + the
      // old bbArg (that is now directly yielded) will canonicalize away.
      rewriter.startRootUpdate(yieldOp);
      yieldOp.setOperand(idx, bbArg);
      rewriter.finalizeRootUpdate(yieldOp);
    }
    if (replacements.empty())
      return failure();

    // We want to replace a subset of the results of `forOp`. rewriter.replaceOp
    // replaces the whole op and erase it unconditionally. This is wrong for
    // `forOp` as it generally contains ops with side effects.
    // Instead, use `rewriter.replaceOpWithIf`.
    SmallVector<Value> newResults;
    newResults.reserve(forOp.getNumResults());
    for (Value v : forOp.getResults()) {
      auto it = replacements.find(v);
      newResults.push_back((it != replacements.end()) ? it->second : v);
    }
    unsigned idx = 0;
    rewriter.replaceOpWithIf(forOp, newResults, [&](OpOperand &op) {
      return op.get() != newResults[idx++];
    });
    return success();
  }
};
} // namespace

void ForOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<ForOpIterArgsFolder, SimplifyTrivialLoops,
              LastTensorLoadCanonicalization, ForOpTensorCastFolder>(context);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

bool mlir::scf::insideMutuallyExclusiveBranches(Operation *a, Operation *b) {
  assert(a && "expected non-empty operation");
  assert(b && "expected non-empty operation");

  IfOp ifOp = a->getParentOfType<IfOp>();
  while (ifOp) {
    // Check if b is inside ifOp. (We already know that a is.)
    if (ifOp->isProperAncestor(b))
      // b is contained in ifOp. a and b are in mutually exclusive branches if
      // they are in different blocks of ifOp.
      return static_cast<bool>(ifOp.thenBlock()->findAncestorOpInBlock(*a)) !=
             static_cast<bool>(ifOp.thenBlock()->findAncestorOpInBlock(*b));
    // Check next enclosing IfOp.
    ifOp = ifOp->getParentOfType<IfOp>();
  }

  // Could not find a common IfOp among a's and b's ancestors.
  return false;
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 bool withElseRegion) {
  build(builder, result, /*resultTypes=*/llvm::None, cond, withElseRegion);
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value cond, bool withElseRegion) {
  auto addTerminator = [&](OpBuilder &nested, Location loc) {
    if (resultTypes.empty())
      IfOp::ensureTerminator(*nested.getInsertionBlock()->getParent(), nested,
                             loc);
  };

  build(builder, result, resultTypes, cond, addTerminator,
        withElseRegion ? addTerminator
                       : function_ref<void(OpBuilder &, Location)>());
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value cond,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  assert(thenBuilder && "the builder callback for 'then' must be present");

  result.addOperands(cond);
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  Region *elseRegion = result.addRegion();
  if (!elseBuilder)
    return;

  builder.createBlock(elseRegion);
  elseBuilder(builder, result.location);
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  build(builder, result, TypeRange(), cond, thenBuilder, elseBuilder);
}

static LogicalResult verify(IfOp op) {
  if (op.getNumResults() != 0 && op.getElseRegion().empty())
    return op.emitOpError("must have an else block if defining values");

  return RegionBranchOpInterface::verifyTypes(op);
}

static ParseResult parseIfOp(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
  Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();
  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    IfOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, IfOp op) {
  bool printBlockTerminators = false;

  p << " " << op.getCondition();
  if (!op.getResults().empty()) {
    p << " -> (" << op.getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p.printRegion(op.getThenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = op.getElseRegion();
  if (!elseRegion.empty()) {
    p << " else";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printBlockTerminators);
  }

  p.printOptionalAttrDict(op->getAttrs());
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void IfOp::getSuccessorRegions(Optional<unsigned> index,
                               ArrayRef<Attribute> operands,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  bool condition;
  if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
    condition = condAttr.getValue().isOneValue();
  } else {
    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&getThenRegion()));
    // If the else region does not exist, it is not a viable successor.
    if (elseRegion)
      regions.push_back(RegionSuccessor(elseRegion));
    return;
  }

  // Add the successor regions using the condition.
  regions.push_back(RegionSuccessor(condition ? &getThenRegion() : elseRegion));
}

/// If the condition is a constant, returns 1 for the executed block and 0 for
/// the other. Otherwise, returns `kUnknownNumRegionInvocations` for both
/// successors.
void IfOp::getNumRegionInvocations(ArrayRef<Attribute> operands,
                                   SmallVectorImpl<int64_t> &countPerRegion) {
  if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
    // If the condition is true, `then` is executed once and `else` zero times,
    // and vice-versa.
    bool cond = condAttr.getValue().isOneValue();
    countPerRegion.assign(1, cond ? 1 : 0);
    countPerRegion.push_back(cond ? 0 : 1);
  } else {
    // Non-constant condition: unknown invocations for both successors.
    countPerRegion.assign(2, kUnknownNumRegionInvocations);
  }
}

LogicalResult IfOp::fold(ArrayRef<Attribute> operands,
                         SmallVectorImpl<OpFoldResult> &results) {
  // if (!c) then A() else B() -> if c then B() else A()
  if (getElseRegion().empty())
    return failure();

  arith::XOrIOp xorStmt = getCondition().getDefiningOp<arith::XOrIOp>();
  if (!xorStmt)
    return failure();

  if (!matchPattern(xorStmt.getRhs(), m_One()))
    return failure();

  getConditionMutable().assign(xorStmt.getLhs());
  Block *thenBlock = &getThenRegion().front();
  // It would be nicer to use iplist::swap, but that has no implemented
  // callbacks See: https://llvm.org/doxygen/ilist_8h_source.html#l00224
  getThenRegion().getBlocks().splice(getThenRegion().getBlocks().begin(),
                                     getElseRegion().getBlocks());
  getElseRegion().getBlocks().splice(getElseRegion().getBlocks().begin(),
                                     getThenRegion().getBlocks(), thenBlock);
  return success();
}

namespace {
// Pattern to remove unused IfOp results.
struct RemoveUnusedResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  void transferBody(Block *source, Block *dest, ArrayRef<OpResult> usedResults,
                    PatternRewriter &rewriter) const {
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest);
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    SmallVector<Value, 4> usedOperands;
    llvm::transform(usedResults, std::back_inserter(usedOperands),
                    [&](OpResult result) {
                      return yieldOp.getOperand(result.getResultNumber());
                    });
    rewriter.updateRootInPlace(yieldOp,
                               [&]() { yieldOp->setOperands(usedOperands); });
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Compute the list of used results.
    SmallVector<OpResult, 4> usedResults;
    llvm::copy_if(op.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    // Replace the operation if only a subset of its results have uses.
    if (usedResults.size() == op.getNumResults())
      return failure();

    // Compute the result types of the replacement operation.
    SmallVector<Type, 4> newTypes;
    llvm::transform(usedResults, std::back_inserter(newTypes),
                    [](OpResult result) { return result.getType(); });

    // Create a replacement operation with empty then and else regions.
    auto emptyBuilder = [](OpBuilder &, Location) {};
    auto newOp = rewriter.create<IfOp>(op.getLoc(), newTypes, op.getCondition(),
                                       emptyBuilder, emptyBuilder);

    // Move the bodies and replace the terminators (note there is a then and
    // an else region since the operation returns results).
    transferBody(op.getBody(0), newOp.getBody(0), usedResults, rewriter);
    transferBody(op.getBody(1), newOp.getBody(1), usedResults, rewriter);

    // Replace the operation by the new one.
    SmallVector<Value, 4> repResults(op.getNumResults());
    for (const auto &en : llvm::enumerate(usedResults))
      repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
    rewriter.replaceOp(op, repResults);
    return success();
  }
};

struct RemoveStaticCondition : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    auto constant = op.getCondition().getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return failure();

    if (constant.getValue().cast<BoolAttr>().getValue())
      replaceOpWithRegion(rewriter, op, op.getThenRegion());
    else if (!op.getElseRegion().empty())
      replaceOpWithRegion(rewriter, op, op.getElseRegion());
    else
      rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertTrivialIfToSelect : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0)
      return failure();

    if (!llvm::hasSingleElement(op.getThenRegion().front()) ||
        !llvm::hasSingleElement(op.getElseRegion().front()))
      return failure();

    auto cond = op.getCondition();
    auto thenYieldArgs =
        cast<scf::YieldOp>(op.getThenRegion().front().getTerminator())
            .getOperands();
    auto elseYieldArgs =
        cast<scf::YieldOp>(op.getElseRegion().front().getTerminator())
            .getOperands();
    SmallVector<Value> results(op->getNumResults());
    assert(thenYieldArgs.size() == results.size());
    assert(elseYieldArgs.size() == results.size());
    for (const auto &it :
         llvm::enumerate(llvm::zip(thenYieldArgs, elseYieldArgs))) {
      Value trueVal = std::get<0>(it.value());
      Value falseVal = std::get<1>(it.value());
      if (trueVal == falseVal)
        results[it.index()] = trueVal;
      else
        results[it.index()] =
            rewriter.create<SelectOp>(op.getLoc(), cond, trueVal, falseVal);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Allow the true region of an if to assume the condition is true
/// and vice versa. For example:
///
///   scf.if %cmp {
///      print(%cmp)
///   }
///
///  becomes
///
///   scf.if %cmp {
///      print(true)
///   }
///
struct ConditionPropagation : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Early exit if the condition is constant since replacing a constant
    // in the body with another constant isn't a simplification.
    if (op.getCondition().getDefiningOp<arith::ConstantOp>())
      return failure();

    bool changed = false;
    mlir::Type i1Ty = rewriter.getI1Type();

    // These variables serve to prevent creating duplicate constants
    // and hold constant true or false values.
    Value constantTrue = nullptr;
    Value constantFalse = nullptr;

    for (OpOperand &use :
         llvm::make_early_inc_range(op.getCondition().getUses())) {
      if (op.getThenRegion().isAncestor(use.getOwner()->getParentRegion())) {
        changed = true;

        if (!constantTrue)
          constantTrue = rewriter.create<arith::ConstantOp>(
              op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 1));

        rewriter.updateRootInPlace(use.getOwner(),
                                   [&]() { use.set(constantTrue); });
      } else if (op.getElseRegion().isAncestor(
                     use.getOwner()->getParentRegion())) {
        changed = true;

        if (!constantFalse)
          constantFalse = rewriter.create<arith::ConstantOp>(
              op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 0));

        rewriter.updateRootInPlace(use.getOwner(),
                                   [&]() { use.set(constantFalse); });
      }
    }

    return success(changed);
  }
};

/// Remove any statements from an if that are equivalent to the condition
/// or its negation. For example:
///
///    %res:2 = scf.if %cmp {
///       yield something(), true
///    } else {
///       yield something2(), false
///    }
///    print(%res#1)
///
///  becomes
///    %res = scf.if %cmp {
///       yield something()
///    } else {
///       yield something2()
///    }
///    print(%cmp)
///
/// Additionally if both branches yield the same value, replace all uses
/// of the result with the yielded value.
///
///    %res:2 = scf.if %cmp {
///       yield something(), %arg1
///    } else {
///       yield something2(), %arg1
///    }
///    print(%res#1)
///
///  becomes
///    %res = scf.if %cmp {
///       yield something()
///    } else {
///       yield something2()
///    }
///    print(%arg1)
///
struct ReplaceIfYieldWithConditionOrValue : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Early exit if there are no results that could be replaced.
    if (op.getNumResults() == 0)
      return failure();

    auto trueYield =
        cast<scf::YieldOp>(op.getThenRegion().back().getTerminator());
    auto falseYield =
        cast<scf::YieldOp>(op.getElseRegion().back().getTerminator());

    rewriter.setInsertionPoint(op->getBlock(),
                               op.getOperation()->getIterator());
    bool changed = false;
    Type i1Ty = rewriter.getI1Type();
    for (auto tup : llvm::zip(trueYield.getResults(), falseYield.getResults(),
                              op.getResults())) {
      Value trueResult, falseResult, opResult;
      std::tie(trueResult, falseResult, opResult) = tup;

      if (trueResult == falseResult) {
        if (!opResult.use_empty()) {
          opResult.replaceAllUsesWith(trueResult);
          changed = true;
        }
        continue;
      }

      auto trueYield = trueResult.getDefiningOp<arith::ConstantOp>();
      if (!trueYield)
        continue;

      if (!trueYield.getType().isInteger(1))
        continue;

      auto falseYield = falseResult.getDefiningOp<arith::ConstantOp>();
      if (!falseYield)
        continue;

      bool trueVal = trueYield.getValue().cast<BoolAttr>().getValue();
      bool falseVal = falseYield.getValue().cast<BoolAttr>().getValue();
      if (!trueVal && falseVal) {
        if (!opResult.use_empty()) {
          Value notCond = rewriter.create<arith::XOrIOp>(
              op.getLoc(), op.getCondition(),
              rewriter.create<arith::ConstantOp>(
                  op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 1)));
          opResult.replaceAllUsesWith(notCond);
          changed = true;
        }
      }
      if (trueVal && !falseVal) {
        if (!opResult.use_empty()) {
          opResult.replaceAllUsesWith(op.getCondition());
          changed = true;
        }
      }
    }
    return success(changed);
  }
};

/// Merge any consecutive scf.if's with the same condition.
///
///    scf.if %cond {
///       firstCodeTrue();...
///    } else {
///       firstCodeFalse();...
///    }
///    %res = scf.if %cond {
///       secondCodeTrue();...
///    } else {
///       secondCodeFalse();...
///    }
///
///  becomes
///    %res = scf.if %cmp {
///       firstCodeTrue();...
///       secondCodeTrue();...
///    } else {
///       firstCodeFalse();...
///       secondCodeFalse();...
///    }
struct CombineIfs : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp nextIf,
                                PatternRewriter &rewriter) const override {
    Block *parent = nextIf->getBlock();
    if (nextIf == &parent->front())
      return failure();

    auto prevIf = dyn_cast<IfOp>(nextIf->getPrevNode());
    if (!prevIf)
      return failure();

    if (nextIf.getCondition() != prevIf.getCondition())
      return failure();

    // Don't permit merging if a result of the first if is used
    // within the second.
    if (llvm::any_of(prevIf->getUsers(),
                     [&](Operation *user) { return nextIf->isAncestor(user); }))
      return failure();

    SmallVector<Type> mergedTypes(prevIf.getResultTypes());
    llvm::append_range(mergedTypes, nextIf.getResultTypes());

    IfOp combinedIf = rewriter.create<IfOp>(
        nextIf.getLoc(), mergedTypes, nextIf.getCondition(), /*hasElse=*/false);
    rewriter.eraseBlock(&combinedIf.getThenRegion().back());

    YieldOp thenYield = prevIf.thenYield();
    YieldOp thenYield2 = nextIf.thenYield();

    combinedIf.getThenRegion().getBlocks().splice(
        combinedIf.getThenRegion().getBlocks().begin(),
        prevIf.getThenRegion().getBlocks());

    rewriter.mergeBlocks(nextIf.thenBlock(), combinedIf.thenBlock());
    rewriter.setInsertionPointToEnd(combinedIf.thenBlock());

    SmallVector<Value> mergedYields(thenYield.getOperands());
    llvm::append_range(mergedYields, thenYield2.getOperands());
    rewriter.create<YieldOp>(thenYield2.getLoc(), mergedYields);
    rewriter.eraseOp(thenYield);
    rewriter.eraseOp(thenYield2);

    combinedIf.getElseRegion().getBlocks().splice(
        combinedIf.getElseRegion().getBlocks().begin(),
        prevIf.getElseRegion().getBlocks());

    if (!nextIf.getElseRegion().empty()) {
      if (combinedIf.getElseRegion().empty()) {
        combinedIf.getElseRegion().getBlocks().splice(
            combinedIf.getElseRegion().getBlocks().begin(),
            nextIf.getElseRegion().getBlocks());
      } else {
        YieldOp elseYield = combinedIf.elseYield();
        YieldOp elseYield2 = nextIf.elseYield();
        rewriter.mergeBlocks(nextIf.elseBlock(), combinedIf.elseBlock());

        rewriter.setInsertionPointToEnd(combinedIf.elseBlock());

        SmallVector<Value> mergedElseYields(elseYield.getOperands());
        llvm::append_range(mergedElseYields, elseYield2.getOperands());

        rewriter.create<YieldOp>(elseYield2.getLoc(), mergedElseYields);
        rewriter.eraseOp(elseYield);
        rewriter.eraseOp(elseYield2);
      }
    }

    SmallVector<Value> prevValues;
    SmallVector<Value> nextValues;
    for (const auto &pair : llvm::enumerate(combinedIf.getResults())) {
      if (pair.index() < prevIf.getNumResults())
        prevValues.push_back(pair.value());
      else
        nextValues.push_back(pair.value());
    }
    rewriter.replaceOp(prevIf, prevValues);
    rewriter.replaceOp(nextIf, nextValues);
    return success();
  }
};

/// Pattern to remove an empty else branch.
struct RemoveEmptyElseBranch : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Cannot remove else region when there are operation results.
    if (ifOp.getNumResults())
      return failure();
    Block *elseBlock = ifOp.elseBlock();
    if (!elseBlock || !llvm::hasSingleElement(*elseBlock))
      return failure();
    auto newIfOp = rewriter.cloneWithoutRegions(ifOp);
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());
    rewriter.eraseOp(ifOp);
    return success();
  }
};

/// Convert nested `if`s into `arith.andi` + single `if`.
///
///    scf.if %arg0 {
///      scf.if %arg1 {
///        ...
///        scf.yield
///      }
///      scf.yield
///    }
///  becomes
///
///    %0 = arith.andi %arg0, %arg1
///    scf.if %0 {
///      ...
///      scf.yield
///    }
struct CombineNestedIfs : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Both `if` ops must not yield results and have only `then` block.
    if (op->getNumResults() != 0 || op.elseBlock())
      return failure();

    auto nestedOps = op.thenBlock()->without_terminator();
    // Nested `if` must be the only op in block.
    if (!llvm::hasSingleElement(nestedOps))
      return failure();

    auto nestedIf = dyn_cast<IfOp>(*nestedOps.begin());
    if (!nestedIf || nestedIf->getNumResults() != 0 || nestedIf.elseBlock())
      return failure();

    Location loc = op.getLoc();
    Value newCondition = rewriter.create<arith::AndIOp>(
        loc, op.getCondition(), nestedIf.getCondition());
    auto newIf = rewriter.create<IfOp>(loc, newCondition);
    Block *newIfBlock = newIf.thenBlock();
    rewriter.eraseOp(newIfBlock->getTerminator());
    rewriter.mergeBlocks(nestedIf.thenBlock(), newIfBlock);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add<CombineIfs, CombineNestedIfs, ConditionPropagation,
              ConvertTrivialIfToSelect, RemoveEmptyElseBranch,
              RemoveStaticCondition, RemoveUnusedResults,
              ReplaceIfYieldWithConditionOrValue>(context);
}

Block *IfOp::thenBlock() { return &getThenRegion().back(); }
YieldOp IfOp::thenYield() { return cast<YieldOp>(&thenBlock()->back()); }
Block *IfOp::elseBlock() {
  Region &r = getElseRegion();
  if (r.empty())
    return nullptr;
  return &r.back();
}
YieldOp IfOp::elseYield() { return cast<YieldOp>(&elseBlock()->back()); }

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, ValueRange lowerBounds,
    ValueRange upperBounds, ValueRange steps, ValueRange initVals,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addOperands(initVals);
  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lowerBounds.size()),
                                static_cast<int32_t>(upperBounds.size()),
                                static_cast<int32_t>(steps.size()),
                                static_cast<int32_t>(initVals.size())}));
  result.addTypes(initVals.getTypes());

  OpBuilder::InsertionGuard guard(builder);
  unsigned numIVs = steps.size();
  SmallVector<Type, 8> argTypes(numIVs, builder.getIndexType());
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion, {}, argTypes);

  if (bodyBuilderFn) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilderFn(builder, result.location,
                  bodyBlock->getArguments().take_front(numIVs),
                  bodyBlock->getArguments().drop_front(numIVs));
  }
  ParallelOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, ValueRange lowerBounds,
    ValueRange upperBounds, ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
  // Only pass a non-null wrapper if bodyBuilderFn is non-null itself. Make sure
  // we don't capture a reference to a temporary by constructing the lambda at
  // function level.
  auto wrappedBuilderFn = [&bodyBuilderFn](OpBuilder &nestedBuilder,
                                           Location nestedLoc, ValueRange ivs,
                                           ValueRange) {
    bodyBuilderFn(nestedBuilder, nestedLoc, ivs);
  };
  function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)> wrapper;
  if (bodyBuilderFn)
    wrapper = wrappedBuilderFn;

  build(builder, result, lowerBounds, upperBounds, steps, ValueRange(),
        wrapper);
}

static LogicalResult verify(ParallelOp op) {
  // Check that there is at least one value in lowerBound, upperBound and step.
  // It is sufficient to test only step, because it is ensured already that the
  // number of elements in lowerBound, upperBound and step are the same.
  Operation::operand_range stepValues = op.getStep();
  if (stepValues.empty())
    return op.emitOpError(
        "needs at least one tuple element for lowerBound, upperBound and step");

  // Check whether all constant step values are positive.
  for (Value stepValue : stepValues)
    if (auto cst = stepValue.getDefiningOp<arith::ConstantIndexOp>())
      if (cst.value() <= 0)
        return op.emitOpError("constant step operand must be positive");

  // Check that the body defines the same number of block arguments as the
  // number of tuple elements in step.
  Block *body = op.getBody();
  if (body->getNumArguments() != stepValues.size())
    return op.emitOpError()
           << "expects the same number of induction variables: "
           << body->getNumArguments()
           << " as bound and step values: " << stepValues.size();
  for (auto arg : body->getArguments())
    if (!arg.getType().isIndex())
      return op.emitOpError(
          "expects arguments for the induction variable to be of index type");

  // Check that the yield has no results
  Operation *yield = body->getTerminator();
  if (yield->getNumOperands() != 0)
    return yield->emitOpError() << "not allowed to have operands inside '"
                                << ParallelOp::getOperationName() << "'";

  // Check that the number of results is the same as the number of ReduceOps.
  SmallVector<ReduceOp, 4> reductions(body->getOps<ReduceOp>());
  auto resultsSize = op.getResults().size();
  auto reductionsSize = reductions.size();
  auto initValsSize = op.getInitVals().size();
  if (resultsSize != reductionsSize)
    return op.emitOpError()
           << "expects number of results: " << resultsSize
           << " to be the same as number of reductions: " << reductionsSize;
  if (resultsSize != initValsSize)
    return op.emitOpError()
           << "expects number of results: " << resultsSize
           << " to be the same as number of initial values: " << initValsSize;

  // Check that the types of the results and reductions are the same.
  for (auto resultAndReduce : llvm::zip(op.getResults(), reductions)) {
    auto resultType = std::get<0>(resultAndReduce).getType();
    auto reduceOp = std::get<1>(resultAndReduce);
    auto reduceType = reduceOp.getOperand().getType();
    if (resultType != reduceType)
      return reduceOp.emitOpError()
             << "expects type of reduce: " << reduceType
             << " to be the same as result type: " << resultType;
  }
  return success();
}

static ParseResult parseParallelOp(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::OperandType, 4> ivs;
  if (parser.parseRegionArgumentList(ivs, /*requiredOperandCount=*/-1,
                                     OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::OperandType, 4> lower;
  if (parser.parseEqual() ||
      parser.parseOperandList(lower, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(lower, builder.getIndexType(), result.operands))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> upper;
  if (parser.parseKeyword("to") ||
      parser.parseOperandList(upper, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return failure();

  // Parse step values.
  SmallVector<OpAsmParser::OperandType, 4> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, builder.getIndexType(), result.operands))
    return failure();

  // Parse init values.
  SmallVector<OpAsmParser::OperandType, 4> initVals;
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    if (parser.parseOperandList(initVals, /*requiredOperandCount=*/-1,
                                OpAsmParser::Delimiter::Paren))
      return failure();
  }

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  SmallVector<Type, 4> types(ivs.size(), builder.getIndexType());
  if (parser.parseRegion(*body, ivs, types))
    return failure();

  // Set `operand_segment_sizes` attribute.
  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lower.size()),
                                static_cast<int32_t>(upper.size()),
                                static_cast<int32_t>(steps.size()),
                                static_cast<int32_t>(initVals.size())}));

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (!initVals.empty())
    parser.resolveOperands(initVals, result.types, parser.getNameLoc(),
                           result.operands);
  // Add a terminator if none was parsed.
  ForOp::ensureTerminator(*body, builder, result.location);

  return success();
}

static void print(OpAsmPrinter &p, ParallelOp op) {
  p << " (" << op.getBody()->getArguments() << ") = (" << op.getLowerBound()
    << ") to (" << op.getUpperBound() << ") step (" << op.getStep() << ")";
  if (!op.getInitVals().empty())
    p << " init (" << op.getInitVals() << ")";
  p.printOptionalArrowTypeList(op.getResultTypes());
  p.printRegion(op.getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      op->getAttrs(), /*elidedAttrs=*/ParallelOp::getOperandSegmentSizeAttr());
}

Region &ParallelOp::getLoopBody() { return getRegion(); }

bool ParallelOp::isDefinedOutsideOfLoop(Value value) {
  return !getRegion().isAncestor(value.getParentRegion());
}

LogicalResult ParallelOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(*this);
  return success();
}

ParallelOp mlir::scf::getParallelForInductionVarOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return ParallelOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<ParallelOp>(containingOp);
}

namespace {
// Collapse loop dimensions that perform a single iteration.
struct CollapseSingleIterationLoops : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {
    BlockAndValueMapping mapping;
    // Compute new loop bounds that omit all single-iteration loop dimensions.
    SmallVector<Value, 2> newLowerBounds;
    SmallVector<Value, 2> newUpperBounds;
    SmallVector<Value, 2> newSteps;
    newLowerBounds.reserve(op.getLowerBound().size());
    newUpperBounds.reserve(op.getUpperBound().size());
    newSteps.reserve(op.getStep().size());
    for (auto dim : llvm::zip(op.getLowerBound(), op.getUpperBound(),
                              op.getStep(), op.getInductionVars())) {
      Value lowerBound, upperBound, step, iv;
      std::tie(lowerBound, upperBound, step, iv) = dim;
      // Collect the statically known loop bounds.
      auto lowerBoundConstant =
          dyn_cast_or_null<arith::ConstantIndexOp>(lowerBound.getDefiningOp());
      auto upperBoundConstant =
          dyn_cast_or_null<arith::ConstantIndexOp>(upperBound.getDefiningOp());
      auto stepConstant =
          dyn_cast_or_null<arith::ConstantIndexOp>(step.getDefiningOp());
      // Replace the loop induction variable by the lower bound if the loop
      // performs a single iteration. Otherwise, copy the loop bounds.
      if (lowerBoundConstant && upperBoundConstant && stepConstant &&
          (upperBoundConstant.value() - lowerBoundConstant.value()) > 0 &&
          (upperBoundConstant.value() - lowerBoundConstant.value()) <=
              stepConstant.value()) {
        mapping.map(iv, lowerBound);
      } else {
        newLowerBounds.push_back(lowerBound);
        newUpperBounds.push_back(upperBound);
        newSteps.push_back(step);
      }
    }
    // Exit if none of the loop dimensions perform a single iteration.
    if (newLowerBounds.size() == op.getLowerBound().size())
      return failure();

    if (newLowerBounds.empty()) {
      // All of the loop dimensions perform a single iteration. Inline
      // loop body and nested ReduceOp's
      SmallVector<Value> results;
      results.reserve(op.getInitVals().size());
      for (auto &bodyOp : op.getLoopBody().front().without_terminator()) {
        auto reduce = dyn_cast<ReduceOp>(bodyOp);
        if (!reduce) {
          rewriter.clone(bodyOp, mapping);
          continue;
        }
        Block &reduceBlock = reduce.getReductionOperator().front();
        auto initValIndex = results.size();
        mapping.map(reduceBlock.getArgument(0), op.getInitVals()[initValIndex]);
        mapping.map(reduceBlock.getArgument(1),
                    mapping.lookupOrDefault(reduce.getOperand()));
        for (auto &reduceBodyOp : reduceBlock.without_terminator())
          rewriter.clone(reduceBodyOp, mapping);

        auto result = mapping.lookupOrDefault(
            cast<ReduceReturnOp>(reduceBlock.getTerminator()).getResult());
        results.push_back(result);
      }
      rewriter.replaceOp(op, results);
      return success();
    }
    // Replace the parallel loop by lower-dimensional parallel loop.
    auto newOp =
        rewriter.create<ParallelOp>(op.getLoc(), newLowerBounds, newUpperBounds,
                                    newSteps, op.getInitVals(), nullptr);
    // Clone the loop body and remap the block arguments of the collapsed loops
    // (inlining does not support a cancellable block argument mapping).
    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                               newOp.getRegion().begin(), mapping);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

/// Removes parallel loops in which at least one lower/upper bound pair consists
/// of the same values - such loops have an empty iteration domain.
struct RemoveEmptyParallelLoops : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {
    for (auto dim : llvm::zip(op.getLowerBound(), op.getUpperBound())) {
      if (std::get<0>(dim) == std::get<1>(dim)) {
        rewriter.replaceOp(op, op.getInitVals());
        return success();
      }
    }
    return failure();
  }
};

struct MergeNestedParallelLoops : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {
    Block &outerBody = op.getLoopBody().front();
    if (!llvm::hasSingleElement(outerBody.without_terminator()))
      return failure();

    auto innerOp = dyn_cast<ParallelOp>(outerBody.front());
    if (!innerOp)
      return failure();

    auto hasVal = [](const auto &range, Value val) {
      return llvm::find(range, val) != range.end();
    };

    for (auto val : outerBody.getArguments())
      if (hasVal(innerOp.getLowerBound(), val) ||
          hasVal(innerOp.getUpperBound(), val) ||
          hasVal(innerOp.getStep(), val))
        return failure();

    // Reductions are not supported yet.
    if (!op.getInitVals().empty() || !innerOp.getInitVals().empty())
      return failure();

    auto bodyBuilder = [&](OpBuilder &builder, Location /*loc*/,
                           ValueRange iterVals, ValueRange) {
      Block &innerBody = innerOp.getLoopBody().front();
      assert(iterVals.size() ==
             (outerBody.getNumArguments() + innerBody.getNumArguments()));
      BlockAndValueMapping mapping;
      mapping.map(outerBody.getArguments(),
                  iterVals.take_front(outerBody.getNumArguments()));
      mapping.map(innerBody.getArguments(),
                  iterVals.take_back(innerBody.getNumArguments()));
      for (Operation &op : innerBody.without_terminator())
        builder.clone(op, mapping);
    };

    auto concatValues = [](const auto &first, const auto &second) {
      SmallVector<Value> ret;
      ret.reserve(first.size() + second.size());
      ret.assign(first.begin(), first.end());
      ret.append(second.begin(), second.end());
      return ret;
    };

    auto newLowerBounds =
        concatValues(op.getLowerBound(), innerOp.getLowerBound());
    auto newUpperBounds =
        concatValues(op.getUpperBound(), innerOp.getUpperBound());
    auto newSteps = concatValues(op.getStep(), innerOp.getStep());

    rewriter.replaceOpWithNewOp<ParallelOp>(op, newLowerBounds, newUpperBounds,
                                            newSteps, llvm::None, bodyBuilder);
    return success();
  }
};

} // namespace

void ParallelOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<CollapseSingleIterationLoops, RemoveEmptyParallelLoops,
              MergeNestedParallelLoops>(context);
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

void ReduceOp::build(
    OpBuilder &builder, OperationState &result, Value operand,
    function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuilderFn) {
  auto type = operand.getType();
  result.addOperands(operand);

  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion, {}, ArrayRef<Type>{type, type});
  if (bodyBuilderFn)
    bodyBuilderFn(builder, result.location, body->getArgument(0),
                  body->getArgument(1));
}

static LogicalResult verify(ReduceOp op) {
  // The region of a ReduceOp has two arguments of the same type as its operand.
  auto type = op.getOperand().getType();
  Block &block = op.getReductionOperator().front();
  if (block.empty())
    return op.emitOpError("the block inside reduce should not be empty");
  if (block.getNumArguments() != 2 ||
      llvm::any_of(block.getArguments(), [&](const BlockArgument &arg) {
        return arg.getType() != type;
      }))
    return op.emitOpError()
           << "expects two arguments to reduce block of type " << type;

  // Check that the block is terminated by a ReduceReturnOp.
  if (!isa<ReduceReturnOp>(block.getTerminator()))
    return op.emitOpError("the block inside reduce should be terminated with a "
                          "'scf.reduce.return' op");

  return success();
}

static ParseResult parseReduceOp(OpAsmParser &parser, OperationState &result) {
  // Parse an opening `(` followed by the reduced value followed by `)`
  OpAsmParser::OperandType operand;
  if (parser.parseLParen() || parser.parseOperand(operand) ||
      parser.parseRParen())
    return failure();

  Type resultType;
  // Parse the type of the operand (and also what reduce computes on).
  if (parser.parseColonType(resultType) ||
      parser.resolveOperand(operand, resultType, result.operands))
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, ReduceOp op) {
  p << "(" << op.getOperand() << ") ";
  p << " : " << op.getOperand().getType();
  p.printRegion(op.getReductionOperator());
}

//===----------------------------------------------------------------------===//
// ReduceReturnOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReduceReturnOp op) {
  // The type of the return value should be the same type as the type of the
  // operand of the enclosing ReduceOp.
  auto reduceOp = cast<ReduceOp>(op->getParentOp());
  Type reduceType = reduceOp.getOperand().getType();
  if (reduceType != op.getResult().getType())
    return op.emitOpError() << "needs to have type " << reduceType
                            << " (the type of the enclosing ReduceOp)";
  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

OperandRange WhileOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0 &&
         "WhileOp is expected to branch only to the first region");

  return getInits();
}

ConditionOp WhileOp::getConditionOp() {
  return cast<ConditionOp>(getBefore().front().getTerminator());
}

YieldOp WhileOp::getYieldOp() {
  return cast<YieldOp>(getAfter().front().getTerminator());
}

Block::BlockArgListType WhileOp::getBeforeArguments() {
  return getBefore().front().getArguments();
}

Block::BlockArgListType WhileOp::getAfterArguments() {
  return getAfter().front().getArguments();
}

void WhileOp::getSuccessorRegions(Optional<unsigned> index,
                                  ArrayRef<Attribute> operands,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  (void)operands;

  if (!index.hasValue()) {
    regions.emplace_back(&getBefore(), getBefore().getArguments());
    return;
  }

  assert(*index < 2 && "there are only two regions in a WhileOp");
  if (*index == 0) {
    regions.emplace_back(&getAfter(), getAfter().getArguments());
    regions.emplace_back(getResults());
    return;
  }

  regions.emplace_back(&getBefore(), getBefore().getArguments());
}

/// Parses a `while` op.
///
/// op ::= `scf.while` assignments `:` function-type region `do` region
///         `attributes` attribute-dict
/// initializer ::= /* empty */ | `(` assignment-list `)`
/// assignment-list ::= assignment | assignment `,` assignment-list
/// assignment ::= ssa-value `=` ssa-value
static ParseResult parseWhileOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> regionArgs, operands;
  Region *before = result.addRegion();
  Region *after = result.addRegion();

  OptionalParseResult listResult =
      parser.parseOptionalAssignmentList(regionArgs, operands);
  if (listResult.hasValue() && failed(listResult.getValue()))
    return failure();

  FunctionType functionType;
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  if (failed(parser.parseColonType(functionType)))
    return failure();

  result.addTypes(functionType.getResults());

  if (functionType.getNumInputs() != operands.size()) {
    return parser.emitError(typeLoc)
           << "expected as many input types as operands "
           << "(expected " << operands.size() << " got "
           << functionType.getNumInputs() << ")";
  }

  // Resolve input operands.
  if (failed(parser.resolveOperands(operands, functionType.getInputs(),
                                    parser.getCurrentLocation(),
                                    result.operands)))
    return failure();

  return failure(
      parser.parseRegion(*before, regionArgs, functionType.getInputs()) ||
      parser.parseKeyword("do") || parser.parseRegion(*after) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes));
}

/// Prints a `while` op.
static void print(OpAsmPrinter &p, scf::WhileOp op) {
  printInitializationList(p, op.getBefore().front().getArguments(),
                          op.getInits(), " ");
  p << " : ";
  p.printFunctionalType(op.getInits().getTypes(), op.getResults().getTypes());
  p.printRegion(op.getBefore(), /*printEntryBlockArgs=*/false);
  p << " do";
  p.printRegion(op.getAfter());
  p.printOptionalAttrDictWithKeyword(op->getAttrs());
}

/// Verifies that two ranges of types match, i.e. have the same number of
/// entries and that types are pairwise equals. Reports errors on the given
/// operation in case of mismatch.
template <typename OpTy>
static LogicalResult verifyTypeRangesMatch(OpTy op, TypeRange left,
                                           TypeRange right, StringRef message) {
  if (left.size() != right.size())
    return op.emitOpError("expects the same number of ") << message;

  for (unsigned i = 0, e = left.size(); i < e; ++i) {
    if (left[i] != right[i]) {
      InFlightDiagnostic diag = op.emitOpError("expects the same types for ")
                                << message;
      diag.attachNote() << "for argument " << i << ", found " << left[i]
                        << " and " << right[i];
      return diag;
    }
  }

  return success();
}

/// Verifies that the first block of the given `region` is terminated by a
/// YieldOp. Reports errors on the given operation if it is not the case.
template <typename TerminatorTy>
static TerminatorTy verifyAndGetTerminator(scf::WhileOp op, Region &region,
                                           StringRef errorMessage) {
  Operation *terminatorOperation = region.front().getTerminator();
  if (auto yield = dyn_cast_or_null<TerminatorTy>(terminatorOperation))
    return yield;

  auto diag = op.emitOpError(errorMessage);
  if (terminatorOperation)
    diag.attachNote(terminatorOperation->getLoc()) << "terminator here";
  return nullptr;
}

static LogicalResult verify(scf::WhileOp op) {
  if (failed(RegionBranchOpInterface::verifyTypes(op)))
    return failure();

  auto beforeTerminator = verifyAndGetTerminator<scf::ConditionOp>(
      op, op.getBefore(),
      "expects the 'before' region to terminate with 'scf.condition'");
  if (!beforeTerminator)
    return failure();

  auto afterTerminator = verifyAndGetTerminator<scf::YieldOp>(
      op, op.getAfter(),
      "expects the 'after' region to terminate with 'scf.yield'");
  return success(afterTerminator != nullptr);
}

namespace {
/// Replace uses of the condition within the do block with true, since otherwise
/// the block would not be evaluated.
///
/// scf.while (..) : (i1, ...) -> ... {
///  %condition = call @evaluate_condition() : () -> i1
///  scf.condition(%condition) %condition : i1, ...
/// } do {
/// ^bb0(%arg0: i1, ...):
///    use(%arg0)
///    ...
///
/// becomes
/// scf.while (..) : (i1, ...) -> ... {
///  %condition = call @evaluate_condition() : () -> i1
///  scf.condition(%condition) %condition : i1, ...
/// } do {
/// ^bb0(%arg0: i1, ...):
///    use(%true)
///    ...
struct WhileConditionTruth : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = op.getConditionOp();

    // These variables serve to prevent creating duplicate constants
    // and hold constant true or false values.
    Value constantTrue = nullptr;

    bool replaced = false;
    for (auto yieldedAndBlockArgs :
         llvm::zip(term.getArgs(), op.getAfterArguments())) {
      if (std::get<0>(yieldedAndBlockArgs) == term.getCondition()) {
        if (!std::get<1>(yieldedAndBlockArgs).use_empty()) {
          if (!constantTrue)
            constantTrue = rewriter.create<arith::ConstantOp>(
                op.getLoc(), term.getCondition().getType(),
                rewriter.getBoolAttr(true));

          std::get<1>(yieldedAndBlockArgs).replaceAllUsesWith(constantTrue);
          replaced = true;
        }
      }
    }
    return success(replaced);
  }
};

/// Remove WhileOp results that are also unused in 'after' block.
///
///  %0:2 = scf.while () : () -> (i32, i64) {
///    %condition = "test.condition"() : () -> i1
///    %v1 = "test.get_some_value"() : () -> i32
///    %v2 = "test.get_some_value"() : () -> i64
///    scf.condition(%condition) %v1, %v2 : i32, i64
///  } do {
///  ^bb0(%arg0: i32, %arg1: i64):
///    "test.use"(%arg0) : (i32) -> ()
///    scf.yield
///  }
///  return %0#0 : i32
///
/// becomes
///  %0 = scf.while () : () -> (i32) {
///    %condition = "test.condition"() : () -> i1
///    %v1 = "test.get_some_value"() : () -> i32
///    %v2 = "test.get_some_value"() : () -> i64
///    scf.condition(%condition) %v1 : i32
///  } do {
///  ^bb0(%arg0: i32):
///    "test.use"(%arg0) : (i32) -> ()
///    scf.yield
///  }
///  return %0 : i32
struct WhileUnusedResult : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = op.getConditionOp();
    auto afterArgs = op.getAfterArguments();
    auto termArgs = term.getArgs();

    // Collect results mapping, new terminator args and new result types.
    SmallVector<unsigned> newResultsIndices;
    SmallVector<Type> newResultTypes;
    SmallVector<Value> newTermArgs;
    bool needUpdate = false;
    for (const auto &it :
         llvm::enumerate(llvm::zip(op.getResults(), afterArgs, termArgs))) {
      auto i = static_cast<unsigned>(it.index());
      Value result = std::get<0>(it.value());
      Value afterArg = std::get<1>(it.value());
      Value termArg = std::get<2>(it.value());
      if (result.use_empty() && afterArg.use_empty()) {
        needUpdate = true;
      } else {
        newResultsIndices.emplace_back(i);
        newTermArgs.emplace_back(termArg);
        newResultTypes.emplace_back(result.getType());
      }
    }

    if (!needUpdate)
      return failure();

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                               newTermArgs);
    }

    auto newWhile =
        rewriter.create<WhileOp>(op.getLoc(), newResultTypes, op.getInits());

    Block &newAfterBlock = *rewriter.createBlock(
        &newWhile.getAfter(), /*insertPt*/ {}, newResultTypes);

    // Build new results list and new after block args (unused entries will be
    // null).
    SmallVector<Value> newResults(op.getNumResults());
    SmallVector<Value> newAfterBlockArgs(op.getNumResults());
    for (const auto &it : llvm::enumerate(newResultsIndices)) {
      newResults[it.value()] = newWhile.getResult(it.index());
      newAfterBlockArgs[it.value()] = newAfterBlock.getArgument(it.index());
    }

    rewriter.inlineRegionBefore(op.getBefore(), newWhile.getBefore(),
                                newWhile.getBefore().begin());

    Block &afterBlock = op.getAfter().front();
    rewriter.mergeBlocks(&afterBlock, &newAfterBlock, newAfterBlockArgs);

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

/// Replace operations equivalent to the condition in the do block with true,
/// since otherwise the block would not be evaluated.
///
/// scf.while (..) : (i32, ...) -> ... {
///  %z = ... : i32
///  %condition = cmpi pred %z, %a
///  scf.condition(%condition) %z : i32, ...
/// } do {
/// ^bb0(%arg0: i32, ...):
///    %condition2 = cmpi pred %arg0, %a
///    use(%condition2)
///    ...
///
/// becomes
/// scf.while (..) : (i32, ...) -> ... {
///  %z = ... : i32
///  %condition = cmpi pred %z, %a
///  scf.condition(%condition) %z : i32, ...
/// } do {
/// ^bb0(%arg0: i32, ...):
///    use(%true)
///    ...
struct WhileCmpCond : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    using namespace scf;
    auto cond = op.getConditionOp();
    auto cmp = cond.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return failure();
    bool changed = false;
    for (auto tup :
         llvm::zip(cond.getArgs(), op.getAfter().front().getArguments())) {
      for (size_t opIdx = 0; opIdx < 2; opIdx++) {
        if (std::get<0>(tup) != cmp.getOperand(opIdx))
          continue;
        for (OpOperand &u :
             llvm::make_early_inc_range(std::get<1>(tup).getUses())) {
          auto cmp2 = dyn_cast<arith::CmpIOp>(u.getOwner());
          if (!cmp2)
            continue;
          // For a binary operator 1-opIdx gets the other side.
          if (cmp2.getOperand(1 - opIdx) != cmp.getOperand(1 - opIdx))
            continue;
          bool samePredicate;
          if (cmp2.getPredicate() == cmp.getPredicate())
            samePredicate = true;
          else if (cmp2.getPredicate() ==
                   arith::invertPredicate(cmp.getPredicate()))
            samePredicate = false;
          else
            continue;

          rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(cmp2, samePredicate,
                                                            1);
          changed = true;
        }
      }
    }
    return success(changed);
  }
};

struct WhileUnusedArg : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {

    if (!llvm::any_of(op.getBeforeArguments(),
                      [](Value arg) { return arg.use_empty(); }))
      return failure();

    YieldOp yield = op.getYieldOp();

    // Collect results mapping, new terminator args and new result types.
    SmallVector<Value> newYields;
    SmallVector<Value> newInits;
    SmallVector<unsigned> argsToErase;
    for (const auto &it : llvm::enumerate(llvm::zip(
             op.getBeforeArguments(), yield.getOperands(), op.getInits()))) {
      Value beforeArg = std::get<0>(it.value());
      Value yieldValue = std::get<1>(it.value());
      Value initValue = std::get<2>(it.value());
      if (beforeArg.use_empty()) {
        argsToErase.push_back(it.index());
      } else {
        newYields.emplace_back(yieldValue);
        newInits.emplace_back(initValue);
      }
    }

    if (argsToErase.size() == 0)
      return failure();

    rewriter.startRootUpdate(op);
    op.getBefore().front().eraseArguments(argsToErase);
    rewriter.finalizeRootUpdate(op);

    WhileOp replacement =
        rewriter.create<WhileOp>(op.getLoc(), op.getResultTypes(), newInits);
    replacement.getBefore().takeBody(op.getBefore());
    replacement.getAfter().takeBody(op.getAfter());
    rewriter.replaceOp(op, replacement.getResults());

    rewriter.setInsertionPoint(yield);
    rewriter.replaceOpWithNewOp<YieldOp>(yield, newYields);
    return success();
  }
};
} // namespace

void WhileOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<WhileConditionTruth, WhileUnusedResult, WhileCmpCond,
                 WhileUnusedArg>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/SCF/SCFOps.cpp.inc"

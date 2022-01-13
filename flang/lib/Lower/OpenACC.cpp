//===-- OpenACC.cpp -- OpenACC directive lowering -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenACC.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

static const Fortran::parser::Name *
getDesignatorNameIfDataRef(const Fortran::parser::Designator &designator) {
  const auto *dataRef{std::get_if<Fortran::parser::DataRef>(&designator.u)};
  return dataRef ? std::get_if<Fortran::parser::Name>(&dataRef->u) : nullptr;
}

static void genObjectList(const Fortran::parser::AccObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          SmallVectorImpl<Value> &operands) {
  for (const auto &accObject : objectList.v) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const auto *name = getDesignatorNameIfDataRef(designator)) {
                const auto variable = converter.getSymbolAddress(*name->symbol);
                operands.push_back(variable);
              }
            },
            [&](const Fortran::parser::Name &name) {
              const auto variable = converter.getSymbolAddress(*name.symbol);
              operands.push_back(variable);
            }},
        accObject.u);
  }
}

template <typename Clause>
static void
genObjectListWithModifier(const Clause *x,
                          Fortran::lower::AbstractConverter &converter,
                          Fortran::parser::AccDataModifier::Modifier mod,
                          SmallVectorImpl<Value> &operandsWithModifier,
                          SmallVectorImpl<Value> &operands) {
  const Fortran::parser::AccObjectListWithModifier &listWithModifier = x->v;
  const Fortran::parser::AccObjectList &accObjectList =
      std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
  const auto &modifier =
      std::get<std::optional<Fortran::parser::AccDataModifier>>(
          listWithModifier.t);
  if (modifier && (*modifier).v == mod) {
    genObjectList(accObjectList, converter, operandsWithModifier);
  } else {
    genObjectList(accObjectList, converter, operands);
  }
}

static void addOperands(SmallVectorImpl<Value> &operands,
                        SmallVectorImpl<int32_t> &operandSegments,
                        const SmallVectorImpl<Value> &clauseOperands) {
  operands.append(clauseOperands.begin(), clauseOperands.end());
  operandSegments.push_back(clauseOperands.size());
}

static void addOperand(SmallVectorImpl<Value> &operands,
                       SmallVectorImpl<int32_t> &operandSegments,
                       const Value &clauseOperand) {
  if (clauseOperand) {
    operands.push_back(clauseOperand);
    operandSegments.push_back(1);
  } else {
    operandSegments.push_back(0);
  }
}

template <typename Op, typename Terminator>
static Op createRegionOp(Fortran::lower::FirOpBuilder &builder,
                         mlir::Location loc,
                         const SmallVectorImpl<Value> &operands,
                         const SmallVectorImpl<int32_t> &operandSegments) {
  llvm::ArrayRef<mlir::Type> argTy;
  Op op = builder.create<Op>(loc, argTy, operands);
  builder.createBlock(&op.getRegion());
  auto &block = op.getRegion().back();
  builder.setInsertionPointToStart(&block);
  builder.create<Terminator>(loc);

  op->setAttr(Op::getOperandSegmentSizeAttr(),
              builder.getI32VectorAttr(operandSegments));

  // Place the insertion point to the start of the first block.
  builder.setInsertionPointToStart(&block);

  return op;
}

template <typename Op>
static Op createSimpleOp(Fortran::lower::FirOpBuilder &builder,
                         mlir::Location loc,
                         const SmallVectorImpl<Value> &operands,
                         const SmallVectorImpl<int32_t> &operandSegments) {
  llvm::ArrayRef<mlir::Type> argTy;
  Op op = builder.create<Op>(loc, argTy, operands);
  op->setAttr(Op::getOperandSegmentSizeAttr(),
              builder.getI32VectorAttr(operandSegments));
  return op;
}

static void genACC(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {

  const auto &beginLoopDirective =
      std::get<Fortran::parser::AccBeginLoopDirective>(loopConstruct.t);
  const auto &loopDirective =
      std::get<Fortran::parser::AccLoopDirective>(beginLoopDirective.t);

  if (loopDirective.v == llvm::acc::ACCD_loop) {
    auto &firOpBuilder = converter.getFirOpBuilder();
    auto currentLocation = converter.getCurrentLocation();

    // Add attribute extracted from clauses.
    const auto &accClauseList =
        std::get<Fortran::parser::AccClauseList>(beginLoopDirective.t);

    mlir::Value workerNum;
    mlir::Value vectorLength;
    mlir::Value gangNum;
    mlir::Value gangStatic;
    SmallVector<Value, 2> tileOperands, privateOperands, reductionOperands;
    std::int64_t executionMapping = mlir::acc::OpenACCExecMapping::NONE;

    // Lower clauses values mapped to operands.
    for (const auto &clause : accClauseList.v) {
      if (const auto *gangClause =
              std::get_if<Fortran::parser::AccClause::Gang>(&clause.u)) {
        if (gangClause->v) {
          const Fortran::parser::AccGangArgument &x = *gangClause->v;
          if (const auto &gangNumValue =
                  std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                      x.t)) {
            gangNum = fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(gangNumValue.value())));
          }
          if (const auto &gangStaticValue =
                  std::get<std::optional<Fortran::parser::AccSizeExpr>>(x.t)) {
            const auto &expr =
                std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                    gangStaticValue.value().t);
            if (expr) {
              gangStatic = fir::getBase(
                  converter.genExprValue(*Fortran::semantics::GetExpr(*expr)));
            } else {
              // * was passed as value and will be represented as a -1 constant
              // integer.
              gangStatic = firOpBuilder.createIntegerConstant(
                  currentLocation, firOpBuilder.getIntegerType(32),
                  /* STAR */ -1);
            }
          }
        }
        executionMapping |= mlir::acc::OpenACCExecMapping::GANG;
      } else if (const auto *workerClause =
                     std::get_if<Fortran::parser::AccClause::Worker>(
                         &clause.u)) {
        if (workerClause->v) {
          workerNum = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(*workerClause->v)));
        }
        executionMapping |= mlir::acc::OpenACCExecMapping::WORKER;
      } else if (const auto *vectorClause =
                     std::get_if<Fortran::parser::AccClause::Vector>(
                         &clause.u)) {
        if (vectorClause->v) {
          vectorLength = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(*vectorClause->v)));
        }
        executionMapping |= mlir::acc::OpenACCExecMapping::VECTOR;
      } else if (const auto *tileClause =
                     std::get_if<Fortran::parser::AccClause::Tile>(&clause.u)) {
        const Fortran::parser::AccTileExprList &accTileExprList = tileClause->v;
        for (const auto &accTileExpr : accTileExprList.v) {
          const auto &expr =
              std::get<std::optional<Fortran::parser::ScalarIntConstantExpr>>(
                  accTileExpr.t);
          if (expr) {
            tileOperands.push_back(fir::getBase(
                converter.genExprValue(*Fortran::semantics::GetExpr(*expr))));
          } else {
            // * was passed as value and will be represented as a -1 constant
            // integer.
            mlir::Value tileStar = firOpBuilder.createIntegerConstant(
                currentLocation, firOpBuilder.getIntegerType(32),
                /* STAR */ -1);
            tileOperands.push_back(tileStar);
          }
        }
      } else if (const auto *privateClause =
                     std::get_if<Fortran::parser::AccClause::Private>(
                         &clause.u)) {
        genObjectList(privateClause->v, converter, privateOperands);
      }
      // Reduction clause is left out for the moment as the clause will probably
      // end up having its own operation.
    }

    // Prepare the operand segement size attribute and the operands value range.
    SmallVector<Value, 8> operands;
    SmallVector<int32_t, 8> operandSegments;
    addOperand(operands, operandSegments, gangNum);
    addOperand(operands, operandSegments, gangStatic);
    addOperand(operands, operandSegments, workerNum);
    addOperand(operands, operandSegments, vectorLength);
    addOperands(operands, operandSegments, tileOperands);
    addOperands(operands, operandSegments, privateOperands);
    addOperands(operands, operandSegments, reductionOperands);

    auto loopOp = createRegionOp<mlir::acc::LoopOp, mlir::acc::YieldOp>(
        firOpBuilder, currentLocation, operands, operandSegments);

    loopOp->setAttr(mlir::acc::LoopOp::getExecutionMappingAttrName(),
                    firOpBuilder.getI64IntegerAttr(executionMapping));

    // Lower clauses mapped to attributes
    for (const auto &clause : accClauseList.v) {
      if (const auto *collapseClause =
              std::get_if<Fortran::parser::AccClause::Collapse>(&clause.u)) {
        const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
        const auto collapseValue = Fortran::evaluate::ToInt64(*expr);
        if (collapseValue) {
          loopOp->setAttr(mlir::acc::LoopOp::getCollapseAttrName(),
                          firOpBuilder.getI64IntegerAttr(*collapseValue));
        }
      } else if (std::get_if<Fortran::parser::AccClause::Seq>(&clause.u)) {
        loopOp->setAttr(mlir::acc::LoopOp::getSeqAttrName(),
                        firOpBuilder.getUnitAttr());
      } else if (std::get_if<Fortran::parser::AccClause::Independent>(
                     &clause.u)) {
        loopOp->setAttr(mlir::acc::LoopOp::getIndependentAttrName(),
                        firOpBuilder.getUnitAttr());
      } else if (std::get_if<Fortran::parser::AccClause::Auto>(&clause.u)) {
        loopOp->setAttr(mlir::acc::LoopOp::getAutoAttrName(),
                        firOpBuilder.getUnitAttr());
      }
    }
  }
}

static void
genACCParallelOp(Fortran::lower::AbstractConverter &converter,
                 const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value async;
  mlir::Value numGangs;
  mlir::Value numWorkers;
  mlir::Value vectorLength;
  mlir::Value ifCond;
  mlir::Value selfCond;
  SmallVector<Value, 2> waitOperands, reductionOperands, copyOperands,
      copyinOperands, copyinReadonlyOperands, copyoutOperands,
      copyoutZeroOperands, createOperands, createZeroOperands, noCreateOperands,
      presentOperands, devicePtrOperands, attachOperands, privateOperands,
      firstprivateOperands;

  // Async, wait and self clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addSelfAttr = false;

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const auto &clause : accClauseList.v) {
    if (const auto *asyncClause =
            std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      const auto &asyncClauseValue = asyncClause->v;
      if (asyncClauseValue) { // async has a value.
        async = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*asyncClauseValue)));
      } else {
        addAsyncAttr = true;
      }
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      const auto &waitClauseValue = waitClause->v;
      if (waitClauseValue) { // wait has a value.
        const Fortran::parser::AccWaitArgument &waitArg = *waitClauseValue;
        const std::list<Fortran::parser::ScalarIntExpr> &waitList =
            std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
        for (const Fortran::parser::ScalarIntExpr &value : waitList) {
          Value v = fir::getBase(
              converter.genExprValue(*Fortran::semantics::GetExpr(value)));
          waitOperands.push_back(v);
        }
      } else {
        addWaitAttr = true;
      }
    } else if (const auto *numGangsClause =
                   std::get_if<Fortran::parser::AccClause::NumGangs>(
                       &clause.u)) {
      numGangs = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numGangsClause->v)));
    } else if (const auto *numWorkersClause =
                   std::get_if<Fortran::parser::AccClause::NumWorkers>(
                       &clause.u)) {
      numWorkers = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numWorkersClause->v)));
    } else if (const auto *vectorLengthClause =
                   std::get_if<Fortran::parser::AccClause::VectorLength>(
                       &clause.u)) {
      vectorLength = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(vectorLengthClause->v)));
    } else if (const auto *ifClause =
                   std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      Value cond = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(ifClause->v)));
      ifCond = firOpBuilder.createConvert(currentLocation,
                                          firOpBuilder.getI1Type(), cond);
    } else if (const auto *selfClause =
                   std::get_if<Fortran::parser::AccClause::Self>(&clause.u)) {
      const Fortran::parser::AccSelfClause &accSelfClause = selfClause->v;
      if (const auto *optCondition =
              std::get_if<std::optional<Fortran::parser::ScalarLogicalExpr>>(
                  &accSelfClause.u)) {
        if (*optCondition) {
          Value cond = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(*optCondition)));
          selfCond = firOpBuilder.createConvert(currentLocation,
                                                firOpBuilder.getI1Type(), cond);
        } else {
          addSelfAttr = true;
        }
      }
    } else if (const auto *copyClause =
                   std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
      genObjectList(copyClause->v, converter, copyOperands);
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Copyin>(
          copyinClause, converter,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          copyinReadonlyOperands, copyinOperands);
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Copyout>(
          copyoutClause, converter,
          Fortran::parser::AccDataModifier::Modifier::Zero, copyoutZeroOperands,
          copyoutOperands);
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Create>(
          createClause, converter,
          Fortran::parser::AccDataModifier::Modifier::Zero, createZeroOperands,
          createOperands);
    } else if (const auto *noCreateClause =
                   std::get_if<Fortran::parser::AccClause::NoCreate>(
                       &clause.u)) {
      genObjectList(noCreateClause->v, converter, noCreateOperands);
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      genObjectList(presentClause->v, converter, presentOperands);
    } else if (const auto *devicePtrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genObjectList(devicePtrClause->v, converter, devicePtrOperands);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      genObjectList(attachClause->v, converter, attachOperands);
    } else if (const auto *privateClause =
                   std::get_if<Fortran::parser::AccClause::Private>(
                       &clause.u)) {
      genObjectList(privateClause->v, converter, privateOperands);
    } else if (const auto *firstprivateClause =
                   std::get_if<Fortran::parser::AccClause::Firstprivate>(
                       &clause.u)) {
      genObjectList(firstprivateClause->v, converter, firstprivateOperands);
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  SmallVector<Value, 8> operands;
  SmallVector<int32_t, 8> operandSegments;
  addOperand(operands, operandSegments, async);
  addOperands(operands, operandSegments, waitOperands);
  addOperand(operands, operandSegments, numGangs);
  addOperand(operands, operandSegments, numWorkers);
  addOperand(operands, operandSegments, vectorLength);
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, selfCond);
  addOperands(operands, operandSegments, reductionOperands);
  addOperands(operands, operandSegments, copyOperands);
  addOperands(operands, operandSegments, copyinOperands);
  addOperands(operands, operandSegments, copyinReadonlyOperands);
  addOperands(operands, operandSegments, copyoutOperands);
  addOperands(operands, operandSegments, copyoutZeroOperands);
  addOperands(operands, operandSegments, createOperands);
  addOperands(operands, operandSegments, createZeroOperands);
  addOperands(operands, operandSegments, noCreateOperands);
  addOperands(operands, operandSegments, presentOperands);
  addOperands(operands, operandSegments, devicePtrOperands);
  addOperands(operands, operandSegments, attachOperands);
  addOperands(operands, operandSegments, privateOperands);
  addOperands(operands, operandSegments, firstprivateOperands);

  auto parallelOp = createRegionOp<mlir::acc::ParallelOp, mlir::acc::YieldOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    parallelOp->setAttr(mlir::acc::ParallelOp::getAsyncAttrName(),
                        firOpBuilder.getUnitAttr());
  if (addWaitAttr)
    parallelOp->setAttr(mlir::acc::ParallelOp::getWaitAttrName(),
                        firOpBuilder.getUnitAttr());
  if (addSelfAttr)
    parallelOp->setAttr(mlir::acc::ParallelOp::getSelfAttrName(),
                        firOpBuilder.getUnitAttr());
}

static void genACCDataOp(Fortran::lower::AbstractConverter &converter,
                         const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond;
  SmallVector<Value, 2> copyOperands, copyinOperands, copyinReadonlyOperands,
      copyoutOperands, copyoutZeroOperands, createOperands, createZeroOperands,
      noCreateOperands, presentOperands, deviceptrOperands, attachOperands;

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const auto &clause : accClauseList.v) {
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      Value cond = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(ifClause->v)));
      ifCond = firOpBuilder.createConvert(currentLocation,
                                          firOpBuilder.getI1Type(), cond);
    } else if (const auto *copyClause =
                   std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
      genObjectList(copyClause->v, converter, copyOperands);
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Copyin>(
          copyinClause, converter,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          copyinReadonlyOperands, copyinOperands);
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Copyout>(
          copyoutClause, converter,
          Fortran::parser::AccDataModifier::Modifier::Zero, copyoutZeroOperands,
          copyoutOperands);
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Create>(
          createClause, converter,
          Fortran::parser::AccDataModifier::Modifier::Zero, createZeroOperands,
          createOperands);
    } else if (const auto *noCreateClause =
                   std::get_if<Fortran::parser::AccClause::NoCreate>(
                       &clause.u)) {
      genObjectList(noCreateClause->v, converter, noCreateOperands);
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      genObjectList(presentClause->v, converter, presentOperands);
    } else if (const auto *deviceptrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genObjectList(deviceptrClause->v, converter, deviceptrOperands);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      genObjectList(attachClause->v, converter, attachOperands);
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  SmallVector<Value, 8> operands;
  SmallVector<int32_t, 8> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperands(operands, operandSegments, copyOperands);
  addOperands(operands, operandSegments, copyinOperands);
  addOperands(operands, operandSegments, copyinReadonlyOperands);
  addOperands(operands, operandSegments, copyoutOperands);
  addOperands(operands, operandSegments, copyoutZeroOperands);
  addOperands(operands, operandSegments, createOperands);
  addOperands(operands, operandSegments, createZeroOperands);
  addOperands(operands, operandSegments, noCreateOperands);
  addOperands(operands, operandSegments, presentOperands);
  addOperands(operands, operandSegments, deviceptrOperands);
  addOperands(operands, operandSegments, attachOperands);

  createRegionOp<mlir::acc::DataOp, mlir::acc::TerminatorOp>(
      firOpBuilder, currentLocation, operands, operandSegments);
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::AccBeginBlockDirective>(blockConstruct.t);
  const auto &blockDirective =
      std::get<Fortran::parser::AccBlockDirective>(beginBlockDirective.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(beginBlockDirective.t);

  if (blockDirective.v == llvm::acc::ACCD_parallel) {
    genACCParallelOp(converter, accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_data) {
    genACCDataOp(converter, accClauseList);
  }
}

static void
genACCEnterDataOp(Fortran::lower::AbstractConverter &converter,
                  const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, async, waitDevnum;
  SmallVector<Value, 2> copyinOperands, createOperands, createZeroOperands,
      attachOperands, waitOperands;

  // Async, wait and self clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const auto &clause : accClauseList.v) {
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      mlir::Value cond = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(ifClause->v)));
      ifCond = firOpBuilder.createConvert(currentLocation,
                                          firOpBuilder.getI1Type(), cond);
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      const auto &asyncClauseValue = asyncClause->v;
      if (asyncClauseValue) { // async has a value.
        async = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*asyncClauseValue)));
      } else {
        addAsyncAttr = true;
      }
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      const auto &waitClauseValue = waitClause->v;
      if (waitClauseValue) { // wait has a value.
        const Fortran::parser::AccWaitArgument &waitArg = *waitClauseValue;
        const std::list<Fortran::parser::ScalarIntExpr> &waitList =
            std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
        for (const Fortran::parser::ScalarIntExpr &value : waitList) {
          mlir::Value v = fir::getBase(
              converter.genExprValue(*Fortran::semantics::GetExpr(value)));
          waitOperands.push_back(v);
        }

        const std::optional<Fortran::parser::ScalarIntExpr> &waitDevnumValue =
            std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
        if (waitDevnumValue)
          waitDevnum = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(*waitDevnumValue)));
      } else {
        addWaitAttr = true;
      }
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          copyinClause->v;
      const Fortran::parser::AccObjectList &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      genObjectList(accObjectList, converter, copyinOperands);
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Create>(
          createClause, converter,
          Fortran::parser::AccDataModifier::Modifier::Zero, createZeroOperands,
          createOperands);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      genObjectList(attachClause->v, converter, attachOperands);
    } else {
      llvm::report_fatal_error(
          "Unknown clause in ENTER DATA directive lowering");
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  SmallVector<mlir::Value, 16> operands;
  SmallVector<int32_t, 8> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, copyinOperands);
  addOperands(operands, operandSegments, createOperands);
  addOperands(operands, operandSegments, createZeroOperands);
  addOperands(operands, operandSegments, attachOperands);

  auto enterDataOp = createSimpleOp<mlir::acc::EnterDataOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    enterDataOp.asyncAttr(firOpBuilder.getUnitAttr());
  if (addWaitAttr)
    enterDataOp.waitAttr(firOpBuilder.getUnitAttr());
}

static void
genACCExitDataOp(Fortran::lower::AbstractConverter &converter,
                 const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, async, waitDevnum;
  SmallVector<Value, 2> copyoutOperands, deleteOperands, detachOperands,
      waitOperands;

  // Async and wait clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addFinalizeAttr = false;

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const auto &clause : accClauseList.v) {
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      Value cond = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(ifClause->v)));
      ifCond = firOpBuilder.createConvert(currentLocation,
                                          firOpBuilder.getI1Type(), cond);
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      const auto &asyncClauseValue = asyncClause->v;
      if (asyncClauseValue) { // async has a value.
        async = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*asyncClauseValue)));
      } else {
        addAsyncAttr = true;
      }
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      const auto &waitClauseValue = waitClause->v;
      if (waitClauseValue) { // wait has a value.
        const Fortran::parser::AccWaitArgument &waitArg = *waitClauseValue;
        const std::list<Fortran::parser::ScalarIntExpr> &waitList =
            std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
        for (const Fortran::parser::ScalarIntExpr &value : waitList) {
          Value v = fir::getBase(
              converter.genExprValue(*Fortran::semantics::GetExpr(value)));
          waitOperands.push_back(v);
        }

        const std::optional<Fortran::parser::ScalarIntExpr> &waitDevnumValue =
            std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
        if (waitDevnumValue)
          waitDevnum = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(*waitDevnumValue)));
      } else {
        addWaitAttr = true;
      }
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          copyoutClause->v;
      const Fortran::parser::AccObjectList &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      genObjectList(accObjectList, converter, copyoutOperands);
    } else if (const auto *deleteClause =
                   std::get_if<Fortran::parser::AccClause::Delete>(&clause.u)) {
      genObjectList(deleteClause->v, converter, deleteOperands);
    } else if (const auto *detachClause =
                   std::get_if<Fortran::parser::AccClause::Detach>(&clause.u)) {
      genObjectList(detachClause->v, converter, detachOperands);
    } else if (std::get_if<Fortran::parser::AccClause::Finalize>(&clause.u)) {
      addFinalizeAttr = true;
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  SmallVector<mlir::Value, 14> operands;
  SmallVector<int32_t, 7> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, copyoutOperands);
  addOperands(operands, operandSegments, deleteOperands);
  addOperands(operands, operandSegments, detachOperands);

  auto exitDataOp = createSimpleOp<mlir::acc::ExitDataOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    exitDataOp.asyncAttr(firOpBuilder.getUnitAttr());
  if (addWaitAttr)
    exitDataOp.waitAttr(firOpBuilder.getUnitAttr());
  if (addFinalizeAttr)
    exitDataOp.finalizeAttr(firOpBuilder.getUnitAttr());
}

template <typename Op>
static void
genACCInitShutdownOp(Fortran::lower::AbstractConverter &converter,
                     const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, deviceNum;
  SmallVector<Value, 2> deviceTypeOperands;

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const auto &clause : accClauseList.v) {
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      mlir::Value cond = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(ifClause->v)));
      ifCond = firOpBuilder.createConvert(currentLocation,
                                          firOpBuilder.getI1Type(), cond);
    } else if (const auto *deviceNumClause =
                   std::get_if<Fortran::parser::AccClause::DeviceNum>(
                       &clause.u)) {
      deviceNum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(deviceNumClause->v)));
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {

      const auto &deviceTypeValue = deviceTypeClause->v;
      if (deviceTypeValue) {
        for (const auto &scalarIntExpr : *deviceTypeValue) {
          mlir::Value expr = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(scalarIntExpr)));
          deviceTypeOperands.push_back(expr);
        }
      } else {
        // * was passed as value and will be represented as a -1 constant
        // integer.
        mlir::Value star = firOpBuilder.createIntegerConstant(
            currentLocation, firOpBuilder.getIntegerType(32), /* STAR */ -1);
        deviceTypeOperands.push_back(star);
      }
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  SmallVector<mlir::Value, 6> operands;
  SmallVector<int32_t, 3> operandSegments;
  addOperands(operands, operandSegments, deviceTypeOperands);
  addOperand(operands, operandSegments, deviceNum);
  addOperand(operands, operandSegments, ifCond);

  createSimpleOp<Op>(firOpBuilder, currentLocation, operands, operandSegments);
}

static void
genACCUpdateOp(Fortran::lower::AbstractConverter &converter,
               const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, async, waitDevnum;
  SmallVector<Value, 2> hostOperands, deviceOperands, waitOperands,
      deviceTypeOperands;

  // Async and wait clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addIfPresentAttr = false;

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const auto &clause : accClauseList.v) {
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      mlir::Value cond = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(ifClause->v)));
      ifCond = firOpBuilder.createConvert(currentLocation,
                                          firOpBuilder.getI1Type(), cond);
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      const auto &asyncClauseValue = asyncClause->v;
      if (asyncClauseValue) { // async has a value.
        async = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*asyncClauseValue)));
      } else {
        addAsyncAttr = true;
      }
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      const auto &waitClauseValue = waitClause->v;
      if (waitClauseValue) { // wait has a value.
        const Fortran::parser::AccWaitArgument &waitArg = *waitClauseValue;
        const std::list<Fortran::parser::ScalarIntExpr> &waitList =
            std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
        for (const Fortran::parser::ScalarIntExpr &value : waitList) {
          mlir::Value v = fir::getBase(
              converter.genExprValue(*Fortran::semantics::GetExpr(value)));
          waitOperands.push_back(v);
        }

        const std::optional<Fortran::parser::ScalarIntExpr> &waitDevnumValue =
            std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
        if (waitDevnumValue)
          waitDevnum = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(*waitDevnumValue)));
      } else {
        addWaitAttr = true;
      }
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {

      const auto &deviceTypeValue = deviceTypeClause->v;
      if (deviceTypeValue) {
        for (const auto &scalarIntExpr : *deviceTypeValue) {
          mlir::Value expr = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(scalarIntExpr)));
          deviceTypeOperands.push_back(expr);
        }
      } else {
        // * was passed as value and will be represented as a -1 constant
        // integer.
        mlir::Value star = firOpBuilder.createIntegerConstant(
            currentLocation, firOpBuilder.getIntegerType(32), /* STAR */ -1);
        deviceTypeOperands.push_back(star);
      }
    } else if (const auto *hostClause =
                   std::get_if<Fortran::parser::AccClause::Host>(&clause.u)) {
      genObjectList(hostClause->v, converter, hostOperands);
    } else if (const auto *deviceClause =
                   std::get_if<Fortran::parser::AccClause::Device>(&clause.u)) {
      genObjectList(deviceClause->v, converter, deviceOperands);
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  SmallVector<mlir::Value, 14> operands;
  SmallVector<int32_t, 7> operandSegments;
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, deviceTypeOperands);
  addOperand(operands, operandSegments, ifCond);
  addOperands(operands, operandSegments, hostOperands);
  addOperands(operands, operandSegments, deviceOperands);

  auto updateOp = createSimpleOp<mlir::acc::UpdateOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    updateOp.asyncAttr(firOpBuilder.getUnitAttr());
  if (addWaitAttr)
    updateOp.waitAttr(firOpBuilder.getUnitAttr());
  if (addIfPresentAttr)
    updateOp.ifPresentAttr(firOpBuilder.getUnitAttr());
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCStandaloneConstruct &standaloneConstruct) {
  const auto &standaloneDirective =
      std::get<Fortran::parser::AccStandaloneDirective>(standaloneConstruct.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(standaloneConstruct.t);

  if (standaloneDirective.v == llvm::acc::Directive::ACCD_enter_data) {
    genACCEnterDataOp(converter, accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_exit_data) {
    genACCExitDataOp(converter, accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_init) {
    genACCInitShutdownOp<mlir::acc::InitOp>(converter, accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_shutdown) {
    genACCInitShutdownOp<mlir::acc::ShutdownOp>(converter, accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_set) {
    TODO(converter.genLocation(), "OpenACC set directive not lowered yet!");
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_update) {
    genACCUpdateOp(converter, accClauseList);
  }
}

static void genACC(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {

  const auto &waitArgument =
      std::get<std::optional<Fortran::parser::AccWaitArgument>>(
          waitConstruct.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(waitConstruct.t);

  mlir::Value ifCond, waitDevnum, async;
  SmallVector<mlir::Value, 2> waitOperands;

  // Async clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();

  if (waitArgument) { // wait has a value.
    const Fortran::parser::AccWaitArgument &waitArg = *waitArgument;
    const std::list<Fortran::parser::ScalarIntExpr> &waitList =
        std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    for (const Fortran::parser::ScalarIntExpr &value : waitList) {
      mlir::Value v = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(value)));
      waitOperands.push_back(v);
    }

    const std::optional<Fortran::parser::ScalarIntExpr> &waitDevnumValue =
        std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    if (waitDevnumValue)
      waitDevnum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(*waitDevnumValue)));
  }

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const auto &clause : accClauseList.v) {
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      mlir::Value cond = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(ifClause->v)));
      ifCond = firOpBuilder.createConvert(currentLocation,
                                          firOpBuilder.getI1Type(), cond);
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      const auto &asyncClauseValue = asyncClause->v;
      if (asyncClauseValue) { // async has a value.
        async = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*asyncClauseValue)));
      } else {
        addAsyncAttr = true;
      }
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  SmallVector<mlir::Value, 8> operands;
  SmallVector<int32_t, 4> operandSegments;
  addOperands(operands, operandSegments, waitOperands);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperand(operands, operandSegments, ifCond);

  auto waitOp = createSimpleOp<mlir::acc::WaitOp>(firOpBuilder, currentLocation,
                                                  operands, operandSegments);

  if (addAsyncAttr)
    waitOp.asyncAttr(firOpBuilder.getUnitAttr());
}

void Fortran::lower::genOpenACCConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenACCConstruct &accConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenACCBlockConstruct &blockConstruct) {
            genACC(converter, eval, blockConstruct);
          },
          [&](const Fortran::parser::OpenACCCombinedConstruct
                  &combinedConstruct) {
            TODO(converter.genLocation(),
                 "OpenACC Combined construct not lowered yet!");
          },
          [&](const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {
            genACC(converter, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenACCStandaloneConstruct
                  &standaloneConstruct) {
            genACC(converter, eval, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenACCRoutineConstruct
                  &routineConstruct) {
            TODO(converter.genLocation(),
                 "OpenACC Routine construct not lowered yet!");
          },
          [&](const Fortran::parser::OpenACCCacheConstruct &cacheConstruct) {
            TODO(converter.genLocation(),
                 "OpenACC Cache construct not lowered yet!");
          },
          [&](const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {
            genACC(converter, eval, waitConstruct);
          },
          [&](const Fortran::parser::OpenACCAtomicConstruct &atomicConstruct) {
            TODO(converter.genLocation(),
                 "OpenACC Atomic construct not lowered yet!");
          },
      },
      accConstruct.u);
}

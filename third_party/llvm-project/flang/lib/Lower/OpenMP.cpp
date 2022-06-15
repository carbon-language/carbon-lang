//===-- OpenMP.cpp -- Open MP directive lowering --------------------------===//
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

#include "flang/Lower/OpenMP.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace mlir;

int64_t Fortran::lower::getCollapseValue(
    const Fortran::parser::OmpClauseList &clauseList) {
  for (const auto &clause : clauseList.v) {
    if (const auto &collapseClause =
            std::get_if<Fortran::parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      return Fortran::evaluate::ToInt64(*expr).value();
    }
  }
  return 1;
}

static const Fortran::parser::Name *
getDesignatorNameIfDataRef(const Fortran::parser::Designator &designator) {
  const auto *dataRef = std::get_if<Fortran::parser::DataRef>(&designator.u);
  return dataRef ? std::get_if<Fortran::parser::Name>(&dataRef->u) : nullptr;
}

static Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject) {
  Fortran::semantics::Symbol *sym = nullptr;
  std::visit(Fortran::common::visitors{
                 [&](const Fortran::parser::Designator &designator) {
                   if (const Fortran::parser::Name *name =
                           getDesignatorNameIfDataRef(designator)) {
                     sym = name->symbol;
                   }
                 },
                 [&](const Fortran::parser::Name &name) { sym = name.symbol; }},
             ompObject.u);
  return sym;
}

template <typename T>
static void createPrivateVarSyms(Fortran::lower::AbstractConverter &converter,
                                 const T *clause) {
  const Fortran::parser::OmpObjectList &ompObjectList = clause->v;
  for (const Fortran::parser::OmpObject &ompObject : ompObjectList.v) {
    Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
    // Privatization for symbols which are pre-determined (like loop index
    // variables) happen separately, for everything else privatize here.
    if (sym->test(Fortran::semantics::Symbol::Flag::OmpPreDetermined))
      continue;
    if constexpr (std::is_same_v<T, Fortran::parser::OmpClause::Firstprivate>) {
      converter.copyHostAssociateVar(*sym);
    } else {
      bool success = converter.createHostAssociateVarClone(*sym);
      (void)success;
      assert(success && "Privatization failed due to existing binding");
    }
  }
}

static void privatizeVars(Fortran::lower::AbstractConverter &converter,
                          const Fortran::parser::OmpClauseList &opClauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (const auto &privateClause =
            std::get_if<Fortran::parser::OmpClause::Private>(&clause.u)) {
      createPrivateVarSyms(converter, privateClause);
    } else if (const auto &firstPrivateClause =
                   std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                       &clause.u)) {
      createPrivateVarSyms(converter, firstPrivateClause);
    }
  }
  firOpBuilder.restoreInsertionPoint(insPt);
}

/// The COMMON block is a global structure. \p commonValue is the base address
/// of the the COMMON block. As the offset from the symbol \p sym, generate the
/// COMMON block member value (commonValue + offset) for the symbol.
/// FIXME: Share the code with `instantiateCommon` in ConvertVariable.cpp.
static mlir::Value
genCommonBlockMember(Fortran::lower::AbstractConverter &converter,
                     const Fortran::semantics::Symbol &sym,
                     mlir::Value commonValue) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::IntegerType i8Ty = firOpBuilder.getIntegerType(8);
  mlir::Type i8Ptr = firOpBuilder.getRefType(i8Ty);
  mlir::Type seqTy = firOpBuilder.getRefType(firOpBuilder.getVarLenSeqTy(i8Ty));
  mlir::Value base =
      firOpBuilder.createConvert(currentLocation, seqTy, commonValue);
  std::size_t byteOffset = sym.GetUltimate().offset();
  mlir::Value offs = firOpBuilder.createIntegerConstant(
      currentLocation, firOpBuilder.getIndexType(), byteOffset);
  mlir::Value varAddr = firOpBuilder.create<fir::CoordinateOp>(
      currentLocation, i8Ptr, base, mlir::ValueRange{offs});
  mlir::Type symType = converter.genType(sym);
  return firOpBuilder.createConvert(currentLocation,
                                    firOpBuilder.getRefType(symType), varAddr);
}

// Get the extended value for \p val by extracting additional variable
// information from \p base.
static fir::ExtendedValue getExtendedValue(fir::ExtendedValue base,
                                           mlir::Value val) {
  return base.match(
      [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
        return fir::MutableBoxValue(val, box.nonDeferredLenParams(), {});
      },
      [&](const auto &) -> fir::ExtendedValue {
        return fir::substBase(base, val);
      });
}

static void threadPrivatizeVars(Fortran::lower::AbstractConverter &converter,
                                Fortran::lower::pft::Evaluation &eval) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  auto insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  // Get the original ThreadprivateOp corresponding to the symbol and use the
  // symbol value from that opeartion to create one ThreadprivateOp copy
  // operation inside the parallel region.
  auto genThreadprivateOp = [&](Fortran::lower::SymbolRef sym) -> mlir::Value {
    mlir::Value symOriThreadprivateValue = converter.getSymbolAddress(sym);
    mlir::Operation *op = symOriThreadprivateValue.getDefiningOp();
    assert(mlir::isa<mlir::omp::ThreadprivateOp>(op) &&
           "The threadprivate operation not created");
    mlir::Value symValue =
        mlir::dyn_cast<mlir::omp::ThreadprivateOp>(op).sym_addr();
    return firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  };

  llvm::SetVector<const Fortran::semantics::Symbol *> threadprivateSyms;
  converter.collectSymbolSet(
      eval, threadprivateSyms,
      Fortran::semantics::Symbol::Flag::OmpThreadprivate);

  // For a COMMON block, the ThreadprivateOp is generated for itself instead of
  // its members, so only bind the value of the new copied ThreadprivateOp
  // inside the parallel region to the common block symbol only once for
  // multiple members in one COMMON block.
  llvm::SetVector<const Fortran::semantics::Symbol *> commonSyms;
  for (std::size_t i = 0; i < threadprivateSyms.size(); i++) {
    auto sym = threadprivateSyms[i];
    mlir::Value symThreadprivateValue;
    if (const Fortran::semantics::Symbol *common =
            Fortran::semantics::FindCommonBlockContaining(sym->GetUltimate())) {
      mlir::Value commonThreadprivateValue;
      if (commonSyms.contains(common)) {
        commonThreadprivateValue = converter.getSymbolAddress(*common);
      } else {
        commonThreadprivateValue = genThreadprivateOp(*common);
        converter.bindSymbol(*common, commonThreadprivateValue);
        commonSyms.insert(common);
      }
      symThreadprivateValue =
          genCommonBlockMember(converter, *sym, commonThreadprivateValue);
    } else {
      symThreadprivateValue = genThreadprivateOp(*sym);
    }

    fir::ExtendedValue sexv = converter.getSymbolExtendedValue(*sym);
    fir::ExtendedValue symThreadprivateExv =
        getExtendedValue(sexv, symThreadprivateValue);
    converter.bindSymbol(*sym, symThreadprivateExv);
  }

  firOpBuilder.restoreInsertionPoint(insPt);
}

static void genObjectList(const Fortran::parser::OmpObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          llvm::SmallVectorImpl<Value> &operands) {
  auto addOperands = [&](Fortran::lower::SymbolRef sym) {
    const mlir::Value variable = converter.getSymbolAddress(sym);
    if (variable) {
      operands.push_back(variable);
    } else {
      if (const auto *details =
              sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
        operands.push_back(converter.getSymbolAddress(details->symbol()));
        converter.copySymbolBinding(details->symbol(), sym);
      }
    }
  };
  for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
    Fortran::semantics::Symbol *sym = getOmpObjectSymbol(ompObject);
    addOperands(*sym);
  }
}

static mlir::Type getLoopVarType(Fortran::lower::AbstractConverter &converter,
                                 std::size_t loopVarTypeSize) {
  // OpenMP runtime requires 32-bit or 64-bit loop variables.
  loopVarTypeSize = loopVarTypeSize * 8;
  if (loopVarTypeSize < 32) {
    loopVarTypeSize = 32;
  } else if (loopVarTypeSize > 64) {
    loopVarTypeSize = 64;
    mlir::emitWarning(converter.getCurrentLocation(),
                      "OpenMP loop iteration variable cannot have more than 64 "
                      "bits size and will be narrowed into 64 bits.");
  }
  assert((loopVarTypeSize == 32 || loopVarTypeSize == 64) &&
         "OpenMP loop iteration variable size must be transformed into 32-bit "
         "or 64-bit");
  return converter.getFirOpBuilder().getIntegerType(loopVarTypeSize);
}

/// Create empty blocks for the current region.
/// These blocks replace blocks parented to an enclosing region.
void createEmptyRegionBlocks(
    fir::FirOpBuilder &firOpBuilder,
    std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
  auto *region = &firOpBuilder.getRegion();
  for (auto &eval : evaluationList) {
    if (eval.block) {
      if (eval.block->empty()) {
        eval.block->erase();
        eval.block = firOpBuilder.createBlock(region);
      } else {
        [[maybe_unused]] auto &terminatorOp = eval.block->back();
        assert((mlir::isa<mlir::omp::TerminatorOp>(terminatorOp) ||
                mlir::isa<mlir::omp::YieldOp>(terminatorOp)) &&
               "expected terminator op");
      }
    }
    if (!eval.isDirective() && eval.hasNestedEvaluations())
      createEmptyRegionBlocks(firOpBuilder, eval.getNestedEvaluations());
  }
}

/// Create the body (block) for an OpenMP Operation.
///
/// \param [in]    op - the operation the body belongs to.
/// \param [inout] converter - converter to use for the clauses.
/// \param [in]    loc - location in source code.
/// \param [in]    eval - current PFT node/evaluation.
/// \oaran [in]    clauses - list of clauses to process.
/// \param [in]    args - block arguments (induction variable[s]) for the
////                      region.
/// \param [in]    outerCombined - is this an outer operation - prevents
///                                privatization.
template <typename Op>
static void
createBodyOfOp(Op &op, Fortran::lower::AbstractConverter &converter,
               mlir::Location &loc, Fortran::lower::pft::Evaluation &eval,
               const Fortran::parser::OmpClauseList *clauses = nullptr,
               const SmallVector<const Fortran::semantics::Symbol *> &args = {},
               bool outerCombined = false) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  // If an argument for the region is provided then create the block with that
  // argument. Also update the symbol's address with the mlir argument value.
  // e.g. For loops the argument is the induction variable. And all further
  // uses of the induction variable should use this mlir value.
  mlir::Operation *storeOp = nullptr;
  if (args.size()) {
    std::size_t loopVarTypeSize = 0;
    for (const Fortran::semantics::Symbol *arg : args)
      loopVarTypeSize = std::max(loopVarTypeSize, arg->GetUltimate().size());
    mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
    SmallVector<Type> tiv;
    SmallVector<Location> locs;
    for (int i = 0; i < (int)args.size(); i++) {
      tiv.push_back(loopVarType);
      locs.push_back(loc);
    }
    firOpBuilder.createBlock(&op.getRegion(), {}, tiv, locs);
    int argIndex = 0;
    // The argument is not currently in memory, so make a temporary for the
    // argument, and store it there, then bind that location to the argument.
    for (const Fortran::semantics::Symbol *arg : args) {
      mlir::Value val =
          fir::getBase(op.getRegion().front().getArgument(argIndex));
      mlir::Value temp = firOpBuilder.createTemporary(
          loc, loopVarType,
          llvm::ArrayRef<mlir::NamedAttribute>{
              Fortran::lower::getAdaptToByRefAttr(firOpBuilder)});
      storeOp = firOpBuilder.create<fir::StoreOp>(loc, val, temp);
      converter.bindSymbol(*arg, temp);
      argIndex++;
    }
  } else {
    firOpBuilder.createBlock(&op.getRegion());
  }
  // Set the insert for the terminator operation to go at the end of the
  // block - this is either empty or the block with the stores above,
  // the end of the block works for both.
  mlir::Block &block = op.getRegion().back();
  firOpBuilder.setInsertionPointToEnd(&block);

  // If it is an unstructured region and is not the outer region of a combined
  // construct, create empty blocks for all evaluations.
  if (eval.lowerAsUnstructured() && !outerCombined)
    createEmptyRegionBlocks(firOpBuilder, eval.getNestedEvaluations());

  // Insert the terminator.
  if constexpr (std::is_same_v<Op, omp::WsLoopOp> ||
                std::is_same_v<Op, omp::SimdLoopOp>) {
    mlir::ValueRange results;
    firOpBuilder.create<mlir::omp::YieldOp>(loc, results);
  } else {
    firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
  }

  // Reset the insert point to before the terminator.
  if (storeOp)
    firOpBuilder.setInsertionPointAfter(storeOp);
  else
    firOpBuilder.setInsertionPointToStart(&block);

  // Handle privatization. Do not privatize if this is the outer operation.
  if (clauses && !outerCombined)
    privatizeVars(converter, *clauses);

  if (std::is_same_v<Op, omp::ParallelOp>)
    threadPrivatizeVars(converter, eval);
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPSimpleStandaloneConstruct
                       &simpleStandaloneConstruct) {
  const auto &directive =
      std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
          simpleStandaloneConstruct.t);
  switch (directive.v) {
  default:
    break;
  case llvm::omp::Directive::OMPD_barrier:
    converter.getFirOpBuilder().create<mlir::omp::BarrierOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    converter.getFirOpBuilder().create<mlir::omp::TaskwaitOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    converter.getFirOpBuilder().create<mlir::omp::TaskyieldOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    TODO(converter.getCurrentLocation(), "OMPD_target_enter_data");
  case llvm::omp::Directive::OMPD_target_exit_data:
    TODO(converter.getCurrentLocation(), "OMPD_target_exit_data");
  case llvm::omp::Directive::OMPD_target_update:
    TODO(converter.getCurrentLocation(), "OMPD_target_update");
  case llvm::omp::Directive::OMPD_ordered:
    TODO(converter.getCurrentLocation(), "OMPD_ordered");
  }
}

static void
genAllocateClause(Fortran::lower::AbstractConverter &converter,
                  const Fortran::parser::OmpAllocateClause &ompAllocateClause,
                  SmallVector<Value> &allocatorOperands,
                  SmallVector<Value> &allocateOperands) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  Fortran::lower::StatementContext stmtCtx;

  mlir::Value allocatorOperand;
  const Fortran::parser::OmpObjectList &ompObjectList =
      std::get<Fortran::parser::OmpObjectList>(ompAllocateClause.t);
  const auto &allocatorValue =
      std::get<std::optional<Fortran::parser::OmpAllocateClause::Allocator>>(
          ompAllocateClause.t);
  // Check if allocate clause has allocator specified. If so, add it
  // to list of allocators, otherwise, add default allocator to
  // list of allocators.
  if (allocatorValue) {
    allocatorOperand = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(allocatorValue->v), stmtCtx));
    allocatorOperands.insert(allocatorOperands.end(), ompObjectList.v.size(),
                             allocatorOperand);
  } else {
    allocatorOperand = firOpBuilder.createIntegerConstant(
        currentLocation, firOpBuilder.getI32Type(), 1);
    allocatorOperands.insert(allocatorOperands.end(), ompObjectList.v.size(),
                             allocatorOperand);
  }
  genObjectList(ompObjectList, converter, allocateOperands);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPSimpleStandaloneConstruct
                  &simpleStandaloneConstruct) {
            genOMP(converter, eval, simpleStandaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
            SmallVector<Value, 4> operandRange;
            if (const auto &ompObjectList =
                    std::get<std::optional<Fortran::parser::OmpObjectList>>(
                        flushConstruct.t))
              genObjectList(*ompObjectList, converter, operandRange);
            const auto &memOrderClause = std::get<std::optional<
                std::list<Fortran::parser::OmpMemoryOrderClause>>>(
                flushConstruct.t);
            if (memOrderClause.has_value() && memOrderClause->size() > 0)
              TODO(converter.getCurrentLocation(),
                   "Handle OmpMemoryOrderClause");
            converter.getFirOpBuilder().create<mlir::omp::FlushOp>(
                converter.getCurrentLocation(), operandRange);
          },
          [&](const Fortran::parser::OpenMPCancelConstruct &cancelConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
          [&](const Fortran::parser::OpenMPCancellationPointConstruct
                  &cancellationPointConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
      },
      standaloneConstruct.u);
}

static omp::ClauseProcBindKindAttr genProcBindKindAttr(
    fir::FirOpBuilder &firOpBuilder,
    const Fortran::parser::OmpClause::ProcBind *procBindClause) {
  omp::ClauseProcBindKind pbKind;
  switch (procBindClause->v.v) {
  case Fortran::parser::OmpProcBindClause::Type::Master:
    pbKind = omp::ClauseProcBindKind::Master;
    break;
  case Fortran::parser::OmpProcBindClause::Type::Close:
    pbKind = omp::ClauseProcBindKind::Close;
    break;
  case Fortran::parser::OmpProcBindClause::Type::Spread:
    pbKind = omp::ClauseProcBindKind::Spread;
    break;
  case Fortran::parser::OmpProcBindClause::Type::Primary:
    pbKind = omp::ClauseProcBindKind::Primary;
    break;
  }
  return omp::ClauseProcBindKindAttr::get(firOpBuilder.getContext(), pbKind);
}

/* When parallel is used in a combined construct, then use this function to
 * create the parallel operation. It handles the parallel specific clauses
 * and leaves the rest for handling at the inner operations.
 * TODO: Refactor clause handling
 */
template <typename Directive>
static void
createCombinedParallelOp(Fortran::lower::AbstractConverter &converter,
                         Fortran::lower::pft::Evaluation &eval,
                         const Directive &directive) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  Fortran::lower::StatementContext stmtCtx;
  llvm::ArrayRef<mlir::Type> argTy;
  mlir::Value ifClauseOperand, numThreadsClauseOperand;
  SmallVector<Value> allocatorOperands, allocateOperands;
  mlir::omp::ClauseProcBindKindAttr procBindKindAttr;
  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(directive.t);
  // TODO: Handle the following clauses
  // 1. default
  // 2. copyin
  // Note: rest of the clauses are handled when the inner operation is created
  for (const Fortran::parser::OmpClause &clause : opClauseList.v) {
    if (const auto &ifClause =
            std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->v.t);
      mlir::Value ifVal = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
      ifClauseOperand = firOpBuilder.createConvert(
          currentLocation, firOpBuilder.getI1Type(), ifVal);
    } else if (const auto &numThreadsClause =
                   std::get_if<Fortran::parser::OmpClause::NumThreads>(
                       &clause.u)) {
      numThreadsClauseOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numThreadsClause->v), stmtCtx));
    } else if (const auto &procBindClause =
                   std::get_if<Fortran::parser::OmpClause::ProcBind>(
                       &clause.u)) {
      procBindKindAttr = genProcBindKindAttr(firOpBuilder, procBindClause);
    }
  }
  // Create and insert the operation.
  auto parallelOp = firOpBuilder.create<mlir::omp::ParallelOp>(
      currentLocation, argTy, ifClauseOperand, numThreadsClauseOperand,
      allocateOperands, allocatorOperands, /*reduction_vars=*/ValueRange(),
      /*reductions=*/nullptr, procBindKindAttr);

  createBodyOfOp<omp::ParallelOp>(parallelOp, converter, currentLocation, eval,
                                  &opClauseList, /*iv=*/{},
                                  /*isCombined=*/true);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &blockDirective =
      std::get<Fortran::parser::OmpBlockDirective>(beginBlockDirective.t);
  const auto &endBlockDirective =
      std::get<Fortran::parser::OmpEndBlockDirective>(blockConstruct.t);
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  Fortran::lower::StatementContext stmtCtx;
  llvm::ArrayRef<mlir::Type> argTy;
  mlir::Value ifClauseOperand, numThreadsClauseOperand, finalClauseOperand,
      priorityClauseOperand;
  mlir::omp::ClauseProcBindKindAttr procBindKindAttr;
  SmallVector<Value> allocateOperands, allocatorOperands;
  mlir::UnitAttr nowaitAttr, untiedAttr, mergeableAttr;

  const auto &opClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginBlockDirective.t);
  for (const auto &clause : opClauseList.v) {
    if (const auto &ifClause =
            std::get_if<Fortran::parser::OmpClause::If>(&clause.u)) {
      auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->v.t);
      mlir::Value ifVal = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
      ifClauseOperand = firOpBuilder.createConvert(
          currentLocation, firOpBuilder.getI1Type(), ifVal);
    } else if (const auto &numThreadsClause =
                   std::get_if<Fortran::parser::OmpClause::NumThreads>(
                       &clause.u)) {
      // OMPIRBuilder expects `NUM_THREAD` clause as a `Value`.
      numThreadsClauseOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numThreadsClause->v), stmtCtx));
    } else if (const auto &procBindClause =
                   std::get_if<Fortran::parser::OmpClause::ProcBind>(
                       &clause.u)) {
      procBindKindAttr = genProcBindKindAttr(firOpBuilder, procBindClause);
    } else if (const auto &allocateClause =
                   std::get_if<Fortran::parser::OmpClause::Allocate>(
                       &clause.u)) {
      genAllocateClause(converter, allocateClause->v, allocatorOperands,
                        allocateOperands);
    } else if (std::get_if<Fortran::parser::OmpClause::Private>(&clause.u) ||
               std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                   &clause.u)) {
      // Privatisation clauses are handled elsewhere.
      continue;
    } else if (std::get_if<Fortran::parser::OmpClause::Threads>(&clause.u)) {
      // Nothing needs to be done for threads clause.
      continue;
    } else if (const auto &finalClause =
                   std::get_if<Fortran::parser::OmpClause::Final>(&clause.u)) {
      mlir::Value finalVal = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(finalClause->v), stmtCtx));
      finalClauseOperand = firOpBuilder.createConvert(
          currentLocation, firOpBuilder.getI1Type(), finalVal);
    } else if (std::get_if<Fortran::parser::OmpClause::Untied>(&clause.u)) {
      untiedAttr = firOpBuilder.getUnitAttr();
    } else if (std::get_if<Fortran::parser::OmpClause::Mergeable>(&clause.u)) {
      mergeableAttr = firOpBuilder.getUnitAttr();
    } else if (const auto &priorityClause =
                   std::get_if<Fortran::parser::OmpClause::Priority>(
                       &clause.u)) {
      priorityClauseOperand = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(priorityClause->v), stmtCtx));
    } else {
      TODO(currentLocation, "OpenMP Block construct clauses");
    }
  }

  for (const auto &clause :
       std::get<Fortran::parser::OmpClauseList>(endBlockDirective.t).v) {
    if (std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u))
      nowaitAttr = firOpBuilder.getUnitAttr();
  }

  if (blockDirective.v == llvm::omp::OMPD_parallel) {
    // Create and insert the operation.
    auto parallelOp = firOpBuilder.create<mlir::omp::ParallelOp>(
        currentLocation, argTy, ifClauseOperand, numThreadsClauseOperand,
        allocateOperands, allocatorOperands, /*reduction_vars=*/ValueRange(),
        /*reductions=*/nullptr, procBindKindAttr);
    createBodyOfOp<omp::ParallelOp>(parallelOp, converter, currentLocation,
                                    eval, &opClauseList);
  } else if (blockDirective.v == llvm::omp::OMPD_master) {
    auto masterOp =
        firOpBuilder.create<mlir::omp::MasterOp>(currentLocation, argTy);
    createBodyOfOp<omp::MasterOp>(masterOp, converter, currentLocation, eval);
  } else if (blockDirective.v == llvm::omp::OMPD_single) {
    auto singleOp = firOpBuilder.create<mlir::omp::SingleOp>(
        currentLocation, allocateOperands, allocatorOperands, nowaitAttr);
    createBodyOfOp<omp::SingleOp>(singleOp, converter, currentLocation, eval);
  } else if (blockDirective.v == llvm::omp::OMPD_ordered) {
    auto orderedOp = firOpBuilder.create<mlir::omp::OrderedRegionOp>(
        currentLocation, /*simd=*/nullptr);
    createBodyOfOp<omp::OrderedRegionOp>(orderedOp, converter, currentLocation,
                                         eval);
  } else if (blockDirective.v == llvm::omp::OMPD_task) {
    auto taskOp = firOpBuilder.create<mlir::omp::TaskOp>(
        currentLocation, ifClauseOperand, finalClauseOperand, untiedAttr,
        mergeableAttr, /*in_reduction_vars=*/ValueRange(),
        /*in_reductions=*/nullptr, priorityClauseOperand, allocateOperands,
        allocatorOperands);
    createBodyOfOp(taskOp, converter, currentLocation, eval, &opClauseList);
  } else {
    TODO(converter.getCurrentLocation(), "Unhandled block directive");
  }
}

static mlir::omp::ScheduleModifier
translateModifier(const Fortran::parser::OmpScheduleModifierType &m) {
  switch (m.v) {
  case Fortran::parser::OmpScheduleModifierType::ModType::Monotonic:
    return mlir::omp::ScheduleModifier::monotonic;
  case Fortran::parser::OmpScheduleModifierType::ModType::Nonmonotonic:
    return mlir::omp::ScheduleModifier::nonmonotonic;
  case Fortran::parser::OmpScheduleModifierType::ModType::Simd:
    return mlir::omp::ScheduleModifier::simd;
  }
  return mlir::omp::ScheduleModifier::none;
}

static mlir::omp::ScheduleModifier
getScheduleModifier(const Fortran::parser::OmpScheduleClause &x) {
  const auto &modifier =
      std::get<std::optional<Fortran::parser::OmpScheduleModifier>>(x.t);
  // The input may have the modifier any order, so we look for one that isn't
  // SIMD. If modifier is not set at all, fall down to the bottom and return
  // "none".
  if (modifier) {
    const auto &modType1 =
        std::get<Fortran::parser::OmpScheduleModifier::Modifier1>(modifier->t);
    if (modType1.v.v ==
        Fortran::parser::OmpScheduleModifierType::ModType::Simd) {
      const auto &modType2 = std::get<
          std::optional<Fortran::parser::OmpScheduleModifier::Modifier2>>(
          modifier->t);
      if (modType2 &&
          modType2->v.v !=
              Fortran::parser::OmpScheduleModifierType::ModType::Simd)
        return translateModifier(modType2->v);

      return mlir::omp::ScheduleModifier::none;
    }

    return translateModifier(modType1.v);
  }
  return mlir::omp::ScheduleModifier::none;
}

static mlir::omp::ScheduleModifier
getSIMDModifier(const Fortran::parser::OmpScheduleClause &x) {
  const auto &modifier =
      std::get<std::optional<Fortran::parser::OmpScheduleModifier>>(x.t);
  // Either of the two possible modifiers in the input can be the SIMD modifier,
  // so look in either one, and return simd if we find one. Not found = return
  // "none".
  if (modifier) {
    const auto &modType1 =
        std::get<Fortran::parser::OmpScheduleModifier::Modifier1>(modifier->t);
    if (modType1.v.v == Fortran::parser::OmpScheduleModifierType::ModType::Simd)
      return mlir::omp::ScheduleModifier::simd;

    const auto &modType2 = std::get<
        std::optional<Fortran::parser::OmpScheduleModifier::Modifier2>>(
        modifier->t);
    if (modType2 && modType2->v.v ==
                        Fortran::parser::OmpScheduleModifierType::ModType::Simd)
      return mlir::omp::ScheduleModifier::simd;
  }
  return mlir::omp::ScheduleModifier::none;
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  llvm::SmallVector<mlir::Value> lowerBound, upperBound, step, linearVars,
      linearStepVars, reductionVars;
  mlir::Value scheduleChunkClauseOperand;
  mlir::Attribute scheduleClauseOperand, collapseClauseOperand,
      noWaitClauseOperand, orderedClauseOperand, orderClauseOperand;
  const auto &loopOpClauseList = std::get<Fortran::parser::OmpClauseList>(
      std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t).t);

  const auto ompDirective =
      std::get<Fortran::parser::OmpLoopDirective>(
          std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t).t)
          .v;
  if (llvm::omp::OMPD_parallel_do == ompDirective) {
    createCombinedParallelOp<Fortran::parser::OmpBeginLoopDirective>(
        converter, eval,
        std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t));
  } else if (llvm::omp::OMPD_do != ompDirective &&
             llvm::omp::OMPD_simd != ompDirective) {
    TODO(converter.getCurrentLocation(), "Construct enclosing do loop");
  }

  // Collect the loops to collapse.
  auto *doConstructEval = &eval.getFirstNestedEvaluation();

  std::int64_t collapseValue =
      Fortran::lower::getCollapseValue(loopOpClauseList);
  std::size_t loopVarTypeSize = 0;
  SmallVector<const Fortran::semantics::Symbol *> iv;
  do {
    auto *doLoop = &doConstructEval->getFirstNestedEvaluation();
    auto *doStmt = doLoop->getIf<Fortran::parser::NonLabelDoStmt>();
    assert(doStmt && "Expected do loop to be in the nested evaluation");
    const auto &loopControl =
        std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
    const Fortran::parser::LoopControl::Bounds *bounds =
        std::get_if<Fortran::parser::LoopControl::Bounds>(&loopControl->u);
    assert(bounds && "Expected bounds for worksharing do loop");
    Fortran::lower::StatementContext stmtCtx;
    lowerBound.push_back(fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(bounds->lower), stmtCtx)));
    upperBound.push_back(fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(bounds->upper), stmtCtx)));
    if (bounds->step) {
      step.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(bounds->step), stmtCtx)));
    } else { // If `step` is not present, assume it as `1`.
      step.push_back(firOpBuilder.createIntegerConstant(
          currentLocation, firOpBuilder.getIntegerType(32), 1));
    }
    iv.push_back(bounds->name.thing.symbol);
    loopVarTypeSize = std::max(loopVarTypeSize,
                               bounds->name.thing.symbol->GetUltimate().size());

    collapseValue--;
    doConstructEval =
        &*std::next(doConstructEval->getNestedEvaluations().begin());
  } while (collapseValue > 0);

  for (const auto &clause : loopOpClauseList.v) {
    if (const auto &scheduleClause =
            std::get_if<Fortran::parser::OmpClause::Schedule>(&clause.u)) {
      if (const auto &chunkExpr =
              std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                  scheduleClause->v.t)) {
        if (const auto *expr = Fortran::semantics::GetExpr(*chunkExpr)) {
          Fortran::lower::StatementContext stmtCtx;
          scheduleChunkClauseOperand =
              fir::getBase(converter.genExprValue(*expr, stmtCtx));
        }
      }
    }
  }

  // The types of lower bound, upper bound, and step are converted into the
  // type of the loop variable if necessary.
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  for (unsigned it = 0; it < (unsigned)lowerBound.size(); it++) {
    lowerBound[it] = firOpBuilder.createConvert(currentLocation, loopVarType,
                                                lowerBound[it]);
    upperBound[it] = firOpBuilder.createConvert(currentLocation, loopVarType,
                                                upperBound[it]);
    step[it] =
        firOpBuilder.createConvert(currentLocation, loopVarType, step[it]);
  }

  // 2.9.3.1 SIMD construct
  // TODO: Support all the clauses
  if (llvm::omp::OMPD_simd == ompDirective) {
    TypeRange resultType;
    auto SimdLoopOp = firOpBuilder.create<mlir::omp::SimdLoopOp>(
        currentLocation, resultType, lowerBound, upperBound, step);
    createBodyOfOp<omp::SimdLoopOp>(SimdLoopOp, converter, currentLocation,
                                    eval, &loopOpClauseList, iv);
    return;
  }

  // FIXME: Add support for following clauses:
  // 1. linear
  // 2. order
  auto wsLoopOp = firOpBuilder.create<mlir::omp::WsLoopOp>(
      currentLocation, lowerBound, upperBound, step, linearVars, linearStepVars,
      reductionVars, /*reductions=*/nullptr,
      scheduleClauseOperand.dyn_cast_or_null<omp::ClauseScheduleKindAttr>(),
      scheduleChunkClauseOperand, /*schedule_modifiers=*/nullptr,
      /*simd_modifier=*/nullptr,
      collapseClauseOperand.dyn_cast_or_null<IntegerAttr>(),
      noWaitClauseOperand.dyn_cast_or_null<UnitAttr>(),
      orderedClauseOperand.dyn_cast_or_null<IntegerAttr>(),
      orderClauseOperand.dyn_cast_or_null<omp::ClauseOrderKindAttr>(),
      /*inclusive=*/firOpBuilder.getUnitAttr());

  // Handle attribute based clauses.
  for (const Fortran::parser::OmpClause &clause : loopOpClauseList.v) {
    if (const auto &orderedClause =
            std::get_if<Fortran::parser::OmpClause::Ordered>(&clause.u)) {
      if (orderedClause->v.has_value()) {
        const auto *expr = Fortran::semantics::GetExpr(orderedClause->v);
        const std::optional<std::int64_t> orderedClauseValue =
            Fortran::evaluate::ToInt64(*expr);
        wsLoopOp.ordered_valAttr(
            firOpBuilder.getI64IntegerAttr(*orderedClauseValue));
      } else {
        wsLoopOp.ordered_valAttr(firOpBuilder.getI64IntegerAttr(0));
      }
    } else if (const auto &collapseClause =
                   std::get_if<Fortran::parser::OmpClause::Collapse>(
                       &clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      const std::optional<std::int64_t> collapseValue =
          Fortran::evaluate::ToInt64(*expr);
      wsLoopOp.collapse_valAttr(firOpBuilder.getI64IntegerAttr(*collapseValue));
    } else if (const auto &scheduleClause =
                   std::get_if<Fortran::parser::OmpClause::Schedule>(
                       &clause.u)) {
      mlir::MLIRContext *context = firOpBuilder.getContext();
      const auto &scheduleType = scheduleClause->v;
      const auto &scheduleKind =
          std::get<Fortran::parser::OmpScheduleClause::ScheduleType>(
              scheduleType.t);
      switch (scheduleKind) {
      case Fortran::parser::OmpScheduleClause::ScheduleType::Static:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Static));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Dynamic:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Dynamic));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Guided:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Guided));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Auto:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Auto));
        break;
      case Fortran::parser::OmpScheduleClause::ScheduleType::Runtime:
        wsLoopOp.schedule_valAttr(omp::ClauseScheduleKindAttr::get(
            context, omp::ClauseScheduleKind::Runtime));
        break;
      }
      mlir::omp::ScheduleModifier scheduleModifier =
          getScheduleModifier(scheduleClause->v);
      if (scheduleModifier != mlir::omp::ScheduleModifier::none)
        wsLoopOp.schedule_modifierAttr(
            omp::ScheduleModifierAttr::get(context, scheduleModifier));
      if (getSIMDModifier(scheduleClause->v) !=
          mlir::omp::ScheduleModifier::none)
        wsLoopOp.simd_modifierAttr(firOpBuilder.getUnitAttr());
    }
  }
  // In FORTRAN `nowait` clause occur at the end of `omp do` directive.
  // i.e
  // !$omp do
  // <...>
  // !$omp end do nowait
  if (const auto &endClauseList =
          std::get<std::optional<Fortran::parser::OmpEndLoopDirective>>(
              loopConstruct.t)) {
    const auto &clauseList =
        std::get<Fortran::parser::OmpClauseList>((*endClauseList).t);
    for (const Fortran::parser::OmpClause &clause : clauseList.v)
      if (std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u))
        wsLoopOp.nowaitAttr(firOpBuilder.getUnitAttr());
  }

  createBodyOfOp<omp::WsLoopOp>(wsLoopOp, converter, currentLocation, eval,
                                &loopOpClauseList, iv);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPCriticalConstruct &criticalConstruct) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  std::string name;
  const Fortran::parser::OmpCriticalDirective &cd =
      std::get<Fortran::parser::OmpCriticalDirective>(criticalConstruct.t);
  if (std::get<std::optional<Fortran::parser::Name>>(cd.t).has_value()) {
    name =
        std::get<std::optional<Fortran::parser::Name>>(cd.t).value().ToString();
  }

  uint64_t hint = 0;
  const auto &clauseList = std::get<Fortran::parser::OmpClauseList>(cd.t);
  for (const Fortran::parser::OmpClause &clause : clauseList.v)
    if (auto hintClause =
            std::get_if<Fortran::parser::OmpClause::Hint>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(hintClause->v);
      hint = *Fortran::evaluate::ToInt64(*expr);
      break;
    }

  mlir::omp::CriticalOp criticalOp = [&]() {
    if (name.empty()) {
      return firOpBuilder.create<mlir::omp::CriticalOp>(currentLocation,
                                                        FlatSymbolRefAttr());
    } else {
      mlir::ModuleOp module = firOpBuilder.getModule();
      mlir::OpBuilder modBuilder(module.getBodyRegion());
      auto global = module.lookupSymbol<mlir::omp::CriticalDeclareOp>(name);
      if (!global)
        global = modBuilder.create<mlir::omp::CriticalDeclareOp>(
            currentLocation, name, hint);
      return firOpBuilder.create<mlir::omp::CriticalOp>(
          currentLocation, mlir::FlatSymbolRefAttr::get(
                               firOpBuilder.getContext(), global.sym_name()));
    }
  }();
  createBodyOfOp<omp::CriticalOp>(criticalOp, converter, currentLocation, eval);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPSectionConstruct &sectionConstruct) {

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  mlir::omp::SectionOp sectionOp =
      firOpBuilder.create<mlir::omp::SectionOp>(currentLocation);
  createBodyOfOp<omp::SectionOp>(sectionOp, converter, currentLocation, eval);
}

// TODO: Add support for reduction
static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPSectionsConstruct &sectionsConstruct) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  SmallVector<Value> reductionVars, allocateOperands, allocatorOperands;
  mlir::UnitAttr noWaitClauseOperand;
  const auto &sectionsClauseList = std::get<Fortran::parser::OmpClauseList>(
      std::get<Fortran::parser::OmpBeginSectionsDirective>(sectionsConstruct.t)
          .t);
  for (const Fortran::parser::OmpClause &clause : sectionsClauseList.v) {

    // Reduction Clause
    if (std::get_if<Fortran::parser::OmpClause::Reduction>(&clause.u)) {
      TODO(currentLocation, "OMPC_Reduction");

      // Allocate clause
    } else if (const auto &allocateClause =
                   std::get_if<Fortran::parser::OmpClause::Allocate>(
                       &clause.u)) {
      genAllocateClause(converter, allocateClause->v, allocatorOperands,
                        allocateOperands);
    }
  }
  const auto &endSectionsClauseList =
      std::get<Fortran::parser::OmpEndSectionsDirective>(sectionsConstruct.t);
  const auto &clauseList =
      std::get<Fortran::parser::OmpClauseList>(endSectionsClauseList.t);
  for (const auto &clause : clauseList.v) {
    // Nowait clause
    if (std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u)) {
      noWaitClauseOperand = firOpBuilder.getUnitAttr();
    }
  }

  llvm::omp::Directive dir =
      std::get<Fortran::parser::OmpSectionsDirective>(
          std::get<Fortran::parser::OmpBeginSectionsDirective>(
              sectionsConstruct.t)
              .t)
          .v;

  // Parallel Sections Construct
  if (dir == llvm::omp::Directive::OMPD_parallel_sections) {
    createCombinedParallelOp<Fortran::parser::OmpBeginSectionsDirective>(
        converter, eval,
        std::get<Fortran::parser::OmpBeginSectionsDirective>(
            sectionsConstruct.t));
    auto sectionsOp = firOpBuilder.create<mlir::omp::SectionsOp>(
        currentLocation, /*reduction_vars*/ ValueRange(),
        /*reductions=*/nullptr, allocateOperands, allocatorOperands,
        /*nowait=*/nullptr);
    createBodyOfOp(sectionsOp, converter, currentLocation, eval);

    // Sections Construct
  } else if (dir == llvm::omp::Directive::OMPD_sections) {
    auto sectionsOp = firOpBuilder.create<mlir::omp::SectionsOp>(
        currentLocation, reductionVars, /*reductions = */ nullptr,
        allocateOperands, allocatorOperands, noWaitClauseOperand);
    createBodyOfOp<omp::SectionsOp>(sectionsOp, converter, currentLocation,
                                    eval);
  }
}

static void genOmpAtomicHintAndMemoryOrderClauses(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::OmpAtomicClauseList &clauseList,
    mlir::IntegerAttr &hint,
    mlir::omp::ClauseMemoryOrderKindAttr &memory_order) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  for (const auto &clause : clauseList.v) {
    if (auto ompClause = std::get_if<Fortran::parser::OmpClause>(&clause.u)) {
      if (auto hintClause =
              std::get_if<Fortran::parser::OmpClause::Hint>(&ompClause->u)) {
        const auto *expr = Fortran::semantics::GetExpr(hintClause->v);
        uint64_t hintExprValue = *Fortran::evaluate::ToInt64(*expr);
        hint = firOpBuilder.getI64IntegerAttr(hintExprValue);
      }
    } else if (auto ompMemoryOrderClause =
                   std::get_if<Fortran::parser::OmpMemoryOrderClause>(
                       &clause.u)) {
      if (std::get_if<Fortran::parser::OmpClause::Acquire>(
              &ompMemoryOrderClause->v.u)) {
        memory_order = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Acquire);
      } else if (std::get_if<Fortran::parser::OmpClause::Relaxed>(
                     &ompMemoryOrderClause->v.u)) {
        memory_order = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Relaxed);
      } else if (std::get_if<Fortran::parser::OmpClause::SeqCst>(
                     &ompMemoryOrderClause->v.u)) {
        memory_order = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Seq_cst);
      } else if (std::get_if<Fortran::parser::OmpClause::Release>(
                     &ompMemoryOrderClause->v.u)) {
        memory_order = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(), omp::ClauseMemoryOrderKind::Release);
      }
    }
  }
}

static void
genOmpAtomicWrite(Fortran::lower::AbstractConverter &converter,
                  Fortran::lower::pft::Evaluation &eval,
                  const Fortran::parser::OmpAtomicWrite &atomicWrite) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  // Get the value and address of atomic write operands.
  const Fortran::parser::OmpAtomicClauseList &rightHandClauseList =
      std::get<2>(atomicWrite.t);
  const Fortran::parser::OmpAtomicClauseList &leftHandClauseList =
      std::get<0>(atomicWrite.t);
  const auto &assignmentStmtExpr =
      std::get<Fortran::parser::Expr>(std::get<3>(atomicWrite.t).statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<3>(atomicWrite.t).statement.t);
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value value = fir::getBase(converter.genExprValue(
      *Fortran::semantics::GetExpr(assignmentStmtExpr), stmtCtx));
  mlir::Value address = fir::getBase(converter.genExprAddr(
      *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx));
  // If no hint clause is specified, the effect is as if
  // hint(omp_sync_hint_none) had been specified.
  mlir::IntegerAttr hint = nullptr;
  mlir::omp::ClauseMemoryOrderKindAttr memory_order = nullptr;
  genOmpAtomicHintAndMemoryOrderClauses(converter, leftHandClauseList, hint,
                                        memory_order);
  genOmpAtomicHintAndMemoryOrderClauses(converter, rightHandClauseList, hint,
                                        memory_order);
  firOpBuilder.create<mlir::omp::AtomicWriteOp>(currentLocation, address, value,
                                                hint, memory_order);
}

static void genOmpAtomicRead(Fortran::lower::AbstractConverter &converter,
                             Fortran::lower::pft::Evaluation &eval,
                             const Fortran::parser::OmpAtomicRead &atomicRead) {
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  // Get the address of atomic read operands.
  const Fortran::parser::OmpAtomicClauseList &rightHandClauseList =
      std::get<2>(atomicRead.t);
  const Fortran::parser::OmpAtomicClauseList &leftHandClauseList =
      std::get<0>(atomicRead.t);
  const auto &assignmentStmtExpr =
      std::get<Fortran::parser::Expr>(std::get<3>(atomicRead.t).statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<3>(atomicRead.t).statement.t);
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value from_address = fir::getBase(converter.genExprAddr(
      *Fortran::semantics::GetExpr(assignmentStmtExpr), stmtCtx));
  mlir::Value to_address = fir::getBase(converter.genExprAddr(
      *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx));
  // If no hint clause is specified, the effect is as if
  // hint(omp_sync_hint_none) had been specified.
  mlir::IntegerAttr hint = nullptr;
  mlir::omp::ClauseMemoryOrderKindAttr memory_order = nullptr;
  genOmpAtomicHintAndMemoryOrderClauses(converter, leftHandClauseList, hint,
                                        memory_order);
  genOmpAtomicHintAndMemoryOrderClauses(converter, rightHandClauseList, hint,
                                        memory_order);
  firOpBuilder.create<mlir::omp::AtomicReadOp>(currentLocation, from_address,
                                               to_address, hint, memory_order);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
  std::visit(Fortran::common::visitors{
                 [&](const Fortran::parser::OmpAtomicRead &atomicRead) {
                   genOmpAtomicRead(converter, eval, atomicRead);
                 },
                 [&](const Fortran::parser::OmpAtomicWrite &atomicWrite) {
                   genOmpAtomicWrite(converter, eval, atomicWrite);
                 },
                 [&](const auto &) {
                   TODO(converter.getCurrentLocation(),
                        "Atomic update & capture");
                 },
             },
             atomicConstruct.u);
}

void Fortran::lower::genOpenMPConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPConstruct &ompConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenMPStandaloneConstruct
                  &standaloneConstruct) {
            genOMP(converter, eval, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionsConstruct
                  &sectionsConstruct) {
            genOMP(converter, eval, sectionsConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionConstruct &sectionConstruct) {
            genOMP(converter, eval, sectionConstruct);
          },
          [&](const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {
            genOMP(converter, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPExecutableAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPExecutableAllocate");
          },
          [&](const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
            genOMP(converter, eval, blockConstruct);
          },
          [&](const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
            genOMP(converter, eval, atomicConstruct);
          },
          [&](const Fortran::parser::OpenMPCriticalConstruct
                  &criticalConstruct) {
            genOMP(converter, eval, criticalConstruct);
          },
      },
      ompConstruct.u);
}

void Fortran::lower::genThreadprivateOp(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  const Fortran::semantics::Symbol &sym = var.getSymbol();
  mlir::Value symThreadprivateValue;
  if (const Fortran::semantics::Symbol *common =
          Fortran::semantics::FindCommonBlockContaining(sym.GetUltimate())) {
    mlir::Value commonValue = converter.getSymbolAddress(*common);
    if (mlir::isa<mlir::omp::ThreadprivateOp>(commonValue.getDefiningOp())) {
      // Generate ThreadprivateOp for a common block instead of its members and
      // only do it once for a common block.
      return;
    }
    // Generate ThreadprivateOp and rebind the common block.
    mlir::Value commonThreadprivateValue =
        firOpBuilder.create<mlir::omp::ThreadprivateOp>(
            currentLocation, commonValue.getType(), commonValue);
    converter.bindSymbol(*common, commonThreadprivateValue);
    // Generate the threadprivate value for the common block member.
    symThreadprivateValue =
        genCommonBlockMember(converter, sym, commonThreadprivateValue);
  } else {
    mlir::Value symValue = converter.getSymbolAddress(sym);
    symThreadprivateValue = firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  }

  fir::ExtendedValue sexv = converter.getSymbolExtendedValue(sym);
  fir::ExtendedValue symThreadprivateExv =
      getExtendedValue(sexv, symThreadprivateValue);
  converter.bindSymbol(sym, symThreadprivateExv);
}

void Fortran::lower::genOpenMPDeclarativeConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &ompDeclConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &declarativeAllocate) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPDeclareReductionConstruct
                  &declareReductionConstruct) {
            TODO(converter.getCurrentLocation(),
                 "OpenMPDeclareReductionConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareSimdConstruct
                  &declareSimdConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclareSimdConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct
                  &declareTargetConstruct) {
            TODO(converter.getCurrentLocation(),
                 "OpenMPDeclareTargetConstruct");
          },
          [&](const Fortran::parser::OpenMPThreadprivate &threadprivate) {
            // The directive is lowered when instantiating the variable to
            // support the case of threadprivate variable declared in module.
          },
      },
      ompDeclConstruct.u);
}

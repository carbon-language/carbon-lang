//===--------- ScopInfo.cpp  - Create Scops from LLVM IR ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a static control flow region.
//
// The pass creates a polyhedral description of the Scops detected by the Scop
// detection derived from their LLVM-IR code.
//
// This representation is shared among several tools in the polyhedral
// community, which are e.g. Cloog, Pluto, Loopo, Graphite.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "polly/TempScopInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/Debug.h"
#include "isl/aff.h"
#include "isl/constraint.h"
#include "isl/local_space.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/printer.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include "isl/val.h"
#include <sstream>
#include <string>
#include <vector>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-scops"

STATISTIC(ScopFound, "Number of valid Scops");
STATISTIC(RichScopFound, "Number of Scops containing a loop");

// Multiplicative reductions can be disabled separately as these kind of
// operations can overflow easily. Additive reductions and bit operations
// are in contrast pretty stable.
static cl::opt<bool> DisableMultiplicativeReductions(
    "polly-disable-multiplicative-reductions",
    cl::desc("Disable multiplicative reductions"), cl::Hidden, cl::ZeroOrMore,
    cl::init(false), cl::cat(PollyCategory));

static cl::opt<unsigned> RunTimeChecksMaxParameters(
    "polly-rtc-max-parameters",
    cl::desc("The maximal number of parameters allowed in RTCs."), cl::Hidden,
    cl::ZeroOrMore, cl::init(8), cl::cat(PollyCategory));

static cl::opt<unsigned> RunTimeChecksMaxArraysPerGroup(
    "polly-rtc-max-arrays-per-group",
    cl::desc("The maximal number of arrays to compare in each alias group."),
    cl::Hidden, cl::ZeroOrMore, cl::init(20), cl::cat(PollyCategory));

/// Translate a 'const SCEV *' expression in an isl_pw_aff.
struct SCEVAffinator : public SCEVVisitor<SCEVAffinator, isl_pw_aff *> {
public:
  /// @brief Translate a 'const SCEV *' to an isl_pw_aff.
  ///
  /// @param Stmt The location at which the scalar evolution expression
  ///             is evaluated.
  /// @param Expr The expression that is translated.
  static __isl_give isl_pw_aff *getPwAff(ScopStmt *Stmt, const SCEV *Expr);

private:
  isl_ctx *Ctx;
  int NbLoopSpaces;
  const Scop *S;

  SCEVAffinator(const ScopStmt *Stmt);
  int getLoopDepth(const Loop *L);

  __isl_give isl_pw_aff *visit(const SCEV *Expr);
  __isl_give isl_pw_aff *visitConstant(const SCEVConstant *Expr);
  __isl_give isl_pw_aff *visitTruncateExpr(const SCEVTruncateExpr *Expr);
  __isl_give isl_pw_aff *visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr);
  __isl_give isl_pw_aff *visitSignExtendExpr(const SCEVSignExtendExpr *Expr);
  __isl_give isl_pw_aff *visitAddExpr(const SCEVAddExpr *Expr);
  __isl_give isl_pw_aff *visitMulExpr(const SCEVMulExpr *Expr);
  __isl_give isl_pw_aff *visitUDivExpr(const SCEVUDivExpr *Expr);
  __isl_give isl_pw_aff *visitAddRecExpr(const SCEVAddRecExpr *Expr);
  __isl_give isl_pw_aff *visitSMaxExpr(const SCEVSMaxExpr *Expr);
  __isl_give isl_pw_aff *visitUMaxExpr(const SCEVUMaxExpr *Expr);
  __isl_give isl_pw_aff *visitUnknown(const SCEVUnknown *Expr);
  __isl_give isl_pw_aff *visitSDivInstruction(Instruction *SDiv);

  friend struct SCEVVisitor<SCEVAffinator, isl_pw_aff *>;
};

SCEVAffinator::SCEVAffinator(const ScopStmt *Stmt)
    : Ctx(Stmt->getIslCtx()), NbLoopSpaces(Stmt->getNumIterators()),
      S(Stmt->getParent()) {}

__isl_give isl_pw_aff *SCEVAffinator::getPwAff(ScopStmt *Stmt,
                                               const SCEV *Scev) {
  Scop *S = Stmt->getParent();
  const Region *Reg = &S->getRegion();

  S->addParams(getParamsInAffineExpr(Reg, Scev, *S->getSE()));

  SCEVAffinator Affinator(Stmt);
  return Affinator.visit(Scev);
}

__isl_give isl_pw_aff *SCEVAffinator::visit(const SCEV *Expr) {
  // In case the scev is a valid parameter, we do not further analyze this
  // expression, but create a new parameter in the isl_pw_aff. This allows us
  // to treat subexpressions that we cannot translate into an piecewise affine
  // expression, as constant parameters of the piecewise affine expression.
  if (isl_id *Id = S->getIdForParam(Expr)) {
    isl_space *Space = isl_space_set_alloc(Ctx, 1, NbLoopSpaces);
    Space = isl_space_set_dim_id(Space, isl_dim_param, 0, Id);

    isl_set *Domain = isl_set_universe(isl_space_copy(Space));
    isl_aff *Affine = isl_aff_zero_on_domain(isl_local_space_from_space(Space));
    Affine = isl_aff_add_coefficient_si(Affine, isl_dim_param, 0, 1);

    return isl_pw_aff_alloc(Domain, Affine);
  }

  return SCEVVisitor<SCEVAffinator, isl_pw_aff *>::visit(Expr);
}

__isl_give isl_pw_aff *SCEVAffinator::visitConstant(const SCEVConstant *Expr) {
  ConstantInt *Value = Expr->getValue();
  isl_val *v;

  // LLVM does not define if an integer value is interpreted as a signed or
  // unsigned value. Hence, without further information, it is unknown how
  // this value needs to be converted to GMP. At the moment, we only support
  // signed operations. So we just interpret it as signed. Later, there are
  // two options:
  //
  // 1. We always interpret any value as signed and convert the values on
  //    demand.
  // 2. We pass down the signedness of the calculation and use it to interpret
  //    this constant correctly.
  v = isl_valFromAPInt(Ctx, Value->getValue(), /* isSigned */ true);

  isl_space *Space = isl_space_set_alloc(Ctx, 0, NbLoopSpaces);
  isl_local_space *ls = isl_local_space_from_space(Space);
  return isl_pw_aff_from_aff(isl_aff_val_on_domain(ls, v));
}

__isl_give isl_pw_aff *
SCEVAffinator::visitTruncateExpr(const SCEVTruncateExpr *Expr) {
  llvm_unreachable("SCEVTruncateExpr not yet supported");
}

__isl_give isl_pw_aff *
SCEVAffinator::visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
  llvm_unreachable("SCEVZeroExtendExpr not yet supported");
}

__isl_give isl_pw_aff *
SCEVAffinator::visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
  // Assuming the value is signed, a sign extension is basically a noop.
  // TODO: Reconsider this as soon as we support unsigned values.
  return visit(Expr->getOperand());
}

__isl_give isl_pw_aff *SCEVAffinator::visitAddExpr(const SCEVAddExpr *Expr) {
  isl_pw_aff *Sum = visit(Expr->getOperand(0));

  for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
    isl_pw_aff *NextSummand = visit(Expr->getOperand(i));
    Sum = isl_pw_aff_add(Sum, NextSummand);
  }

  // TODO: Check for NSW and NUW.

  return Sum;
}

__isl_give isl_pw_aff *SCEVAffinator::visitMulExpr(const SCEVMulExpr *Expr) {
  // Divide Expr into a constant part and the rest. Then visit both and multiply
  // the result to obtain the representation for Expr. While the second part of
  // ConstantAndLeftOverPair might still be a SCEVMulExpr we will not get to
  // this point again. The reason is that if it is a multiplication it consists
  // only of parameters and we will stop in the visit(const SCEV *) function and
  // return the isl_pw_aff for that parameter.
  auto ConstantAndLeftOverPair = extractConstantFactor(Expr, *S->getSE());
  return isl_pw_aff_mul(visit(ConstantAndLeftOverPair.first),
                        visit(ConstantAndLeftOverPair.second));
}

__isl_give isl_pw_aff *SCEVAffinator::visitUDivExpr(const SCEVUDivExpr *Expr) {
  llvm_unreachable("SCEVUDivExpr not yet supported");
}

__isl_give isl_pw_aff *
SCEVAffinator::visitAddRecExpr(const SCEVAddRecExpr *Expr) {
  assert(Expr->isAffine() && "Only affine AddRecurrences allowed");

  auto Flags = Expr->getNoWrapFlags();

  // Directly generate isl_pw_aff for Expr if 'start' is zero.
  if (Expr->getStart()->isZero()) {
    assert(S->getRegion().contains(Expr->getLoop()) &&
           "Scop does not contain the loop referenced in this AddRec");

    isl_pw_aff *Start = visit(Expr->getStart());
    isl_pw_aff *Step = visit(Expr->getOperand(1));
    isl_space *Space = isl_space_set_alloc(Ctx, 0, NbLoopSpaces);
    isl_local_space *LocalSpace = isl_local_space_from_space(Space);

    int loopDimension = getLoopDepth(Expr->getLoop());

    isl_aff *LAff = isl_aff_set_coefficient_si(
        isl_aff_zero_on_domain(LocalSpace), isl_dim_in, loopDimension, 1);
    isl_pw_aff *LPwAff = isl_pw_aff_from_aff(LAff);

    // TODO: Do we need to check for NSW and NUW?
    return isl_pw_aff_add(Start, isl_pw_aff_mul(Step, LPwAff));
  }

  // Translate AddRecExpr from '{start, +, inc}' into 'start + {0, +, inc}'
  // if 'start' is not zero.
  // TODO: Using the original SCEV no-wrap flags is not always safe, however
  //       as our code generation is reordering the expression anyway it doesn't
  //       really matter.
  ScalarEvolution &SE = *S->getSE();
  const SCEV *ZeroStartExpr =
      SE.getAddRecExpr(SE.getConstant(Expr->getStart()->getType(), 0),
                       Expr->getStepRecurrence(SE), Expr->getLoop(), Flags);

  isl_pw_aff *ZeroStartResult = visit(ZeroStartExpr);
  isl_pw_aff *Start = visit(Expr->getStart());

  return isl_pw_aff_add(ZeroStartResult, Start);
}

__isl_give isl_pw_aff *SCEVAffinator::visitSMaxExpr(const SCEVSMaxExpr *Expr) {
  isl_pw_aff *Max = visit(Expr->getOperand(0));

  for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
    isl_pw_aff *NextOperand = visit(Expr->getOperand(i));
    Max = isl_pw_aff_max(Max, NextOperand);
  }

  return Max;
}

__isl_give isl_pw_aff *SCEVAffinator::visitUMaxExpr(const SCEVUMaxExpr *Expr) {
  llvm_unreachable("SCEVUMaxExpr not yet supported");
}

__isl_give isl_pw_aff *SCEVAffinator::visitSDivInstruction(Instruction *SDiv) {
  assert(SDiv->getOpcode() == Instruction::SDiv && "Assumed SDiv instruction!");
  auto *SE = S->getSE();

  auto *Divisor = SDiv->getOperand(1);
  auto *DivisorSCEV = SE->getSCEV(Divisor);
  auto *DivisorPWA = visit(DivisorSCEV);
  assert(isa<ConstantInt>(Divisor) &&
         "SDiv is no parameter but has a non-constant RHS.");

  auto *Dividend = SDiv->getOperand(0);
  auto *DividendSCEV = SE->getSCEV(Dividend);
  auto *DividendPWA = visit(DividendSCEV);
  return isl_pw_aff_tdiv_q(DividendPWA, DivisorPWA);
}

__isl_give isl_pw_aff *SCEVAffinator::visitUnknown(const SCEVUnknown *Expr) {
  if (Instruction *I = dyn_cast<Instruction>(Expr->getValue())) {
    switch (I->getOpcode()) {
    case Instruction::SDiv:
      return visitSDivInstruction(I);
    default:
      break; // Fall through.
    }
  }

  llvm_unreachable(
      "Unknowns SCEV was neither parameter nor a valid instruction.");
}

int SCEVAffinator::getLoopDepth(const Loop *L) {
  Loop *outerLoop = S->getRegion().outermostLoopInRegion(const_cast<Loop *>(L));
  assert(outerLoop && "Scop does not contain this loop");
  return L->getLoopDepth() - outerLoop->getLoopDepth();
}

/// @brief Add the bounds of @p Range to the set @p S for dimension @p dim.
static __isl_give isl_set *addRangeBoundsToSet(__isl_take isl_set *S,
                                               const ConstantRange &Range,
                                               int dim,
                                               enum isl_dim_type type) {
  isl_val *V;
  isl_ctx *ctx = isl_set_get_ctx(S);

  bool useLowerUpperBound = Range.isSignWrappedSet() && !Range.isFullSet();
  const auto LB = useLowerUpperBound ? Range.getLower() : Range.getSignedMin();
  V = isl_valFromAPInt(ctx, LB, true);
  isl_set *SLB = isl_set_lower_bound_val(isl_set_copy(S), type, dim, V);

  const auto UB = useLowerUpperBound ? Range.getUpper() : Range.getSignedMax();
  V = isl_valFromAPInt(ctx, UB, true);
  if (useLowerUpperBound)
    V = isl_val_sub_ui(V, 1);
  isl_set *SUB = isl_set_upper_bound_val(S, type, dim, V);

  if (useLowerUpperBound)
    return isl_set_union(SLB, SUB);
  else
    return isl_set_intersect(SLB, SUB);
}

ScopArrayInfo::ScopArrayInfo(Value *BasePtr, Type *ElementType, isl_ctx *Ctx,
                             const SmallVector<const SCEV *, 4> &DimensionSizes)
    : BasePtr(BasePtr), ElementType(ElementType),
      DimensionSizes(DimensionSizes) {
  const std::string BasePtrName = getIslCompatibleName("MemRef_", BasePtr, "");
  Id = isl_id_alloc(Ctx, BasePtrName.c_str(), this);
}

ScopArrayInfo::~ScopArrayInfo() { isl_id_free(Id); }

std::string ScopArrayInfo::getName() const { return isl_id_get_name(Id); }

int ScopArrayInfo::getElemSizeInBytes() const {
  return ElementType->getPrimitiveSizeInBits() / 8;
}

isl_id *ScopArrayInfo::getBasePtrId() const { return isl_id_copy(Id); }

void ScopArrayInfo::dump() const { print(errs()); }

void ScopArrayInfo::print(raw_ostream &OS) const {
  OS.indent(8) << *getElementType() << " " << getName() << "[*]";
  for (unsigned u = 0; u < getNumberOfDimensions(); u++)
    OS << "[" << *DimensionSizes[u] << "]";
  OS << " // Element size " << getElemSizeInBytes() << "\n";
}

const ScopArrayInfo *
ScopArrayInfo::getFromAccessFunction(__isl_keep isl_pw_multi_aff *PMA) {
  isl_id *Id = isl_pw_multi_aff_get_tuple_id(PMA, isl_dim_out);
  assert(Id && "Output dimension didn't have an ID");
  return getFromId(Id);
}

const ScopArrayInfo *ScopArrayInfo::getFromId(isl_id *Id) {
  void *User = isl_id_get_user(Id);
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  isl_id_free(Id);
  return SAI;
}

const std::string
MemoryAccess::getReductionOperatorStr(MemoryAccess::ReductionType RT) {
  switch (RT) {
  case MemoryAccess::RT_NONE:
    llvm_unreachable("Requested a reduction operator string for a memory "
                     "access which isn't a reduction");
  case MemoryAccess::RT_ADD:
    return "+";
  case MemoryAccess::RT_MUL:
    return "*";
  case MemoryAccess::RT_BOR:
    return "|";
  case MemoryAccess::RT_BXOR:
    return "^";
  case MemoryAccess::RT_BAND:
    return "&";
  }
  llvm_unreachable("Unknown reduction type");
  return "";
}

/// @brief Return the reduction type for a given binary operator
static MemoryAccess::ReductionType getReductionType(const BinaryOperator *BinOp,
                                                    const Instruction *Load) {
  if (!BinOp)
    return MemoryAccess::RT_NONE;
  switch (BinOp->getOpcode()) {
  case Instruction::FAdd:
    if (!BinOp->hasUnsafeAlgebra())
      return MemoryAccess::RT_NONE;
  // Fall through
  case Instruction::Add:
    return MemoryAccess::RT_ADD;
  case Instruction::Or:
    return MemoryAccess::RT_BOR;
  case Instruction::Xor:
    return MemoryAccess::RT_BXOR;
  case Instruction::And:
    return MemoryAccess::RT_BAND;
  case Instruction::FMul:
    if (!BinOp->hasUnsafeAlgebra())
      return MemoryAccess::RT_NONE;
  // Fall through
  case Instruction::Mul:
    if (DisableMultiplicativeReductions)
      return MemoryAccess::RT_NONE;
    return MemoryAccess::RT_MUL;
  default:
    return MemoryAccess::RT_NONE;
  }
}
//===----------------------------------------------------------------------===//

MemoryAccess::~MemoryAccess() {
  isl_id_free(Id);
  isl_map_free(AccessRelation);
  isl_map_free(newAccessRelation);
}

static MemoryAccess::AccessType getMemoryAccessType(const IRAccess &Access) {
  switch (Access.getType()) {
  case IRAccess::READ:
    return MemoryAccess::READ;
  case IRAccess::MUST_WRITE:
    return MemoryAccess::MUST_WRITE;
  case IRAccess::MAY_WRITE:
    return MemoryAccess::MAY_WRITE;
  }
  llvm_unreachable("Unknown IRAccess type!");
}

const ScopArrayInfo *MemoryAccess::getScopArrayInfo() const {
  isl_id *ArrayId = getArrayId();
  void *User = isl_id_get_user(ArrayId);
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  isl_id_free(ArrayId);
  return SAI;
}

__isl_give isl_id *MemoryAccess::getArrayId() const {
  return isl_map_get_tuple_id(AccessRelation, isl_dim_out);
}

__isl_give isl_pw_multi_aff *MemoryAccess::applyScheduleToAccessRelation(
    __isl_take isl_union_map *USchedule) const {
  isl_map *Schedule, *ScheduledAccRel;
  isl_union_set *UDomain;

  UDomain = isl_union_set_from_set(getStatement()->getDomain());
  USchedule = isl_union_map_intersect_domain(USchedule, UDomain);
  Schedule = isl_map_from_union_map(USchedule);
  ScheduledAccRel = isl_map_apply_domain(getAccessRelation(), Schedule);
  return isl_pw_multi_aff_from_map(ScheduledAccRel);
}

__isl_give isl_map *MemoryAccess::getOriginalAccessRelation() const {
  return isl_map_copy(AccessRelation);
}

std::string MemoryAccess::getOriginalAccessRelationStr() const {
  return stringFromIslObj(AccessRelation);
}

__isl_give isl_space *MemoryAccess::getOriginalAccessRelationSpace() const {
  return isl_map_get_space(AccessRelation);
}

__isl_give isl_map *MemoryAccess::getNewAccessRelation() const {
  return isl_map_copy(newAccessRelation);
}

__isl_give isl_basic_map *
MemoryAccess::createBasicAccessMap(ScopStmt *Statement) {
  isl_space *Space = isl_space_set_alloc(Statement->getIslCtx(), 0, 1);
  Space = isl_space_align_params(Space, Statement->getDomainSpace());

  return isl_basic_map_from_domain_and_range(
      isl_basic_set_universe(Statement->getDomainSpace()),
      isl_basic_set_universe(Space));
}

// Formalize no out-of-bound access assumption
//
// When delinearizing array accesses we optimistically assume that the
// delinearized accesses do not access out of bound locations (the subscript
// expression of each array evaluates for each statement instance that is
// executed to a value that is larger than zero and strictly smaller than the
// size of the corresponding dimension). The only exception is the outermost
// dimension for which we do not need to assume any upper bound.  At this point
// we formalize this assumption to ensure that at code generation time the
// relevant run-time checks can be generated.
//
// To find the set of constraints necessary to avoid out of bound accesses, we
// first build the set of data locations that are not within array bounds. We
// then apply the reverse access relation to obtain the set of iterations that
// may contain invalid accesses and reduce this set of iterations to the ones
// that are actually executed by intersecting them with the domain of the
// statement. If we now project out all loop dimensions, we obtain a set of
// parameters that may cause statement instances to be executed that may
// possibly yield out of bound memory accesses. The complement of these
// constraints is the set of constraints that needs to be assumed to ensure such
// statement instances are never executed.
void MemoryAccess::assumeNoOutOfBound(const IRAccess &Access) {
  isl_space *Space = isl_space_range(getOriginalAccessRelationSpace());
  isl_set *Outside = isl_set_empty(isl_space_copy(Space));
  for (int i = 1, Size = Access.Subscripts.size(); i < Size; ++i) {
    isl_local_space *LS = isl_local_space_from_space(isl_space_copy(Space));
    isl_pw_aff *Var =
        isl_pw_aff_var_on_domain(isl_local_space_copy(LS), isl_dim_set, i);
    isl_pw_aff *Zero = isl_pw_aff_zero_on_domain(LS);

    isl_set *DimOutside;

    DimOutside = isl_pw_aff_lt_set(isl_pw_aff_copy(Var), Zero);
    isl_pw_aff *SizeE = SCEVAffinator::getPwAff(Statement, Access.Sizes[i - 1]);

    SizeE = isl_pw_aff_drop_dims(SizeE, isl_dim_in, 0,
                                 Statement->getNumIterators());
    SizeE = isl_pw_aff_add_dims(SizeE, isl_dim_in,
                                isl_space_dim(Space, isl_dim_set));
    SizeE = isl_pw_aff_set_tuple_id(SizeE, isl_dim_in,
                                    isl_space_get_tuple_id(Space, isl_dim_set));

    DimOutside = isl_set_union(DimOutside, isl_pw_aff_le_set(SizeE, Var));

    Outside = isl_set_union(Outside, DimOutside);
  }

  Outside = isl_set_apply(Outside, isl_map_reverse(getAccessRelation()));
  Outside = isl_set_intersect(Outside, Statement->getDomain());
  Outside = isl_set_params(Outside);
  Outside = isl_set_complement(Outside);
  Statement->getParent()->addAssumption(Outside);
  isl_space_free(Space);
}

void MemoryAccess::computeBoundsOnAccessRelation(unsigned ElementSize) {
  ScalarEvolution *SE = Statement->getParent()->getSE();

  Value *Ptr = getPointerOperand(*getAccessInstruction());
  if (!Ptr || !SE->isSCEVable(Ptr->getType()))
    return;

  auto *PtrSCEV = SE->getSCEV(Ptr);
  if (isa<SCEVCouldNotCompute>(PtrSCEV))
    return;

  auto *BasePtrSCEV = SE->getPointerBase(PtrSCEV);
  if (BasePtrSCEV && !isa<SCEVCouldNotCompute>(BasePtrSCEV))
    PtrSCEV = SE->getMinusSCEV(PtrSCEV, BasePtrSCEV);

  const ConstantRange &Range = SE->getSignedRange(PtrSCEV);
  if (Range.isFullSet())
    return;

  bool isWrapping = Range.isSignWrappedSet();
  unsigned BW = Range.getBitWidth();
  const auto LB = isWrapping ? Range.getLower() : Range.getSignedMin();
  const auto UB = isWrapping ? Range.getUpper() : Range.getSignedMax();

  auto Min = LB.sdiv(APInt(BW, ElementSize));
  auto Max = (UB - APInt(BW, 1)).sdiv(APInt(BW, ElementSize));

  isl_set *AccessRange = isl_map_range(isl_map_copy(AccessRelation));
  AccessRange =
      addRangeBoundsToSet(AccessRange, ConstantRange(Min, Max), 0, isl_dim_set);
  AccessRelation = isl_map_intersect_range(AccessRelation, AccessRange);
}

__isl_give isl_map *MemoryAccess::foldAccess(const IRAccess &Access,
                                             __isl_take isl_map *AccessRelation,
                                             ScopStmt *Statement) {
  int Size = Access.Subscripts.size();

  for (int i = Size - 2; i >= 0; --i) {
    isl_space *Space;
    isl_map *MapOne, *MapTwo;
    isl_pw_aff *DimSize = SCEVAffinator::getPwAff(Statement, Access.Sizes[i]);

    isl_space *SpaceSize = isl_pw_aff_get_space(DimSize);
    isl_pw_aff_free(DimSize);
    isl_id *ParamId = isl_space_get_dim_id(SpaceSize, isl_dim_param, 0);

    Space = isl_map_get_space(AccessRelation);
    Space = isl_space_map_from_set(isl_space_range(Space));
    Space = isl_space_align_params(Space, SpaceSize);

    int ParamLocation = isl_space_find_dim_by_id(Space, isl_dim_param, ParamId);
    isl_id_free(ParamId);

    MapOne = isl_map_universe(isl_space_copy(Space));
    for (int j = 0; j < Size; ++j)
      MapOne = isl_map_equate(MapOne, isl_dim_in, j, isl_dim_out, j);
    MapOne = isl_map_lower_bound_si(MapOne, isl_dim_in, i + 1, 0);

    MapTwo = isl_map_universe(isl_space_copy(Space));
    for (int j = 0; j < Size; ++j)
      if (j < i || j > i + 1)
        MapTwo = isl_map_equate(MapTwo, isl_dim_in, j, isl_dim_out, j);

    isl_local_space *LS = isl_local_space_from_space(Space);
    isl_constraint *C;
    C = isl_equality_alloc(isl_local_space_copy(LS));
    C = isl_constraint_set_constant_si(C, -1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_in, i, 1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_out, i, -1);
    MapTwo = isl_map_add_constraint(MapTwo, C);
    C = isl_equality_alloc(LS);
    C = isl_constraint_set_coefficient_si(C, isl_dim_in, i + 1, 1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_out, i + 1, -1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_param, ParamLocation, 1);
    MapTwo = isl_map_add_constraint(MapTwo, C);
    MapTwo = isl_map_upper_bound_si(MapTwo, isl_dim_in, i + 1, -1);

    MapOne = isl_map_union(MapOne, MapTwo);
    AccessRelation = isl_map_apply_range(AccessRelation, MapOne);
  }
  return AccessRelation;
}

MemoryAccess::MemoryAccess(const IRAccess &Access, Instruction *AccInst,
                           ScopStmt *Statement, const ScopArrayInfo *SAI,
                           int Identifier)
    : AccType(getMemoryAccessType(Access)), Statement(Statement), Inst(AccInst),
      newAccessRelation(nullptr) {

  isl_ctx *Ctx = Statement->getIslCtx();
  BaseAddr = Access.getBase();
  BaseName = getIslCompatibleName("MemRef_", getBaseAddr(), "");

  isl_id *BaseAddrId = SAI->getBasePtrId();

  auto IdName = "__polly_array_ref_ " + std::to_string(Identifier);
  Id = isl_id_alloc(Ctx, IdName.c_str(), nullptr);

  if (!Access.isAffine()) {
    // We overapproximate non-affine accesses with a possible access to the
    // whole array. For read accesses it does not make a difference, if an
    // access must or may happen. However, for write accesses it is important to
    // differentiate between writes that must happen and writes that may happen.
    AccessRelation = isl_map_from_basic_map(createBasicAccessMap(Statement));
    AccessRelation =
        isl_map_set_tuple_id(AccessRelation, isl_dim_out, BaseAddrId);

    computeBoundsOnAccessRelation(Access.getElemSizeInBytes());
    return;
  }

  isl_space *Space = isl_space_alloc(Ctx, 0, Statement->getNumIterators(), 0);
  AccessRelation = isl_map_universe(Space);

  for (int i = 0, Size = Access.Subscripts.size(); i < Size; ++i) {
    isl_pw_aff *Affine =
        SCEVAffinator::getPwAff(Statement, Access.Subscripts[i]);

    if (Size == 1) {
      // For the non delinearized arrays, divide the access function of the last
      // subscript by the size of the elements in the array.
      //
      // A stride one array access in C expressed as A[i] is expressed in
      // LLVM-IR as something like A[i * elementsize]. This hides the fact that
      // two subsequent values of 'i' index two values that are stored next to
      // each other in memory. By this division we make this characteristic
      // obvious again.
      isl_val *v = isl_val_int_from_si(Ctx, Access.getElemSizeInBytes());
      Affine = isl_pw_aff_scale_down_val(Affine, v);
    }

    isl_map *SubscriptMap = isl_map_from_pw_aff(Affine);

    AccessRelation = isl_map_flat_range_product(AccessRelation, SubscriptMap);
  }

  AccessRelation = foldAccess(Access, AccessRelation, Statement);

  Space = Statement->getDomainSpace();
  AccessRelation = isl_map_set_tuple_id(
      AccessRelation, isl_dim_in, isl_space_get_tuple_id(Space, isl_dim_set));
  AccessRelation =
      isl_map_set_tuple_id(AccessRelation, isl_dim_out, BaseAddrId);

  assumeNoOutOfBound(Access);
  AccessRelation = isl_map_gist_domain(AccessRelation, Statement->getDomain());
  isl_space_free(Space);
}

void MemoryAccess::realignParams() {
  isl_space *ParamSpace = Statement->getParent()->getParamSpace();
  AccessRelation = isl_map_align_params(AccessRelation, ParamSpace);
}

const std::string MemoryAccess::getReductionOperatorStr() const {
  return MemoryAccess::getReductionOperatorStr(getReductionType());
}

__isl_give isl_id *MemoryAccess::getId() const { return isl_id_copy(Id); }

raw_ostream &polly::operator<<(raw_ostream &OS,
                               MemoryAccess::ReductionType RT) {
  if (RT == MemoryAccess::RT_NONE)
    OS << "NONE";
  else
    OS << MemoryAccess::getReductionOperatorStr(RT);
  return OS;
}

void MemoryAccess::print(raw_ostream &OS) const {
  switch (AccType) {
  case READ:
    OS.indent(12) << "ReadAccess :=\t";
    break;
  case MUST_WRITE:
    OS.indent(12) << "MustWriteAccess :=\t";
    break;
  case MAY_WRITE:
    OS.indent(12) << "MayWriteAccess :=\t";
    break;
  }
  OS << "[Reduction Type: " << getReductionType() << "] ";
  OS << "[Scalar: " << isScalar() << "]\n";
  OS.indent(16) << getOriginalAccessRelationStr() << ";\n";
}

void MemoryAccess::dump() const { print(errs()); }

// Create a map in the size of the provided set domain, that maps from the
// one element of the provided set domain to another element of the provided
// set domain.
// The mapping is limited to all points that are equal in all but the last
// dimension and for which the last dimension of the input is strict smaller
// than the last dimension of the output.
//
//   getEqualAndLarger(set[i0, i1, ..., iX]):
//
//   set[i0, i1, ..., iX] -> set[o0, o1, ..., oX]
//     : i0 = o0, i1 = o1, ..., i(X-1) = o(X-1), iX < oX
//
static isl_map *getEqualAndLarger(isl_space *setDomain) {
  isl_space *Space = isl_space_map_from_set(setDomain);
  isl_map *Map = isl_map_universe(Space);
  unsigned lastDimension = isl_map_dim(Map, isl_dim_in) - 1;

  // Set all but the last dimension to be equal for the input and output
  //
  //   input[i0, i1, ..., iX] -> output[o0, o1, ..., oX]
  //     : i0 = o0, i1 = o1, ..., i(X-1) = o(X-1)
  for (unsigned i = 0; i < lastDimension; ++i)
    Map = isl_map_equate(Map, isl_dim_in, i, isl_dim_out, i);

  // Set the last dimension of the input to be strict smaller than the
  // last dimension of the output.
  //
  //   input[?,?,?,...,iX] -> output[?,?,?,...,oX] : iX < oX
  Map = isl_map_order_lt(Map, isl_dim_in, lastDimension, isl_dim_out,
                         lastDimension);
  return Map;
}

__isl_give isl_set *
MemoryAccess::getStride(__isl_take const isl_map *Schedule) const {
  isl_map *S = const_cast<isl_map *>(Schedule);
  isl_map *AccessRelation = getAccessRelation();
  isl_space *Space = isl_space_range(isl_map_get_space(S));
  isl_map *NextScatt = getEqualAndLarger(Space);

  S = isl_map_reverse(S);
  NextScatt = isl_map_lexmin(NextScatt);

  NextScatt = isl_map_apply_range(NextScatt, isl_map_copy(S));
  NextScatt = isl_map_apply_range(NextScatt, isl_map_copy(AccessRelation));
  NextScatt = isl_map_apply_domain(NextScatt, S);
  NextScatt = isl_map_apply_domain(NextScatt, AccessRelation);

  isl_set *Deltas = isl_map_deltas(NextScatt);
  return Deltas;
}

bool MemoryAccess::isStrideX(__isl_take const isl_map *Schedule,
                             int StrideWidth) const {
  isl_set *Stride, *StrideX;
  bool IsStrideX;

  Stride = getStride(Schedule);
  StrideX = isl_set_universe(isl_set_get_space(Stride));
  StrideX = isl_set_fix_si(StrideX, isl_dim_set, 0, StrideWidth);
  IsStrideX = isl_set_is_equal(Stride, StrideX);

  isl_set_free(StrideX);
  isl_set_free(Stride);

  return IsStrideX;
}

bool MemoryAccess::isStrideZero(const isl_map *Schedule) const {
  return isStrideX(Schedule, 0);
}

bool MemoryAccess::isScalar() const {
  return isl_map_n_out(AccessRelation) == 0;
}

bool MemoryAccess::isStrideOne(const isl_map *Schedule) const {
  return isStrideX(Schedule, 1);
}

void MemoryAccess::setNewAccessRelation(isl_map *newAccess) {
  isl_map_free(newAccessRelation);
  newAccessRelation = newAccess;
}

//===----------------------------------------------------------------------===//

isl_map *ScopStmt::getSchedule() const { return isl_map_copy(Schedule); }

void ScopStmt::restrictDomain(__isl_take isl_set *NewDomain) {
  assert(isl_set_is_subset(NewDomain, Domain) &&
         "New domain is not a subset of old domain!");
  isl_set_free(Domain);
  Domain = NewDomain;
  Schedule = isl_map_intersect_domain(Schedule, isl_set_copy(Domain));
}

void ScopStmt::setSchedule(__isl_take isl_map *NewSchedule) {
  assert(NewSchedule && "New schedule is nullptr");
  isl_map_free(Schedule);
  Schedule = NewSchedule;
}

void ScopStmt::buildSchedule(SmallVectorImpl<unsigned> &ScheduleVec) {
  unsigned NbIterators = getNumIterators();
  unsigned NbScheduleDims = Parent.getMaxLoopDepth() * 2 + 1;

  isl_space *Space = isl_space_set_alloc(getIslCtx(), 0, NbScheduleDims);

  Schedule = isl_map_from_domain_and_range(isl_set_universe(getDomainSpace()),
                                           isl_set_universe(Space));

  // Loop dimensions.
  for (unsigned i = 0; i < NbIterators; ++i)
    Schedule = isl_map_equate(Schedule, isl_dim_out, 2 * i + 1, isl_dim_in, i);

  // Constant dimensions
  for (unsigned i = 0; i < NbIterators + 1; ++i)
    Schedule = isl_map_fix_si(Schedule, isl_dim_out, 2 * i, ScheduleVec[i]);

  // Fill schedule dimensions.
  for (unsigned i = 2 * NbIterators + 1; i < NbScheduleDims; ++i)
    Schedule = isl_map_fix_si(Schedule, isl_dim_out, i, 0);

  Schedule = isl_map_align_params(Schedule, Parent.getParamSpace());
}

void ScopStmt::buildAccesses(TempScop &tempScop, BasicBlock *Block,
                             bool isApproximated) {
  AccFuncSetType *AFS = tempScop.getAccessFunctions(Block);
  if (!AFS)
    return;

  for (auto &AccessPair : *AFS) {
    IRAccess &Access = AccessPair.first;
    Instruction *AccessInst = AccessPair.second;

    Type *ElementType = getAccessInstType(AccessInst);
    const ScopArrayInfo *SAI = getParent()->getOrCreateScopArrayInfo(
        Access.getBase(), ElementType, Access.Sizes);

    if (isApproximated && Access.isWrite())
      Access.setMayWrite();

    MemoryAccessList *&MAL = InstructionToAccess[AccessInst];
    if (!MAL)
      MAL = new MemoryAccessList();
    MAL->emplace_front(Access, AccessInst, this, SAI, MemAccs.size());
    MemAccs.push_back(&MAL->front());
  }
}

void ScopStmt::realignParams() {
  for (MemoryAccess *MA : *this)
    MA->realignParams();

  Domain = isl_set_align_params(Domain, Parent.getParamSpace());
  Schedule = isl_map_align_params(Schedule, Parent.getParamSpace());
}

__isl_give isl_set *ScopStmt::buildConditionSet(const Comparison &Comp) {
  isl_pw_aff *L = SCEVAffinator::getPwAff(this, Comp.getLHS());
  isl_pw_aff *R = SCEVAffinator::getPwAff(this, Comp.getRHS());

  switch (Comp.getPred()) {
  case ICmpInst::ICMP_EQ:
    return isl_pw_aff_eq_set(L, R);
  case ICmpInst::ICMP_NE:
    return isl_pw_aff_ne_set(L, R);
  case ICmpInst::ICMP_SLT:
    return isl_pw_aff_lt_set(L, R);
  case ICmpInst::ICMP_SLE:
    return isl_pw_aff_le_set(L, R);
  case ICmpInst::ICMP_SGT:
    return isl_pw_aff_gt_set(L, R);
  case ICmpInst::ICMP_SGE:
    return isl_pw_aff_ge_set(L, R);
  case ICmpInst::ICMP_ULT:
    return isl_pw_aff_lt_set(L, R);
  case ICmpInst::ICMP_UGT:
    return isl_pw_aff_gt_set(L, R);
  case ICmpInst::ICMP_ULE:
    return isl_pw_aff_le_set(L, R);
  case ICmpInst::ICMP_UGE:
    return isl_pw_aff_ge_set(L, R);
  default:
    llvm_unreachable("Non integer predicate not supported");
  }
}

__isl_give isl_set *ScopStmt::addLoopBoundsToDomain(__isl_take isl_set *Domain,
                                                    TempScop &tempScop) {
  isl_space *Space;
  isl_local_space *LocalSpace;

  Space = isl_set_get_space(Domain);
  LocalSpace = isl_local_space_from_space(Space);

  ScalarEvolution *SE = getParent()->getSE();
  for (int i = 0, e = getNumIterators(); i != e; ++i) {
    isl_aff *Zero = isl_aff_zero_on_domain(isl_local_space_copy(LocalSpace));
    isl_pw_aff *IV =
        isl_pw_aff_from_aff(isl_aff_set_coefficient_si(Zero, isl_dim_in, i, 1));

    // 0 <= IV.
    isl_set *LowerBound = isl_pw_aff_nonneg_set(isl_pw_aff_copy(IV));
    Domain = isl_set_intersect(Domain, LowerBound);

    // IV <= LatchExecutions.
    const Loop *L = getLoopForDimension(i);
    const SCEV *LatchExecutions = SE->getBackedgeTakenCount(L);
    isl_pw_aff *UpperBound = SCEVAffinator::getPwAff(this, LatchExecutions);
    isl_set *UpperBoundSet = isl_pw_aff_le_set(IV, UpperBound);
    Domain = isl_set_intersect(Domain, UpperBoundSet);
  }

  isl_local_space_free(LocalSpace);
  return Domain;
}

__isl_give isl_set *ScopStmt::addConditionsToDomain(__isl_take isl_set *Domain,
                                                    TempScop &tempScop,
                                                    const Region &CurRegion) {
  const Region *TopRegion = tempScop.getMaxRegion().getParent(),
               *CurrentRegion = &CurRegion;
  const BasicBlock *BranchingBB = BB ? BB : R->getEntry();

  do {
    if (BranchingBB != CurrentRegion->getEntry()) {
      if (const BBCond *Condition = tempScop.getBBCond(BranchingBB))
        for (const auto &C : *Condition) {
          isl_set *ConditionSet = buildConditionSet(C);
          Domain = isl_set_intersect(Domain, ConditionSet);
        }
    }
    BranchingBB = CurrentRegion->getEntry();
    CurrentRegion = CurrentRegion->getParent();
  } while (TopRegion != CurrentRegion);

  return Domain;
}

__isl_give isl_set *ScopStmt::buildDomain(TempScop &tempScop,
                                          const Region &CurRegion) {
  isl_space *Space;
  isl_set *Domain;
  isl_id *Id;

  Space = isl_space_set_alloc(getIslCtx(), 0, getNumIterators());

  Id = isl_id_alloc(getIslCtx(), getBaseName(), this);

  Domain = isl_set_universe(Space);
  Domain = addLoopBoundsToDomain(Domain, tempScop);
  Domain = addConditionsToDomain(Domain, tempScop, CurRegion);
  Domain = isl_set_set_tuple_id(Domain, Id);

  return Domain;
}

void ScopStmt::deriveAssumptionsFromGEP(GetElementPtrInst *GEP) {
  int Dimension = 0;
  isl_ctx *Ctx = Parent.getIslCtx();
  isl_local_space *LSpace = isl_local_space_from_space(getDomainSpace());
  Type *Ty = GEP->getPointerOperandType();
  ScalarEvolution &SE = *Parent.getSE();

  if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
    Dimension = 1;
    Ty = PtrTy->getElementType();
  }

  while (auto ArrayTy = dyn_cast<ArrayType>(Ty)) {
    unsigned int Operand = 1 + Dimension;

    if (GEP->getNumOperands() <= Operand)
      break;

    const SCEV *Expr = SE.getSCEV(GEP->getOperand(Operand));

    if (isAffineExpr(&Parent.getRegion(), Expr, SE)) {
      isl_pw_aff *AccessOffset = SCEVAffinator::getPwAff(this, Expr);
      AccessOffset =
          isl_pw_aff_set_tuple_id(AccessOffset, isl_dim_in, getDomainId());

      isl_pw_aff *DimSize = isl_pw_aff_from_aff(isl_aff_val_on_domain(
          isl_local_space_copy(LSpace),
          isl_val_int_from_si(Ctx, ArrayTy->getNumElements())));

      isl_set *OutOfBound = isl_pw_aff_ge_set(AccessOffset, DimSize);
      OutOfBound = isl_set_intersect(getDomain(), OutOfBound);
      OutOfBound = isl_set_params(OutOfBound);
      isl_set *InBound = isl_set_complement(OutOfBound);
      isl_set *Executed = isl_set_params(getDomain());

      // A => B == !A or B
      isl_set *InBoundIfExecuted =
          isl_set_union(isl_set_complement(Executed), InBound);

      Parent.addAssumption(InBoundIfExecuted);
    }

    Dimension += 1;
    Ty = ArrayTy->getElementType();
  }

  isl_local_space_free(LSpace);
}

void ScopStmt::deriveAssumptions(BasicBlock *Block) {
  for (Instruction &Inst : *Block)
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&Inst))
      deriveAssumptionsFromGEP(GEP);
}

ScopStmt::ScopStmt(Scop &parent, TempScop &tempScop, const Region &CurRegion,
                   Region &R, SmallVectorImpl<Loop *> &Nest,
                   SmallVectorImpl<unsigned> &ScheduleVec)
    : Parent(parent), BB(nullptr), R(&R), Build(nullptr),
      NestLoops(Nest.size()) {
  // Setup the induction variables.
  for (unsigned i = 0, e = Nest.size(); i < e; ++i)
    NestLoops[i] = Nest[i];

  BaseName = getIslCompatibleName("Stmt_(", R.getNameStr(), ")");

  Domain = buildDomain(tempScop, CurRegion);
  buildSchedule(ScheduleVec);

  BasicBlock *EntryBB = R.getEntry();
  for (BasicBlock *Block : R.blocks()) {
    buildAccesses(tempScop, Block, Block != EntryBB);
    deriveAssumptions(Block);
  }
  checkForReductions();
}

ScopStmt::ScopStmt(Scop &parent, TempScop &tempScop, const Region &CurRegion,
                   BasicBlock &bb, SmallVectorImpl<Loop *> &Nest,
                   SmallVectorImpl<unsigned> &ScheduleVec)
    : Parent(parent), BB(&bb), R(nullptr), Build(nullptr),
      NestLoops(Nest.size()) {
  // Setup the induction variables.
  for (unsigned i = 0, e = Nest.size(); i < e; ++i)
    NestLoops[i] = Nest[i];

  BaseName = getIslCompatibleName("Stmt_", &bb, "");

  Domain = buildDomain(tempScop, CurRegion);
  buildSchedule(ScheduleVec);
  buildAccesses(tempScop, BB);
  deriveAssumptions(BB);
  checkForReductions();
}

/// @brief Collect loads which might form a reduction chain with @p StoreMA
///
/// Check if the stored value for @p StoreMA is a binary operator with one or
/// two loads as operands. If the binary operand is commutative & associative,
/// used only once (by @p StoreMA) and its load operands are also used only
/// once, we have found a possible reduction chain. It starts at an operand
/// load and includes the binary operator and @p StoreMA.
///
/// Note: We allow only one use to ensure the load and binary operator cannot
///       escape this block or into any other store except @p StoreMA.
void ScopStmt::collectCandiateReductionLoads(
    MemoryAccess *StoreMA, SmallVectorImpl<MemoryAccess *> &Loads) {
  auto *Store = dyn_cast<StoreInst>(StoreMA->getAccessInstruction());
  if (!Store)
    return;

  // Skip if there is not one binary operator between the load and the store
  auto *BinOp = dyn_cast<BinaryOperator>(Store->getValueOperand());
  if (!BinOp)
    return;

  // Skip if the binary operators has multiple uses
  if (BinOp->getNumUses() != 1)
    return;

  // Skip if the opcode of the binary operator is not commutative/associative
  if (!BinOp->isCommutative() || !BinOp->isAssociative())
    return;

  // Skip if the binary operator is outside the current SCoP
  if (BinOp->getParent() != Store->getParent())
    return;

  // Skip if it is a multiplicative reduction and we disabled them
  if (DisableMultiplicativeReductions &&
      (BinOp->getOpcode() == Instruction::Mul ||
       BinOp->getOpcode() == Instruction::FMul))
    return;

  // Check the binary operator operands for a candidate load
  auto *PossibleLoad0 = dyn_cast<LoadInst>(BinOp->getOperand(0));
  auto *PossibleLoad1 = dyn_cast<LoadInst>(BinOp->getOperand(1));
  if (!PossibleLoad0 && !PossibleLoad1)
    return;

  // A load is only a candidate if it cannot escape (thus has only this use)
  if (PossibleLoad0 && PossibleLoad0->getNumUses() == 1)
    if (PossibleLoad0->getParent() == Store->getParent())
      Loads.push_back(lookupAccessFor(PossibleLoad0));
  if (PossibleLoad1 && PossibleLoad1->getNumUses() == 1)
    if (PossibleLoad1->getParent() == Store->getParent())
      Loads.push_back(lookupAccessFor(PossibleLoad1));
}

/// @brief Check for reductions in this ScopStmt
///
/// Iterate over all store memory accesses and check for valid binary reduction
/// like chains. For all candidates we check if they have the same base address
/// and there are no other accesses which overlap with them. The base address
/// check rules out impossible reductions candidates early. The overlap check,
/// together with the "only one user" check in collectCandiateReductionLoads,
/// guarantees that none of the intermediate results will escape during
/// execution of the loop nest. We basically check here that no other memory
/// access can access the same memory as the potential reduction.
void ScopStmt::checkForReductions() {
  SmallVector<MemoryAccess *, 2> Loads;
  SmallVector<std::pair<MemoryAccess *, MemoryAccess *>, 4> Candidates;

  // First collect candidate load-store reduction chains by iterating over all
  // stores and collecting possible reduction loads.
  for (MemoryAccess *StoreMA : MemAccs) {
    if (StoreMA->isRead())
      continue;

    Loads.clear();
    collectCandiateReductionLoads(StoreMA, Loads);
    for (MemoryAccess *LoadMA : Loads)
      Candidates.push_back(std::make_pair(LoadMA, StoreMA));
  }

  // Then check each possible candidate pair.
  for (const auto &CandidatePair : Candidates) {
    bool Valid = true;
    isl_map *LoadAccs = CandidatePair.first->getAccessRelation();
    isl_map *StoreAccs = CandidatePair.second->getAccessRelation();

    // Skip those with obviously unequal base addresses.
    if (!isl_map_has_equal_space(LoadAccs, StoreAccs)) {
      isl_map_free(LoadAccs);
      isl_map_free(StoreAccs);
      continue;
    }

    // And check if the remaining for overlap with other memory accesses.
    isl_map *AllAccsRel = isl_map_union(LoadAccs, StoreAccs);
    AllAccsRel = isl_map_intersect_domain(AllAccsRel, getDomain());
    isl_set *AllAccs = isl_map_range(AllAccsRel);

    for (MemoryAccess *MA : MemAccs) {
      if (MA == CandidatePair.first || MA == CandidatePair.second)
        continue;

      isl_map *AccRel =
          isl_map_intersect_domain(MA->getAccessRelation(), getDomain());
      isl_set *Accs = isl_map_range(AccRel);

      if (isl_set_has_equal_space(AllAccs, Accs) || isl_set_free(Accs)) {
        isl_set *OverlapAccs = isl_set_intersect(Accs, isl_set_copy(AllAccs));
        Valid = Valid && isl_set_is_empty(OverlapAccs);
        isl_set_free(OverlapAccs);
      }
    }

    isl_set_free(AllAccs);
    if (!Valid)
      continue;

    const LoadInst *Load =
        dyn_cast<const LoadInst>(CandidatePair.first->getAccessInstruction());
    MemoryAccess::ReductionType RT =
        getReductionType(dyn_cast<BinaryOperator>(Load->user_back()), Load);

    // If no overlapping access was found we mark the load and store as
    // reduction like.
    CandidatePair.first->markAsReductionLike(RT);
    CandidatePair.second->markAsReductionLike(RT);
  }
}

std::string ScopStmt::getDomainStr() const { return stringFromIslObj(Domain); }

std::string ScopStmt::getScheduleStr() const {
  return stringFromIslObj(Schedule);
}

unsigned ScopStmt::getNumParams() const { return Parent.getNumParams(); }

unsigned ScopStmt::getNumIterators() const { return NestLoops.size(); }

unsigned ScopStmt::getNumSchedule() const {
  return isl_map_dim(Schedule, isl_dim_out);
}

const char *ScopStmt::getBaseName() const { return BaseName.c_str(); }

const Loop *ScopStmt::getLoopForDimension(unsigned Dimension) const {
  return NestLoops[Dimension];
}

isl_ctx *ScopStmt::getIslCtx() const { return Parent.getIslCtx(); }

__isl_give isl_set *ScopStmt::getDomain() const { return isl_set_copy(Domain); }

__isl_give isl_space *ScopStmt::getDomainSpace() const {
  return isl_set_get_space(Domain);
}

__isl_give isl_id *ScopStmt::getDomainId() const {
  return isl_set_get_tuple_id(Domain);
}

ScopStmt::~ScopStmt() {
  DeleteContainerSeconds(InstructionToAccess);
  isl_set_free(Domain);
  isl_map_free(Schedule);
}

void ScopStmt::print(raw_ostream &OS) const {
  OS << "\t" << getBaseName() << "\n";
  OS.indent(12) << "Domain :=\n";

  if (Domain) {
    OS.indent(16) << getDomainStr() << ";\n";
  } else
    OS.indent(16) << "n/a\n";

  OS.indent(12) << "Schedule :=\n";

  if (Domain) {
    OS.indent(16) << getScheduleStr() << ";\n";
  } else
    OS.indent(16) << "n/a\n";

  for (MemoryAccess *Access : MemAccs)
    Access->print(OS);
}

void ScopStmt::dump() const { print(dbgs()); }

//===----------------------------------------------------------------------===//
/// Scop class implement

void Scop::setContext(__isl_take isl_set *NewContext) {
  NewContext = isl_set_align_params(NewContext, isl_set_get_space(Context));
  isl_set_free(Context);
  Context = NewContext;
}

void Scop::addParams(std::vector<const SCEV *> NewParameters) {
  for (const SCEV *Parameter : NewParameters) {
    Parameter = extractConstantFactor(Parameter, *SE).second;
    if (ParameterIds.find(Parameter) != ParameterIds.end())
      continue;

    int dimension = Parameters.size();

    Parameters.push_back(Parameter);
    ParameterIds[Parameter] = dimension;
  }
}

__isl_give isl_id *Scop::getIdForParam(const SCEV *Parameter) const {
  ParamIdType::const_iterator IdIter = ParameterIds.find(Parameter);

  if (IdIter == ParameterIds.end())
    return nullptr;

  std::string ParameterName;

  if (const SCEVUnknown *ValueParameter = dyn_cast<SCEVUnknown>(Parameter)) {
    Value *Val = ValueParameter->getValue();
    ParameterName = Val->getName();
  }

  if (ParameterName == "" || ParameterName.substr(0, 2) == "p_")
    ParameterName = "p_" + utostr_32(IdIter->second);

  return isl_id_alloc(getIslCtx(), ParameterName.c_str(),
                      const_cast<void *>((const void *)Parameter));
}

void Scop::buildContext() {
  isl_space *Space = isl_space_params_alloc(IslCtx, 0);
  Context = isl_set_universe(isl_space_copy(Space));
  AssumedContext = isl_set_universe(Space);
}

void Scop::addParameterBounds() {
  for (const auto &ParamID : ParameterIds) {
    int dim = ParamID.second;

    ConstantRange SRange = SE->getSignedRange(ParamID.first);

    Context = addRangeBoundsToSet(Context, SRange, dim, isl_dim_param);
  }
}

void Scop::realignParams() {
  // Add all parameters into a common model.
  isl_space *Space = isl_space_params_alloc(IslCtx, ParameterIds.size());

  for (const auto &ParamID : ParameterIds) {
    const SCEV *Parameter = ParamID.first;
    isl_id *id = getIdForParam(Parameter);
    Space = isl_space_set_dim_id(Space, isl_dim_param, ParamID.second, id);
  }

  // Align the parameters of all data structures to the model.
  Context = isl_set_align_params(Context, Space);

  for (ScopStmt &Stmt : *this)
    Stmt.realignParams();
}

void Scop::simplifyAssumedContext() {
  // The parameter constraints of the iteration domains give us a set of
  // constraints that need to hold for all cases where at least a single
  // statement iteration is executed in the whole scop. We now simplify the
  // assumed context under the assumption that such constraints hold and at
  // least a single statement iteration is executed. For cases where no
  // statement instances are executed, the assumptions we have taken about
  // the executed code do not matter and can be changed.
  //
  // WARNING: This only holds if the assumptions we have taken do not reduce
  //          the set of statement instances that are executed. Otherwise we
  //          may run into a case where the iteration domains suggest that
  //          for a certain set of parameter constraints no code is executed,
  //          but in the original program some computation would have been
  //          performed. In such a case, modifying the run-time conditions and
  //          possibly influencing the run-time check may cause certain scops
  //          to not be executed.
  //
  // Example:
  //
  //   When delinearizing the following code:
  //
  //     for (long i = 0; i < 100; i++)
  //       for (long j = 0; j < m; j++)
  //         A[i+p][j] = 1.0;
  //
  //   we assume that the condition m <= 0 or (m >= 1 and p >= 0) holds as
  //   otherwise we would access out of bound data. Now, knowing that code is
  //   only executed for the case m >= 0, it is sufficient to assume p >= 0.
  AssumedContext =
      isl_set_gist_params(AssumedContext, isl_union_set_params(getDomains()));
  AssumedContext = isl_set_gist_params(AssumedContext, getContext());
}

/// @brief Add the minimal/maximal access in @p Set to @p User.
static isl_stat buildMinMaxAccess(__isl_take isl_set *Set, void *User) {
  Scop::MinMaxVectorTy *MinMaxAccesses = (Scop::MinMaxVectorTy *)User;
  isl_pw_multi_aff *MinPMA, *MaxPMA;
  isl_pw_aff *LastDimAff;
  isl_aff *OneAff;
  unsigned Pos;

  // Restrict the number of parameters involved in the access as the lexmin/
  // lexmax computation will take too long if this number is high.
  //
  // Experiments with a simple test case using an i7 4800MQ:
  //
  //  #Parameters involved | Time (in sec)
  //            6          |     0.01
  //            7          |     0.04
  //            8          |     0.12
  //            9          |     0.40
  //           10          |     1.54
  //           11          |     6.78
  //           12          |    30.38
  //
  if (isl_set_n_param(Set) > RunTimeChecksMaxParameters) {
    unsigned InvolvedParams = 0;
    for (unsigned u = 0, e = isl_set_n_param(Set); u < e; u++)
      if (isl_set_involves_dims(Set, isl_dim_param, u, 1))
        InvolvedParams++;

    if (InvolvedParams > RunTimeChecksMaxParameters) {
      isl_set_free(Set);
      return isl_stat_error;
    }
  }

  Set = isl_set_remove_divs(Set);

  MinPMA = isl_set_lexmin_pw_multi_aff(isl_set_copy(Set));
  MaxPMA = isl_set_lexmax_pw_multi_aff(isl_set_copy(Set));

  MinPMA = isl_pw_multi_aff_coalesce(MinPMA);
  MaxPMA = isl_pw_multi_aff_coalesce(MaxPMA);

  // Adjust the last dimension of the maximal access by one as we want to
  // enclose the accessed memory region by MinPMA and MaxPMA. The pointer
  // we test during code generation might now point after the end of the
  // allocated array but we will never dereference it anyway.
  assert(isl_pw_multi_aff_dim(MaxPMA, isl_dim_out) &&
         "Assumed at least one output dimension");
  Pos = isl_pw_multi_aff_dim(MaxPMA, isl_dim_out) - 1;
  LastDimAff = isl_pw_multi_aff_get_pw_aff(MaxPMA, Pos);
  OneAff = isl_aff_zero_on_domain(
      isl_local_space_from_space(isl_pw_aff_get_domain_space(LastDimAff)));
  OneAff = isl_aff_add_constant_si(OneAff, 1);
  LastDimAff = isl_pw_aff_add(LastDimAff, isl_pw_aff_from_aff(OneAff));
  MaxPMA = isl_pw_multi_aff_set_pw_aff(MaxPMA, Pos, LastDimAff);

  MinMaxAccesses->push_back(std::make_pair(MinPMA, MaxPMA));

  isl_set_free(Set);
  return isl_stat_ok;
}

static __isl_give isl_set *getAccessDomain(MemoryAccess *MA) {
  isl_set *Domain = MA->getStatement()->getDomain();
  Domain = isl_set_project_out(Domain, isl_dim_set, 0, isl_set_n_dim(Domain));
  return isl_set_reset_tuple_id(Domain);
}

bool Scop::buildAliasGroups(AliasAnalysis &AA) {
  // To create sound alias checks we perform the following steps:
  //   o) Use the alias analysis and an alias set tracker to build alias sets
  //      for all memory accesses inside the SCoP.
  //   o) For each alias set we then map the aliasing pointers back to the
  //      memory accesses we know, thus obtain groups of memory accesses which
  //      might alias.
  //   o) We divide each group based on the domains of the minimal/maximal
  //      accesses. That means two minimal/maximal accesses are only in a group
  //      if their access domains intersect, otherwise they are in different
  //      ones.
  //   o) We split groups such that they contain at most one read only base
  //      address.
  //   o) For each group with more than one base pointer we then compute minimal
  //      and maximal accesses to each array in this group.
  using AliasGroupTy = SmallVector<MemoryAccess *, 4>;

  AliasSetTracker AST(AA);

  DenseMap<Value *, MemoryAccess *> PtrToAcc;
  DenseSet<Value *> HasWriteAccess;
  for (ScopStmt &Stmt : *this) {

    // Skip statements with an empty domain as they will never be executed.
    isl_set *StmtDomain = Stmt.getDomain();
    bool StmtDomainEmpty = isl_set_is_empty(StmtDomain);
    isl_set_free(StmtDomain);
    if (StmtDomainEmpty)
      continue;

    for (MemoryAccess *MA : Stmt) {
      if (MA->isScalar())
        continue;
      if (!MA->isRead())
        HasWriteAccess.insert(MA->getBaseAddr());
      Instruction *Acc = MA->getAccessInstruction();
      PtrToAcc[getPointerOperand(*Acc)] = MA;
      AST.add(Acc);
    }
  }

  SmallVector<AliasGroupTy, 4> AliasGroups;
  for (AliasSet &AS : AST) {
    if (AS.isMustAlias() || AS.isForwardingAliasSet())
      continue;
    AliasGroupTy AG;
    for (auto PR : AS)
      AG.push_back(PtrToAcc[PR.getValue()]);
    assert(AG.size() > 1 &&
           "Alias groups should contain at least two accesses");
    AliasGroups.push_back(std::move(AG));
  }

  // Split the alias groups based on their domain.
  for (unsigned u = 0; u < AliasGroups.size(); u++) {
    AliasGroupTy NewAG;
    AliasGroupTy &AG = AliasGroups[u];
    AliasGroupTy::iterator AGI = AG.begin();
    isl_set *AGDomain = getAccessDomain(*AGI);
    while (AGI != AG.end()) {
      MemoryAccess *MA = *AGI;
      isl_set *MADomain = getAccessDomain(MA);
      if (isl_set_is_disjoint(AGDomain, MADomain)) {
        NewAG.push_back(MA);
        AGI = AG.erase(AGI);
        isl_set_free(MADomain);
      } else {
        AGDomain = isl_set_union(AGDomain, MADomain);
        AGI++;
      }
    }
    if (NewAG.size() > 1)
      AliasGroups.push_back(std::move(NewAG));
    isl_set_free(AGDomain);
  }

  MapVector<const Value *, SmallPtrSet<MemoryAccess *, 8>> ReadOnlyPairs;
  SmallPtrSet<const Value *, 4> NonReadOnlyBaseValues;
  for (AliasGroupTy &AG : AliasGroups) {
    NonReadOnlyBaseValues.clear();
    ReadOnlyPairs.clear();

    if (AG.size() < 2) {
      AG.clear();
      continue;
    }

    for (auto II = AG.begin(); II != AG.end();) {
      Value *BaseAddr = (*II)->getBaseAddr();
      if (HasWriteAccess.count(BaseAddr)) {
        NonReadOnlyBaseValues.insert(BaseAddr);
        II++;
      } else {
        ReadOnlyPairs[BaseAddr].insert(*II);
        II = AG.erase(II);
      }
    }

    // If we don't have read only pointers check if there are at least two
    // non read only pointers, otherwise clear the alias group.
    if (ReadOnlyPairs.empty()) {
      if (NonReadOnlyBaseValues.size() <= 1)
        AG.clear();
      continue;
    }

    // If we don't have non read only pointers clear the alias group.
    if (NonReadOnlyBaseValues.empty()) {
      AG.clear();
      continue;
    }

    // If we have both read only and non read only base pointers we combine
    // the non read only ones with exactly one read only one at a time into a
    // new alias group and clear the old alias group in the end.
    for (const auto &ReadOnlyPair : ReadOnlyPairs) {
      AliasGroupTy AGNonReadOnly = AG;
      for (MemoryAccess *MA : ReadOnlyPair.second)
        AGNonReadOnly.push_back(MA);
      AliasGroups.push_back(std::move(AGNonReadOnly));
    }
    AG.clear();
  }

  for (AliasGroupTy &AG : AliasGroups) {
    if (AG.empty())
      continue;

    MinMaxVectorTy *MinMaxAccesses = new MinMaxVectorTy();
    MinMaxAccesses->reserve(AG.size());

    isl_union_map *Accesses = isl_union_map_empty(getParamSpace());
    for (MemoryAccess *MA : AG)
      Accesses = isl_union_map_add_map(Accesses, MA->getAccessRelation());
    Accesses = isl_union_map_intersect_domain(Accesses, getDomains());

    isl_union_set *Locations = isl_union_map_range(Accesses);
    Locations = isl_union_set_intersect_params(Locations, getAssumedContext());
    Locations = isl_union_set_coalesce(Locations);
    Locations = isl_union_set_detect_equalities(Locations);
    bool Valid = (0 == isl_union_set_foreach_set(Locations, buildMinMaxAccess,
                                                 MinMaxAccesses));
    isl_union_set_free(Locations);
    MinMaxAliasGroups.push_back(MinMaxAccesses);

    if (!Valid)
      return false;
  }

  // Bail out if the number of values we need to compare is too large.
  // This is important as the number of comparisions grows quadratically with
  // the number of values we need to compare.
  for (const auto *Values : MinMaxAliasGroups)
    if (Values->size() > RunTimeChecksMaxArraysPerGroup)
      return false;

  return true;
}

static unsigned getMaxLoopDepthInRegion(const Region &R, LoopInfo &LI,
                                        ScopDetection &SD) {

  const ScopDetection::BoxedLoopsSetTy *BoxedLoops = SD.getBoxedLoops(&R);

  unsigned MinLD = INT_MAX, MaxLD = 0;
  for (BasicBlock *BB : R.blocks()) {
    if (Loop *L = LI.getLoopFor(BB)) {
      if (!R.contains(L))
        continue;
      if (BoxedLoops && BoxedLoops->count(L))
        continue;
      unsigned LD = L->getLoopDepth();
      MinLD = std::min(MinLD, LD);
      MaxLD = std::max(MaxLD, LD);
    }
  }

  // Handle the case that there is no loop in the SCoP first.
  if (MaxLD == 0)
    return 1;

  assert(MinLD >= 1 && "Minimal loop depth should be at least one");
  assert(MaxLD >= MinLD &&
         "Maximal loop depth was smaller than mininaml loop depth?");
  return MaxLD - MinLD + 1;
}

void Scop::dropConstantScheduleDims() {
  isl_union_map *FullSchedule = getSchedule();

  if (isl_union_map_n_map(FullSchedule) == 0) {
    isl_union_map_free(FullSchedule);
    return;
  }

  isl_set *ScheduleSpace =
      isl_set_from_union_set(isl_union_map_range(FullSchedule));
  isl_map *DropDimMap = isl_set_identity(isl_set_copy(ScheduleSpace));

  int NumDimsDropped = 0;
  for (unsigned i = 0; i < isl_set_dim(ScheduleSpace, isl_dim_set); i += 2) {
    isl_val *FixedVal =
        isl_set_plain_get_val_if_fixed(ScheduleSpace, isl_dim_set, i);
    if (isl_val_is_int(FixedVal)) {
      DropDimMap =
          isl_map_project_out(DropDimMap, isl_dim_out, i - NumDimsDropped, 1);
      NumDimsDropped++;
    }
    isl_val_free(FixedVal);
  }

  for (ScopStmt &Stmt : *this) {
    isl_map *Schedule = Stmt.getSchedule();
    Schedule = isl_map_apply_range(Schedule, isl_map_copy(DropDimMap));
    Stmt.setSchedule(Schedule);
  }
  isl_set_free(ScheduleSpace);
  isl_map_free(DropDimMap);
}

Scop::Scop(TempScop &tempScop, LoopInfo &LI, ScalarEvolution &ScalarEvolution,
           ScopDetection &SD, isl_ctx *Context)
    : SE(&ScalarEvolution), R(tempScop.getMaxRegion()), IsOptimized(false),
      MaxLoopDepth(getMaxLoopDepthInRegion(tempScop.getMaxRegion(), LI, SD)) {
  IslCtx = Context;

  buildContext();

  SmallVector<Loop *, 8> NestLoops;
  SmallVector<unsigned, 8> Schedule;

  Schedule.assign(MaxLoopDepth + 1, 0);

  // Build the iteration domain, access functions and schedule functions
  // traversing the region tree.
  buildScop(tempScop, getRegion(), NestLoops, Schedule, LI, SD);

  realignParams();
  addParameterBounds();
  simplifyAssumedContext();
  dropConstantScheduleDims();

  assert(NestLoops.empty() && "NestLoops not empty at top level!");
}

Scop::~Scop() {
  isl_set_free(Context);
  isl_set_free(AssumedContext);

  // Free the alias groups
  for (MinMaxVectorTy *MinMaxAccesses : MinMaxAliasGroups) {
    for (MinMaxAccessTy &MMA : *MinMaxAccesses) {
      isl_pw_multi_aff_free(MMA.first);
      isl_pw_multi_aff_free(MMA.second);
    }
    delete MinMaxAccesses;
  }
}

const ScopArrayInfo *
Scop::getOrCreateScopArrayInfo(Value *BasePtr, Type *AccessType,
                               const SmallVector<const SCEV *, 4> &Sizes) {
  auto &SAI = ScopArrayInfoMap[BasePtr];
  if (!SAI)
    SAI.reset(new ScopArrayInfo(BasePtr, AccessType, getIslCtx(), Sizes));
  return SAI.get();
}

const ScopArrayInfo *Scop::getScopArrayInfo(Value *BasePtr) {
  const ScopArrayInfo *SAI = ScopArrayInfoMap[BasePtr].get();
  assert(SAI && "No ScopArrayInfo available for this base pointer");
  return SAI;
}

std::string Scop::getContextStr() const { return stringFromIslObj(Context); }
std::string Scop::getAssumedContextStr() const {
  return stringFromIslObj(AssumedContext);
}

std::string Scop::getNameStr() const {
  std::string ExitName, EntryName;
  raw_string_ostream ExitStr(ExitName);
  raw_string_ostream EntryStr(EntryName);

  R.getEntry()->printAsOperand(EntryStr, false);
  EntryStr.str();

  if (R.getExit()) {
    R.getExit()->printAsOperand(ExitStr, false);
    ExitStr.str();
  } else
    ExitName = "FunctionExit";

  return EntryName + "---" + ExitName;
}

__isl_give isl_set *Scop::getContext() const { return isl_set_copy(Context); }
__isl_give isl_space *Scop::getParamSpace() const {
  return isl_set_get_space(Context);
}

__isl_give isl_set *Scop::getAssumedContext() const {
  return isl_set_copy(AssumedContext);
}

void Scop::addAssumption(__isl_take isl_set *Set) {
  AssumedContext = isl_set_intersect(AssumedContext, Set);
  AssumedContext = isl_set_coalesce(AssumedContext);
}

void Scop::printContext(raw_ostream &OS) const {
  OS << "Context:\n";

  if (!Context) {
    OS.indent(4) << "n/a\n\n";
    return;
  }

  OS.indent(4) << getContextStr() << "\n";

  OS.indent(4) << "Assumed Context:\n";
  if (!AssumedContext) {
    OS.indent(4) << "n/a\n\n";
    return;
  }

  OS.indent(4) << getAssumedContextStr() << "\n";

  for (const SCEV *Parameter : Parameters) {
    int Dim = ParameterIds.find(Parameter)->second;
    OS.indent(4) << "p" << Dim << ": " << *Parameter << "\n";
  }
}

void Scop::printAliasAssumptions(raw_ostream &OS) const {
  OS.indent(4) << "Alias Groups (" << MinMaxAliasGroups.size() << "):\n";
  if (MinMaxAliasGroups.empty()) {
    OS.indent(8) << "n/a\n";
    return;
  }
  for (MinMaxVectorTy *MinMaxAccesses : MinMaxAliasGroups) {
    OS.indent(8) << "[[";
    for (MinMaxAccessTy &MinMacAccess : *MinMaxAccesses)
      OS << " <" << MinMacAccess.first << ", " << MinMacAccess.second << ">";
    OS << " ]]\n";
  }
}

void Scop::printStatements(raw_ostream &OS) const {
  OS << "Statements {\n";

  for (const ScopStmt &Stmt : *this)
    OS.indent(4) << Stmt;

  OS.indent(4) << "}\n";
}

void Scop::printArrayInfo(raw_ostream &OS) const {
  OS << "Arrays {\n";

  for (auto &Array : arrays())
    Array.second->print(OS);

  OS.indent(4) << "}\n";
}

void Scop::print(raw_ostream &OS) const {
  OS.indent(4) << "Function: " << getRegion().getEntry()->getParent()->getName()
               << "\n";
  OS.indent(4) << "Region: " << getNameStr() << "\n";
  OS.indent(4) << "Max Loop Depth:  " << getMaxLoopDepth() << "\n";
  printContext(OS.indent(4));
  printArrayInfo(OS.indent(4));
  printAliasAssumptions(OS);
  printStatements(OS.indent(4));
}

void Scop::dump() const { print(dbgs()); }

isl_ctx *Scop::getIslCtx() const { return IslCtx; }

__isl_give isl_union_set *Scop::getDomains() {
  isl_union_set *Domain = isl_union_set_empty(getParamSpace());

  for (ScopStmt &Stmt : *this)
    Domain = isl_union_set_add_set(Domain, Stmt.getDomain());

  return Domain;
}

__isl_give isl_union_map *Scop::getMustWrites() {
  isl_union_map *Write = isl_union_map_empty(getParamSpace());

  for (ScopStmt &Stmt : *this) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isMustWrite())
        continue;

      isl_set *Domain = Stmt.getDomain();
      isl_map *AccessDomain = MA->getAccessRelation();
      AccessDomain = isl_map_intersect_domain(AccessDomain, Domain);
      Write = isl_union_map_add_map(Write, AccessDomain);
    }
  }
  return isl_union_map_coalesce(Write);
}

__isl_give isl_union_map *Scop::getMayWrites() {
  isl_union_map *Write = isl_union_map_empty(getParamSpace());

  for (ScopStmt &Stmt : *this) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isMayWrite())
        continue;

      isl_set *Domain = Stmt.getDomain();
      isl_map *AccessDomain = MA->getAccessRelation();
      AccessDomain = isl_map_intersect_domain(AccessDomain, Domain);
      Write = isl_union_map_add_map(Write, AccessDomain);
    }
  }
  return isl_union_map_coalesce(Write);
}

__isl_give isl_union_map *Scop::getWrites() {
  isl_union_map *Write = isl_union_map_empty(getParamSpace());

  for (ScopStmt &Stmt : *this) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isWrite())
        continue;

      isl_set *Domain = Stmt.getDomain();
      isl_map *AccessDomain = MA->getAccessRelation();
      AccessDomain = isl_map_intersect_domain(AccessDomain, Domain);
      Write = isl_union_map_add_map(Write, AccessDomain);
    }
  }
  return isl_union_map_coalesce(Write);
}

__isl_give isl_union_map *Scop::getReads() {
  isl_union_map *Read = isl_union_map_empty(getParamSpace());

  for (ScopStmt &Stmt : *this) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isRead())
        continue;

      isl_set *Domain = Stmt.getDomain();
      isl_map *AccessDomain = MA->getAccessRelation();

      AccessDomain = isl_map_intersect_domain(AccessDomain, Domain);
      Read = isl_union_map_add_map(Read, AccessDomain);
    }
  }
  return isl_union_map_coalesce(Read);
}

__isl_give isl_union_map *Scop::getSchedule() {
  isl_union_map *Schedule = isl_union_map_empty(getParamSpace());

  for (ScopStmt &Stmt : *this)
    Schedule = isl_union_map_add_map(Schedule, Stmt.getSchedule());

  return isl_union_map_coalesce(Schedule);
}

bool Scop::restrictDomains(__isl_take isl_union_set *Domain) {
  bool Changed = false;
  for (ScopStmt &Stmt : *this) {
    isl_union_set *StmtDomain = isl_union_set_from_set(Stmt.getDomain());
    isl_union_set *NewStmtDomain = isl_union_set_intersect(
        isl_union_set_copy(StmtDomain), isl_union_set_copy(Domain));

    if (isl_union_set_is_subset(StmtDomain, NewStmtDomain)) {
      isl_union_set_free(StmtDomain);
      isl_union_set_free(NewStmtDomain);
      continue;
    }

    Changed = true;

    isl_union_set_free(StmtDomain);
    NewStmtDomain = isl_union_set_coalesce(NewStmtDomain);

    if (isl_union_set_is_empty(NewStmtDomain)) {
      Stmt.restrictDomain(isl_set_empty(Stmt.getDomainSpace()));
      isl_union_set_free(NewStmtDomain);
    } else
      Stmt.restrictDomain(isl_set_from_union_set(NewStmtDomain));
  }
  isl_union_set_free(Domain);
  return Changed;
}

ScalarEvolution *Scop::getSE() const { return SE; }

bool Scop::isTrivialBB(BasicBlock *BB, TempScop &tempScop) {
  if (tempScop.getAccessFunctions(BB))
    return false;

  return true;
}

void Scop::addScopStmt(BasicBlock *BB, Region *R, TempScop &tempScop,
                       const Region &CurRegion,
                       SmallVectorImpl<Loop *> &NestLoops,
                       SmallVectorImpl<unsigned> &ScheduleVec) {
  if (BB) {
    Stmts.emplace_back(*this, tempScop, CurRegion, *BB, NestLoops, ScheduleVec);
    StmtMap[BB] = &Stmts.back();
  } else {
    assert(R && "Either basic block or a region expected.");
    Stmts.emplace_back(*this, tempScop, CurRegion, *R, NestLoops, ScheduleVec);
    auto *Ptr = &Stmts.back();
    for (BasicBlock *BB : R->blocks())
      StmtMap[BB] = Ptr;
  }

  // Increasing the Schedule function is OK for the moment, because
  // we are using a depth first iterator and the program is well structured.
  ++ScheduleVec[NestLoops.size()];
}

void Scop::buildScop(TempScop &tempScop, const Region &CurRegion,
                     SmallVectorImpl<Loop *> &NestLoops,
                     SmallVectorImpl<unsigned> &ScheduleVec, LoopInfo &LI,
                     ScopDetection &SD) {
  if (SD.isNonAffineSubRegion(&CurRegion, &getRegion()))
    return addScopStmt(nullptr, const_cast<Region *>(&CurRegion), tempScop,
                       CurRegion, NestLoops, ScheduleVec);

  Loop *L = castToLoop(CurRegion, LI);

  if (L)
    NestLoops.push_back(L);

  unsigned loopDepth = NestLoops.size();
  assert(ScheduleVec.size() > loopDepth && "Schedule not big enough!");

  for (Region::const_element_iterator I = CurRegion.element_begin(),
                                      E = CurRegion.element_end();
       I != E; ++I)
    if (I->isSubRegion()) {
      buildScop(tempScop, *I->getNodeAs<Region>(), NestLoops, ScheduleVec, LI,
                SD);
    } else {
      BasicBlock *BB = I->getNodeAs<BasicBlock>();

      if (isTrivialBB(BB, tempScop))
        continue;

      addScopStmt(BB, nullptr, tempScop, CurRegion, NestLoops, ScheduleVec);
    }

  if (!L)
    return;

  // Exiting a loop region.
  ScheduleVec[loopDepth] = 0;
  NestLoops.pop_back();
  ++ScheduleVec[loopDepth - 1];
}

ScopStmt *Scop::getStmtForBasicBlock(BasicBlock *BB) const {
  auto StmtMapIt = StmtMap.find(BB);
  if (StmtMapIt == StmtMap.end())
    return nullptr;
  return StmtMapIt->second;
}

//===----------------------------------------------------------------------===//
ScopInfo::ScopInfo() : RegionPass(ID), scop(0) {
  ctx = isl_ctx_alloc();
  isl_options_set_on_error(ctx, ISL_ON_ERROR_ABORT);
}

ScopInfo::~ScopInfo() {
  clear();
  isl_ctx_free(ctx);
}

void ScopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<RegionInfoPass>();
  AU.addRequired<ScalarEvolution>();
  AU.addRequired<ScopDetection>();
  AU.addRequired<TempScopInfo>();
  AU.addRequired<AliasAnalysis>();
  AU.setPreservesAll();
}

bool ScopInfo::runOnRegion(Region *R, RGPassManager &RGM) {
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  ScopDetection &SD = getAnalysis<ScopDetection>();
  ScalarEvolution &SE = getAnalysis<ScalarEvolution>();

  TempScop *tempScop = getAnalysis<TempScopInfo>().getTempScop(R);

  // This region is no Scop.
  if (!tempScop) {
    scop = nullptr;
    return false;
  }

  scop = new Scop(*tempScop, LI, SE, SD, ctx);

  DEBUG(scop->print(dbgs()));

  if (!PollyUseRuntimeAliasChecks) {
    // Statistics.
    ++ScopFound;
    if (scop->getMaxLoopDepth() > 0)
      ++RichScopFound;
    return false;
  }

  // If a problem occurs while building the alias groups we need to delete
  // this SCoP and pretend it wasn't valid in the first place.
  if (scop->buildAliasGroups(AA)) {
    // Statistics.
    ++ScopFound;
    if (scop->getMaxLoopDepth() > 0)
      ++RichScopFound;
    return false;
  }

  DEBUG(dbgs()
        << "\n\nNOTE: Run time checks for " << scop->getNameStr()
        << " could not be created as the number of parameters involved is too "
           "high. The SCoP will be "
           "dismissed.\nUse:\n\t--polly-rtc-max-parameters=X\nto adjust the "
           "maximal number of parameters but be advised that the compile time "
           "might increase exponentially.\n\n");

  delete scop;
  scop = nullptr;
  return false;
}

char ScopInfo::ID = 0;

Pass *polly::createScopInfoPass() { return new ScopInfo(); }

INITIALIZE_PASS_BEGIN(ScopInfo, "polly-scops",
                      "Polly - Create polyhedral description of Scops", false,
                      false);
INITIALIZE_AG_DEPENDENCY(AliasAnalysis);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_DEPENDENCY(TempScopInfo);
INITIALIZE_PASS_END(ScopInfo, "polly-scops",
                    "Polly - Create polyhedral description of Scops", false,
                    false)

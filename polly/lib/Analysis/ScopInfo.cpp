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
// This represantation is shared among several tools in the polyhedral
// community, which are e.g. Cloog, Pluto, Loopo, Graphite.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopInfo.h"

#include "polly/TempScopInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ScopHelper.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "polly-scops"
#include "llvm/Support/Debug.h"

#include "isl/constraint.h"
#include "isl/set.h"
#include "isl/map.h"
#include "isl/aff.h"
#include "isl/printer.h"
#include <sstream>
#include <string>
#include <vector>

using namespace llvm;
using namespace polly;

STATISTIC(ScopFound,  "Number of valid Scops");
STATISTIC(RichScopFound,   "Number of Scops containing a loop");

/// Convert an int into a string.
static std::string convertInt(int number)
{
  if (number == 0)
    return "0";
  std::string temp = "";
  std::string returnvalue = "";
  while (number > 0)
  {
    temp += number % 10 + 48;
    number /= 10;
  }
  for (unsigned i = 0; i < temp.length(); i++)
    returnvalue+=temp[temp.length() - i - 1];
  return returnvalue;
}

static isl_set *set_remove_dim_ids(isl_set *set) {
  isl_ctx *ctx = isl_set_get_ctx(set);
  int numParams = isl_set_n_param(set);
  isl_printer *p = isl_printer_to_str(ctx);
  isl_set *new_set;
  char *str;
  const char *name = isl_set_get_tuple_name(set);

  p = isl_printer_set_output_format (p, ISL_FORMAT_EXT_POLYLIB);
  p = isl_printer_print_set(p, set);

  str = isl_printer_get_str (p);
  new_set = isl_set_read_from_str (ctx, str, numParams);
  new_set = isl_set_set_tuple_name(new_set, name);
  free (str);
  isl_set_free (set);
  isl_printer_free (p);
  return new_set;
}

static isl_map *map_remove_dim_ids(isl_map *map) {
  isl_ctx *ctx = isl_map_get_ctx(map);
  int numParams = isl_map_n_param(map);
  isl_printer *p = isl_printer_to_str(ctx);
  char *str;
  isl_map *new_map;
  const char *name_in = isl_map_get_tuple_name(map, isl_dim_in);
  const char *name_out = isl_map_get_tuple_name(map, isl_dim_out);

  p = isl_printer_set_output_format (p, ISL_FORMAT_EXT_POLYLIB);
  p = isl_printer_print_map(p, map);

  str = isl_printer_get_str (p);
  new_map = isl_map_read_from_str (ctx, str, numParams);
  new_map = isl_map_set_tuple_name(new_map, isl_dim_in, name_in);
  new_map = isl_map_set_tuple_name(new_map, isl_dim_out, name_out);
  isl_map_free (map);
  free (str);
  isl_printer_free (p);
  return new_map;
}

/// Translate a SCEVExpression into an isl_pw_aff object.
struct SCEVAffinator : public SCEVVisitor<SCEVAffinator, isl_pw_aff*> {
private:
  isl_ctx *ctx;
  int NbLoopDims;
  const Scop *scop;

  /// baseAdress is set if we analyze a memory access. It holds the base address
  /// of this memory access.
  const Value *baseAddress;

public:
  static isl_pw_aff *getPwAff(const ScopStmt *stmt, const SCEV *scev,
                              const Value *baseAddress) {
    SCEVAffinator Affinator(stmt, baseAddress);
    return Affinator.visit(scev);
  }

  isl_pw_aff *visit(const SCEV *scev) {
    // In case the scev is contained in our list of parameters, we do not
    // further analyze this expression, but create a new parameter in the
    // isl_pw_aff. This allows us to treat subexpressions that we cannot
    // translate into an piecewise affine expression, as constant parameters of
    // the piecewise affine expression.
    int i = 0;
    for (Scop::const_param_iterator PI = scop->param_begin(),
         PE = scop->param_end(); PI != PE; ++PI) {
      if (*PI == scev) {
        isl_id *ID = isl_id_alloc(ctx, ("p" + convertInt(i)).c_str(),
                                  (void *) scev);
        isl_dim *Dim = isl_dim_set_alloc(ctx, 1, NbLoopDims);
        Dim = isl_dim_set_dim_id(Dim, isl_dim_param, 0, ID);

        isl_set *Domain = isl_set_universe(isl_dim_copy(Dim));
        isl_aff *Affine = isl_aff_zero(isl_local_space_from_dim(Dim));
        Affine = isl_aff_add_coefficient_si(Affine, isl_dim_param, 0, 1);

        return isl_pw_aff_alloc(Domain, Affine);
      }
      i++;
    }

    return SCEVVisitor<SCEVAffinator, isl_pw_aff*>::visit(scev);
  }

  SCEVAffinator(const ScopStmt *stmt, const Value *baseAddress) :
    ctx(stmt->getParent()->getCtx()),
    NbLoopDims(stmt->getNumIterators()),
    scop(stmt->getParent()),
    baseAddress(baseAddress) {};

  __isl_give isl_pw_aff *visitConstant(const SCEVConstant *Constant) {
    ConstantInt *Value = Constant->getValue();
    isl_int v;
    isl_int_init(v);

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
    MPZ_from_APInt(v, Value->getValue(), /* isSigned */ true);

    isl_dim *dim = isl_dim_set_alloc(ctx, 0, NbLoopDims);
    isl_local_space *ls = isl_local_space_from_dim(isl_dim_copy(dim));
    isl_aff *Affine = isl_aff_zero(ls);
    isl_set *Domain = isl_set_universe(dim);

    Affine = isl_aff_add_constant(Affine, v);
    isl_int_clear(v);

    return isl_pw_aff_alloc(Domain, Affine);
  }

  __isl_give isl_pw_aff *visitTruncateExpr(const SCEVTruncateExpr* Expr) {
    assert(0 && "Not yet supported");
  }

  __isl_give isl_pw_aff *visitZeroExtendExpr(const SCEVZeroExtendExpr * Expr) {
    assert(0 && "Not yet supported");
  }

  __isl_give isl_pw_aff *visitSignExtendExpr(const SCEVSignExtendExpr* Expr) {
    // Assuming the value is signed, a sign extension is basically a noop.
    // TODO: Reconsider this as soon as we support unsigned values.
    return visit(Expr->getOperand());
  }

  __isl_give isl_pw_aff *visitAddExpr(const SCEVAddExpr* Expr) {
    isl_pw_aff *Sum = visit(Expr->getOperand(0));

    for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
      isl_pw_aff *NextSummand = visit(Expr->getOperand(i));
      Sum = isl_pw_aff_add(Sum, NextSummand);
    }

    // TODO: Check for NSW and NUW.

    return Sum;
  }

  __isl_give isl_pw_aff *visitMulExpr(const SCEVMulExpr* Expr) {
    isl_pw_aff *Product = visit(Expr->getOperand(0));

    for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
      isl_pw_aff *NextOperand = visit(Expr->getOperand(i));

      if (!isl_pw_aff_is_cst(Product) && !isl_pw_aff_is_cst(NextOperand)) {
        isl_pw_aff_free(Product);
        isl_pw_aff_free(NextOperand);
        return NULL;
      }

      Product = isl_pw_aff_mul(Product, NextOperand);
    }

    // TODO: Check for NSW and NUW.
    return Product;
  }

  __isl_give isl_pw_aff *visitUDivExpr(const SCEVUDivExpr* Expr) {
    assert(0 && "Not yet supported");
  }

  int getLoopDepth(const Loop *L) {
    Loop *outerLoop =
      scop->getRegion().outermostLoopInRegion(const_cast<Loop*>(L));
    return L->getLoopDepth() - outerLoop->getLoopDepth();
  }

  __isl_give isl_pw_aff *visitAddRecExpr(const SCEVAddRecExpr* Expr) {
    assert(Expr->isAffine() && "Only affine AddRecurrences allowed");

    isl_pw_aff *Start = visit(Expr->getStart());
    isl_pw_aff *Step = visit(Expr->getOperand(1));
    isl_dim *Dim = isl_dim_set_alloc (ctx, 0, NbLoopDims);
    isl_local_space *LocalSpace = isl_local_space_from_dim (Dim);

    int loopDimension = getLoopDepth(Expr->getLoop());

    isl_aff *LAff = isl_aff_set_coefficient_si (isl_aff_zero (LocalSpace),
                                                isl_dim_set, loopDimension, 1);
    isl_pw_aff *LPwAff = isl_pw_aff_from_aff(LAff);

    // TODO: Do we need to check for NSW and NUW?
    return isl_pw_aff_add(Start, isl_pw_aff_mul(Step, LPwAff));
  }

  __isl_give isl_pw_aff *visitSMaxExpr(const SCEVSMaxExpr* Expr) {
    isl_pw_aff *Max = visit(Expr->getOperand(0));

    for (int i = 1, e = Expr->getNumOperands(); i < e; ++i) {
      isl_pw_aff *NextOperand = visit(Expr->getOperand(i));
      Max = isl_pw_aff_max(Max, NextOperand);
    }

    return Max;
  }

  __isl_give isl_pw_aff *visitUMaxExpr(const SCEVUMaxExpr* Expr) {
    assert(0 && "Not yet supported");
  }

  __isl_give isl_pw_aff *visitUnknown(const SCEVUnknown* Expr) {
    Value *Value = Expr->getValue();

    isl_dim *Dim;

    /// If baseAddress is set, we ignore its Value object in the scev and do not
    /// add it to the isl_pw_aff. This is because it is regarded as defining the
    /// name of an array, in contrast to its array subscript.
    if (baseAddress != Value) {
      isl_id *ID = isl_id_alloc(ctx, Value->getNameStr().c_str(), Value);
      Dim = isl_dim_set_alloc(ctx, 1, NbLoopDims);
      Dim = isl_dim_set_dim_id(Dim, isl_dim_param, 0, ID);
    } else {
      Dim = isl_dim_set_alloc(ctx, 0, NbLoopDims);
    }

    isl_set *Domain = isl_set_universe(isl_dim_copy(Dim));
    isl_aff *Affine = isl_aff_zero(isl_local_space_from_dim(Dim));

    if (baseAddress != Value)
      Affine = isl_aff_add_coefficient_si(Affine, isl_dim_param, 0, 1);

    return isl_pw_aff_alloc(Domain, Affine);
  }
};

//===----------------------------------------------------------------------===//

MemoryAccess::~MemoryAccess() {
  isl_map_free(AccessRelation);
  isl_map_free(newAccessRelation);
}

static void replace(std::string& str, const std::string& find,
                    const std::string& replace) {
  size_t pos = 0;
  while((pos = str.find(find, pos)) != std::string::npos)
  {
    str.replace(pos, find.length(), replace);
    pos += replace.length();
  }
}

static void makeIslCompatible(std::string& str) {
  replace(str, ".", "_");
  replace(str, "\"", "_");
}

void MemoryAccess::setBaseName() {
  raw_string_ostream OS(BaseName);
  WriteAsOperand(OS, getBaseAddr(), false);
  BaseName = OS.str();

  // Remove the % in the name. This is not supported by isl.
  BaseName.erase(0,1);
  makeIslCompatible(BaseName);
  BaseName = "MemRef_" + BaseName;
}

std::string MemoryAccess::getAccessFunctionStr() const {
  return stringFromIslObj(getAccessFunction());
}

isl_basic_map *MemoryAccess::createBasicAccessMap(ScopStmt *Statement) {
  isl_dim *dim = isl_dim_alloc(Statement->getIslContext(),
                               Statement->getNumParams(),
                               Statement->getNumIterators(), 1);
  setBaseName();

  dim = isl_dim_set_tuple_name(dim, isl_dim_out, getBaseName().c_str());
  dim = isl_dim_set_tuple_name(dim, isl_dim_in, Statement->getBaseName());

  return isl_basic_map_universe(dim);
}

MemoryAccess::MemoryAccess(const SCEVAffFunc &AffFunc, ScopStmt *Statement) {
  newAccessRelation = NULL;
  BaseAddr = AffFunc.getBaseAddr();
  Type = AffFunc.isRead() ? Read : Write;
  statement = Statement;

  setBaseName();

  isl_pw_aff *Affine = SCEVAffinator::getPwAff(Statement, AffFunc.OriginalSCEV,
                                               AffFunc.getBaseAddr());

  // Devide the access function by the size of the elements in the array.
  //
  // A stride one array access in C expressed as A[i] is expressed in LLVM-IR
  // as something like A[i * elementsize]. This hides the fact that two
  // subsequent values of 'i' index two values that are stored next to each
  // other in memory. By this devision we make this characteristic obvious
  // again.
  isl_int v;
  isl_int_init(v);
  isl_int_set_si(v, AffFunc.getElemSizeInBytes());
  Affine = isl_pw_aff_scale_down(Affine, v);
  isl_int_clear(v);

  AccessRelation = isl_map_from_pw_aff(Affine);
  AccessRelation = isl_map_set_tuple_name(AccessRelation, isl_dim_in,
                                          Statement->getBaseName());
  AccessRelation = isl_map_set_tuple_name(AccessRelation, isl_dim_out,
                                          getBaseName().c_str());

  isl_dim *Model = isl_set_get_dim(Statement->getParent()->getContext());
  AccessRelation = isl_map_align_params(AccessRelation, Model);

  // FIXME: Temporarily remove dimension ids.
  AccessRelation = map_remove_dim_ids(AccessRelation);
}

MemoryAccess::MemoryAccess(const Value *BaseAddress, ScopStmt *Statement) {
  newAccessRelation = NULL;
  BaseAddr = BaseAddress;
  Type = Read;
  statement = Statement;

  isl_basic_map *BasicAccessMap = createBasicAccessMap(Statement);
  AccessRelation = isl_map_from_basic_map(BasicAccessMap);
}

void MemoryAccess::print(raw_ostream &OS) const {
  OS.indent(12) << (isRead() ? "Read" : "Write") << "Access := \n";
  OS.indent(16) << getAccessFunctionStr() << ";\n";
}

void MemoryAccess::dump() const {
  print(errs());
}

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
static isl_map *getEqualAndLarger(isl_dim *setDomain) {
  isl_dim *mapDomain = isl_dim_map_from_set(setDomain);
  isl_basic_map *bmap = isl_basic_map_universe(mapDomain);

  // Set all but the last dimension to be equal for the input and output
  //
  //   input[i0, i1, ..., iX] -> output[o0, o1, ..., oX]
  //     : i0 = o0, i1 = o1, ..., i(X-1) = o(X-1)
  for (unsigned i = 0; i < isl_basic_map_n_in(bmap) - 1; ++i) {
    isl_int v;
    isl_int_init(v);
    isl_constraint *c = isl_equality_alloc(isl_basic_map_get_dim(bmap));

    isl_int_set_si(v, 1);
    isl_constraint_set_coefficient(c, isl_dim_in, i, v);
    isl_int_set_si(v, -1);
    isl_constraint_set_coefficient(c, isl_dim_out, i, v);

    bmap = isl_basic_map_add_constraint(bmap, c);

    isl_int_clear(v);
  }

  // Set the last dimension of the input to be strict smaller than the
  // last dimension of the output.
  //
  //   input[?,?,?,...,iX] -> output[?,?,?,...,oX] : iX < oX
  //
  unsigned lastDimension = isl_basic_map_n_in(bmap) - 1;
  isl_int v;
  isl_int_init(v);
  isl_constraint *c = isl_inequality_alloc(isl_basic_map_get_dim(bmap));
  isl_int_set_si(v, -1);
  isl_constraint_set_coefficient(c, isl_dim_in, lastDimension, v);
  isl_int_set_si(v, 1);
  isl_constraint_set_coefficient(c, isl_dim_out, lastDimension, v);
  isl_int_set_si(v, -1);
  isl_constraint_set_constant(c, v);
  isl_int_clear(v);

  bmap = isl_basic_map_add_constraint(bmap, c);

  return isl_map_from_basic_map(bmap);
}

isl_set *MemoryAccess::getStride(const isl_set *domainSubset) const {
  isl_map *accessRelation = isl_map_copy(getAccessFunction());
  isl_set *scatteringDomain = isl_set_copy(const_cast<isl_set*>(domainSubset));
  isl_map *scattering = isl_map_copy(getStatement()->getScattering());

  scattering = isl_map_reverse(scattering);
  int difference = isl_map_n_in(scattering) - isl_set_n_dim(scatteringDomain);
  scattering = isl_map_project_out(scattering, isl_dim_in,
                                   isl_set_n_dim(scatteringDomain),
                                   difference);

  // Remove all names of the scattering dimensions, as the names may be lost
  // anyways during the project. This leads to consistent results.
  scattering = isl_map_set_tuple_name(scattering, isl_dim_in, "");
  scatteringDomain = isl_set_set_tuple_name(scatteringDomain, "");

  isl_map *nextScatt = getEqualAndLarger(isl_set_get_dim(scatteringDomain));
  nextScatt = isl_map_lexmin(nextScatt);

  scattering = isl_map_intersect_domain(scattering, scatteringDomain);

  nextScatt = isl_map_apply_range(nextScatt, isl_map_copy(scattering));
  nextScatt = isl_map_apply_range(nextScatt, isl_map_copy(accessRelation));
  nextScatt = isl_map_apply_domain(nextScatt, scattering);
  nextScatt = isl_map_apply_domain(nextScatt, accessRelation);

  return isl_map_deltas(nextScatt);
}

bool MemoryAccess::isStrideZero(const isl_set *domainSubset) const {
  isl_set *stride = getStride(domainSubset);
  isl_constraint *c = isl_equality_alloc(isl_set_get_dim(stride));

  isl_int v;
  isl_int_init(v);
  isl_int_set_si(v, 1);
  isl_constraint_set_coefficient(c, isl_dim_set, 0, v);
  isl_int_set_si(v, 0);
  isl_constraint_set_constant(c, v);
  isl_int_clear(v);

  isl_basic_set *bset = isl_basic_set_universe(isl_set_get_dim(stride));

  bset = isl_basic_set_add_constraint(bset, c);
  isl_set *strideZero = isl_set_from_basic_set(bset);

  return isl_set_is_equal(stride, strideZero);
}

bool MemoryAccess::isStrideOne(const isl_set *domainSubset) const {
  isl_set *stride = getStride(domainSubset);
  isl_constraint *c = isl_equality_alloc(isl_set_get_dim(stride));

  isl_int v;
  isl_int_init(v);
  isl_int_set_si(v, 1);
  isl_constraint_set_coefficient(c, isl_dim_set, 0, v);
  isl_int_set_si(v, -1);
  isl_constraint_set_constant(c, v);
  isl_int_clear(v);

  isl_basic_set *bset = isl_basic_set_universe(isl_set_get_dim(stride));

  bset = isl_basic_set_add_constraint(bset, c);
  isl_set *strideZero = isl_set_from_basic_set(bset);

  return isl_set_is_equal(stride, strideZero);
}

void MemoryAccess::setNewAccessFunction(isl_map *newAccess) {
  newAccessRelation = newAccess;
}

//===----------------------------------------------------------------------===//
void ScopStmt::buildScattering(SmallVectorImpl<unsigned> &Scatter) {
  unsigned NumberOfIterators = getNumIterators();
  unsigned ScatDim = Parent.getMaxLoopDepth() * 2 + 1;
  isl_dim *dim = isl_dim_alloc(Parent.getCtx(), Parent.getNumParams(),
                               NumberOfIterators, ScatDim);
  dim = isl_dim_set_tuple_name(dim, isl_dim_out, "scattering");
  dim = isl_dim_set_tuple_name(dim, isl_dim_in, getBaseName());
  isl_basic_map *bmap = isl_basic_map_universe(isl_dim_copy(dim));
  isl_int v;
  isl_int_init(v);

  // Loop dimensions.
  for (unsigned i = 0; i < NumberOfIterators; ++i) {
    isl_constraint *c = isl_equality_alloc(isl_dim_copy(dim));
    isl_int_set_si(v, 1);
    isl_constraint_set_coefficient(c, isl_dim_out, 2 * i + 1, v);
    isl_int_set_si(v, -1);
    isl_constraint_set_coefficient(c, isl_dim_in, i, v);

    bmap = isl_basic_map_add_constraint(bmap, c);
  }

  // Constant dimensions
  for (unsigned i = 0; i < NumberOfIterators + 1; ++i) {
    isl_constraint *c = isl_equality_alloc(isl_dim_copy(dim));
    isl_int_set_si(v, -1);
    isl_constraint_set_coefficient(c, isl_dim_out, 2 * i, v);
    isl_int_set_si(v, Scatter[i]);
    isl_constraint_set_constant(c, v);

    bmap = isl_basic_map_add_constraint(bmap, c);
  }

  // Fill scattering dimensions.
  for (unsigned i = 2 * NumberOfIterators + 1; i < ScatDim ; ++i) {
    isl_constraint *c = isl_equality_alloc(isl_dim_copy(dim));
    isl_int_set_si(v, 1);
    isl_constraint_set_coefficient(c, isl_dim_out, i, v);
    isl_int_set_si(v, 0);
    isl_constraint_set_constant(c, v);

    bmap = isl_basic_map_add_constraint(bmap, c);
  }

  isl_int_clear(v);
  isl_dim_free(dim);
  Scattering = isl_map_from_basic_map(bmap);
}

void ScopStmt::buildAccesses(TempScop &tempScop, const Region &CurRegion) {
  const AccFuncSetType *AccFuncs = tempScop.getAccessFunctions(BB);

  for (AccFuncSetType::const_iterator I = AccFuncs->begin(),
       E = AccFuncs->end(); I != E; ++I) {
    MemAccs.push_back(new MemoryAccess(I->first, this));
    InstructionToAccess[I->second] = MemAccs.back();
  }
}

isl_set *ScopStmt::toConditionSet(const Comparison &Comp, isl_dim *dim) const {
  isl_pw_aff *LHS = SCEVAffinator::getPwAff(this, Comp.getLHS()->OriginalSCEV,
                                            0);
  isl_pw_aff *RHS = SCEVAffinator::getPwAff(this, Comp.getRHS()->OriginalSCEV,
                                            0);

  isl_set *set;

  switch (Comp.getPred()) {
  case ICmpInst::ICMP_EQ:
    set = isl_pw_aff_eq_set(LHS, RHS);
    break;
  case ICmpInst::ICMP_NE:
    set = isl_pw_aff_ne_set(LHS, RHS);
    break;
  case ICmpInst::ICMP_SLT:
    set = isl_pw_aff_lt_set(LHS, RHS);
    break;
  case ICmpInst::ICMP_SLE:
    set = isl_pw_aff_le_set(LHS, RHS);
    break;
  case ICmpInst::ICMP_SGT:
    set = isl_pw_aff_gt_set(LHS, RHS);
    break;
  case ICmpInst::ICMP_SGE:
    set = isl_pw_aff_ge_set(LHS, RHS);
    break;
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_UGE:
    llvm_unreachable("Unsigned comparisons not yet supported");
  default:
    llvm_unreachable("Non integer predicate not supported");
  }

  set = isl_set_set_tuple_name(set, isl_dim_get_tuple_name(dim, isl_dim_set));

  return set;
}

isl_set *ScopStmt::toUpperLoopBound(const SCEVAffFunc &UpperBound, isl_dim *Dim,
				    unsigned BoundedDimension) const {
  // FIXME: We should choose a consistent scheme of when to name the dimensions.
  isl_dim *UnnamedDim = isl_dim_copy(Dim);
  UnnamedDim = isl_dim_set_tuple_name(UnnamedDim, isl_dim_set, 0);
  isl_local_space *LocalSpace = isl_local_space_from_dim (UnnamedDim);
  isl_aff *LAff = isl_aff_set_coefficient_si (isl_aff_zero (LocalSpace),
                                              isl_dim_set, BoundedDimension, 1);
  isl_pw_aff *BoundedDim = isl_pw_aff_from_aff(LAff);
  isl_pw_aff *Bound = SCEVAffinator::getPwAff(this, UpperBound.OriginalSCEV, 0);
  isl_set *set = isl_pw_aff_le_set(BoundedDim, Bound);
  set = isl_set_set_tuple_name(set, isl_dim_get_tuple_name(Dim, isl_dim_set));
  return set;
}

void ScopStmt::buildIterationDomainFromLoops(TempScop &tempScop) {
  isl_dim *dim = isl_dim_set_alloc(Parent.getCtx(), 0,
                                   getNumIterators());
  dim = isl_dim_set_tuple_name(dim, isl_dim_set, getBaseName());

  Domain = isl_set_universe(isl_dim_copy(dim));
  Domain = isl_set_align_params(Domain, isl_set_get_dim(Parent.getContext()));

  isl_int v;
  isl_int_init(v);

  for (int i = 0, e = getNumIterators(); i != e; ++i) {
    // Lower bound: IV >= 0.
    isl_basic_set *bset = isl_basic_set_universe(isl_dim_copy(dim));
    isl_constraint *c = isl_inequality_alloc(isl_dim_copy(dim));
    isl_int_set_si(v, 1);
    isl_constraint_set_coefficient(c, isl_dim_set, i, v);
    bset = isl_basic_set_add_constraint(bset, c);
    Domain = isl_set_intersect(Domain, isl_set_from_basic_set(bset));

    // Upper bound: IV <= NumberOfIterations.
    const Loop *L = getLoopForDimension(i);
    const SCEVAffFunc &UpperBound = tempScop.getLoopBound(L);
    isl_set *UpperBoundSet = toUpperLoopBound(UpperBound, isl_dim_copy(dim), i);
    Domain = isl_set_intersect(Domain, UpperBoundSet);
  }

  isl_int_clear(v);
}

void ScopStmt::addConditionsToDomain(TempScop &tempScop,
                                     const Region &CurRegion) {
  isl_dim *dim = isl_set_get_dim(Domain);
  const Region *TopR = tempScop.getMaxRegion().getParent(),
               *CurR = &CurRegion;
  const BasicBlock *CurEntry = BB;

  // Build BB condition constrains, by traveling up the region tree.
  do {
    assert(CurR && "We exceed the top region?");
    // Skip when multiple regions share the same entry.
    if (CurEntry != CurR->getEntry()) {
      if (const BBCond *Cnd = tempScop.getBBCond(CurEntry))
        for (BBCond::const_iterator I = Cnd->begin(), E = Cnd->end();
             I != E; ++I) {
          isl_set *c = toConditionSet(*I, dim);
          Domain = isl_set_intersect(Domain, c);
        }
    }
    CurEntry = CurR->getEntry();
    CurR = CurR->getParent();
  } while (TopR != CurR);

  isl_dim_free(dim);
}

void ScopStmt::buildIterationDomain(TempScop &tempScop, const Region &CurRegion)
{
  buildIterationDomainFromLoops(tempScop);
  addConditionsToDomain(tempScop, CurRegion);
}

ScopStmt::ScopStmt(Scop &parent, TempScop &tempScop,
                   const Region &CurRegion, BasicBlock &bb,
                   SmallVectorImpl<Loop*> &NestLoops,
                   SmallVectorImpl<unsigned> &Scatter)
  : Parent(parent), BB(&bb), IVS(NestLoops.size()) {
  // Setup the induction variables.
  for (unsigned i = 0, e = NestLoops.size(); i < e; ++i) {
    PHINode *PN = NestLoops[i]->getCanonicalInductionVariable();
    assert(PN && "Non canonical IV in Scop!");
    IVS[i] = std::make_pair(PN, NestLoops[i]);
  }

  raw_string_ostream OS(BaseName);
  WriteAsOperand(OS, &bb, false);
  BaseName = OS.str();

  // Remove the % in the name. This is not supported by isl.
  BaseName.erase(0, 1);
  makeIslCompatible(BaseName);
  BaseName = "Stmt_" + BaseName;

  buildIterationDomain(tempScop, CurRegion);
  buildScattering(Scatter);
  buildAccesses(tempScop, CurRegion);

  IsReduction = tempScop.is_Reduction(*BB);

  // FIXME: Temporarily remove dimension ids.
  Scattering = map_remove_dim_ids(Scattering);
  Domain = set_remove_dim_ids(Domain);
}

ScopStmt::ScopStmt(Scop &parent, SmallVectorImpl<unsigned> &Scatter)
  : Parent(parent), BB(NULL), IVS(0) {

  BaseName = "FinalRead";

  // Build iteration domain.
  std::string IterationDomainString = "{[i0] : i0 = 0}";
  Domain = isl_set_read_from_str(Parent.getCtx(), IterationDomainString.c_str(),
                                 -1);
  Domain = isl_set_add_dims(Domain, isl_dim_param, Parent.getNumParams());
  Domain = isl_set_set_tuple_name(Domain, getBaseName());

  // Build scattering.
  unsigned ScatDim = Parent.getMaxLoopDepth() * 2 + 1;
  isl_dim *dim = isl_dim_alloc(Parent.getCtx(), Parent.getNumParams(), 1,
                               ScatDim);
  dim = isl_dim_set_tuple_name(dim, isl_dim_out, "scattering");
  dim = isl_dim_set_tuple_name(dim, isl_dim_in, getBaseName());
  isl_basic_map *bmap = isl_basic_map_universe(isl_dim_copy(dim));
  isl_int v;
  isl_int_init(v);

  isl_constraint *c = isl_equality_alloc(dim);
  isl_int_set_si(v, -1);
  isl_constraint_set_coefficient(c, isl_dim_out, 0, v);

  // TODO: This is incorrect. We should not use a very large number to ensure
  // that this statement is executed last.
  isl_int_set_si(v, 200000000);
  isl_constraint_set_constant(c, v);

  bmap = isl_basic_map_add_constraint(bmap, c);
  isl_int_clear(v);
  Scattering = isl_map_from_basic_map(bmap);

  // Build memory accesses, use SetVector to keep the order of memory accesses
  // and prevent the same memory access inserted more than once.
  SetVector<const Value*> BaseAddressSet;

  for (Scop::const_iterator SI = Parent.begin(), SE = Parent.end(); SI != SE;
       ++SI) {
    ScopStmt *Stmt = *SI;

    for (MemoryAccessVec::const_iterator I = Stmt->memacc_begin(),
         E = Stmt->memacc_end(); I != E; ++I)
      BaseAddressSet.insert((*I)->getBaseAddr());
  }

  for (SetVector<const Value*>::iterator BI = BaseAddressSet.begin(),
       BE = BaseAddressSet.end(); BI != BE; ++BI)
    MemAccs.push_back(new MemoryAccess(*BI, this));

  IsReduction = false;
}

std::string ScopStmt::getDomainStr() const {
  isl_set *domain = getDomain();
  std::string string = stringFromIslObj(domain);
  isl_set_free(domain);
  return string;
}

std::string ScopStmt::getScatteringStr() const {
  return stringFromIslObj(getScattering());
}

unsigned ScopStmt::getNumParams() const {
  return Parent.getNumParams();
}

unsigned ScopStmt::getNumIterators() const {
  // The final read has one dimension with one element.
  if (!BB)
    return 1;

  return IVS.size();
}

unsigned ScopStmt::getNumScattering() const {
  return isl_map_dim(Scattering, isl_dim_out);
}

const char *ScopStmt::getBaseName() const { return BaseName.c_str(); }

const PHINode *ScopStmt::getInductionVariableForDimension(unsigned Dimension)
  const {
  return IVS[Dimension].first;
}

const Loop *ScopStmt::getLoopForDimension(unsigned Dimension) const {
  return IVS[Dimension].second;
}

const SCEVAddRecExpr *ScopStmt::getSCEVForDimension(unsigned Dimension)
  const {
  PHINode *PN =
    const_cast<PHINode*>(getInductionVariableForDimension(Dimension));
  return cast<SCEVAddRecExpr>(getParent()->getSE()->getSCEV(PN));
}

isl_ctx *ScopStmt::getIslContext() {
  return Parent.getCtx();
}

isl_set *ScopStmt::getDomain() const {
  return isl_set_copy(Domain);
}

ScopStmt::~ScopStmt() {
  while (!MemAccs.empty()) {
    delete MemAccs.back();
    MemAccs.pop_back();
  }

  isl_set_free(Domain);
  isl_map_free(Scattering);
}

void ScopStmt::print(raw_ostream &OS) const {
  OS << "\t" << getBaseName() << "\n";

  OS.indent(12) << "Domain :=\n";

  if (Domain) {
    OS.indent(16) << getDomainStr() << ";\n";
  } else
    OS.indent(16) << "n/a\n";

  OS.indent(12) << "Scattering :=\n";

  if (Domain) {
    OS.indent(16) << getScatteringStr() << ";\n";
  } else
    OS.indent(16) << "n/a\n";

  for (MemoryAccessVec::const_iterator I = MemAccs.begin(), E = MemAccs.end();
      I != E; ++I)
    (*I)->print(OS);
}

void ScopStmt::dump() const { print(dbgs()); }

//===----------------------------------------------------------------------===//
/// Scop class implement
Scop::Scop(TempScop &tempScop, LoopInfo &LI, ScalarEvolution &ScalarEvolution)
           : SE(&ScalarEvolution), R(tempScop.getMaxRegion()),
           MaxLoopDepth(tempScop.getMaxLoopDepth()) {
  isl_ctx *ctx = isl_ctx_alloc();

  ParamSetType &Params = tempScop.getParamSet();
  Parameters.insert(Parameters.begin(), Params.begin(), Params.end());

  isl_dim *dim = isl_dim_set_alloc(ctx, getNumParams(), 0);

  int i = 0;
  for (ParamSetType::iterator PI = Params.begin(), PE = Params.end();
       PI != PE; ++PI) {
    const SCEV *scev = *PI;
    isl_id *id = isl_id_alloc(ctx,
                              ("p" + convertInt(i)).c_str(),
                              (void *) scev);
    dim = isl_dim_set_dim_id(dim, isl_dim_param, i, id);
    i++;
  }

  // TODO: Insert relations between parameters.
  // TODO: Insert constraints on parameters.
  Context = isl_set_universe (dim);

  SmallVector<Loop*, 8> NestLoops;
  SmallVector<unsigned, 8> Scatter;

  Scatter.assign(MaxLoopDepth + 1, 0);

  // Build the iteration domain, access functions and scattering functions
  // traversing the region tree.
  buildScop(tempScop, getRegion(), NestLoops, Scatter, LI);
  Stmts.push_back(new ScopStmt(*this, Scatter));

  // FIXME: Temporarily remove dimension ids
  Context = set_remove_dim_ids(Context);

  assert(NestLoops.empty() && "NestLoops not empty at top level!");
}

Scop::~Scop() {
  isl_set_free(Context);

  // Free the statements;
  for (iterator I = begin(), E = end(); I != E; ++I)
    delete *I;

  // Do we need a singleton to manage this?
  //isl_ctx_free(ctx);
}

std::string Scop::getContextStr() const {
    return stringFromIslObj(getContext());
}

std::string Scop::getNameStr() const {
  std::string ExitName, EntryName;
  raw_string_ostream ExitStr(ExitName);
  raw_string_ostream EntryStr(EntryName);

  WriteAsOperand(EntryStr, R.getEntry(), false);
  EntryStr.str();

  if (R.getExit()) {
    WriteAsOperand(ExitStr, R.getExit(), false);
    ExitStr.str();
  } else
    ExitName = "FunctionExit";

  return EntryName + "---" + ExitName;
}

void Scop::printContext(raw_ostream &OS) const {
  OS << "Context:\n";

  if (!Context) {
    OS.indent(4) << "n/a\n\n";
    return;
  }

  OS.indent(4) << getContextStr() << "\n";
}

void Scop::printStatements(raw_ostream &OS) const {
  OS << "Statements {\n";

  for (const_iterator SI = begin(), SE = end();SI != SE; ++SI)
    OS.indent(4) << (**SI);

  OS.indent(4) << "}\n";
}


void Scop::print(raw_ostream &OS) const {
  printContext(OS.indent(4));
  printStatements(OS.indent(4));
}

void Scop::dump() const { print(dbgs()); }

isl_ctx *Scop::getCtx() const { return isl_set_get_ctx(Context); }

ScalarEvolution *Scop::getSE() const { return SE; }

bool Scop::isTrivialBB(BasicBlock *BB, TempScop &tempScop) {
  if (tempScop.getAccessFunctions(BB))
    return false;

  return true;
}

void Scop::buildScop(TempScop &tempScop,
                      const Region &CurRegion,
                      SmallVectorImpl<Loop*> &NestLoops,
                      SmallVectorImpl<unsigned> &Scatter,
                      LoopInfo &LI) {
  Loop *L = castToLoop(CurRegion, LI);

  if (L)
    NestLoops.push_back(L);

  unsigned loopDepth = NestLoops.size();
  assert(Scatter.size() > loopDepth && "Scatter not big enough!");

  for (Region::const_element_iterator I = CurRegion.element_begin(),
       E = CurRegion.element_end(); I != E; ++I)
    if (I->isSubRegion())
      buildScop(tempScop, *(I->getNodeAs<Region>()), NestLoops, Scatter, LI);
    else {
      BasicBlock *BB = I->getNodeAs<BasicBlock>();

      if (isTrivialBB(BB, tempScop))
        continue;

      Stmts.push_back(new ScopStmt(*this, tempScop, CurRegion, *BB, NestLoops,
                                   Scatter));

      // Increasing the Scattering function is OK for the moment, because
      // we are using a depth first iterator and the program is well structured.
      ++Scatter[loopDepth];
    }

  if (!L)
    return;

  // Exiting a loop region.
  Scatter[loopDepth] = 0;
  NestLoops.pop_back();
  ++Scatter[loopDepth-1];
}

//===----------------------------------------------------------------------===//

void ScopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfo>();
  AU.addRequired<RegionInfo>();
  AU.addRequired<ScalarEvolution>();
  AU.addRequired<TempScopInfo>();
  AU.setPreservesAll();
}

bool ScopInfo::runOnRegion(Region *R, RGPassManager &RGM) {
  LoopInfo &LI = getAnalysis<LoopInfo>();
  ScalarEvolution &SE = getAnalysis<ScalarEvolution>();

  TempScop *tempScop = getAnalysis<TempScopInfo>().getTempScop(R);

  // This region is no Scop.
  if (!tempScop) {
    scop = 0;
    return false;
  }

  // Statistics.
  ++ScopFound;
  if (tempScop->getMaxLoopDepth() > 0) ++RichScopFound;

  scop = new Scop(*tempScop, LI, SE);

  return false;
}

char ScopInfo::ID = 0;


static RegisterPass<ScopInfo>
X("polly-scops", "Polly - Create polyhedral description of Scops");

Pass *polly::createScopInfoPass() {
  return new ScopInfo();
}

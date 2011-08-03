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
#include <sstream>
#include <string>
#include <vector>

using namespace llvm;
using namespace polly;

STATISTIC(ScopFound,  "Number of valid Scops");
STATISTIC(RichScopFound,   "Number of Scops containing a loop");


//===----------------------------------------------------------------------===//
static void setCoefficient(const SCEV *Coeff, mpz_t v, bool negative,
                           bool isSigned = true) {
  if (Coeff) {
    const SCEVConstant *C = dyn_cast<SCEVConstant>(Coeff);
    const APInt &CI = C->getValue()->getValue();
    MPZ_from_APInt(v, negative ? (-CI) : CI, isSigned);
  } else
    isl_int_set_si(v, 0);
}

static isl_map *getValueOf(const SCEVAffFunc &AffFunc,
                           const ScopStmt *Statement, isl_dim *dim) {

  const SmallVectorImpl<const SCEV*> &Params =
    Statement->getParent()->getParams();
  unsigned num_in = Statement->getNumIterators(), num_param = Params.size();

  const char *dimname = isl_dim_get_tuple_name(dim, isl_dim_set);
  dim = isl_dim_alloc(isl_dim_get_ctx(dim), num_param,
                      isl_dim_size(dim, isl_dim_set), 1);
  dim = isl_dim_set_tuple_name(dim, isl_dim_in, dimname);

  assert((AffFunc.getType() == SCEVAffFunc::Eq
          || AffFunc.getType() == SCEVAffFunc::ReadMem
          || AffFunc.getType() == SCEVAffFunc::WriteMem)
          && "AffFunc is not an equality");

  isl_constraint *c = isl_equality_alloc(isl_dim_copy(dim));

  isl_int v;
  isl_int_init(v);

  // Set single output dimension.
  isl_int_set_si(v, -1);
  isl_constraint_set_coefficient(c, isl_dim_out, 0, v);

  // Set the coefficient for induction variables.
  for (unsigned i = 0, e = num_in; i != e; ++i) {
    setCoefficient(AffFunc.getCoeff(Statement->getSCEVForDimension(i)), v,
                   false, AffFunc.isSigned());
    isl_constraint_set_coefficient(c, isl_dim_in, i, v);
  }

  // Set the coefficient of parameters
  for (unsigned i = 0, e = num_param; i != e; ++i) {
    setCoefficient(AffFunc.getCoeff(Params[i]), v, false, AffFunc.isSigned());
    isl_constraint_set_coefficient(c, isl_dim_param, i, v);
  }

  // Set the constant.
  setCoefficient(AffFunc.getTransComp(), v, false, AffFunc.isSigned());
  isl_constraint_set_constant(c, v);
  isl_int_clear(v);

  isl_basic_map *BasicMap = isl_basic_map_universe(isl_dim_copy(dim));
  BasicMap = isl_basic_map_add_constraint(BasicMap, c);
  return isl_map_from_basic_map(BasicMap);
}
//===----------------------------------------------------------------------===//

MemoryAccess::~MemoryAccess() {
  isl_map_free(getAccessFunction());
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

  isl_dim *dim = isl_dim_set_alloc(Statement->getIslContext(),
                                   Statement->getNumParams(),
                                   Statement->getNumIterators());
  dim = isl_dim_set_tuple_name(dim, isl_dim_set, Statement->getBaseName());

  AccessRelation = getValueOf(AffFunc, Statement, dim);

  // Devide the access function by the size of the elements in the function.
  isl_dim *dim2 = isl_dim_alloc(Statement->getIslContext(),
                                    Statement->getNumParams(), 1, 1);
  isl_basic_map *bmap = isl_basic_map_universe(isl_dim_copy(dim2));
  isl_constraint *c = isl_equality_alloc(dim2);
  isl_int v;
  isl_int_init(v);
  isl_int_set_si(v, -1);
  isl_constraint_set_coefficient(c, isl_dim_in, 0, v);
  isl_int_set_si(v, AffFunc.getElemSizeInBytes());
  isl_constraint_set_coefficient(c, isl_dim_out, 0, v);

  bmap = isl_basic_map_add_constraint(bmap, c);
  isl_map* dataSizeMap = isl_map_from_basic_map(bmap);

  AccessRelation = isl_map_apply_range(AccessRelation, dataSizeMap);

  AccessRelation = isl_map_set_tuple_name(AccessRelation, isl_dim_out,
                                          getBaseName().c_str());
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

static isl_map *MapValueToLHS(isl_ctx *Context, unsigned ParameterNumber) {
  std::string MapString;
  isl_map *Map;

  MapString = "{[i0] -> [i0, o1]}";
  Map = isl_map_read_from_str(Context, MapString.c_str(), -1);
  return isl_map_add_dims(Map, isl_dim_param, ParameterNumber);
}

static isl_map *MapValueToRHS(isl_ctx *Context, unsigned ParameterNumber) {
  std::string MapString;
  isl_map *Map;

  MapString = "{[i0] -> [o0, i0]}";
  Map = isl_map_read_from_str(Context, MapString.c_str(), -1);
  return isl_map_add_dims(Map, isl_dim_param, ParameterNumber);
}

static isl_set *getComparison(isl_ctx *Context, const ICmpInst::Predicate Pred,
                              unsigned ParameterNumber) {
  std::string SetString;

  switch (Pred) {
  case ICmpInst::ICMP_EQ:
    SetString = "{[i0, i1] : i0 = i1}";
    break;
  case ICmpInst::ICMP_NE:
    SetString = "{[i0, i1] : i0 + 1 <= i1; [i0, i1] : i0 - 1 >= i1}";
    break;
  case ICmpInst::ICMP_SLT:
    SetString = "{[i0, i1] : i0 + 1 <= i1}";
    break;
  case ICmpInst::ICMP_ULT:
    SetString = "{[i0, i1] : i0 + 1 <= i1}";
    break;
  case ICmpInst::ICMP_SGT:
    SetString = "{[i0, i1] : i0 >= i1 + 1}";
    break;
  case ICmpInst::ICMP_UGT:
    SetString = "{[i0, i1] : i0 >= i1 + 1}";
    break;
  case ICmpInst::ICMP_SLE:
    SetString = "{[i0, i1] : i0 <= i1}";
    break;
  case ICmpInst::ICMP_ULE:
    SetString = "{[i0, i1] : i0 <= i1}";
    break;
  case ICmpInst::ICMP_SGE:
    SetString = "{[i0, i1] : i0 >= i1}";
    break;
  case ICmpInst::ICMP_UGE:
    SetString = "{[i0, i1] : i0 >= i1}";
    break;
  default:
    llvm_unreachable("Non integer predicate not supported");
  }

  isl_set *Set = isl_set_read_from_str(Context, SetString.c_str(), -1);
  return isl_set_add_dims(Set, isl_dim_param, ParameterNumber);
}

static isl_set *compareValues(isl_map *LeftValue, isl_map *RightValue,
                              const ICmpInst::Predicate Predicate) {
  isl_ctx *Context = isl_map_get_ctx(LeftValue);
  unsigned NumberOfParameters = isl_map_n_param(LeftValue);

  isl_map *MapToLHS = MapValueToLHS(Context, NumberOfParameters);
  isl_map *MapToRHS = MapValueToRHS(Context, NumberOfParameters);

  isl_map *LeftValueAtLHS = isl_map_apply_range(LeftValue, MapToLHS);
  isl_map *RightValueAtRHS = isl_map_apply_range(RightValue, MapToRHS);

  isl_map *BothValues = isl_map_intersect(LeftValueAtLHS, RightValueAtRHS);
  isl_set *Comparison = getComparison(Context, Predicate, NumberOfParameters);

  isl_map *ComparedValues = isl_map_intersect_range(BothValues, Comparison);
  return isl_map_domain(ComparedValues);
}

isl_set *ScopStmt::toConditionSet(const Comparison &Comp, isl_dim *dim) const {
  isl_map *LHSValue = getValueOf(*Comp.getLHS(), this, dim);
  isl_map *RHSValue = getValueOf(*Comp.getRHS(), this, dim);

  return compareValues(LHSValue, RHSValue, Comp.getPred());
}

isl_set *ScopStmt::toUpperLoopBound(const SCEVAffFunc &UpperBound, isl_dim *dim,
				    unsigned BoundedDimension) const {
  // Set output dimension to bounded dimension.
  isl_dim *RHSDim = isl_dim_alloc(Parent.getCtx(), getNumParams(),
                                  getNumIterators(), 1);
  RHSDim = isl_dim_set_tuple_name(RHSDim, isl_dim_in, getBaseName());
  isl_constraint *c = isl_equality_alloc(isl_dim_copy(RHSDim));
  isl_int v;
  isl_int_init(v);
  isl_int_set_si(v, 1);
  isl_constraint_set_coefficient(c, isl_dim_in, BoundedDimension, v);
  isl_int_set_si(v, -1);
  isl_constraint_set_coefficient(c, isl_dim_out, 0, v);
  isl_int_clear(v);
  isl_basic_map *bmap = isl_basic_map_universe(RHSDim);
  bmap = isl_basic_map_add_constraint(bmap, c);

  isl_map *LHSValue = isl_map_from_basic_map(bmap);

  isl_map *RHSValue = getValueOf(UpperBound, this, dim);

  return compareValues(LHSValue, RHSValue, ICmpInst::ICMP_SLE);
}

void ScopStmt::buildIterationDomainFromLoops(TempScop &tempScop) {
  isl_dim *dim = isl_dim_set_alloc(Parent.getCtx(), getNumParams(),
                                   getNumIterators());
  dim = isl_dim_set_tuple_name(dim, isl_dim_set, getBaseName());

  Domain = isl_set_universe(isl_dim_copy(dim));

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

//===------ CodeGeneration.cpp - Code generate the Scops. -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The CodeGeneration pass takes a Scop created by ScopInfo and translates it
// back to LLVM-IR using Cloog.
//
// The Scop describes the high level memory behaviour of a control flow region.
// Transformation passes can update the schedule (execution order) of statements
// in the Scop. Cloog is used to generate an abstract syntax tree (clast) that
// reflects the updated execution order. This clast is used to create new
// LLVM-IR that is computational equivalent to the original control flow region,
// but executes its code in the new execution order defined by the changed
// scattering.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "polly-codegen"

#include "polly/Cloog.h"
#include "polly/CodeGeneration.h"
#include "polly/Dependences.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"
#include "polly/TempScopInfo.h"
#include "polly/Support/GICHelper.h"

#include "llvm/Module.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define CLOOG_INT_GMP 1
#include "cloog/cloog.h"
#include "cloog/isl/cloog.h"

#include "isl/aff.h"

#include <vector>
#include <utility>

using namespace polly;
using namespace llvm;

struct isl_set;

namespace polly {

bool EnablePollyVector;

static cl::opt<bool, true>
Vector("enable-polly-vector",
       cl::desc("Enable polly vector code generation"), cl::Hidden,
       cl::location(EnablePollyVector), cl::init(false));

static cl::opt<bool>
OpenMP("enable-polly-openmp",
       cl::desc("Generate OpenMP parallel code"), cl::Hidden,
       cl::value_desc("OpenMP code generation enabled if true"),
       cl::init(false));

static cl::opt<bool>
AtLeastOnce("enable-polly-atLeastOnce",
       cl::desc("Give polly the hint, that every loop is executed at least"
                "once"), cl::Hidden,
       cl::value_desc("OpenMP code generation enabled if true"),
       cl::init(false));

static cl::opt<bool>
Aligned("enable-polly-aligned",
       cl::desc("Assumed aligned memory accesses."), cl::Hidden,
       cl::value_desc("OpenMP code generation enabled if true"),
       cl::init(false));

typedef DenseMap<const Value*, Value*> ValueMapT;
typedef DenseMap<const char*, Value*> CharMapT;
typedef std::vector<ValueMapT> VectorValueMapT;
typedef struct {
  Value *Result;
  IRBuilder<> *Builder;
}IslPwAffUserInfo;

// Create a new loop.
//
// @param Builder The builder used to create the loop.  It also defines the
//                place where to create the loop.
// @param UB      The upper bound of the loop iv.
// @param Stride  The number by which the loop iv is incremented after every
//                iteration.
static Value *createLoop(IRBuilder<> *Builder, Value *LB, Value *UB,
                         APInt Stride, DominatorTree *DT, Pass *P,
                         BasicBlock **AfterBlock) {
  Function *F = Builder->GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  BasicBlock *PreheaderBB = Builder->GetInsertBlock();
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.loop_header", F);
  BasicBlock *BodyBB = BasicBlock::Create(Context, "polly.loop_body", F);
  BasicBlock *AfterBB = SplitBlock(PreheaderBB, Builder->GetInsertPoint()++, P);
  AfterBB->setName("polly.loop_after");

  PreheaderBB->getTerminator()->setSuccessor(0, HeaderBB);
  DT->addNewBlock(HeaderBB, PreheaderBB);

  Builder->SetInsertPoint(HeaderBB);

  // Use the type of upper and lower bound.
  assert(LB->getType() == UB->getType()
         && "Different types for upper and lower bound.");

  IntegerType *LoopIVType = dyn_cast<IntegerType>(UB->getType());
  assert(LoopIVType && "UB is not integer?");

  // IV
  PHINode *IV = Builder->CreatePHI(LoopIVType, 2, "polly.loopiv");
  IV->addIncoming(LB, PreheaderBB);

  // IV increment.
  Value *StrideValue = ConstantInt::get(LoopIVType,
                                        Stride.zext(LoopIVType->getBitWidth()));
  Value *IncrementedIV = Builder->CreateAdd(IV, StrideValue,
                                            "polly.next_loopiv");

  // Exit condition.
  Value *CMP;
  if (AtLeastOnce) { // At least on iteration.
    UB = Builder->CreateAdd(UB, Builder->getInt64(1));
    CMP = Builder->CreateICmpNE(IV, UB);
  } else { // Maybe not executed at all.
    CMP = Builder->CreateICmpSLE(IV, UB);
  }

  Builder->CreateCondBr(CMP, BodyBB, AfterBB);
  DT->addNewBlock(BodyBB, HeaderBB);

  Builder->SetInsertPoint(BodyBB);
  Builder->CreateBr(HeaderBB);
  IV->addIncoming(IncrementedIV, BodyBB);
  DT->changeImmediateDominator(AfterBB, HeaderBB);

  Builder->SetInsertPoint(BodyBB->begin());
  *AfterBlock = AfterBB;

  return IV;
}

class IslGenerator;

class IslGenerator {
public:
  IslGenerator(IRBuilder<> &Builder) : Builder(Builder) {}
  Value *generateIslInt(__isl_take isl_int Int);
  Value *generateIslAff(__isl_take isl_aff *Aff);
  Value *generateIslPwAff(__isl_take isl_pw_aff *PwAff);

private:
  typedef struct {
    Value *Result;
    class IslGenerator *Generator;
  } IslGenInfo;

  IRBuilder<> &Builder;
  static int mergeIslAffValues(__isl_take isl_set *Set,
                               __isl_take isl_aff *Aff, void *User);
};

Value *IslGenerator::generateIslInt(isl_int Int) {
  mpz_t IntMPZ;
  mpz_init(IntMPZ);
  isl_int_get_gmp(Int, IntMPZ);
  Value *IntValue = Builder.getInt(APInt_from_MPZ(IntMPZ));
  mpz_clear(IntMPZ);
  return IntValue;
}

Value *IslGenerator::generateIslAff(__isl_take isl_aff *Aff) {
  assert(isl_aff_is_cst(Aff) && "Only constant access functions supported");
  Value *ConstValue;
  isl_int ConstIsl;

  isl_int_init(ConstIsl);
  isl_aff_get_constant(Aff, &ConstIsl);
  ConstValue = generateIslInt(ConstIsl);

  isl_int_clear(ConstIsl);
  isl_aff_free(Aff);

  return ConstValue;
}

int IslGenerator::mergeIslAffValues(__isl_take isl_set *Set,
                                    __isl_take isl_aff *Aff, void *User) {
  IslGenInfo *GenInfo = (IslGenInfo *)User;

  assert((GenInfo->Result == NULL) && "Result is already set."
         "Currently only single isl_aff is supported");
  assert(isl_set_plain_is_universe(Set)
         && "Code generation failed because the set is not universe");

  GenInfo->Result = GenInfo->Generator->generateIslAff(Aff);

  isl_set_free(Set);
  return 0;
}

Value *IslGenerator::generateIslPwAff(__isl_take isl_pw_aff *PwAff) {
  IslGenInfo User;
  User.Result = NULL;
  User.Generator = this;
  isl_pw_aff_foreach_piece(PwAff, mergeIslAffValues, &User);
  assert(User.Result && "Code generation for isl_pw_aff failed");

  isl_pw_aff_free(PwAff);
  return User.Result;
}

class BlockGenerator {
public:
  static void generate(IRBuilder<> &B, ValueMapT &ValueMap,
                       ScopStmt &Stmt, __isl_keep isl_set *Domain, Pass *P) {
    BlockGenerator Generator(B, ValueMap, Stmt, Domain, P);
    Generator.copyBB();
  }

protected:
  IRBuilder<> &Builder;
  ValueMapT &VMap;
  Scop &S;
  ScopStmt &Statement;
  isl_set *ScatteringDomain;
  Pass *P;

  BlockGenerator(IRBuilder<> &B, ValueMapT &vmap, ScopStmt &Stmt,
                 __isl_keep isl_set *Domain, Pass *p);

  Value *getOperand(const Value *oldOperand, ValueMapT &BBMap);

  void copyInstScalar(const Instruction *Inst, ValueMapT &BBMap);

  /// @brief Get the memory access offset to be added to the base address
  std::vector<Value*> getMemoryAccessIndex(__isl_keep isl_map *AccessRelation,
                                            Value *BaseAddress);

  /// @brief Get the new operand address according to the changed access in
  ///        JSCOP file.
  Value *getNewAccessOperand(__isl_keep isl_map *NewAccessRelation,
                             Value *BaseAddress, const Value *OldOperand,
                             ValueMapT &BBMap);

  /// @brief Generate the operand address
  Value *generateLocationAccessed(const Instruction *Inst,
                                  const Value *Pointer, ValueMapT &BBMap);

  Value *generateScalarLoad(const LoadInst *load, ValueMapT &BBMap);

  void copyInstruction(const Instruction *Inst, ValueMapT &ScalarMap);

  void copyBB();
};

BlockGenerator::BlockGenerator(IRBuilder<> &B, ValueMapT &vmap, ScopStmt &Stmt,
                               __isl_keep isl_set *domain, Pass *P) :
  Builder(B), VMap(vmap), S(*Stmt.getParent()), Statement(Stmt),
  ScatteringDomain(domain), P(P) {}

Value *BlockGenerator::getOperand(const Value *OldOperand, ValueMapT &BBMap) {
  const Instruction *OpInst = dyn_cast<Instruction>(OldOperand);

  if (!OpInst)
    return const_cast<Value*>(OldOperand);

  // IVS and Parameters.
  if (VMap.count(OldOperand)) {
    Value *NewOperand = VMap[OldOperand];

    // Insert a cast if types are different
    if (OldOperand->getType()->getScalarSizeInBits()
        < NewOperand->getType()->getScalarSizeInBits())
      NewOperand = Builder.CreateTruncOrBitCast(NewOperand,
                                                OldOperand->getType());

    return NewOperand;
  }

  // Instructions calculated in the current BB.
  if (BBMap.count(OldOperand)) {
    return BBMap[OldOperand];
  }

  // Ignore instructions that are referencing ops in the old BB. These
  // instructions are unused. They where replace by new ones during
  // createIndependentBlocks().
  if (S.getRegion().contains(OpInst->getParent()))
    return NULL;

  return const_cast<Value*>(OldOperand);
}

void BlockGenerator::copyInstScalar(const Instruction *Inst, ValueMapT &BBMap) {
  Instruction *NewInst = Inst->clone();

  // Replace old operands with the new ones.
  for (Instruction::const_op_iterator OI = Inst->op_begin(),
       OE = Inst->op_end(); OI != OE; ++OI) {
    Value *OldOperand = *OI;
    Value *NewOperand = getOperand(OldOperand, BBMap);

    if (!NewOperand) {
      assert(!isa<StoreInst>(NewInst)
             && "Store instructions are always needed!");
      delete NewInst;
      return;
    }

    NewInst->replaceUsesOfWith(OldOperand, NewOperand);
  }

  Builder.Insert(NewInst);
  BBMap[Inst] = NewInst;

  if (!NewInst->getType()->isVoidTy())
    NewInst->setName("p_" + Inst->getName());
}

std::vector <Value*> BlockGenerator::getMemoryAccessIndex(
  __isl_keep isl_map *AccessRelation, Value *BaseAddress) {
  assert((isl_map_dim(AccessRelation, isl_dim_out) == 1)
         && "Only single dimensional access functions supported");

  isl_pw_aff *PwAff = isl_map_dim_max(isl_map_copy(AccessRelation), 0);
  IslGenerator IslGen(Builder);
  Value *OffsetValue = IslGen.generateIslPwAff(PwAff);

  PointerType *BaseAddressType = dyn_cast<PointerType>(
    BaseAddress->getType());
  Type *ArrayTy = BaseAddressType->getElementType();
  Type *ArrayElementType = dyn_cast<ArrayType>(ArrayTy)->getElementType();
  OffsetValue = Builder.CreateSExtOrBitCast(OffsetValue, ArrayElementType);

  std::vector<Value*> IndexArray;
  Value *NullValue = Constant::getNullValue(ArrayElementType);
  IndexArray.push_back(NullValue);
  IndexArray.push_back(OffsetValue);
  return IndexArray;
}

Value *BlockGenerator::getNewAccessOperand(
  __isl_keep isl_map *NewAccessRelation, Value *BaseAddress, const Value
  *OldOperand, ValueMapT &BBMap) {
  std::vector<Value*> IndexArray = getMemoryAccessIndex(NewAccessRelation,
                                                        BaseAddress);
  Value *NewOperand = Builder.CreateGEP(BaseAddress, IndexArray,
                                        "p_newarrayidx_");
  return NewOperand;
}

Value *BlockGenerator::generateLocationAccessed(const Instruction *Inst,
                                                const Value *Pointer,
                                                ValueMapT &BBMap ) {
  MemoryAccess &Access = Statement.getAccessFor(Inst);
  isl_map *CurrentAccessRelation = Access.getAccessRelation();
  isl_map *NewAccessRelation = Access.getNewAccessRelation();

  assert(isl_map_has_equal_space(CurrentAccessRelation, NewAccessRelation)
         && "Current and new access function use different spaces");

  Value *NewPointer;

  if (!NewAccessRelation) {
    NewPointer = getOperand(Pointer, BBMap);
  } else {
    Value *BaseAddress = const_cast<Value*>(Access.getBaseAddr());
    NewPointer = getNewAccessOperand(NewAccessRelation, BaseAddress, Pointer,
                                     BBMap);
  }

  isl_map_free(CurrentAccessRelation);
  isl_map_free(NewAccessRelation);
  return NewPointer;
}

Value *BlockGenerator::generateScalarLoad(const LoadInst *Load,
                                          ValueMapT &BBMap) {
  const Value *Pointer = Load->getPointerOperand();
  const Instruction *Inst = dyn_cast<Instruction>(Load);
  Value *NewPointer = generateLocationAccessed(Inst, Pointer, BBMap);
  Value *ScalarLoad = Builder.CreateLoad(NewPointer,
                                         Load->getName() + "_p_scalar_");
  return ScalarLoad;
}

void BlockGenerator::copyInstruction(const Instruction *Inst,
                                     ValueMapT &ScalarMap) {
  // Terminator instructions control the control flow. They are explicitly
  // expressed in the clast and do not need to be copied.
  if (Inst->isTerminator())
    return;

  if (const LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
    ScalarMap[Load] = generateScalarLoad(Load, ScalarMap);
    return;
  }

  copyInstScalar(Inst, ScalarMap);
}


void BlockGenerator::copyBB() {
  BasicBlock *BB = Statement.getBasicBlock();
  BasicBlock *CopyBB = SplitBlock(Builder.GetInsertBlock(),
                                  Builder.GetInsertPoint(), P);
  CopyBB->setName("polly.stmt." + BB->getName());
  Builder.SetInsertPoint(CopyBB->begin());

  ValueMapT ScalarBlockMap;

  for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end(); II != IE;
       ++II)
      copyInstruction(II, ScalarBlockMap);
}

class VectorBlockGenerator : BlockGenerator {
public:
  static void generate(IRBuilder<> &B, ValueMapT &ValueMap,
                       VectorValueMapT &VectorMaps, ScopStmt &Stmt,
                       __isl_keep isl_set *Domain, Pass *P) {
    VectorBlockGenerator Generator(B, ValueMap, VectorMaps, Stmt, Domain, P);
    Generator.copyBB();
  }

private:
  VectorValueMapT &ValueMaps;

  VectorBlockGenerator(IRBuilder<> &B, ValueMapT &vmap, VectorValueMapT &vmaps,
                 ScopStmt &Stmt, __isl_keep isl_set *domain, Pass *p);

  int getVectorWidth();

  Value *makeVectorOperand(Value *Operand);

  Value *getOperand(const Value *oldOperand, ValueMapT &BBMap,
                    ValueMapT *VectorMap = 0);

  Type *getVectorPtrTy(const Value *V, int Width);

  /// @brief Load a vector from a set of adjacent scalars
  ///
  /// In case a set of scalars is known to be next to each other in memory,
  /// create a vector load that loads those scalars
  ///
  /// %vector_ptr= bitcast double* %p to <4 x double>*
  /// %vec_full = load <4 x double>* %vector_ptr
  ///
  Value *generateStrideOneLoad(const LoadInst *Load, ValueMapT &BBMap);

  /// @brief Load a vector initialized from a single scalar in memory
  ///
  /// In case all elements of a vector are initialized to the same
  /// scalar value, this value is loaded and shuffeled into all elements
  /// of the vector.
  ///
  /// %splat_one = load <1 x double>* %p
  /// %splat = shufflevector <1 x double> %splat_one, <1 x
  ///       double> %splat_one, <4 x i32> zeroinitializer
  ///
  Value *generateStrideZeroLoad(const LoadInst *Load, ValueMapT &BBMap);

  /// @Load a vector from scalars distributed in memory
  ///
  /// In case some scalars a distributed randomly in memory. Create a vector
  /// by loading each scalar and by inserting one after the other into the
  /// vector.
  ///
  /// %scalar_1= load double* %p_1
  /// %vec_1 = insertelement <2 x double> undef, double %scalar_1, i32 0
  /// %scalar 2 = load double* %p_2
  /// %vec_2 = insertelement <2 x double> %vec_1, double %scalar_1, i32 1
  ///
  Value *generateUnknownStrideLoad(const LoadInst *Load,
                                   VectorValueMapT &ScalarMaps);


  void generateLoad(const LoadInst *Load, ValueMapT &VectorMap,
                    VectorValueMapT &ScalarMaps);

  void copyUnaryInst(const UnaryInstruction *Inst, ValueMapT &BBMap,
                           ValueMapT &VectorMap);

  void copyBinaryInst(const BinaryOperator *Inst, ValueMapT &BBMap,
                        ValueMapT &VectorMap);

  void copyStore(const StoreInst *Store, ValueMapT &BBMap,
                       ValueMapT &VectorMap, VectorValueMapT &ScalarMaps);

  bool hasVectorOperands(const Instruction *Inst, ValueMapT &VectorMap);

  void copyInstruction(const Instruction *Inst, ValueMapT &VectorMap,
                       VectorValueMapT &ScalarMaps);

  // Insert a copy of a basic block in the newly generated code.
  void copyBB();
};

VectorBlockGenerator::VectorBlockGenerator(IRBuilder<> &B, ValueMapT &vmap,
                               VectorValueMapT &vmaps, ScopStmt &Stmt,
                               __isl_keep isl_set *domain, Pass *P)
    : BlockGenerator(B, vmap, Stmt, domain, P), ValueMaps(vmaps) {
    assert(ValueMaps.size() > 1 && "Only one vector lane found");
  }

Value *VectorBlockGenerator::makeVectorOperand(Value *Operand) {
  int VectorWidth = getVectorWidth();
  if (Operand->getType()->isVectorTy())
    return Operand;

  VectorType *VectorType = VectorType::get(Operand->getType(), VectorWidth);
  Value *Vector = UndefValue::get(VectorType);
  Vector = Builder.CreateInsertElement(Vector, Operand, Builder.getInt32(0));

  std::vector<Constant*> Splat;

  for (int i = 0; i < VectorWidth; i++)
    Splat.push_back (Builder.getInt32(0));

  Constant *SplatVector = ConstantVector::get(Splat);

  return Builder.CreateShuffleVector(Vector, Vector, SplatVector);
}

Value *VectorBlockGenerator::getOperand(const Value *OldOperand,
                                        ValueMapT &BBMap,
                                        ValueMapT *VectorMap) {
  const Instruction *OpInst = dyn_cast<Instruction>(OldOperand);

  if (!OpInst)
    return const_cast<Value*>(OldOperand);

  if (VectorMap && VectorMap->count(OldOperand))
    return (*VectorMap)[OldOperand];

  return BlockGenerator::getOperand(OldOperand, BBMap);
}

Type *VectorBlockGenerator::getVectorPtrTy(const Value *Val, int Width) {
  PointerType *PointerTy = dyn_cast<PointerType>(Val->getType());
  assert(PointerTy && "PointerType expected");

  Type *ScalarType = PointerTy->getElementType();
  VectorType *VectorType = VectorType::get(ScalarType, Width);

  return PointerType::getUnqual(VectorType);
}

Value *VectorBlockGenerator::generateStrideOneLoad(const LoadInst *Load,
                                                   ValueMapT &BBMap) {
  const Value *Pointer = Load->getPointerOperand();
  Type *VectorPtrType = getVectorPtrTy(Pointer, getVectorWidth());
  Value *NewPointer = getOperand(Pointer, BBMap);
  Value *VectorPtr = Builder.CreateBitCast(NewPointer, VectorPtrType,
                                           "vector_ptr");
  LoadInst *VecLoad = Builder.CreateLoad(VectorPtr,
                                         Load->getName() + "_p_vec_full");
  if (!Aligned)
    VecLoad->setAlignment(8);

  return VecLoad;
}

Value *VectorBlockGenerator::generateStrideZeroLoad(const LoadInst *Load,
                                                    ValueMapT &BBMap) {
  const Value *Pointer = Load->getPointerOperand();
  Type *VectorPtrType = getVectorPtrTy(Pointer, 1);
  Value *NewPointer = getOperand(Pointer, BBMap);
  Value *VectorPtr = Builder.CreateBitCast(NewPointer, VectorPtrType,
                                           Load->getName() + "_p_vec_p");
  LoadInst *ScalarLoad= Builder.CreateLoad(VectorPtr,
                                           Load->getName() + "_p_splat_one");

  if (!Aligned)
    ScalarLoad->setAlignment(8);

  Constant *SplatVector =
    Constant::getNullValue(VectorType::get(Builder.getInt32Ty(),
                                           getVectorWidth()));

  Value *VectorLoad = Builder.CreateShuffleVector(ScalarLoad, ScalarLoad,
                                                  SplatVector,
                                                  Load->getName()
                                                  + "_p_splat");
  return VectorLoad;
}

Value *VectorBlockGenerator::generateUnknownStrideLoad(const LoadInst *Load,
  VectorValueMapT &ScalarMaps) {
  int VectorWidth = getVectorWidth();
  const Value *Pointer = Load->getPointerOperand();
  VectorType *VectorType = VectorType::get(
    dyn_cast<PointerType>(Pointer->getType())->getElementType(), VectorWidth);

  Value *Vector = UndefValue::get(VectorType);

  for (int i = 0; i < VectorWidth; i++) {
    Value *NewPointer = getOperand(Pointer, ScalarMaps[i]);
    Value *ScalarLoad = Builder.CreateLoad(NewPointer,
                                           Load->getName() + "_p_scalar_");
    Vector = Builder.CreateInsertElement(Vector, ScalarLoad,
                                         Builder.getInt32(i),
                                         Load->getName() + "_p_vec_");
  }

  return Vector;
}

void VectorBlockGenerator::generateLoad(const LoadInst *Load,
                                        ValueMapT &VectorMap,
                                        VectorValueMapT &ScalarMaps) {
  Value *NewLoad;

  MemoryAccess &Access = Statement.getAccessFor(Load);

  assert(ScatteringDomain && "No scattering domain available");

  if (Access.isStrideZero(isl_set_copy(ScatteringDomain)))
    NewLoad = generateStrideZeroLoad(Load, ScalarMaps[0]);
  else if (Access.isStrideOne(isl_set_copy(ScatteringDomain)))
    NewLoad = generateStrideOneLoad(Load, ScalarMaps[0]);
  else
    NewLoad = generateUnknownStrideLoad(Load, ScalarMaps);

  VectorMap[Load] = NewLoad;
}

void VectorBlockGenerator::copyUnaryInst(const UnaryInstruction *Inst,
                                         ValueMapT &BBMap,
                                         ValueMapT &VectorMap) {
  int VectorWidth = getVectorWidth();
  Value *NewOperand = getOperand(Inst->getOperand(0), BBMap, &VectorMap);
  NewOperand = makeVectorOperand(NewOperand);

  assert(isa<CastInst>(Inst) && "Can not generate vector code for instruction");

  const CastInst *Cast = dyn_cast<CastInst>(Inst);
  VectorType *DestType = VectorType::get(Inst->getType(), VectorWidth);
  VectorMap[Inst] = Builder.CreateCast(Cast->getOpcode(), NewOperand, DestType);
}

void VectorBlockGenerator::copyBinaryInst(const BinaryOperator *Inst,
                                          ValueMapT &BBMap,
                                          ValueMapT &VectorMap) {
  Value *OpZero = Inst->getOperand(0);
  Value *OpOne = Inst->getOperand(1);

  Value *NewOpZero, *NewOpOne;
  NewOpZero = getOperand(OpZero, BBMap, &VectorMap);
  NewOpOne = getOperand(OpOne, BBMap, &VectorMap);

  NewOpZero = makeVectorOperand(NewOpZero);
  NewOpOne = makeVectorOperand(NewOpOne);

  Value *NewInst = Builder.CreateBinOp(Inst->getOpcode(), NewOpZero,
                                       NewOpOne,
                                       Inst->getName() + "p_vec");
  VectorMap[Inst] = NewInst;
}

void VectorBlockGenerator::copyStore(const StoreInst *Store, ValueMapT &BBMap,
                                     ValueMapT &VectorMap,
                                     VectorValueMapT &ScalarMaps) {
  int VectorWidth = getVectorWidth();

  MemoryAccess &Access = Statement.getAccessFor(Store);

  assert(ScatteringDomain && "No scattering domain available");

  const Value *Pointer = Store->getPointerOperand();
  Value *Vector = getOperand(Store->getValueOperand(), BBMap, &VectorMap);

  if (Access.isStrideOne(isl_set_copy(ScatteringDomain))) {
    Type *VectorPtrType = getVectorPtrTy(Pointer, VectorWidth);
    Value *NewPointer = getOperand(Pointer, BBMap, &VectorMap);

    Value *VectorPtr = Builder.CreateBitCast(NewPointer, VectorPtrType,
                                             "vector_ptr");
    StoreInst *Store = Builder.CreateStore(Vector, VectorPtr);

    if (!Aligned)
      Store->setAlignment(8);
  } else {
    for (unsigned i = 0; i < ScalarMaps.size(); i++) {
      Value *Scalar = Builder.CreateExtractElement(Vector,
                                                   Builder.getInt32(i));
      Value *NewPointer = getOperand(Pointer, ScalarMaps[i]);
      Builder.CreateStore(Scalar, NewPointer);
    }
  }
}

bool VectorBlockGenerator::hasVectorOperands(const Instruction *Inst,
                                             ValueMapT &VectorMap) {
  for (Instruction::const_op_iterator OI = Inst->op_begin(),
       OE = Inst->op_end(); OI != OE; ++OI)
    if (VectorMap.count(*OI))
      return true;
  return false;
}

int VectorBlockGenerator::getVectorWidth() {
  return ValueMaps.size();
}

void VectorBlockGenerator::copyInstruction(const Instruction *Inst,
                                           ValueMapT &VectorMap,
                                           VectorValueMapT &ScalarMaps) {
  // Terminator instructions control the control flow. They are explicitly
  // expressed in the clast and do not need to be copied.
  if (Inst->isTerminator())
    return;

  if (const LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
    generateLoad(Load, VectorMap, ScalarMaps);
    return;
  }

  if (hasVectorOperands(Inst, VectorMap)) {
    if (const StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
      copyStore(Store, ScalarMaps[0], VectorMap, ScalarMaps);
      return;
    }

    if (const UnaryInstruction *Unary = dyn_cast<UnaryInstruction>(Inst)) {
      copyUnaryInst(Unary, ScalarMaps[0], VectorMap);
      return;
    }

    if (const BinaryOperator *Binary = dyn_cast<BinaryOperator>(Inst)) {
      copyBinaryInst(Binary, ScalarMaps[0], VectorMap);
      return;
    }

    llvm_unreachable("Cannot issue vector code for this instruction");
  }

  for (int VectorLane = 0; VectorLane < getVectorWidth(); VectorLane++)
    copyInstScalar(Inst, ScalarMaps[VectorLane]);
}

void VectorBlockGenerator::copyBB() {
  BasicBlock *BB = Statement.getBasicBlock();
  BasicBlock *CopyBB = SplitBlock(Builder.GetInsertBlock(),
                                  Builder.GetInsertPoint(), P);
  CopyBB->setName("polly.stmt." + BB->getName());
  Builder.SetInsertPoint(CopyBB->begin());

  // Create two maps that store the mapping from the original instructions of
  // the old basic block to their copies in the new basic block. Those maps
  // are basic block local.
  //
  // As vector code generation is supported there is one map for scalar values
  // and one for vector values.
  //
  // In case we just do scalar code generation, the vectorMap is not used and
  // the scalarMap has just one dimension, which contains the mapping.
  //
  // In case vector code generation is done, an instruction may either appear
  // in the vector map once (as it is calculating >vectorwidth< values at a
  // time. Or (if the values are calculated using scalar operations), it
  // appears once in every dimension of the scalarMap.
  VectorValueMapT ScalarBlockMap(getVectorWidth());
  ValueMapT VectorBlockMap;

  for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end();
       II != IE; ++II)
      copyInstruction(II, VectorBlockMap, ScalarBlockMap);
}

/// Class to generate LLVM-IR that calculates the value of a clast_expr.
class ClastExpCodeGen {
  IRBuilder<> &Builder;
  const CharMapT *IVS;

  Value *codegen(const clast_name *e, Type *Ty);
  Value *codegen(const clast_term *e, Type *Ty);
  Value *codegen(const clast_binary *e, Type *Ty);
  Value *codegen(const clast_reduction *r, Type *Ty);
public:

  // A generator for clast expressions.
  //
  // @param B The IRBuilder that defines where the code to calculate the
  //          clast expressions should be inserted.
  // @param IVMAP A Map that translates strings describing the induction
  //              variables to the Values* that represent these variables
  //              on the LLVM side.
  ClastExpCodeGen(IRBuilder<> &B, CharMapT *IVMap);

  // Generates code to calculate a given clast expression.
  //
  // @param e The expression to calculate.
  // @return The Value that holds the result.
  Value *codegen(const clast_expr *e, Type *Ty);

  // @brief Reset the CharMap.
  //
  // This function is called to reset the CharMap to new one, while generating
  // OpenMP code.
  void setIVS(CharMapT *IVSNew);
};

Value *ClastExpCodeGen::codegen(const clast_name *e, Type *Ty) {
  CharMapT::const_iterator I = IVS->find(e->name);

  assert(I != IVS->end() && "Clast name not found");

  return Builder.CreateSExtOrBitCast(I->second, Ty);
}

Value *ClastExpCodeGen::codegen(const clast_term *e, Type *Ty) {
  APInt a = APInt_from_MPZ(e->val);

  Value *ConstOne = ConstantInt::get(Builder.getContext(), a);
  ConstOne = Builder.CreateSExtOrBitCast(ConstOne, Ty);

  if (!e->var)
    return ConstOne;

  Value *var = codegen(e->var, Ty);
  return Builder.CreateMul(ConstOne, var);
}

Value *ClastExpCodeGen::codegen(const clast_binary *e, Type *Ty) {
  Value *LHS = codegen(e->LHS, Ty);

  APInt RHS_AP = APInt_from_MPZ(e->RHS);

  Value *RHS = ConstantInt::get(Builder.getContext(), RHS_AP);
  RHS = Builder.CreateSExtOrBitCast(RHS, Ty);

  switch (e->type) {
  case clast_bin_mod:
    return Builder.CreateSRem(LHS, RHS);
  case clast_bin_fdiv:
    {
      // floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
      Value *One = ConstantInt::get(Ty, 1);
      Value *Zero = ConstantInt::get(Ty, 0);
      Value *Sum1 = Builder.CreateSub(LHS, RHS);
      Value *Sum2 = Builder.CreateAdd(Sum1, One);
      Value *isNegative = Builder.CreateICmpSLT(LHS, Zero);
      Value *Dividend = Builder.CreateSelect(isNegative, Sum2, LHS);
      return Builder.CreateSDiv(Dividend, RHS);
    }
  case clast_bin_cdiv:
    {
      // ceild(n,d) ((n < 0) ? n : (n + d - 1)) / d
      Value *One = ConstantInt::get(Ty, 1);
      Value *Zero = ConstantInt::get(Ty, 0);
      Value *Sum1 = Builder.CreateAdd(LHS, RHS);
      Value *Sum2 = Builder.CreateSub(Sum1, One);
      Value *isNegative = Builder.CreateICmpSLT(LHS, Zero);
      Value *Dividend = Builder.CreateSelect(isNegative, LHS, Sum2);
      return Builder.CreateSDiv(Dividend, RHS);
    }
  case clast_bin_div:
    return Builder.CreateSDiv(LHS, RHS);
  };

  llvm_unreachable("Unknown clast binary expression type");
}

Value *ClastExpCodeGen::codegen(const clast_reduction *r, Type *Ty) {
  assert((   r->type == clast_red_min
             || r->type == clast_red_max
             || r->type == clast_red_sum)
         && "Clast reduction type not supported");
  Value *old = codegen(r->elts[0], Ty);

  for (int i=1; i < r->n; ++i) {
    Value *exprValue = codegen(r->elts[i], Ty);

    switch (r->type) {
    case clast_red_min:
      {
        Value *cmp = Builder.CreateICmpSLT(old, exprValue);
        old = Builder.CreateSelect(cmp, old, exprValue);
        break;
      }
    case clast_red_max:
      {
        Value *cmp = Builder.CreateICmpSGT(old, exprValue);
        old = Builder.CreateSelect(cmp, old, exprValue);
        break;
      }
    case clast_red_sum:
      old = Builder.CreateAdd(old, exprValue);
      break;
    }
  }

  return old;
}

ClastExpCodeGen::ClastExpCodeGen(IRBuilder<> &B, CharMapT *IVMap)
  : Builder(B), IVS(IVMap) {}

Value *ClastExpCodeGen::codegen(const clast_expr *e, Type *Ty) {
  switch(e->type) {
  case clast_expr_name:
    return codegen((const clast_name *)e, Ty);
  case clast_expr_term:
    return codegen((const clast_term *)e, Ty);
  case clast_expr_bin:
    return codegen((const clast_binary *)e, Ty);
  case clast_expr_red:
    return codegen((const clast_reduction *)e, Ty);
  }

  llvm_unreachable("Unknown clast expression!");
}

void ClastExpCodeGen::setIVS(CharMapT *IVSNew) {
  IVS = IVSNew;
}

class ClastStmtCodeGen {
  // The Scop we code generate.
  Scop *S;
  ScalarEvolution &SE;
  DominatorTree *DT;
  ScopDetection *SD;
  Dependences *DP;
  TargetData *TD;
  Pass *P;

  // The Builder specifies the current location to code generate at.
  IRBuilder<> &Builder;

  // Map the Values from the old code to their counterparts in the new code.
  ValueMapT ValueMap;

  // clastVars maps from the textual representation of a clast variable to its
  // current *Value. clast variables are scheduling variables, original
  // induction variables or parameters. They are used either in loop bounds or
  // to define the statement instance that is executed.
  //
  //   for (s = 0; s < n + 3; ++i)
  //     for (t = s; t < m; ++j)
  //       Stmt(i = s + 3 * m, j = t);
  //
  // {s,t,i,j,n,m} is the set of clast variables in this clast.
  CharMapT *clastVars;

  // Codegenerator for clast expressions.
  ClastExpCodeGen ExpGen;

  // Do we currently generate parallel code?
  bool parallelCodeGeneration;

  std::vector<std::string> parallelLoops;

public:

  const std::vector<std::string> &getParallelLoops();

  protected:
  void codegen(const clast_assignment *a);

  void codegen(const clast_assignment *a, ScopStmt *Statement,
               unsigned Dimension, int vectorDim,
               std::vector<ValueMapT> *VectorVMap = 0);

  void codegenSubstitutions(const clast_stmt *Assignment,
                            ScopStmt *Statement, int vectorDim = 0,
                            std::vector<ValueMapT> *VectorVMap = 0);

  void codegen(const clast_user_stmt *u, std::vector<Value*> *IVS = NULL,
               const char *iterator = NULL, isl_set *scatteringDomain = 0);

  void codegen(const clast_block *b);

  /// @brief Create a classical sequential loop.
  void codegenForSequential(const clast_for *f, Value *LowerBound = 0,
                                                Value *UpperBound = 0);

  /// @brief Add a new definition of an openmp subfunction.
  Function *addOpenMPSubfunction(Module *M);

  /// @brief Add values to the OpenMP structure.
  ///
  /// Create the subfunction structure and add the values from the list.
  Value *addValuesToOpenMPStruct(SetVector<Value*> OMPDataVals,
                                 Function *SubFunction);

  /// @brief Create OpenMP structure values.
  ///
  /// Create a list of values that has to be stored into the subfuncition
  /// structure.
  SetVector<Value*> createOpenMPStructValues();

  /// @brief Extract the values from the subfunction parameter.
  ///
  /// Extract the values from the subfunction parameter and update the clast
  /// variables to point to the new values.
  void extractValuesFromOpenMPStruct(CharMapT *clastVarsOMP,
                                     SetVector<Value*> OMPDataVals,
                                     Value *userContext);

  /// @brief Add body to the subfunction.
  void addOpenMPSubfunctionBody(Function *FN, const clast_for *f,
                                Value *structData,
                                SetVector<Value*> OMPDataVals);

  /// @brief Create an OpenMP parallel for loop.
  ///
  /// This loop reflects a loop as if it would have been created by an OpenMP
  /// statement.
  void codegenForOpenMP(const clast_for *f);

  bool isInnermostLoop(const clast_for *f);

  /// @brief Get the number of loop iterations for this loop.
  /// @param f The clast for loop to check.
  int getNumberOfIterations(const clast_for *f);

  /// @brief Create vector instructions for this loop.
  void codegenForVector(const clast_for *f);

  void codegen(const clast_for *f);

  Value *codegen(const clast_equation *eq);

  void codegen(const clast_guard *g);

  void codegen(const clast_stmt *stmt);

  void addParameters(const CloogNames *names);

  public:
  void codegen(const clast_root *r);

  ClastStmtCodeGen(Scop *scop, ScalarEvolution &se, DominatorTree *dt,
                   ScopDetection *sd, Dependences *dp, TargetData *td,
                   IRBuilder<> &B, Pass *P);
};
}

const std::vector<std::string> &ClastStmtCodeGen::getParallelLoops() {
  return parallelLoops;
}

void ClastStmtCodeGen::codegen(const clast_assignment *a) {
  Value *V= ExpGen.codegen(a->RHS, TD->getIntPtrType(Builder.getContext()));
  (*clastVars)[a->LHS] = V;
}

void ClastStmtCodeGen::codegen(const clast_assignment *a, ScopStmt *Statement,
                               unsigned Dimension, int vectorDim,
                               std::vector<ValueMapT> *VectorVMap) {
  Value *RHS = ExpGen.codegen(a->RHS,
                              TD->getIntPtrType(Builder.getContext()));

  assert(!a->LHS && "Statement assignments do not have left hand side");
  const PHINode *PN;
  PN = Statement->getInductionVariableForDimension(Dimension);
  const Value *V = PN;

  if (VectorVMap)
    (*VectorVMap)[vectorDim][V] = RHS;

  ValueMap[V] = RHS;
}

void ClastStmtCodeGen::codegenSubstitutions(const clast_stmt *Assignment,
                                             ScopStmt *Statement, int vectorDim,
  std::vector<ValueMapT> *VectorVMap) {
  int Dimension = 0;

  while (Assignment) {
    assert(CLAST_STMT_IS_A(Assignment, stmt_ass)
           && "Substitions are expected to be assignments");
    codegen((const clast_assignment *)Assignment, Statement, Dimension,
            vectorDim, VectorVMap);
    Assignment = Assignment->next;
    Dimension++;
  }
}

void ClastStmtCodeGen::codegen(const clast_user_stmt *u,
                               std::vector<Value*> *IVS , const char *iterator,
                               isl_set *Domain) {
  ScopStmt *Statement = (ScopStmt *)u->statement->usr;

  if (u->substitutions)
    codegenSubstitutions(u->substitutions, Statement);

  int VectorDimensions = IVS ? IVS->size() : 1;

  if (VectorDimensions == 1) {
    BlockGenerator::generate(Builder, ValueMap, *Statement, Domain, P);
    return;
  }

  VectorValueMapT VectorMap(VectorDimensions);

  if (IVS) {
    assert (u->substitutions && "Substitutions expected!");
    int i = 0;
    for (std::vector<Value*>::iterator II = IVS->begin(), IE = IVS->end();
         II != IE; ++II) {
      (*clastVars)[iterator] = *II;
      codegenSubstitutions(u->substitutions, Statement, i, &VectorMap);
      i++;
    }
  }

  VectorBlockGenerator::generate(Builder, ValueMap, VectorMap, *Statement,
                                 Domain, P);
}

void ClastStmtCodeGen::codegen(const clast_block *b) {
  if (b->body)
    codegen(b->body);
}

void ClastStmtCodeGen::codegenForSequential(const clast_for *f,
                                            Value *LowerBound,
                                            Value *UpperBound) {
  APInt Stride;
  BasicBlock *AfterBB;
  Type *IntPtrTy;

  Stride = APInt_from_MPZ(f->stride);
  IntPtrTy = TD->getIntPtrType(Builder.getContext());

  // The value of lowerbound and upperbound will be supplied, if this
  // function is called while generating OpenMP code. Otherwise get
  // the values.
  assert(!!LowerBound == !!UpperBound && "Either give both bounds or none");

  if (LowerBound == 0) {
    LowerBound = ExpGen.codegen(f->LB, IntPtrTy);
    UpperBound = ExpGen.codegen(f->UB, IntPtrTy);
  }

  Value *IV = createLoop(&Builder, LowerBound, UpperBound, Stride, DT, P,
                         &AfterBB);

  // Add loop iv to symbols.
  (*clastVars)[f->iterator] = IV;

  if (f->body)
    codegen(f->body);

  // Loop is finished, so remove its iv from the live symbols.
  clastVars->erase(f->iterator);
  Builder.SetInsertPoint(AfterBB->begin());
}

Function *ClastStmtCodeGen::addOpenMPSubfunction(Module *M) {
  Function *F = Builder.GetInsertBlock()->getParent();
  std::vector<Type*> Arguments(1, Builder.getInt8PtrTy());
  FunctionType *FT = FunctionType::get(Builder.getVoidTy(), Arguments, false);
  Function *FN = Function::Create(FT, Function::InternalLinkage,
                                  F->getName() + ".omp_subfn", M);
  // Do not run any polly pass on the new function.
  SD->markFunctionAsInvalid(FN);

  Function::arg_iterator AI = FN->arg_begin();
  AI->setName("omp.userContext");

  return FN;
}

Value *ClastStmtCodeGen::addValuesToOpenMPStruct(SetVector<Value*> OMPDataVals,
                                                 Function *SubFunction) {
  std::vector<Type*> structMembers;

  // Create the structure.
  for (unsigned i = 0; i < OMPDataVals.size(); i++)
    structMembers.push_back(OMPDataVals[i]->getType());

  StructType *structTy = StructType::get(Builder.getContext(),
                                         structMembers);
  // Store the values into the structure.
  Value *structData = Builder.CreateAlloca(structTy, 0, "omp.userContext");
  for (unsigned i = 0; i < OMPDataVals.size(); i++) {
    Value *storeAddr = Builder.CreateStructGEP(structData, i);
    Builder.CreateStore(OMPDataVals[i], storeAddr);
  }

  return structData;
}

SetVector<Value*> ClastStmtCodeGen::createOpenMPStructValues() {
  SetVector<Value*> OMPDataVals;

  // Push the clast variables available in the clastVars.
  for (CharMapT::iterator I = clastVars->begin(), E = clastVars->end();
       I != E; I++)
    OMPDataVals.insert(I->second);

  // Push the base addresses of memory references.
  for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    for (SmallVector<MemoryAccess*, 8>::iterator I = Stmt->memacc_begin(),
         E = Stmt->memacc_end(); I != E; ++I) {
      Value *BaseAddr = const_cast<Value*>((*I)->getBaseAddr());
      OMPDataVals.insert((BaseAddr));
    }
  }

  return OMPDataVals;
}

void ClastStmtCodeGen::extractValuesFromOpenMPStruct(CharMapT *clastVarsOMP,
  SetVector<Value*> OMPDataVals, Value *userContext) {
  // Extract the clast variables.
  unsigned i = 0;
  for (CharMapT::iterator I = clastVars->begin(), E = clastVars->end();
       I != E; I++) {
    Value *loadAddr = Builder.CreateStructGEP(userContext, i);
    (*clastVarsOMP)[I->first] = Builder.CreateLoad(loadAddr);
    i++;
  }

  // Extract the base addresses of memory references.
  for (unsigned j = i; j < OMPDataVals.size(); j++) {
    Value *loadAddr = Builder.CreateStructGEP(userContext, j);
    Value *baseAddr = OMPDataVals[j];
    ValueMap[baseAddr] = Builder.CreateLoad(loadAddr);
  }
}

void ClastStmtCodeGen::addOpenMPSubfunctionBody(Function *FN,
                                                const clast_for *f,
                                                Value *structData,
                                                SetVector<Value*> OMPDataVals) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  LLVMContext &Context = FN->getContext();
  IntegerType *intPtrTy = TD->getIntPtrType(Context);

  // Store the previous basic block.
  BasicBlock::iterator PrevInsertPoint = Builder.GetInsertPoint();
  BasicBlock *PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "omp.setup", FN);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "omp.exit", FN);
  BasicBlock *checkNextBB = BasicBlock::Create(Context, "omp.checkNext", FN);
  BasicBlock *loadIVBoundsBB = BasicBlock::Create(Context, "omp.loadIVBounds",
                                                  FN);

  DT->addNewBlock(HeaderBB, PrevBB);
  DT->addNewBlock(ExitBB, HeaderBB);
  DT->addNewBlock(checkNextBB, HeaderBB);
  DT->addNewBlock(loadIVBoundsBB, HeaderBB);

  // Fill up basic block HeaderBB.
  Builder.SetInsertPoint(HeaderBB);
  Value *lowerBoundPtr = Builder.CreateAlloca(intPtrTy, 0,
                                              "omp.lowerBoundPtr");
  Value *upperBoundPtr = Builder.CreateAlloca(intPtrTy, 0,
                                              "omp.upperBoundPtr");
  Value *userContext = Builder.CreateBitCast(FN->arg_begin(),
                                             structData->getType(),
                                             "omp.userContext");

  CharMapT clastVarsOMP;
  extractValuesFromOpenMPStruct(&clastVarsOMP, OMPDataVals, userContext);

  Builder.CreateBr(checkNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(checkNextBB);
  Function *runtimeNextFunction = M->getFunction("GOMP_loop_runtime_next");
  Value *ret1 = Builder.CreateCall2(runtimeNextFunction,
                                    lowerBoundPtr, upperBoundPtr);
  Value *hasNextSchedule = Builder.CreateTrunc(ret1, Builder.getInt1Ty(),
                                               "omp.hasNextScheduleBlock");
  Builder.CreateCondBr(hasNextSchedule, loadIVBoundsBB, ExitBB);

  // Add code to to load the iv bounds for this set of iterations.
  Builder.SetInsertPoint(loadIVBoundsBB);
  Value *lowerBound = Builder.CreateLoad(lowerBoundPtr, "omp.lowerBound");
  Value *upperBound = Builder.CreateLoad(upperBoundPtr, "omp.upperBound");

  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  upperBound = Builder.CreateSub(upperBound, ConstantInt::get(intPtrTy, 1),
                                 "omp.upperBoundAdjusted");

  // Use clastVarsOMP during code generation of the OpenMP subfunction.
  CharMapT *oldClastVars = clastVars;
  clastVars = &clastVarsOMP;
  ExpGen.setIVS(&clastVarsOMP);

  Builder.CreateBr(checkNextBB);
  Builder.SetInsertPoint(--Builder.GetInsertPoint());
  codegenForSequential(f, lowerBound, upperBound);

  // Restore the old clastVars.
  clastVars = oldClastVars;
  ExpGen.setIVS(oldClastVars);

  // Add code to terminate this openmp subfunction.
  Builder.SetInsertPoint(ExitBB);
  Function *endnowaitFunction = M->getFunction("GOMP_loop_end_nowait");
  Builder.CreateCall(endnowaitFunction);
  Builder.CreateRetVoid();

  // Restore the previous insert point.
  Builder.SetInsertPoint(PrevInsertPoint);
}

void ClastStmtCodeGen::codegenForOpenMP(const clast_for *For) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  IntegerType *IntPtrTy = TD->getIntPtrType(Builder.getContext());

  Function *SubFunction = addOpenMPSubfunction(M);
  SetVector<Value*> OMPDataVals = createOpenMPStructValues();
  Value *StructData = addValuesToOpenMPStruct(OMPDataVals, SubFunction);

  addOpenMPSubfunctionBody(SubFunction, For, StructData, OMPDataVals);

  // Create call for GOMP_parallel_loop_runtime_start.
  Value *SubfunctionParam = Builder.CreateBitCast(StructData,
                                                  Builder.getInt8PtrTy(),
                                                  "omp_data");

  Value *NumberOfThreads = Builder.getInt32(0);
  Value *LowerBound = ExpGen.codegen(For->LB, IntPtrTy);
  Value *UpperBound = ExpGen.codegen(For->UB, IntPtrTy);

  // Add one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UpperBound = Builder.CreateAdd(UpperBound, ConstantInt::get(IntPtrTy, 1));
  APInt APStride = APInt_from_MPZ(For->stride);
  Value *Stride = ConstantInt::get(IntPtrTy,
                                   APStride.zext(IntPtrTy->getBitWidth()));

  Value *Arguments[] = { SubFunction, SubfunctionParam, NumberOfThreads,
    LowerBound, UpperBound, Stride};
  Builder.CreateCall(M->getFunction("GOMP_parallel_loop_runtime_start"),
                     Arguments);
  Builder.CreateCall(SubFunction, SubfunctionParam);
  Builder.CreateCall(M->getFunction("GOMP_parallel_end"));
}

bool ClastStmtCodeGen::isInnermostLoop(const clast_for *f) {
  const clast_stmt *stmt = f->body;

  while (stmt) {
    if (!CLAST_STMT_IS_A(stmt, stmt_user))
      return false;

    stmt = stmt->next;
  }

  return true;
}

int ClastStmtCodeGen::getNumberOfIterations(const clast_for *f) {
  isl_set *loopDomain = isl_set_copy(isl_set_from_cloog_domain(f->domain));
  isl_set *tmp = isl_set_copy(loopDomain);

  // Calculate a map similar to the identity map, but with the last input
  // and output dimension not related.
  //  [i0, i1, i2, i3] -> [i0, i1, i2, o0]
  isl_space *Space = isl_set_get_space(loopDomain);
  Space = isl_space_drop_outputs(Space,
                                 isl_set_dim(loopDomain, isl_dim_set) - 2, 1);
  Space = isl_space_map_from_set(Space);
  isl_map *identity = isl_map_identity(Space);
  identity = isl_map_add_dims(identity, isl_dim_in, 1);
  identity = isl_map_add_dims(identity, isl_dim_out, 1);

  isl_map *map = isl_map_from_domain_and_range(tmp, loopDomain);
  map = isl_map_intersect(map, identity);

  isl_map *lexmax = isl_map_lexmax(isl_map_copy(map));
  isl_map *lexmin = isl_map_lexmin(map);
  isl_map *sub = isl_map_sum(lexmax, isl_map_neg(lexmin));

  isl_set *elements = isl_map_range(sub);

  if (!isl_set_is_singleton(elements)) {
    isl_set_free(elements);
    return -1;
  }

  isl_point *p = isl_set_sample_point(elements);

  isl_int v;
  isl_int_init(v);
  isl_point_get_coordinate(p, isl_dim_set, isl_set_n_dim(loopDomain) - 1, &v);
  int numberIterations = isl_int_get_si(v);
  isl_int_clear(v);
  isl_point_free(p);

  return (numberIterations) / isl_int_get_si(f->stride) + 1;
}

void ClastStmtCodeGen::codegenForVector(const clast_for *f) {
  DEBUG(dbgs() << "Vectorizing loop '" << f->iterator << "'\n";);
  int vectorWidth = getNumberOfIterations(f);

  Value *LB = ExpGen.codegen(f->LB,
                             TD->getIntPtrType(Builder.getContext()));

  APInt Stride = APInt_from_MPZ(f->stride);
  IntegerType *LoopIVType = dyn_cast<IntegerType>(LB->getType());
  Stride =  Stride.zext(LoopIVType->getBitWidth());
  Value *StrideValue = ConstantInt::get(LoopIVType, Stride);

  std::vector<Value*> IVS(vectorWidth);
  IVS[0] = LB;

  for (int i = 1; i < vectorWidth; i++)
    IVS[i] = Builder.CreateAdd(IVS[i-1], StrideValue, "p_vector_iv");

  isl_set *scatteringDomain =
    isl_set_copy(isl_set_from_cloog_domain(f->domain));

  // Add loop iv to symbols.
  (*clastVars)[f->iterator] = LB;

  const clast_stmt *stmt = f->body;

  while (stmt) {
    codegen((const clast_user_stmt *)stmt, &IVS, f->iterator,
            scatteringDomain);
    stmt = stmt->next;
  }

  // Loop is finished, so remove its iv from the live symbols.
  isl_set_free(scatteringDomain);
  clastVars->erase(f->iterator);
}

void ClastStmtCodeGen::codegen(const clast_for *f) {
  if ((Vector || OpenMP) && DP->isParallelFor(f)) {
    if (Vector && isInnermostLoop(f) && (-1 != getNumberOfIterations(f))
        && (getNumberOfIterations(f) <= 16)) {
      codegenForVector(f);
      return;
    }

    if (OpenMP && !parallelCodeGeneration) {
      parallelCodeGeneration = true;
      parallelLoops.push_back(f->iterator);
      codegenForOpenMP(f);
      parallelCodeGeneration = false;
      return;
    }
  }

  codegenForSequential(f);
}

Value *ClastStmtCodeGen::codegen(const clast_equation *eq) {
  Value *LHS = ExpGen.codegen(eq->LHS,
                              TD->getIntPtrType(Builder.getContext()));
  Value *RHS = ExpGen.codegen(eq->RHS,
                              TD->getIntPtrType(Builder.getContext()));
  CmpInst::Predicate P;

  if (eq->sign == 0)
    P = ICmpInst::ICMP_EQ;
  else if (eq->sign > 0)
    P = ICmpInst::ICMP_SGE;
  else
    P = ICmpInst::ICMP_SLE;

  return Builder.CreateICmp(P, LHS, RHS);
}

void ClastStmtCodeGen::codegen(const clast_guard *g) {
  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  BasicBlock *CondBB = SplitBlock(Builder.GetInsertBlock(),
                                      Builder.GetInsertPoint(), P);
  CondBB->setName("polly.cond");
  BasicBlock *MergeBB = SplitBlock(CondBB, CondBB->begin(), P);
  MergeBB->setName("polly.merge");
  BasicBlock *ThenBB = BasicBlock::Create(Context, "polly.then", F);

  DT->addNewBlock(ThenBB, CondBB);
  DT->changeImmediateDominator(MergeBB, CondBB);

  CondBB->getTerminator()->eraseFromParent();

  Builder.SetInsertPoint(CondBB);

  Value *Predicate = codegen(&(g->eq[0]));

  for (int i = 1; i < g->n; ++i) {
    Value *TmpPredicate = codegen(&(g->eq[i]));
    Predicate = Builder.CreateAnd(Predicate, TmpPredicate);
  }

  Builder.CreateCondBr(Predicate, ThenBB, MergeBB);
  Builder.SetInsertPoint(ThenBB);
  Builder.CreateBr(MergeBB);
  Builder.SetInsertPoint(ThenBB->begin());

  codegen(g->then);

  Builder.SetInsertPoint(MergeBB->begin());
}

void ClastStmtCodeGen::codegen(const clast_stmt *stmt) {
  if	    (CLAST_STMT_IS_A(stmt, stmt_root))
    assert(false && "No second root statement expected");
  else if (CLAST_STMT_IS_A(stmt, stmt_ass))
    codegen((const clast_assignment *)stmt);
  else if (CLAST_STMT_IS_A(stmt, stmt_user))
    codegen((const clast_user_stmt *)stmt);
  else if (CLAST_STMT_IS_A(stmt, stmt_block))
    codegen((const clast_block *)stmt);
  else if (CLAST_STMT_IS_A(stmt, stmt_for))
    codegen((const clast_for *)stmt);
  else if (CLAST_STMT_IS_A(stmt, stmt_guard))
    codegen((const clast_guard *)stmt);

  if (stmt->next)
    codegen(stmt->next);
}

void ClastStmtCodeGen::addParameters(const CloogNames *names) {
  SCEVExpander Rewriter(SE, "polly");

  int i = 0;
  for (Scop::param_iterator PI = S->param_begin(), PE = S->param_end();
       PI != PE; ++PI) {
    assert(i < names->nb_parameters && "Not enough parameter names");

    const SCEV *Param = *PI;
    Type *Ty = Param->getType();

    Instruction *insertLocation = --(Builder.GetInsertBlock()->end());
    Value *V = Rewriter.expandCodeFor(Param, Ty, insertLocation);
    (*clastVars)[names->parameters[i]] = V;

    ++i;
  }
}

void ClastStmtCodeGen::codegen(const clast_root *r) {
  clastVars = new CharMapT();
  addParameters(r->names);
  ExpGen.setIVS(clastVars);

  parallelCodeGeneration = false;

  const clast_stmt *stmt = (const clast_stmt*) r;
  if (stmt->next)
    codegen(stmt->next);

  delete clastVars;
}

ClastStmtCodeGen::ClastStmtCodeGen(Scop *scop, ScalarEvolution &se,
                                   DominatorTree *dt, ScopDetection *sd,
                                   Dependences *dp, TargetData *td,
                                  IRBuilder<> &B, Pass *P) :
    S(scop), SE(se), DT(dt), SD(sd), DP(dp), TD(td), P(P), Builder(B),
    ExpGen(Builder, NULL) {}

namespace {
class CodeGeneration : public ScopPass {
  Region *region;
  Scop *S;
  DominatorTree *DT;
  ScalarEvolution *SE;
  ScopDetection *SD;
  TargetData *TD;
  RegionInfo *RI;

  std::vector<std::string> parallelLoops;

  public:
  static char ID;

  CodeGeneration() : ScopPass(ID) {}

  // Add the declarations needed by the OpenMP function calls that we insert in
  // OpenMP mode.
  void addOpenMPDeclarations(Module *M)
  {
    IRBuilder<> Builder(M->getContext());
    IntegerType *LongTy = TD->getIntPtrType(M->getContext());

    llvm::GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    if (!M->getFunction("GOMP_parallel_end")) {
      FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
      Function::Create(Ty, Linkage, "GOMP_parallel_end", M);
    }

    if (!M->getFunction("GOMP_parallel_loop_runtime_start")) {
      Type *Params[] = {
        PointerType::getUnqual(FunctionType::get(Builder.getVoidTy(),
                                                 Builder.getInt8PtrTy(),
                                                 false)),
        Builder.getInt8PtrTy(),
        Builder.getInt32Ty(),
        LongTy,
        LongTy,
        LongTy,
      };

      FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
      Function::Create(Ty, Linkage, "GOMP_parallel_loop_runtime_start", M);
    }

    if (!M->getFunction("GOMP_loop_runtime_next")) {
      PointerType *LongPtrTy = PointerType::getUnqual(LongTy);
      Type *Params[] = {
        LongPtrTy,
        LongPtrTy,
      };

      FunctionType *Ty = FunctionType::get(Builder.getInt8Ty(), Params, false);
      Function::Create(Ty, Linkage, "GOMP_loop_runtime_next", M);
    }

    if (!M->getFunction("GOMP_loop_end_nowait")) {
      FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
      Function::Create(Ty, Linkage, "GOMP_loop_end_nowait", M);
    }
  }

  // Split the entry edge of the region and generate a new basic block on this
  // edge. This function also updates ScopInfo and RegionInfo.
  //
  // @param region The region where the entry edge will be splitted.
  BasicBlock *splitEdgeAdvanced(Region *region) {
    BasicBlock *newBlock;
    BasicBlock *splitBlock;

    newBlock = SplitEdge(region->getEnteringBlock(), region->getEntry(), this);

    if (DT->dominates(region->getEntry(), newBlock)) {
      BasicBlock *OldBlock = region->getEntry();
      std::string OldName = OldBlock->getName();

      // Update ScopInfo.
      for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI)
        if ((*SI)->getBasicBlock() == OldBlock) {
          (*SI)->setBasicBlock(newBlock);
          break;
        }

      // Update RegionInfo.
      splitBlock = OldBlock;
      OldBlock->setName("polly.split");
      newBlock->setName(OldName);
      region->replaceEntry(newBlock);
      RI->setRegionFor(newBlock, region);
    } else {
      RI->setRegionFor(newBlock, region->getParent());
      splitBlock = newBlock;
    }

    return splitBlock;
  }

  // Create a split block that branches either to the old code or to a new basic
  // block where the new code can be inserted.
  //
  // @param Builder A builder that will be set to point to a basic block, where
  //                the new code can be generated.
  // @return The split basic block.
  BasicBlock *addSplitAndStartBlock(IRBuilder<> *Builder) {
    BasicBlock *StartBlock, *SplitBlock;

    SplitBlock = splitEdgeAdvanced(region);
    SplitBlock->setName("polly.split_new_and_old");
    Function *F = SplitBlock->getParent();
    StartBlock = BasicBlock::Create(F->getContext(), "polly.start", F);
    SplitBlock->getTerminator()->eraseFromParent();
    Builder->SetInsertPoint(SplitBlock);
    Builder->CreateCondBr(Builder->getTrue(), StartBlock, region->getEntry());
    DT->addNewBlock(StartBlock, SplitBlock);
    Builder->SetInsertPoint(StartBlock);
    return SplitBlock;
  }

  // Merge the control flow of the newly generated code with the existing code.
  //
  // @param SplitBlock The basic block where the control flow was split between
  //                   old and new version of the Scop.
  // @param Builder    An IRBuilder that points to the last instruction of the
  //                   newly generated code.
  void mergeControlFlow(BasicBlock *SplitBlock, IRBuilder<> *Builder) {
    BasicBlock *MergeBlock;
    Region *R = region;

    if (R->getExit()->getSinglePredecessor())
      // No splitEdge required.  A block with a single predecessor cannot have
      // PHI nodes that would complicate life.
      MergeBlock = R->getExit();
    else {
      MergeBlock = SplitEdge(R->getExitingBlock(), R->getExit(), this);
      // SplitEdge will never split R->getExit(), as R->getExit() has more than
      // one predecessor. Hence, mergeBlock is always a newly generated block.
      R->replaceExit(MergeBlock);
    }

    Builder->CreateBr(MergeBlock);
    MergeBlock->setName("polly.merge_new_and_old");

    if (DT->dominates(SplitBlock, MergeBlock))
      DT->changeImmediateDominator(MergeBlock, SplitBlock);
  }

  bool runOnScop(Scop &scop) {
    S = &scop;
    region = &S->getRegion();
    DT = &getAnalysis<DominatorTree>();
    Dependences *DP = &getAnalysis<Dependences>();
    SE = &getAnalysis<ScalarEvolution>();
    SD = &getAnalysis<ScopDetection>();
    TD = &getAnalysis<TargetData>();
    RI = &getAnalysis<RegionInfo>();

    parallelLoops.clear();

    assert(region->isSimple() && "Only simple regions are supported");

    Module *M = region->getEntry()->getParent()->getParent();

    if (OpenMP) addOpenMPDeclarations(M);

    // In the CFG the optimized code of the SCoP is generated next to the
    // original code. Both the new and the original version of the code remain
    // in the CFG. A branch statement decides which version is executed.
    // For now, we always execute the new version (the old one is dead code
    // eliminated by the cleanup passes). In the future we may decide to execute
    // the new version only if certain run time checks succeed. This will be
    // useful to support constructs for which we cannot prove all assumptions at
    // compile time.
    //
    // Before transformation:
    //
    //                        bb0
    //                         |
    //                     orig_scop
    //                         |
    //                        bb1
    //
    // After transformation:
    //                        bb0
    //                         |
    //                  polly.splitBlock
    //                     /       \.
    //                     |     startBlock
    //                     |        |
    //               orig_scop   new_scop
    //                     \      /
    //                      \    /
    //                        bb1 (joinBlock)
    IRBuilder<> builder(region->getEntry());

    // The builder will be set to startBlock.
    BasicBlock *splitBlock = addSplitAndStartBlock(&builder);
    BasicBlock *StartBlock = builder.GetInsertBlock();

    mergeControlFlow(splitBlock, &builder);
    builder.SetInsertPoint(StartBlock->begin());

    ClastStmtCodeGen CodeGen(S, *SE, DT, SD, DP, TD, builder, this);
    CloogInfo &C = getAnalysis<CloogInfo>();
    CodeGen.codegen(C.getClast());

    parallelLoops.insert(parallelLoops.begin(),
                         CodeGen.getParallelLoops().begin(),
                         CodeGen.getParallelLoops().end());

    return true;
  }

  virtual void printScop(raw_ostream &OS) const {
    for (std::vector<std::string>::const_iterator PI = parallelLoops.begin(),
         PE = parallelLoops.end(); PI != PE; ++PI)
      OS << "Parallel loop with iterator '" << *PI << "' generated\n";
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<CloogInfo>();
    AU.addRequired<Dependences>();
    AU.addRequired<DominatorTree>();
    AU.addRequired<RegionInfo>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<ScopDetection>();
    AU.addRequired<ScopInfo>();
    AU.addRequired<TargetData>();

    AU.addPreserved<CloogInfo>();
    AU.addPreserved<Dependences>();

    // FIXME: We do not create LoopInfo for the newly generated loops.
    AU.addPreserved<LoopInfo>();
    AU.addPreserved<DominatorTree>();
    AU.addPreserved<ScopDetection>();
    AU.addPreserved<ScalarEvolution>();

    // FIXME: We do not yet add regions for the newly generated code to the
    //        region tree.
    AU.addPreserved<RegionInfo>();
    AU.addPreserved<TempScopInfo>();
    AU.addPreserved<ScopInfo>();
    AU.addPreservedID(IndependentBlocksID);
  }
};
}

char CodeGeneration::ID = 1;

INITIALIZE_PASS_BEGIN(CodeGeneration, "polly-codegen",
                      "Polly - Create LLVM-IR form SCoPs", false, false)
INITIALIZE_PASS_DEPENDENCY(CloogInfo)
INITIALIZE_PASS_DEPENDENCY(Dependences)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(RegionInfo)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(ScopDetection)
INITIALIZE_PASS_DEPENDENCY(TargetData)
INITIALIZE_PASS_END(CodeGeneration, "polly-codegen",
                      "Polly - Create LLVM-IR form SCoPs", false, false)

Pass *polly::createCodeGenerationPass() {
  return new CodeGeneration();
}

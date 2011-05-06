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

#include "polly/LinkAllPasses.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ScopHelper.h"
#include "polly/Cloog.h"
#include "polly/Dependences.h"
#include "polly/ScopInfo.h"
#include "polly/TempScopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Module.h"
#include "llvm/ADT/SetVector.h"

#define CLOOG_INT_GMP 1
#include "cloog/cloog.h"
#include "cloog/isl/cloog.h"

#include <vector>
#include <utility>

using namespace polly;
using namespace llvm;

struct isl_set;

namespace polly {

static cl::opt<bool>
Vector("enable-polly-vector",
       cl::desc("Enable polly vector code generation"), cl::Hidden,
       cl::value_desc("Vector code generation enabled if true"),
       cl::init(false));

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

static cl::opt<std::string>
CodegenOnly("polly-codegen-only",
            cl::desc("Codegen only this function"), cl::Hidden,
            cl::value_desc("The function name to codegen"),
            cl::ValueRequired, cl::init(""));

typedef DenseMap<const Value*, Value*> ValueMapT;
typedef DenseMap<const char*, Value*> CharMapT;
typedef std::vector<ValueMapT> VectorValueMapT;

// Create a new loop.
//
// @param Builder The builder used to create the loop.  It also defines the
//                place where to create the loop.
// @param UB      The upper bound of the loop iv.
// @param Stride  The number by which the loop iv is incremented after every
//                iteration.
static void createLoop(IRBuilder<> *Builder, Value *LB, Value *UB, APInt Stride,
                PHINode*& IV, BasicBlock*& AfterBB, Value*& IncrementedIV,
                DominatorTree *DT) {
  Function *F = Builder->GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  BasicBlock *PreheaderBB = Builder->GetInsertBlock();
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.loop_header", F);
  BasicBlock *BodyBB = BasicBlock::Create(Context, "polly.loop_body", F);
  AfterBB = BasicBlock::Create(Context, "polly.after_loop", F);

  Builder->CreateBr(HeaderBB);
  DT->addNewBlock(HeaderBB, PreheaderBB);

  Builder->SetInsertPoint(BodyBB);

  Builder->SetInsertPoint(HeaderBB);

  // Use the type of upper and lower bound.
  assert(LB->getType() == UB->getType()
         && "Different types for upper and lower bound.");

  const IntegerType *LoopIVType = dyn_cast<IntegerType>(UB->getType());
  assert(LoopIVType && "UB is not integer?");

  // IV
  IV = Builder->CreatePHI(LoopIVType, 2, "polly.loopiv");
  IV->addIncoming(LB, PreheaderBB);

  // IV increment.
  Value *StrideValue = ConstantInt::get(LoopIVType,
                                        Stride.zext(LoopIVType->getBitWidth()));
  IncrementedIV = Builder->CreateAdd(IV, StrideValue, "polly.next_loopiv");

  // Exit condition.
  if (AtLeastOnce) { // At least on iteration.
    UB = Builder->CreateAdd(UB, Builder->getInt64(1));
    Value *CMP = Builder->CreateICmpEQ(IV, UB);
    Builder->CreateCondBr(CMP, AfterBB, BodyBB);
  } else { // Maybe not executed at all.
    Value *CMP = Builder->CreateICmpSLE(IV, UB);
    Builder->CreateCondBr(CMP, BodyBB, AfterBB);
  }
  DT->addNewBlock(BodyBB, HeaderBB);
  DT->addNewBlock(AfterBB, HeaderBB);

  Builder->SetInsertPoint(BodyBB);
}

class BlockGenerator {
  IRBuilder<> &Builder;
  ValueMapT &VMap;
  VectorValueMapT &ValueMaps;
  Scop &S;
  ScopStmt &statement;
  isl_set *scatteringDomain;

public:
  BlockGenerator(IRBuilder<> &B, ValueMapT &vmap, VectorValueMapT &vmaps,
                 ScopStmt &Stmt, isl_set *domain)
    : Builder(B), VMap(vmap), ValueMaps(vmaps), S(*Stmt.getParent()),
    statement(Stmt), scatteringDomain(domain) {}

  const Region &getRegion() {
    return S.getRegion();
  }

  Value* makeVectorOperand(Value *operand, int vectorWidth) {
    if (operand->getType()->isVectorTy())
      return operand;

    VectorType *vectorType = VectorType::get(operand->getType(), vectorWidth);
    Value *vector = UndefValue::get(vectorType);
    vector = Builder.CreateInsertElement(vector, operand, Builder.getInt32(0));

    std::vector<Constant*> splat;

    for (int i = 0; i < vectorWidth; i++)
      splat.push_back (Builder.getInt32(0));

    Constant *splatVector = ConstantVector::get(splat);

    return Builder.CreateShuffleVector(vector, vector, splatVector);
  }

  Value* getOperand(const Value *OldOperand, ValueMapT &BBMap,
                    ValueMapT *VectorMap = 0) {
    const Instruction *OpInst = dyn_cast<Instruction>(OldOperand);

    if (!OpInst)
      return const_cast<Value*>(OldOperand);

    if (VectorMap && VectorMap->count(OldOperand))
      return (*VectorMap)[OldOperand];

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
    if (getRegion().contains(OpInst->getParent()))
      return NULL;

    return const_cast<Value*>(OldOperand);
  }

  const Type *getVectorPtrTy(const Value *V, int vectorWidth) {
    const PointerType *pointerType = dyn_cast<PointerType>(V->getType());
    assert(pointerType && "PointerType expected");

    const Type *scalarType = pointerType->getElementType();
    VectorType *vectorType = VectorType::get(scalarType, vectorWidth);

    return PointerType::getUnqual(vectorType);
  }

  /// @brief Load a vector from a set of adjacent scalars
  ///
  /// In case a set of scalars is known to be next to each other in memory,
  /// create a vector load that loads those scalars
  ///
  /// %vector_ptr= bitcast double* %p to <4 x double>*
  /// %vec_full = load <4 x double>* %vector_ptr
  ///
  Value *generateStrideOneLoad(const LoadInst *load, ValueMapT &BBMap,
                               int size) {
    const Value *pointer = load->getPointerOperand();
    const Type *vectorPtrType = getVectorPtrTy(pointer, size);
    Value *newPointer = getOperand(pointer, BBMap);
    Value *VectorPtr = Builder.CreateBitCast(newPointer, vectorPtrType,
                                             "vector_ptr");
    LoadInst *VecLoad = Builder.CreateLoad(VectorPtr,
                                        load->getNameStr()
                                        + "_p_vec_full");
    if (!Aligned)
      VecLoad->setAlignment(8);

    return VecLoad;
  }

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
  Value *generateStrideZeroLoad(const LoadInst *load, ValueMapT &BBMap,
                                int size) {
    const Value *pointer = load->getPointerOperand();
    const Type *vectorPtrType = getVectorPtrTy(pointer, 1);
    Value *newPointer = getOperand(pointer, BBMap);
    Value *vectorPtr = Builder.CreateBitCast(newPointer, vectorPtrType,
                                             load->getNameStr() + "_p_vec_p");
    LoadInst *scalarLoad= Builder.CreateLoad(vectorPtr,
                                          load->getNameStr() + "_p_splat_one");

    if (!Aligned)
      scalarLoad->setAlignment(8);

    std::vector<Constant*> splat;

    for (int i = 0; i < size; i++)
      splat.push_back (Builder.getInt32(0));

    Constant *splatVector = ConstantVector::get(splat);

    Value *vectorLoad = Builder.CreateShuffleVector(scalarLoad, scalarLoad,
                                                    splatVector,
                                                    load->getNameStr()
                                                    + "_p_splat");
    return vectorLoad;
  }

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
  Value *generateUnknownStrideLoad(const LoadInst *load,
                                   VectorValueMapT &scalarMaps,
                                   int size) {
    const Value *pointer = load->getPointerOperand();
    VectorType *vectorType = VectorType::get(
      dyn_cast<PointerType>(pointer->getType())->getElementType(), size);

    Value *vector = UndefValue::get(vectorType);

    for (int i = 0; i < size; i++) {
      Value *newPointer = getOperand(pointer, scalarMaps[i]);
      Value *scalarLoad = Builder.CreateLoad(newPointer,
                                             load->getNameStr() + "_p_scalar_");
      vector = Builder.CreateInsertElement(vector, scalarLoad,
                                           Builder.getInt32(i),
                                           load->getNameStr() + "_p_vec_");
    }

    return vector;
  }

  Value *generateScalarLoad(const LoadInst *load, ValueMapT &BBMap) {
    const Value *pointer = load->getPointerOperand();
    Value *newPointer = getOperand(pointer, BBMap);
    Value *scalarLoad = Builder.CreateLoad(newPointer,
                                           load->getNameStr() + "_p_scalar_");
    return scalarLoad;
  }

  /// @brief Load a value (or several values as a vector) from memory.
  void generateLoad(const LoadInst *load, ValueMapT &vectorMap,
                    VectorValueMapT &scalarMaps, int vectorWidth) {

    if (scalarMaps.size() == 1) {
      scalarMaps[0][load] = generateScalarLoad(load, scalarMaps[0]);
      return;
    }

    Value *newLoad;

    MemoryAccess &Access = statement.getAccessFor(load);

    assert(scatteringDomain && "No scattering domain available");

    if (Access.isStrideZero(scatteringDomain))
      newLoad = generateStrideZeroLoad(load, scalarMaps[0], vectorWidth);
    else if (Access.isStrideOne(scatteringDomain))
      newLoad = generateStrideOneLoad(load, scalarMaps[0], vectorWidth);
    else
      newLoad = generateUnknownStrideLoad(load, scalarMaps, vectorWidth);

    vectorMap[load] = newLoad;
  }

  void copyInstruction(const Instruction *Inst, ValueMapT &BBMap,
                       ValueMapT &vectorMap, VectorValueMapT &scalarMaps,
                       int vectorDimension, int vectorWidth) {
    // If this instruction is already in the vectorMap, a vector instruction
    // was already issued, that calculates the values of all dimensions. No
    // need to create any more instructions.
    if (vectorMap.count(Inst))
      return;

    // Terminator instructions control the control flow. They are explicitally
    // expressed in the clast and do not need to be copied.
    if (Inst->isTerminator())
      return;

    if (const LoadInst *load = dyn_cast<LoadInst>(Inst)) {
      generateLoad(load, vectorMap, scalarMaps, vectorWidth);
      return;
    }

    if (const BinaryOperator *binaryInst = dyn_cast<BinaryOperator>(Inst)) {
      Value *opZero = Inst->getOperand(0);
      Value *opOne = Inst->getOperand(1);

      // This is an old instruction that can be ignored.
      if (!opZero && !opOne)
        return;

      bool isVectorOp = vectorMap.count(opZero) || vectorMap.count(opOne);

      if (isVectorOp && vectorDimension > 0)
        return;

      Value *newOpZero, *newOpOne;
      newOpZero = getOperand(opZero, BBMap, &vectorMap);
      newOpOne = getOperand(opOne, BBMap, &vectorMap);


      std::string name;
      if (isVectorOp) {
        newOpZero = makeVectorOperand(newOpZero, vectorWidth);
        newOpOne = makeVectorOperand(newOpOne, vectorWidth);
        name =  Inst->getNameStr() + "p_vec";
      } else
        name = Inst->getNameStr() + "p_sca";

      Value *newInst = Builder.CreateBinOp(binaryInst->getOpcode(), newOpZero,
                                           newOpOne, name);
      if (isVectorOp)
        vectorMap[Inst] = newInst;
      else
        BBMap[Inst] = newInst;

      return;
    }

    if (const StoreInst *store = dyn_cast<StoreInst>(Inst)) {
      if (vectorMap.count(store->getValueOperand()) > 0) {

        // We only need to generate one store if we are in vector mode.
        if (vectorDimension > 0)
          return;

        MemoryAccess &Access = statement.getAccessFor(store);

        assert(scatteringDomain && "No scattering domain available");

        const Value *pointer = store->getPointerOperand();
        Value *vector = getOperand(store->getValueOperand(), BBMap, &vectorMap);

        if (Access.isStrideOne(scatteringDomain)) {
          const Type *vectorPtrType = getVectorPtrTy(pointer, vectorWidth);
          Value *newPointer = getOperand(pointer, BBMap, &vectorMap);

          Value *VectorPtr = Builder.CreateBitCast(newPointer, vectorPtrType,
                                                   "vector_ptr");
          StoreInst *Store = Builder.CreateStore(vector, VectorPtr);

          if (!Aligned)
            Store->setAlignment(8);
        } else {
          for (unsigned i = 0; i < scalarMaps.size(); i++) {
            Value *scalar = Builder.CreateExtractElement(vector,
                                                         Builder.getInt32(i));
            Value *newPointer = getOperand(pointer, scalarMaps[i]);
            Builder.CreateStore(scalar, newPointer);
          }
        }

        return;
      }
    }

    Instruction *NewInst = Inst->clone();

    // Copy the operands in temporary vector, as an in place update
    // fails if an instruction is referencing the same operand twice.
    std::vector<Value*> Operands(NewInst->op_begin(), NewInst->op_end());

    // Replace old operands with the new ones.
    for (std::vector<Value*>::iterator UI = Operands.begin(),
         UE = Operands.end(); UI != UE; ++UI) {
      Value *newOperand = getOperand(*UI, BBMap);

      if (!newOperand) {
        assert(!isa<StoreInst>(NewInst)
               && "Store instructions are always needed!");
        delete NewInst;
        return;
      }

      NewInst->replaceUsesOfWith(*UI, newOperand);
    }

    Builder.Insert(NewInst);
    BBMap[Inst] = NewInst;

    if (!NewInst->getType()->isVoidTy())
      NewInst->setName("p_" + Inst->getName());
  }

  int getVectorSize() {
    return ValueMaps.size();
  }

  bool isVectorBlock() {
    return getVectorSize() > 1;
  }

  // Insert a copy of a basic block in the newly generated code.
  //
  // @param Builder The builder used to insert the code. It also specifies
  //                where to insert the code.
  // @param BB      The basic block to copy
  // @param VMap    A map returning for any old value its new equivalent. This
  //                is used to update the operands of the statements.
  //                For new statements a relation old->new is inserted in this
  //                map.
  void copyBB(BasicBlock *BB, DominatorTree *DT) {
    Function *F = Builder.GetInsertBlock()->getParent();
    LLVMContext &Context = F->getContext();
    BasicBlock *CopyBB = BasicBlock::Create(Context,
                                            "polly.stmt_" + BB->getNameStr(),
                                            F);
    Builder.CreateBr(CopyBB);
    DT->addNewBlock(CopyBB, Builder.GetInsertBlock());
    Builder.SetInsertPoint(CopyBB);

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
    VectorValueMapT scalarBlockMap(getVectorSize());
    ValueMapT vectorBlockMap;

    for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end();
         II != IE; ++II)
      for (int i = 0; i < getVectorSize(); i++) {
        if (isVectorBlock())
          VMap = ValueMaps[i];

        copyInstruction(II, scalarBlockMap[i], vectorBlockMap,
                        scalarBlockMap, i, getVectorSize());
      }
  }
};

/// Class to generate LLVM-IR that calculates the value of a clast_expr.
class ClastExpCodeGen {
  IRBuilder<> &Builder;
  const CharMapT *IVS;

  Value *codegen(const clast_name *e, const Type *Ty) {
    CharMapT::const_iterator I = IVS->find(e->name);

    if (I != IVS->end())
      return Builder.CreateSExtOrBitCast(I->second, Ty);
    else
      llvm_unreachable("Clast name not found");
  }

  Value *codegen(const clast_term *e, const Type *Ty) {
    APInt a = APInt_from_MPZ(e->val);

    Value *ConstOne = ConstantInt::get(Builder.getContext(), a);
    ConstOne = Builder.CreateSExtOrBitCast(ConstOne, Ty);

    if (e->var) {
      Value *var = codegen(e->var, Ty);
      return Builder.CreateMul(ConstOne, var);
    }

    return ConstOne;
  }

  Value *codegen(const clast_binary *e, const Type *Ty) {
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
        Value *One = ConstantInt::get(Builder.getInt1Ty(), 1);
        Value *Zero = ConstantInt::get(Builder.getInt1Ty(), 0);
        One = Builder.CreateZExtOrBitCast(One, Ty);
        Zero = Builder.CreateZExtOrBitCast(Zero, Ty);
        Value *Sum1 = Builder.CreateSub(LHS, RHS);
        Value *Sum2 = Builder.CreateAdd(Sum1, One);
        Value *isNegative = Builder.CreateICmpSLT(LHS, Zero);
        Value *Dividend = Builder.CreateSelect(isNegative, Sum2, LHS);
        return Builder.CreateSDiv(Dividend, RHS);
      }
    case clast_bin_cdiv:
      {
        // ceild(n,d) ((n < 0) ? n : (n + d - 1)) / d
        Value *One = ConstantInt::get(Builder.getInt1Ty(), 1);
        Value *Zero = ConstantInt::get(Builder.getInt1Ty(), 0);
        One = Builder.CreateZExtOrBitCast(One, Ty);
        Zero = Builder.CreateZExtOrBitCast(Zero, Ty);
        Value *Sum1 = Builder.CreateAdd(LHS, RHS);
        Value *Sum2 = Builder.CreateSub(Sum1, One);
        Value *isNegative = Builder.CreateICmpSLT(LHS, Zero);
        Value *Dividend = Builder.CreateSelect(isNegative, LHS, Sum2);
        return Builder.CreateSDiv(Dividend, RHS);
      }
    case clast_bin_div:
      return Builder.CreateSDiv(LHS, RHS);
    default:
      llvm_unreachable("Unknown clast binary expression type");
    };
  }

  Value *codegen(const clast_reduction *r, const Type *Ty) {
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
      default:
        llvm_unreachable("Clast unknown reduction type");
      }
    }

    return old;
  }

public:

  // A generator for clast expressions.
  //
  // @param B The IRBuilder that defines where the code to calculate the
  //          clast expressions should be inserted.
  // @param IVMAP A Map that translates strings describing the induction
  //              variables to the Values* that represent these variables
  //              on the LLVM side.
  ClastExpCodeGen(IRBuilder<> &B, CharMapT *IVMap) : Builder(B), IVS(IVMap) {}

  // Generates code to calculate a given clast expression.
  //
  // @param e The expression to calculate.
  // @return The Value that holds the result.
  Value *codegen(const clast_expr *e, const Type *Ty) {
    switch(e->type) {
      case clast_expr_name:
	return codegen((const clast_name *)e, Ty);
      case clast_expr_term:
	return codegen((const clast_term *)e, Ty);
      case clast_expr_bin:
	return codegen((const clast_binary *)e, Ty);
      case clast_expr_red:
	return codegen((const clast_reduction *)e, Ty);
      default:
        llvm_unreachable("Unknown clast expression!");
    }
  }

  // @brief Reset the CharMap.
  //
  // This function is called to reset the CharMap to new one, while generating
  // OpenMP code.
  void setIVS(CharMapT *IVSNew) {
    IVS = IVSNew;
  }

};

class ClastStmtCodeGen {
  // The Scop we code generate.
  Scop *S;
  ScalarEvolution &SE;
  DominatorTree *DT;
  ScopDetection *SD;
  Dependences *DP;
  TargetData *TD;

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

  const std::vector<std::string> &getParallelLoops() {
    return parallelLoops;
  }

  protected:
  void codegen(const clast_assignment *a) {
    (*clastVars)[a->LHS] = ExpGen.codegen(a->RHS,
      TD->getIntPtrType(Builder.getContext()));
  }

  void codegen(const clast_assignment *a, ScopStmt *Statement,
               unsigned Dimension, int vectorDim,
               std::vector<ValueMapT> *VectorVMap = 0) {
    Value *RHS = ExpGen.codegen(a->RHS,
      TD->getIntPtrType(Builder.getContext()));

    assert(!a->LHS && "Statement assignments do not have left hand side");
    const PHINode *PN;
    PN = Statement->getInductionVariableForDimension(Dimension);
    const Value *V = PN;

    if (PN->getNumOperands() == 2)
      V = *(PN->use_begin());

    if (VectorVMap)
      (*VectorVMap)[vectorDim][V] = RHS;

    ValueMap[V] = RHS;
  }

  void codegenSubstitutions(const clast_stmt *Assignment,
                            ScopStmt *Statement, int vectorDim = 0,
                            std::vector<ValueMapT> *VectorVMap = 0) {
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

  void codegen(const clast_user_stmt *u, std::vector<Value*> *IVS = NULL,
               const char *iterator = NULL, isl_set *scatteringDomain = 0) {
    ScopStmt *Statement = (ScopStmt *)u->statement->usr;
    BasicBlock *BB = Statement->getBasicBlock();

    if (u->substitutions)
      codegenSubstitutions(u->substitutions, Statement);

    int vectorDimensions = IVS ? IVS->size() : 1;

    VectorValueMapT VectorValueMap(vectorDimensions);

    if (IVS) {
      assert (u->substitutions && "Substitutions expected!");
      int i = 0;
      for (std::vector<Value*>::iterator II = IVS->begin(), IE = IVS->end();
           II != IE; ++II) {
        (*clastVars)[iterator] = *II;
        codegenSubstitutions(u->substitutions, Statement, i, &VectorValueMap);
        i++;
      }
    }

    BlockGenerator Generator(Builder, ValueMap, VectorValueMap, *Statement,
                             scatteringDomain);
    Generator.copyBB(BB, DT);
  }

  void codegen(const clast_block *b) {
    if (b->body)
      codegen(b->body);
  }

  /// @brief Create a classical sequential loop.
  void codegenForSequential(const clast_for *f, Value *lowerBound = 0,
                                                Value *upperBound = 0) {
    APInt Stride = APInt_from_MPZ(f->stride);
    PHINode *IV;
    Value *IncrementedIV;
    BasicBlock *AfterBB;
    // The value of lowerbound and upperbound will be supplied, if this
    // function is called while generating OpenMP code. Otherwise get
    // the values.
    assert(((lowerBound && upperBound) || (!lowerBound && !upperBound))
                                && "Either give both bounds or none");
    if (lowerBound == 0 || upperBound == 0) {
        lowerBound = ExpGen.codegen(f->LB,
                                    TD->getIntPtrType(Builder.getContext()));
        upperBound = ExpGen.codegen(f->UB,
                                    TD->getIntPtrType(Builder.getContext()));
    }
    createLoop(&Builder, lowerBound, upperBound, Stride, IV, AfterBB,
               IncrementedIV, DT);

    // Add loop iv to symbols.
    (*clastVars)[f->iterator] = IV;

    if (f->body)
      codegen(f->body);

    // Loop is finished, so remove its iv from the live symbols.
    clastVars->erase(f->iterator);

    BasicBlock *HeaderBB = *pred_begin(AfterBB);
    BasicBlock *LastBodyBB = Builder.GetInsertBlock();
    Builder.CreateBr(HeaderBB);
    IV->addIncoming(IncrementedIV, LastBodyBB);
    Builder.SetInsertPoint(AfterBB);
  }

  /// @brief Add a new definition of an openmp subfunction.
  Function* addOpenMPSubfunction(Module *M) {
    Function *F = Builder.GetInsertBlock()->getParent();
    const std::string &Name = F->getNameStr() + ".omp_subfn";

    std::vector<const Type*> Arguments(1, Builder.getInt8PtrTy());
    FunctionType *FT = FunctionType::get(Builder.getVoidTy(), Arguments, false);
    Function *FN = Function::Create(FT, Function::InternalLinkage, Name, M);
    // Do not run any polly pass on the new function.
    SD->markFunctionAsInvalid(FN);

    Function::arg_iterator AI = FN->arg_begin();
    AI->setName("omp.userContext");

    return FN;
  }

  /// @brief Add values to the OpenMP structure.
  ///
  /// Create the subfunction structure and add the values from the list.
  Value *addValuesToOpenMPStruct(SetVector<Value*> OMPDataVals,
                                 Function *SubFunction) {
    Module *M = Builder.GetInsertBlock()->getParent()->getParent();
    std::vector<const Type*> structMembers;

    // Create the structure.
    for (unsigned i = 0; i < OMPDataVals.size(); i++)
      structMembers.push_back(OMPDataVals[i]->getType());

    const std::string &Name = SubFunction->getNameStr() + ".omp.userContext";
    StructType *structTy = StructType::get(Builder.getContext(),
                                           structMembers);
    M->addTypeName(Name, structTy);

    // Store the values into the structure.
    Value *structData = Builder.CreateAlloca(structTy, 0, "omp.userContext");
    for (unsigned i = 0; i < OMPDataVals.size(); i++) {
      Value *storeAddr = Builder.CreateStructGEP(structData, i);
      Builder.CreateStore(OMPDataVals[i], storeAddr);
    }

    return structData;
  }

  /// @brief Create OpenMP structure values.
  ///
  /// Create a list of values that has to be stored into the subfuncition
  /// structure.
  SetVector<Value*> createOpenMPStructValues() {
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

  /// @brief Extract the values from the subfunction parameter.
  ///
  /// Extract the values from the subfunction parameter and update the clast
  /// variables to point to the new values.
  void extractValuesFromOpenMPStruct(CharMapT *clastVarsOMP,
                                     SetVector<Value*> OMPDataVals,
                                     Value *userContext) {
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

  /// @brief Add body to the subfunction.
  void addOpenMPSubfunctionBody(Function *FN, const clast_for *f,
                                Value *structData,
                                SetVector<Value*> OMPDataVals) {
    Module *M = Builder.GetInsertBlock()->getParent()->getParent();
    LLVMContext &Context = FN->getContext();
    const IntegerType *intPtrTy = TD->getIntPtrType(Context);

    // Store the previous basic block.
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

    codegenForSequential(f, lowerBound, upperBound);

    // Restore the old clastVars.
    clastVars = oldClastVars;
    ExpGen.setIVS(oldClastVars);

    Builder.CreateBr(checkNextBB);

    // Add code to terminate this openmp subfunction.
    Builder.SetInsertPoint(ExitBB);
    Function *endnowaitFunction = M->getFunction("GOMP_loop_end_nowait");
    Builder.CreateCall(endnowaitFunction);
    Builder.CreateRetVoid();

    // Restore the builder back to previous basic block.
    Builder.SetInsertPoint(PrevBB);
  }

  /// @brief Create an OpenMP parallel for loop.
  ///
  /// This loop reflects a loop as if it would have been created by an OpenMP
  /// statement.
  void codegenForOpenMP(const clast_for *f) {
    Module *M = Builder.GetInsertBlock()->getParent()->getParent();
    const IntegerType *intPtrTy = TD->getIntPtrType(Builder.getContext());

    Function *SubFunction = addOpenMPSubfunction(M);
    SetVector<Value*> OMPDataVals = createOpenMPStructValues();
    Value *structData = addValuesToOpenMPStruct(OMPDataVals, SubFunction);

    addOpenMPSubfunctionBody(SubFunction, f, structData, OMPDataVals);

    // Create call for GOMP_parallel_loop_runtime_start.
    Value *subfunctionParam = Builder.CreateBitCast(structData,
                                                    Builder.getInt8PtrTy(),
                                                    "omp_data");

    Value *numberOfThreads = Builder.getInt32(0);
    Value *lowerBound = ExpGen.codegen(f->LB, intPtrTy);
    Value *upperBound = ExpGen.codegen(f->UB, intPtrTy);

    // Add one as the upper bound provided by openmp is a < comparison
    // whereas the codegenForSequential function creates a <= comparison.
    upperBound = Builder.CreateAdd(upperBound, ConstantInt::get(intPtrTy, 1));
    APInt APStride = APInt_from_MPZ(f->stride);
    Value *stride = ConstantInt::get(intPtrTy,
                                     APStride.zext(intPtrTy->getBitWidth()));

    SmallVector<Value *, 6> Arguments;
    Arguments.push_back(SubFunction);
    Arguments.push_back(subfunctionParam);
    Arguments.push_back(numberOfThreads);
    Arguments.push_back(lowerBound);
    Arguments.push_back(upperBound);
    Arguments.push_back(stride);

    Function *parallelStartFunction =
      M->getFunction("GOMP_parallel_loop_runtime_start");
    Builder.CreateCall(parallelStartFunction, Arguments.begin(),
                       Arguments.end());

    // Create call to the subfunction.
    Builder.CreateCall(SubFunction, subfunctionParam);

    // Create call for GOMP_parallel_end.
    Function *FN = M->getFunction("GOMP_parallel_end");
    Builder.CreateCall(FN);
  }

  bool isInnermostLoop(const clast_for *f) {
    const clast_stmt *stmt = f->body;

    while (stmt) {
      if (!CLAST_STMT_IS_A(stmt, stmt_user))
        return false;

      stmt = stmt->next;
    }

    return true;
  }

  /// @brief Get the number of loop iterations for this loop.
  /// @param f The clast for loop to check.
  int getNumberOfIterations(const clast_for *f) {
    isl_set *loopDomain = isl_set_copy(isl_set_from_cloog_domain(f->domain));
    isl_set *tmp = isl_set_copy(loopDomain);

    // Calculate a map similar to the identity map, but with the last input
    // and output dimension not related.
    //  [i0, i1, i2, i3] -> [i0, i1, i2, o0]
    isl_dim *dim = isl_set_get_dim(loopDomain);
    dim = isl_dim_drop_outputs(dim, isl_set_n_dim(loopDomain) - 2, 1);
    dim = isl_dim_map_from_set(dim);
    isl_map *identity = isl_map_identity(dim);
    identity = isl_map_add_dims(identity, isl_dim_in, 1);
    identity = isl_map_add_dims(identity, isl_dim_out, 1);

    isl_map *map = isl_map_from_domain_and_range(tmp, loopDomain);
    map = isl_map_intersect(map, identity);

    isl_map *lexmax = isl_map_lexmax(isl_map_copy(map));
    isl_map *lexmin = isl_map_lexmin(isl_map_copy(map));
    isl_map *sub = isl_map_sum(lexmax, isl_map_neg(lexmin));

    isl_set *elements = isl_map_range(sub);

    if (!isl_set_is_singleton(elements))
      return -1;

    isl_point *p = isl_set_sample_point(elements);

    isl_int v;
    isl_int_init(v);
    isl_point_get_coordinate(p, isl_dim_set, isl_set_n_dim(loopDomain) - 1, &v);
    int numberIterations = isl_int_get_si(v);
    isl_int_clear(v);

    return (numberIterations) / isl_int_get_si(f->stride) + 1;
  }

  /// @brief Create vector instructions for this loop.
  void codegenForVector(const clast_for *f) {
    DEBUG(dbgs() << "Vectorizing loop '" << f->iterator << "'\n";);
    int vectorWidth = getNumberOfIterations(f);

    Value *LB = ExpGen.codegen(f->LB,
      TD->getIntPtrType(Builder.getContext()));

    APInt Stride = APInt_from_MPZ(f->stride);
    const IntegerType *LoopIVType = dyn_cast<IntegerType>(LB->getType());
    Stride =  Stride.zext(LoopIVType->getBitWidth());
    Value *StrideValue = ConstantInt::get(LoopIVType, Stride);

    std::vector<Value*> IVS(vectorWidth);
    IVS[0] = LB;

    for (int i = 1; i < vectorWidth; i++)
      IVS[i] = Builder.CreateAdd(IVS[i-1], StrideValue, "p_vector_iv");

    isl_set *scatteringDomain = isl_set_from_cloog_domain(f->domain);

    // Add loop iv to symbols.
    (*clastVars)[f->iterator] = LB;

    const clast_stmt *stmt = f->body;

    while (stmt) {
      codegen((const clast_user_stmt *)stmt, &IVS, f->iterator,
              scatteringDomain);
      stmt = stmt->next;
    }

    // Loop is finished, so remove its iv from the live symbols.
    clastVars->erase(f->iterator);
  }

  void codegen(const clast_for *f) {
    if (Vector && isInnermostLoop(f) && DP->isParallelFor(f)
        && (-1 != getNumberOfIterations(f))
        && (getNumberOfIterations(f) <= 16)) {
      codegenForVector(f);
    } else if (OpenMP && !parallelCodeGeneration && DP->isParallelFor(f)) {
      parallelCodeGeneration = true;
      parallelLoops.push_back(f->iterator);
      codegenForOpenMP(f);
      parallelCodeGeneration = false;
    } else
      codegenForSequential(f);
  }

  Value *codegen(const clast_equation *eq) {
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

  void codegen(const clast_guard *g) {
    Function *F = Builder.GetInsertBlock()->getParent();
    LLVMContext &Context = F->getContext();
    BasicBlock *ThenBB = BasicBlock::Create(Context, "polly.then", F);
    BasicBlock *MergeBB = BasicBlock::Create(Context, "polly.merge", F);
    DT->addNewBlock(ThenBB, Builder.GetInsertBlock());
    DT->addNewBlock(MergeBB, Builder.GetInsertBlock());

    Value *Predicate = codegen(&(g->eq[0]));

    for (int i = 1; i < g->n; ++i) {
      Value *TmpPredicate = codegen(&(g->eq[i]));
      Predicate = Builder.CreateAnd(Predicate, TmpPredicate);
    }

    Builder.CreateCondBr(Predicate, ThenBB, MergeBB);
    Builder.SetInsertPoint(ThenBB);

    codegen(g->then);

    Builder.CreateBr(MergeBB);
    Builder.SetInsertPoint(MergeBB);
  }

  void codegen(const clast_stmt *stmt) {
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

  void addParameters(const CloogNames *names) {
    SCEVExpander Rewriter(SE);

    // Create an instruction that specifies the location where the parameters
    // are expanded.
    CastInst::CreateIntegerCast(ConstantInt::getTrue(Builder.getContext()),
                                  Builder.getInt16Ty(), false, "insertInst",
                                  Builder.GetInsertBlock());

    int i = 0;
    for (Scop::param_iterator PI = S->param_begin(), PE = S->param_end();
         PI != PE; ++PI) {
      assert(i < names->nb_parameters && "Not enough parameter names");

      const SCEV *Param = *PI;
      const Type *Ty = Param->getType();

      Instruction *insertLocation = --(Builder.GetInsertBlock()->end());
      Value *V = Rewriter.expandCodeFor(Param, Ty, insertLocation);
      (*clastVars)[names->parameters[i]] = V;

      ++i;
    }
  }

  public:
  void codegen(const clast_root *r) {
    clastVars = new CharMapT();
    addParameters(r->names);
    ExpGen.setIVS(clastVars);

    parallelCodeGeneration = false;

    const clast_stmt *stmt = (const clast_stmt*) r;
    if (stmt->next)
      codegen(stmt->next);

    delete clastVars;
  }

  ClastStmtCodeGen(Scop *scop, ScalarEvolution &se, DominatorTree *dt,
                   ScopDetection *sd, Dependences *dp, TargetData *td,
                   IRBuilder<> &B) :
    S(scop), SE(se), DT(dt), SD(sd), DP(dp), TD(td), Builder(B),
    ExpGen(Builder, NULL) {}

};
}

namespace {
class CodeGeneration : public ScopPass {
  Region *region;
  Scop *S;
  DominatorTree *DT;
  ScalarEvolution *SE;
  ScopDetection *SD;
  CloogInfo *C;
  LoopInfo *LI;
  TargetData *TD;

  std::vector<std::string> parallelLoops;

  public:
  static char ID;

  CodeGeneration() : ScopPass(ID) {}

  void createSeSeEdges(Region *R) {
    BasicBlock *newEntry = createSingleEntryEdge(R, this);

    for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI)
      if ((*SI)->getBasicBlock() == R->getEntry())
        (*SI)->setBasicBlock(newEntry);

    createSingleExitEdge(R, this);
  }


  // Adding prototypes required if OpenMP is enabled.
  void addOpenMPDefinitions(IRBuilder<> &Builder)
  {
    Module *M = Builder.GetInsertBlock()->getParent()->getParent();
    LLVMContext &Context = Builder.getContext();
    const IntegerType *intPtrTy = TD->getIntPtrType(Context);

    if (!M->getFunction("GOMP_parallel_end")) {
      FunctionType *FT = FunctionType::get(Type::getVoidTy(Context), false);
      Function::Create(FT, Function::ExternalLinkage, "GOMP_parallel_end", M);
    }

    if (!M->getFunction("GOMP_parallel_loop_runtime_start")) {
      // Type of first argument.
      std::vector<const Type*> Arguments(1, Builder.getInt8PtrTy());
      FunctionType *FnArgTy = FunctionType::get(Builder.getVoidTy(), Arguments,
                                                false);
      PointerType *FnPtrTy = PointerType::getUnqual(FnArgTy);

      std::vector<const Type*> args;
      args.push_back(FnPtrTy);
      args.push_back(Builder.getInt8PtrTy());
      args.push_back(Builder.getInt32Ty());
      args.push_back(intPtrTy);
      args.push_back(intPtrTy);
      args.push_back(intPtrTy);

      FunctionType *type = FunctionType::get(Builder.getVoidTy(), args, false);
      Function::Create(type, Function::ExternalLinkage,
                       "GOMP_parallel_loop_runtime_start", M);
    }

    if (!M->getFunction("GOMP_loop_runtime_next")) {
      PointerType *intLongPtrTy = PointerType::getUnqual(intPtrTy);

      std::vector<const Type*> args;
      args.push_back(intLongPtrTy);
      args.push_back(intLongPtrTy);

      FunctionType *type = FunctionType::get(Builder.getInt8Ty(), args, false);
      Function::Create(type, Function::ExternalLinkage,
                       "GOMP_loop_runtime_next", M);
    }

    if (!M->getFunction("GOMP_loop_end_nowait")) {
      FunctionType *FT = FunctionType::get(Builder.getVoidTy(),
                                           std::vector<const Type*>(), false);
      Function::Create(FT, Function::ExternalLinkage,
		       "GOMP_loop_end_nowait", M);
    }
  }

  bool runOnScop(Scop &scop) {
    S = &scop;
    region = &S->getRegion();
    Region *R = region;
    DT = &getAnalysis<DominatorTree>();
    Dependences *DP = &getAnalysis<Dependences>();
    SE = &getAnalysis<ScalarEvolution>();
    LI = &getAnalysis<LoopInfo>();
    C = &getAnalysis<CloogInfo>();
    SD = &getAnalysis<ScopDetection>();
    TD = &getAnalysis<TargetData>();

    Function *F = R->getEntry()->getParent();

    parallelLoops.clear();

    if (CodegenOnly != "" && CodegenOnly != F->getNameStr()) {
      errs() << "Codegenerating only function '" << CodegenOnly
        << "' skipping '" << F->getNameStr() << "' \n";
      return false;
    }

    createSeSeEdges(R);

    // Create a basic block in which to start code generation.
    BasicBlock *PollyBB = BasicBlock::Create(F->getContext(), "pollyBB", F);
    IRBuilder<> Builder(PollyBB);
    DT->addNewBlock(PollyBB, R->getEntry());

    const clast_root *clast = (const clast_root *) C->getClast();

    ClastStmtCodeGen CodeGen(S, *SE, DT, SD, DP, TD, Builder);

    if (OpenMP)
      addOpenMPDefinitions(Builder);

    CodeGen.codegen(clast);

    // Save the parallel loops generated.
    parallelLoops.insert(parallelLoops.begin(),
                         CodeGen.getParallelLoops().begin(),
                         CodeGen.getParallelLoops().end());

    BasicBlock *AfterScop = *pred_begin(R->getExit());
    Builder.CreateBr(AfterScop);

    BasicBlock *successorBlock = *succ_begin(R->getEntry());

    // Update old PHI nodes to pass LLVM verification.
    std::vector<PHINode*> PHINodes;
    for (BasicBlock::iterator SI = successorBlock->begin(),
         SE = successorBlock->getFirstNonPHI(); SI != SE; ++SI) {
      PHINode *PN = static_cast<PHINode*>(&*SI);
      PHINodes.push_back(PN);
    }

    for (std::vector<PHINode*>::iterator PI = PHINodes.begin(),
         PE = PHINodes.end(); PI != PE; ++PI)
      (*PI)->removeIncomingValue(R->getEntry());

    DT->changeImmediateDominator(AfterScop, Builder.GetInsertBlock());

    BasicBlock *OldRegionEntry = *succ_begin(R->getEntry());

    // Enable the new polly code.
    R->getEntry()->getTerminator()->setSuccessor(0, PollyBB);

    // Remove old Scop nodes from dominator tree.
    std::vector<DomTreeNode*> ToVisit;
    std::vector<DomTreeNode*> Visited;
    ToVisit.push_back(DT->getNode(OldRegionEntry));

    while (!ToVisit.empty()) {
      DomTreeNode *Node = ToVisit.back();

      ToVisit.pop_back();

      if (AfterScop == Node->getBlock())
        continue;

      Visited.push_back(Node);

      std::vector<DomTreeNode*> Children = Node->getChildren();
      ToVisit.insert(ToVisit.end(), Children.begin(), Children.end());
    }

    for (std::vector<DomTreeNode*>::reverse_iterator I = Visited.rbegin(),
         E = Visited.rend(); I != E; ++I)
      DT->eraseNode((*I)->getBlock());

    R->getParent()->removeSubRegion(R);

    // And forget the Scop if we remove the region.
    SD->forgetScop(*R);

    return false;
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
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<LoopInfo>();
    AU.addRequired<RegionInfo>();
    AU.addRequired<ScopDetection>();
    AU.addRequired<ScopInfo>();
    AU.addRequired<TargetData>();

    AU.addPreserved<CloogInfo>();
    AU.addPreserved<Dependences>();
    AU.addPreserved<LoopInfo>();
    AU.addPreserved<DominatorTree>();
    AU.addPreserved<PostDominatorTree>();
    AU.addPreserved<ScopDetection>();
    AU.addPreserved<ScalarEvolution>();
    AU.addPreserved<RegionInfo>();
    AU.addPreserved<TempScopInfo>();
    AU.addPreserved<ScopInfo>();
    AU.addPreservedID(IndependentBlocksID);
  }
};
}

char CodeGeneration::ID = 1;

static RegisterPass<CodeGeneration>
Z("polly-codegen", "Polly - Create LLVM-IR from the polyhedral information");

Pass* polly::createCodeGenerationPass() {
  return new CodeGeneration();
}

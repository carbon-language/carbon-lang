//===------ MemoryBuiltins.cpp - Identify calls to memory builtins --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions identifies calls to builtin functions that allocate
// or free memory.  
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ConstantFolding.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  malloc Call Utility Functions.
//

/// isMalloc - Returns true if the the value is either a malloc call or a
/// bitcast of the result of a malloc call.
bool llvm::isMalloc(const Value* I) {
  return extractMallocCall(I) || extractMallocCallFromBitCast(I);
}

static bool isMallocCall(const CallInst *CI) {
  if (!CI)
    return false;

  const Module* M = CI->getParent()->getParent()->getParent();
  Function *MallocFunc = M->getFunction("malloc");

  if (CI->getOperand(0) != MallocFunc)
    return false;

  // Check malloc prototype.
  // FIXME: workaround for PR5130, this will be obsolete when a nobuiltin 
  // attribute will exist.
  const FunctionType *FTy = MallocFunc->getFunctionType();
  if (FTy->getNumParams() != 1)
    return false;
  if (IntegerType *ITy = dyn_cast<IntegerType>(FTy->param_begin()->get())) {
    if (ITy->getBitWidth() != 32 && ITy->getBitWidth() != 64)
      return false;
    return true;
  }

  return false;
}

/// extractMallocCall - Returns the corresponding CallInst if the instruction
/// is a malloc call.  Since CallInst::CreateMalloc() only creates calls, we
/// ignore InvokeInst here.
const CallInst* llvm::extractMallocCall(const Value* I) {
  const CallInst *CI = dyn_cast<CallInst>(I);
  return (isMallocCall(CI)) ? CI : NULL;
}

CallInst* llvm::extractMallocCall(Value* I) {
  CallInst *CI = dyn_cast<CallInst>(I);
  return (isMallocCall(CI)) ? CI : NULL;
}

static bool isBitCastOfMallocCall(const BitCastInst* BCI) {
  if (!BCI)
    return false;
    
  return isMallocCall(dyn_cast<CallInst>(BCI->getOperand(0)));
}

/// extractMallocCallFromBitCast - Returns the corresponding CallInst if the
/// instruction is a bitcast of the result of a malloc call.
CallInst* llvm::extractMallocCallFromBitCast(Value* I) {
  BitCastInst *BCI = dyn_cast<BitCastInst>(I);
  return (isBitCastOfMallocCall(BCI)) ? cast<CallInst>(BCI->getOperand(0))
                                      : NULL;
}

const CallInst* llvm::extractMallocCallFromBitCast(const Value* I) {
  const BitCastInst *BCI = dyn_cast<BitCastInst>(I);
  return (isBitCastOfMallocCall(BCI)) ? cast<CallInst>(BCI->getOperand(0))
                                      : NULL;
}

/// isConstantOne - Return true only if val is constant int 1.
static bool isConstantOne(Value *val) {
  return isa<ConstantInt>(val) && cast<ConstantInt>(val)->isOne();
}

static Value* isArrayMallocHelper(const CallInst *CI, LLVMContext &Context,
                                  const TargetData* TD) {
  if (!CI)
    return NULL;

  // Type must be known to determine array size.
  const Type* T = getMallocAllocatedType(CI);
  if (!T)
    return NULL;

  Value* MallocArg = CI->getOperand(1);
  ConstantExpr* CO = dyn_cast<ConstantExpr>(MallocArg);
  BinaryOperator* BO = dyn_cast<BinaryOperator>(MallocArg);

  Constant* ElementSize = ConstantExpr::getSizeOf(T);
  ElementSize = ConstantExpr::getTruncOrBitCast(ElementSize, 
                                                MallocArg->getType());
  Constant *FoldedElementSize =
   ConstantFoldConstantExpression(cast<ConstantExpr>(ElementSize), Context, TD);

  // First, check if CI is a non-array malloc.
  if (CO && ((CO == ElementSize) ||
             (FoldedElementSize && (CO == FoldedElementSize))))
    // Match CreateMalloc's use of constant 1 array-size for non-array mallocs.
    return ConstantInt::get(MallocArg->getType(), 1);

  // Second, check if CI is an array malloc whose array size can be determined.
  if (isConstantOne(ElementSize) || 
      (FoldedElementSize && isConstantOne(FoldedElementSize)))
    return MallocArg;

  if (!CO && !BO)
    return NULL;

  Value* Op0 = NULL;
  Value* Op1 = NULL;
  unsigned Opcode = 0;
  if (CO && ((CO->getOpcode() == Instruction::Mul) || 
             (CO->getOpcode() == Instruction::Shl))) {
    Op0 = CO->getOperand(0);
    Op1 = CO->getOperand(1);
    Opcode = CO->getOpcode();
  }
  if (BO && ((BO->getOpcode() == Instruction::Mul) || 
             (BO->getOpcode() == Instruction::Shl))) {
    Op0 = BO->getOperand(0);
    Op1 = BO->getOperand(1);
    Opcode = BO->getOpcode();
  }

  // Determine array size if malloc's argument is the product of a mul or shl.
  if (Op0) {
    if (Opcode == Instruction::Mul) {
      if ((Op1 == ElementSize) ||
          (FoldedElementSize && (Op1 == FoldedElementSize)))
        // ArraySize * ElementSize
        return Op0;
      if ((Op0 == ElementSize) ||
          (FoldedElementSize && (Op0 == FoldedElementSize)))
        // ElementSize * ArraySize
        return Op1;
    }
    if (Opcode == Instruction::Shl) {
      ConstantInt* Op1Int = dyn_cast<ConstantInt>(Op1);
      if (!Op1Int) return NULL;
      Value* Op1Pow = ConstantInt::get(Op1->getType(),
                                       pow(2, Op1Int->getZExtValue()));
      if (Op0 == ElementSize || (FoldedElementSize && Op0 == FoldedElementSize))
        // ArraySize << log2(ElementSize)
        return Op1Pow;
      if (Op1Pow == ElementSize ||
        (FoldedElementSize && Op1Pow == FoldedElementSize))
        // ElementSize << log2(ArraySize)
        return Op0;
    }
  }

  // We could not determine the malloc array size from MallocArg.
  return NULL;
}

/// isArrayMalloc - Returns the corresponding CallInst if the instruction 
/// is a call to malloc whose array size can be determined and the array size
/// is not constant 1.  Otherwise, return NULL.
CallInst* llvm::isArrayMalloc(Value* I, LLVMContext &Context,
                              const TargetData* TD) {
  CallInst *CI = extractMallocCall(I);
  Value* ArraySize = isArrayMallocHelper(CI, Context, TD);

  if (ArraySize &&
      ArraySize != ConstantInt::get(CI->getOperand(1)->getType(), 1))
    return CI;

  // CI is a non-array malloc or we can't figure out that it is an array malloc.
  return NULL;
}

const CallInst* llvm::isArrayMalloc(const Value* I, LLVMContext &Context,
                                    const TargetData* TD) {
  const CallInst *CI = extractMallocCall(I);
  Value* ArraySize = isArrayMallocHelper(CI, Context, TD);

  if (ArraySize &&
      ArraySize != ConstantInt::get(CI->getOperand(1)->getType(), 1))
    return CI;

  // CI is a non-array malloc or we can't figure out that it is an array malloc.
  return NULL;
}

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// This PointerType is the result type of the call's only bitcast use.
/// If there is no unique bitcast use, then return NULL.
const PointerType* llvm::getMallocType(const CallInst* CI) {
  assert(isMalloc(CI) && "GetMallocType and not malloc call");
  
  const BitCastInst* BCI = NULL;
  
  // Determine if CallInst has a bitcast use.
  for (Value::use_const_iterator UI = CI->use_begin(), E = CI->use_end();
       UI != E; )
    if ((BCI = dyn_cast<BitCastInst>(cast<Instruction>(*UI++))))
      break;

  // Malloc call has 1 bitcast use and no other uses, so type is the bitcast's
  // destination type.
  if (BCI && CI->hasOneUse())
    return cast<PointerType>(BCI->getDestTy());

  // Malloc call was not bitcast, so type is the malloc function's return type.
  if (!BCI)
    return cast<PointerType>(CI->getType());

  // Type could not be determined.
  return NULL;
}

/// getMallocAllocatedType - Returns the Type allocated by malloc call. This
/// Type is the result type of the call's only bitcast use. If there is no
/// unique bitcast use, then return NULL.
const Type* llvm::getMallocAllocatedType(const CallInst* CI) {
  const PointerType* PT = getMallocType(CI);
  return PT ? PT->getElementType() : NULL;
}

/// getMallocArraySize - Returns the array size of a malloc call.  If the 
/// argument passed to malloc is a multiple of the size of the malloced type,
/// then return that multiple.  For non-array mallocs, the multiple is
/// constant 1.  Otherwise, return NULL for mallocs whose array size cannot be
/// determined.
Value* llvm::getMallocArraySize(CallInst* CI, LLVMContext &Context,
                                const TargetData* TD) {
  return isArrayMallocHelper(CI, Context, TD);
}

//===----------------------------------------------------------------------===//
//  free Call Utility Functions.
//

/// isFreeCall - Returns true if the the value is a call to the builtin free()
bool llvm::isFreeCall(const Value* I) {
  const CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI)
    return false;

  const Module* M = CI->getParent()->getParent()->getParent();
  Function *FreeFunc = M->getFunction("free");

  if (CI->getOperand(0) != FreeFunc)
    return false;

  // Check free prototype.
  // FIXME: workaround for PR5130, this will be obsolete when a nobuiltin 
  // attribute will exist.
  const FunctionType *FTy = FreeFunc->getFunctionType();
  if (FTy->getReturnType() != Type::getVoidTy(M->getContext()))
    return false;
  if (FTy->getNumParams() != 1)
    return false;
  if (FTy->param_begin()->get() != Type::getInt8PtrTy(M->getContext()))
    return false;

  return true;
}

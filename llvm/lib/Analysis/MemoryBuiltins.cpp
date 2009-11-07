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
#include "llvm/Target/TargetData.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  malloc Call Utility Functions.
//

/// isMalloc - Returns true if the the value is either a malloc call or a
/// bitcast of the result of a malloc call.
bool llvm::isMalloc(const Value *I) {
  return extractMallocCall(I) || extractMallocCallFromBitCast(I);
}

static bool isMallocCall(const CallInst *CI) {
  if (!CI)
    return false;

  Function *Callee = CI->getCalledFunction();
  if (Callee == 0 || !Callee->isDeclaration() || Callee->getName() != "malloc")
    return false;

  // Check malloc prototype.
  // FIXME: workaround for PR5130, this will be obsolete when a nobuiltin 
  // attribute will exist.
  const FunctionType *FTy = Callee->getFunctionType();
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
const CallInst *llvm::extractMallocCall(const Value *I) {
  const CallInst *CI = dyn_cast<CallInst>(I);
  return (isMallocCall(CI)) ? CI : NULL;
}

CallInst *llvm::extractMallocCall(Value *I) {
  CallInst *CI = dyn_cast<CallInst>(I);
  return (isMallocCall(CI)) ? CI : NULL;
}

static bool isBitCastOfMallocCall(const BitCastInst *BCI) {
  if (!BCI)
    return false;
    
  return isMallocCall(dyn_cast<CallInst>(BCI->getOperand(0)));
}

/// extractMallocCallFromBitCast - Returns the corresponding CallInst if the
/// instruction is a bitcast of the result of a malloc call.
CallInst *llvm::extractMallocCallFromBitCast(Value *I) {
  BitCastInst *BCI = dyn_cast<BitCastInst>(I);
  return (isBitCastOfMallocCall(BCI)) ? cast<CallInst>(BCI->getOperand(0))
                                      : NULL;
}

const CallInst *llvm::extractMallocCallFromBitCast(const Value *I) {
  const BitCastInst *BCI = dyn_cast<BitCastInst>(I);
  return (isBitCastOfMallocCall(BCI)) ? cast<CallInst>(BCI->getOperand(0))
                                      : NULL;
}

/// isConstantOne - Return true only if val is constant int 1.
static bool isConstantOne(Value *val) {
  return isa<ConstantInt>(val) && cast<ConstantInt>(val)->isOne();
}

static Value *isArrayMallocHelper(const CallInst *CI, const TargetData *TD) {
  if (!CI)
    return NULL;

  // The size of the malloc's result type must be known to determine array size.
  const Type *T = getMallocAllocatedType(CI);
  if (!T || !T->isSized() || !TD)
    return NULL;

  Value *MallocArg = CI->getOperand(1);
  const Type *ArgType = MallocArg->getType();
  ConstantExpr *CO = dyn_cast<ConstantExpr>(MallocArg);
  BinaryOperator *BO = dyn_cast<BinaryOperator>(MallocArg);

  unsigned ElementSizeInt = TD->getTypeAllocSize(T);
  if (const StructType *ST = dyn_cast<StructType>(T))
    ElementSizeInt = TD->getStructLayout(ST)->getSizeInBytes();
  Constant *ElementSize = ConstantInt::get(ArgType, ElementSizeInt);

  // First, check if CI is a non-array malloc.
  if (CO && CO == ElementSize)
    // Match CreateMalloc's use of constant 1 array-size for non-array mallocs.
    return ConstantInt::get(ArgType, 1);

  // Second, check if CI is an array malloc whose array size can be determined.
  if (isConstantOne(ElementSize))
    return MallocArg;

  if (ConstantInt *CInt = dyn_cast<ConstantInt>(MallocArg))
    if (CInt->getZExtValue() % ElementSizeInt == 0)
      return ConstantInt::get(ArgType, CInt->getZExtValue() / ElementSizeInt);

  if (!CO && !BO)
    return NULL;

  Value *Op0 = NULL;
  Value *Op1 = NULL;
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
      if (Op1 == ElementSize)
        // ArraySize * ElementSize
        return Op0;
      if (Op0 == ElementSize)
        // ElementSize * ArraySize
        return Op1;
    }
    if (Opcode == Instruction::Shl) {
      ConstantInt *Op1CI = dyn_cast<ConstantInt>(Op1);
      if (!Op1CI) return NULL;
      
      APInt Op1Int = Op1CI->getValue();
      uint64_t BitToSet = Op1Int.getLimitedValue(Op1Int.getBitWidth() - 1);
      Value *Op1Pow = ConstantInt::get(Op1CI->getContext(), 
                                  APInt(Op1Int.getBitWidth(), 0).set(BitToSet));
      if (Op0 == ElementSize)
        // ArraySize << log2(ElementSize)
        return Op1Pow;
      if (Op1Pow == ElementSize)
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
CallInst *llvm::isArrayMalloc(Value *I, const TargetData *TD) {
  CallInst *CI = extractMallocCall(I);
  Value *ArraySize = isArrayMallocHelper(CI, TD);

  if (ArraySize &&
      ArraySize != ConstantInt::get(CI->getOperand(1)->getType(), 1))
    return CI;

  // CI is a non-array malloc or we can't figure out that it is an array malloc.
  return NULL;
}

const CallInst *llvm::isArrayMalloc(const Value *I, const TargetData *TD) {
  const CallInst *CI = extractMallocCall(I);
  Value *ArraySize = isArrayMallocHelper(CI, TD);

  if (ArraySize &&
      ArraySize != ConstantInt::get(CI->getOperand(1)->getType(), 1))
    return CI;

  // CI is a non-array malloc or we can't figure out that it is an array malloc.
  return NULL;
}

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// The PointerType depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
const PointerType *llvm::getMallocType(const CallInst *CI) {
  assert(isMalloc(CI) && "GetMallocType and not malloc call");
  
  const PointerType *MallocType = NULL;
  unsigned NumOfBitCastUses = 0;

  // Determine if CallInst has a bitcast use.
  for (Value::use_const_iterator UI = CI->use_begin(), E = CI->use_end();
       UI != E; )
    if (const BitCastInst *BCI = dyn_cast<BitCastInst>(*UI++)) {
      MallocType = cast<PointerType>(BCI->getDestTy());
      NumOfBitCastUses++;
    }

  // Malloc call has 1 bitcast use, so type is the bitcast's destination type.
  if (NumOfBitCastUses == 1)
    return MallocType;

  // Malloc call was not bitcast, so type is the malloc function's return type.
  if (NumOfBitCastUses == 0)
    return cast<PointerType>(CI->getType());

  // Type could not be determined.
  return NULL;
}

/// getMallocAllocatedType - Returns the Type allocated by malloc call.
/// The Type depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
const Type *llvm::getMallocAllocatedType(const CallInst *CI) {
  const PointerType *PT = getMallocType(CI);
  return PT ? PT->getElementType() : NULL;
}

/// getMallocArraySize - Returns the array size of a malloc call.  If the 
/// argument passed to malloc is a multiple of the size of the malloced type,
/// then return that multiple.  For non-array mallocs, the multiple is
/// constant 1.  Otherwise, return NULL for mallocs whose array size cannot be
/// determined.
Value *llvm::getMallocArraySize(CallInst *CI, const TargetData *TD) {
  return isArrayMallocHelper(CI, TD);
}

//===----------------------------------------------------------------------===//
//  free Call Utility Functions.
//

/// isFreeCall - Returns true if the the value is a call to the builtin free()
bool llvm::isFreeCall(const Value *I) {
  const CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI)
    return false;
  Function *Callee = CI->getCalledFunction();
  if (Callee == 0 || !Callee->isDeclaration() || Callee->getName() != "free")
    return false;

  // Check free prototype.
  // FIXME: workaround for PR5130, this will be obsolete when a nobuiltin 
  // attribute will exist.
  const FunctionType *FTy = Callee->getFunctionType();
  if (!FTy->getReturnType()->isVoidTy())
    return false;
  if (FTy->getNumParams() != 1)
    return false;
  if (FTy->param_begin()->get() != Type::getInt8PtrTy(Callee->getContext()))
    return false;

  return true;
}

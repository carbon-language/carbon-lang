//===-- IntrinsicLowering.cpp - Intrinsic Lowering default implementation -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IntrinsicLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

template <class ArgIt>
static void EnsureFunctionExists(Module &M, const char *Name,
                                 ArgIt ArgBegin, ArgIt ArgEnd,
                                 const Type *RetTy) {
  // Insert a correctly-typed definition now.
  std::vector<const Type *> ParamTys;
  for (ArgIt I = ArgBegin; I != ArgEnd; ++I)
    ParamTys.push_back(I->getType());
  M.getOrInsertFunction(Name, FunctionType::get(RetTy, ParamTys, false));
}

static void EnsureFPIntrinsicsExist(Module &M, Function *Fn,
                                    const char *FName,
                                    const char *DName, const char *LDName) {
  // Insert definitions for all the floating point types.
  switch((int)Fn->arg_begin()->getType()->getTypeID()) {
  case Type::FloatTyID:
    EnsureFunctionExists(M, FName, Fn->arg_begin(), Fn->arg_end(),
                         Type::FloatTy);
    break;
  case Type::DoubleTyID:
    EnsureFunctionExists(M, DName, Fn->arg_begin(), Fn->arg_end(),
                         Type::DoubleTy);
    break;
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
    EnsureFunctionExists(M, LDName, Fn->arg_begin(), Fn->arg_end(),
                         Fn->arg_begin()->getType());
    break;
  }
}

/// ReplaceCallWith - This function is used when we want to lower an intrinsic
/// call to a call of an external function.  This handles hard cases such as
/// when there was already a prototype for the external function, and if that
/// prototype doesn't match the arguments we expect to pass in.
template <class ArgIt>
static CallInst *ReplaceCallWith(const char *NewFn, CallInst *CI,
                                 ArgIt ArgBegin, ArgIt ArgEnd,
                                 const Type *RetTy, Constant *&FCache) {
  if (!FCache) {
    // If we haven't already looked up this function, check to see if the
    // program already contains a function with this name.
    Module *M = CI->getParent()->getParent()->getParent();
    // Get or insert the definition now.
    std::vector<const Type *> ParamTys;
    for (ArgIt I = ArgBegin; I != ArgEnd; ++I)
      ParamTys.push_back((*I)->getType());
    FCache = M->getOrInsertFunction(NewFn,
                                    FunctionType::get(RetTy, ParamTys, false));
  }

  IRBuilder<> Builder(CI->getParent(), CI);
  SmallVector<Value *, 8> Args(ArgBegin, ArgEnd);
  CallInst *NewCI = Builder.CreateCall(FCache, Args.begin(), Args.end());
  NewCI->setName(CI->getName());
  if (!CI->use_empty())
    CI->replaceAllUsesWith(NewCI);
  return NewCI;
}

void IntrinsicLowering::AddPrototypes(Module &M) {
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->isDeclaration() && !I->use_empty())
      switch (I->getIntrinsicID()) {
      default: break;
      case Intrinsic::setjmp:
        EnsureFunctionExists(M, "setjmp", I->arg_begin(), I->arg_end(),
                             Type::Int32Ty);
        break;
      case Intrinsic::longjmp:
        EnsureFunctionExists(M, "longjmp", I->arg_begin(), I->arg_end(),
                             Type::VoidTy);
        break;
      case Intrinsic::siglongjmp:
        EnsureFunctionExists(M, "abort", I->arg_end(), I->arg_end(),
                             Type::VoidTy);
        break;
      case Intrinsic::memcpy:
        M.getOrInsertFunction("memcpy", PointerType::getUnqual(Type::Int8Ty),
                              PointerType::getUnqual(Type::Int8Ty), 
                              PointerType::getUnqual(Type::Int8Ty), 
                              TD.getIntPtrType(), (Type *)0);
        break;
      case Intrinsic::memmove:
        M.getOrInsertFunction("memmove", PointerType::getUnqual(Type::Int8Ty),
                              PointerType::getUnqual(Type::Int8Ty), 
                              PointerType::getUnqual(Type::Int8Ty), 
                              TD.getIntPtrType(), (Type *)0);
        break;
      case Intrinsic::memset:
        M.getOrInsertFunction("memset", PointerType::getUnqual(Type::Int8Ty),
                              PointerType::getUnqual(Type::Int8Ty), 
                              Type::Int32Ty, 
                              TD.getIntPtrType(), (Type *)0);
        break;
      case Intrinsic::sqrt:
        EnsureFPIntrinsicsExist(M, I, "sqrtf", "sqrt", "sqrtl");
        break;
      case Intrinsic::sin:
        EnsureFPIntrinsicsExist(M, I, "sinf", "sin", "sinl");
        break;
      case Intrinsic::cos:
        EnsureFPIntrinsicsExist(M, I, "cosf", "cos", "cosl");
        break;
      case Intrinsic::pow:
        EnsureFPIntrinsicsExist(M, I, "powf", "pow", "powl");
        break;
      case Intrinsic::log:
        EnsureFPIntrinsicsExist(M, I, "logf", "log", "logl");
        break;
      case Intrinsic::log2:
        EnsureFPIntrinsicsExist(M, I, "log2f", "log2", "log2l");
        break;
      case Intrinsic::log10:
        EnsureFPIntrinsicsExist(M, I, "log10f", "log10", "log10l");
        break;
      case Intrinsic::exp:
        EnsureFPIntrinsicsExist(M, I, "expf", "exp", "expl");
        break;
      case Intrinsic::exp2:
        EnsureFPIntrinsicsExist(M, I, "exp2f", "exp2", "exp2l");
        break;
      }
}

/// LowerBSWAP - Emit the code to lower bswap of V before the specified
/// instruction IP.
static Value *LowerBSWAP(Value *V, Instruction *IP) {
  assert(V->getType()->isInteger() && "Can't bswap a non-integer type!");

  unsigned BitSize = V->getType()->getPrimitiveSizeInBits();
  
  IRBuilder<> Builder(IP->getParent(), IP);

  switch(BitSize) {
  default: assert(0 && "Unhandled type size of value to byteswap!");
  case 16: {
    Value *Tmp1 = Builder.CreateShl(V, ConstantInt::get(V->getType(), 8),
                                    "bswap.2");
    Value *Tmp2 = Builder.CreateLShr(V, ConstantInt::get(V->getType(), 8),
                                     "bswap.1");
    V = Builder.CreateOr(Tmp1, Tmp2, "bswap.i16");
    break;
  }
  case 32: {
    Value *Tmp4 = Builder.CreateShl(V, ConstantInt::get(V->getType(), 24),
                                    "bswap.4");
    Value *Tmp3 = Builder.CreateShl(V, ConstantInt::get(V->getType(), 8),
                                    "bswap.3");
    Value *Tmp2 = Builder.CreateLShr(V, ConstantInt::get(V->getType(), 8),
                                     "bswap.2");
    Value *Tmp1 = Builder.CreateLShr(V, ConstantInt::get(V->getType(), 24),
                                     "bswap.1");
    Tmp3 = Builder.CreateAnd(Tmp3, ConstantInt::get(Type::Int32Ty, 0xFF0000),
                             "bswap.and3");
    Tmp2 = Builder.CreateAnd(Tmp2, ConstantInt::get(Type::Int32Ty, 0xFF00),
                             "bswap.and2");
    Tmp4 = Builder.CreateOr(Tmp4, Tmp3, "bswap.or1");
    Tmp2 = Builder.CreateOr(Tmp2, Tmp1, "bswap.or2");
    V = Builder.CreateOr(Tmp4, Tmp2, "bswap.i32");
    break;
  }
  case 64: {
    Value *Tmp8 = Builder.CreateShl(V, ConstantInt::get(V->getType(), 56),
                                    "bswap.8");
    Value *Tmp7 = Builder.CreateShl(V, ConstantInt::get(V->getType(), 40),
                                    "bswap.7");
    Value *Tmp6 = Builder.CreateShl(V, ConstantInt::get(V->getType(), 24),
                                    "bswap.6");
    Value *Tmp5 = Builder.CreateShl(V, ConstantInt::get(V->getType(), 8),
                                    "bswap.5");
    Value* Tmp4 = Builder.CreateLShr(V, ConstantInt::get(V->getType(), 8),
                                     "bswap.4");
    Value* Tmp3 = Builder.CreateLShr(V, ConstantInt::get(V->getType(), 24),
                                     "bswap.3");
    Value* Tmp2 = Builder.CreateLShr(V, ConstantInt::get(V->getType(), 40),
                                     "bswap.2");
    Value* Tmp1 = Builder.CreateLShr(V, ConstantInt::get(V->getType(), 56),
                                     "bswap.1");
    Tmp7 = Builder.CreateAnd(Tmp7,
                             ConstantInt::get(Type::Int64Ty,
                                              0xFF000000000000ULL),
                             "bswap.and7");
    Tmp6 = Builder.CreateAnd(Tmp6,
                             ConstantInt::get(Type::Int64Ty,
                                              0xFF0000000000ULL),
                             "bswap.and6");
    Tmp5 = Builder.CreateAnd(Tmp5,
                             ConstantInt::get(Type::Int64Ty, 0xFF00000000ULL),
                             "bswap.and5");
    Tmp4 = Builder.CreateAnd(Tmp4,
                             ConstantInt::get(Type::Int64Ty, 0xFF000000ULL),
                             "bswap.and4");
    Tmp3 = Builder.CreateAnd(Tmp3,
                             ConstantInt::get(Type::Int64Ty, 0xFF0000ULL),
                             "bswap.and3");
    Tmp2 = Builder.CreateAnd(Tmp2,
                             ConstantInt::get(Type::Int64Ty, 0xFF00ULL),
                             "bswap.and2");
    Tmp8 = Builder.CreateOr(Tmp8, Tmp7, "bswap.or1");
    Tmp6 = Builder.CreateOr(Tmp6, Tmp5, "bswap.or2");
    Tmp4 = Builder.CreateOr(Tmp4, Tmp3, "bswap.or3");
    Tmp2 = Builder.CreateOr(Tmp2, Tmp1, "bswap.or4");
    Tmp8 = Builder.CreateOr(Tmp8, Tmp6, "bswap.or5");
    Tmp4 = Builder.CreateOr(Tmp4, Tmp2, "bswap.or6");
    V = Builder.CreateOr(Tmp8, Tmp4, "bswap.i64");
    break;
  }
  }
  return V;
}

/// LowerCTPOP - Emit the code to lower ctpop of V before the specified
/// instruction IP.
static Value *LowerCTPOP(Value *V, Instruction *IP) {
  assert(V->getType()->isInteger() && "Can't ctpop a non-integer type!");

  static const uint64_t MaskValues[6] = {
    0x5555555555555555ULL, 0x3333333333333333ULL,
    0x0F0F0F0F0F0F0F0FULL, 0x00FF00FF00FF00FFULL,
    0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL
  };

  IRBuilder<> Builder(IP->getParent(), IP);

  unsigned BitSize = V->getType()->getPrimitiveSizeInBits();
  unsigned WordSize = (BitSize + 63) / 64;
  Value *Count = ConstantInt::get(V->getType(), 0);

  for (unsigned n = 0; n < WordSize; ++n) {
    Value *PartValue = V;
    for (unsigned i = 1, ct = 0; i < (BitSize>64 ? 64 : BitSize); 
         i <<= 1, ++ct) {
      Value *MaskCst = ConstantInt::get(V->getType(), MaskValues[ct]);
      Value *LHS = Builder.CreateAnd(PartValue, MaskCst, "cppop.and1");
      Value *VShift = Builder.CreateLShr(PartValue,
                                         ConstantInt::get(V->getType(), i),
                                         "ctpop.sh");
      Value *RHS = Builder.CreateAnd(VShift, MaskCst, "cppop.and2");
      PartValue = Builder.CreateAdd(LHS, RHS, "ctpop.step");
    }
    Count = Builder.CreateAdd(PartValue, Count, "ctpop.part");
    if (BitSize > 64) {
      V = Builder.CreateLShr(V, ConstantInt::get(V->getType(), 64),
                             "ctpop.part.sh");
      BitSize -= 64;
    }
  }

  return Count;
}

/// LowerCTLZ - Emit the code to lower ctlz of V before the specified
/// instruction IP.
static Value *LowerCTLZ(Value *V, Instruction *IP) {

  IRBuilder<> Builder(IP->getParent(), IP);

  unsigned BitSize = V->getType()->getPrimitiveSizeInBits();
  for (unsigned i = 1; i < BitSize; i <<= 1) {
    Value *ShVal = ConstantInt::get(V->getType(), i);
    ShVal = Builder.CreateLShr(V, ShVal, "ctlz.sh");
    V = Builder.CreateOr(V, ShVal, "ctlz.step");
  }

  V = Builder.CreateNot(V);
  return LowerCTPOP(V, IP);
}

/// Convert the llvm.part.select.iX.iY intrinsic. This intrinsic takes 
/// three integer arguments. The first argument is the Value from which the
/// bits will be selected. It may be of any bit width. The second and third
/// arguments specify a range of bits to select with the second argument 
/// specifying the low bit and the third argument specifying the high bit. Both
/// must be type i32. The result is the corresponding selected bits from the
/// Value in the same width as the Value (first argument). If the low bit index
/// is higher than the high bit index then the inverse selection is done and 
/// the bits are returned in inverse order. 
/// @brief Lowering of llvm.part.select intrinsic.
static Instruction *LowerPartSelect(CallInst *CI) {
  IRBuilder<> Builder;

  // Make sure we're dealing with a part select intrinsic here
  Function *F = CI->getCalledFunction();
  const FunctionType *FT = F->getFunctionType();
  if (!F->isDeclaration() || !FT->getReturnType()->isInteger() ||
      FT->getNumParams() != 3 || !FT->getParamType(0)->isInteger() ||
      !FT->getParamType(1)->isInteger() || !FT->getParamType(2)->isInteger())
    return CI;

  // Get the intrinsic implementation function by converting all the . to _
  // in the intrinsic's function name and then reconstructing the function
  // declaration.
  std::string Name(F->getName());
  for (unsigned i = 4; i < Name.length(); ++i)
    if (Name[i] == '.')
      Name[i] = '_';
  Module* M = F->getParent();
  F = cast<Function>(M->getOrInsertFunction(Name, FT));
  F->setLinkage(GlobalValue::WeakAnyLinkage);

  // If we haven't defined the impl function yet, do so now
  if (F->isDeclaration()) {

    // Get the arguments to the function
    Function::arg_iterator args = F->arg_begin();
    Value* Val = args++; Val->setName("Val");
    Value* Lo = args++; Lo->setName("Lo");
    Value* Hi = args++; Hi->setName("High");

    // We want to select a range of bits here such that [Hi, Lo] is shifted
    // down to the low bits. However, it is quite possible that Hi is smaller
    // than Lo in which case the bits have to be reversed. 
    
    // Create the blocks we will need for the two cases (forward, reverse)
    BasicBlock* CurBB   = BasicBlock::Create("entry", F);
    BasicBlock *RevSize = BasicBlock::Create("revsize", CurBB->getParent());
    BasicBlock *FwdSize = BasicBlock::Create("fwdsize", CurBB->getParent());
    BasicBlock *Compute = BasicBlock::Create("compute", CurBB->getParent());
    BasicBlock *Reverse = BasicBlock::Create("reverse", CurBB->getParent());
    BasicBlock *RsltBlk = BasicBlock::Create("result",  CurBB->getParent());

    Builder.SetInsertPoint(CurBB);

    // Cast Hi and Lo to the size of Val so the widths are all the same
    if (Hi->getType() != Val->getType())
      Hi = Builder.CreateIntCast(Hi, Val->getType(), /* isSigned */ false,
                                 "tmp");
    if (Lo->getType() != Val->getType())
      Lo = Builder.CreateIntCast(Lo, Val->getType(), /* isSigned */ false,
                                 "tmp");

    // Compute a few things that both cases will need, up front.
    Constant* Zero = ConstantInt::get(Val->getType(), 0);
    Constant* One = ConstantInt::get(Val->getType(), 1);
    Constant* AllOnes = ConstantInt::getAllOnesValue(Val->getType());

    // Compare the Hi and Lo bit positions. This is used to determine 
    // which case we have (forward or reverse)
    Value *Cmp = Builder.CreateICmpULT(Hi, Lo, "less");
    Builder.CreateCondBr(Cmp, RevSize, FwdSize);

    // First, compute the number of bits in the forward case.
    Builder.SetInsertPoint(FwdSize);
    Value* FBitSize = Builder.CreateSub(Hi, Lo, "fbits");
    Builder.CreateBr(Compute);

    // Second, compute the number of bits in the reverse case.
    Builder.SetInsertPoint(RevSize);
    Value* RBitSize = Builder.CreateSub(Lo, Hi, "rbits");
    Builder.CreateBr(Compute);

    // Now, compute the bit range. Start by getting the bitsize and the shift
    // amount (either Hi or Lo) from PHI nodes. Then we compute a mask for 
    // the number of bits we want in the range. We shift the bits down to the 
    // least significant bits, apply the mask to zero out unwanted high bits, 
    // and we have computed the "forward" result. It may still need to be 
    // reversed.
    Builder.SetInsertPoint(Compute);

    // Get the BitSize from one of the two subtractions
    PHINode *BitSize = Builder.CreatePHI(Val->getType(), "bits");
    BitSize->reserveOperandSpace(2);
    BitSize->addIncoming(FBitSize, FwdSize);
    BitSize->addIncoming(RBitSize, RevSize);

    // Get the ShiftAmount as the smaller of Hi/Lo
    PHINode *ShiftAmt = Builder.CreatePHI(Val->getType(), "shiftamt");
    ShiftAmt->reserveOperandSpace(2);
    ShiftAmt->addIncoming(Lo, FwdSize);
    ShiftAmt->addIncoming(Hi, RevSize);

    // Increment the bit size
    Value *BitSizePlusOne = Builder.CreateAdd(BitSize, One, "bits");

    // Create a Mask to zero out the high order bits.
    Value* Mask = Builder.CreateShl(AllOnes, BitSizePlusOne, "mask");
    Mask = Builder.CreateNot(Mask, "mask");

    // Shift the bits down and apply the mask
    Value* FRes = Builder.CreateLShr(Val, ShiftAmt, "fres");
    FRes = Builder.CreateAnd(FRes, Mask, "fres");
    Builder.CreateCondBr(Cmp, Reverse, RsltBlk);

    // In the Reverse block we have the mask already in FRes but we must reverse
    // it by shifting FRes bits right and putting them in RRes by shifting them 
    // in from left.
    Builder.SetInsertPoint(Reverse);

    // First set up our loop counters
    PHINode *Count = Builder.CreatePHI(Val->getType(), "count");
    Count->reserveOperandSpace(2);
    Count->addIncoming(BitSizePlusOne, Compute);

    // Next, get the value that we are shifting.
    PHINode *BitsToShift = Builder.CreatePHI(Val->getType(), "val");
    BitsToShift->reserveOperandSpace(2);
    BitsToShift->addIncoming(FRes, Compute);

    // Finally, get the result of the last computation
    PHINode *RRes = Builder.CreatePHI(Val->getType(), "rres");
    RRes->reserveOperandSpace(2);
    RRes->addIncoming(Zero, Compute);

    // Decrement the counter
    Value *Decr = Builder.CreateSub(Count, One, "decr");
    Count->addIncoming(Decr, Reverse);

    // Compute the Bit that we want to move
    Value *Bit = Builder.CreateAnd(BitsToShift, One, "bit");

    // Compute the new value for next iteration.
    Value *NewVal = Builder.CreateLShr(BitsToShift, One, "rshift");
    BitsToShift->addIncoming(NewVal, Reverse);

    // Shift the bit into the low bits of the result.
    Value *NewRes = Builder.CreateShl(RRes, One, "lshift");
    NewRes = Builder.CreateOr(NewRes, Bit, "addbit");
    RRes->addIncoming(NewRes, Reverse);
    
    // Terminate loop if we've moved all the bits.
    Value *Cond = Builder.CreateICmpEQ(Decr, Zero, "cond");
    Builder.CreateCondBr(Cond, RsltBlk, Reverse);

    // Finally, in the result block, select one of the two results with a PHI
    // node and return the result;
    Builder.SetInsertPoint(RsltBlk);
    PHINode *BitSelect = Builder.CreatePHI(Val->getType(), "part_select");
    BitSelect->reserveOperandSpace(2);
    BitSelect->addIncoming(FRes, Compute);
    BitSelect->addIncoming(NewRes, Reverse);
    Builder.CreateRet(BitSelect);
  }

  // Return a call to the implementation function
  Builder.SetInsertPoint(CI->getParent(), CI);
  CallInst *NewCI = Builder.CreateCall3(F, CI->getOperand(1),
                                        CI->getOperand(2), CI->getOperand(3));
  NewCI->setName(CI->getName());
  return NewCI;
}

/// Convert the llvm.part.set.iX.iY.iZ intrinsic. This intrinsic takes 
/// four integer arguments (iAny %Value, iAny %Replacement, i32 %Low, i32 %High)
/// The first two arguments can be any bit width. The result is the same width
/// as %Value. The operation replaces bits between %Low and %High with the value
/// in %Replacement. If %Replacement is not the same width, it is truncated or
/// zero extended as appropriate to fit the bits being replaced. If %Low is
/// greater than %High then the inverse set of bits are replaced.
/// @brief Lowering of llvm.bit.part.set intrinsic.
static Instruction *LowerPartSet(CallInst *CI) {
  IRBuilder<> Builder;

  // Make sure we're dealing with a part select intrinsic here
  Function *F = CI->getCalledFunction();
  const FunctionType *FT = F->getFunctionType();
  if (!F->isDeclaration() || !FT->getReturnType()->isInteger() ||
      FT->getNumParams() != 4 || !FT->getParamType(0)->isInteger() ||
      !FT->getParamType(1)->isInteger() || !FT->getParamType(2)->isInteger() ||
      !FT->getParamType(3)->isInteger())
    return CI;

  // Get the intrinsic implementation function by converting all the . to _
  // in the intrinsic's function name and then reconstructing the function
  // declaration.
  std::string Name(F->getName());
  for (unsigned i = 4; i < Name.length(); ++i)
    if (Name[i] == '.')
      Name[i] = '_';
  Module* M = F->getParent();
  F = cast<Function>(M->getOrInsertFunction(Name, FT));
  F->setLinkage(GlobalValue::WeakAnyLinkage);

  // If we haven't defined the impl function yet, do so now
  if (F->isDeclaration()) {
    // Get the arguments for the function.
    Function::arg_iterator args = F->arg_begin();
    Value* Val = args++; Val->setName("Val");
    Value* Rep = args++; Rep->setName("Rep");
    Value* Lo  = args++; Lo->setName("Lo");
    Value* Hi  = args++; Hi->setName("Hi");

    // Get some types we need
    const IntegerType* ValTy = cast<IntegerType>(Val->getType());
    const IntegerType* RepTy = cast<IntegerType>(Rep->getType());
    uint32_t RepBits = RepTy->getBitWidth();

    // Constant Definitions
    ConstantInt* RepBitWidth = ConstantInt::get(Type::Int32Ty, RepBits);
    ConstantInt* RepMask = ConstantInt::getAllOnesValue(RepTy);
    ConstantInt* ValMask = ConstantInt::getAllOnesValue(ValTy);
    ConstantInt* One = ConstantInt::get(Type::Int32Ty, 1);
    ConstantInt* ValOne = ConstantInt::get(ValTy, 1);
    ConstantInt* Zero = ConstantInt::get(Type::Int32Ty, 0);
    ConstantInt* ValZero = ConstantInt::get(ValTy, 0);

    // Basic blocks we fill in below.
    BasicBlock* entry = BasicBlock::Create("entry", F, 0);
    BasicBlock* large = BasicBlock::Create("large", F, 0);
    BasicBlock* small = BasicBlock::Create("small", F, 0);
    BasicBlock* reverse = BasicBlock::Create("reverse", F, 0);
    BasicBlock* result = BasicBlock::Create("result", F, 0);

    // BASIC BLOCK: entry
    Builder.SetInsertPoint(entry);
    // First, get the number of bits that we're placing as an i32
    Value* is_forward = Builder.CreateICmpULT(Lo, Hi);
    Value* Hi_pn = Builder.CreateSelect(is_forward, Hi, Lo);
    Value* Lo_pn = Builder.CreateSelect(is_forward, Lo, Hi);
    Value* NumBits = Builder.CreateSub(Hi_pn, Lo_pn);
    NumBits = Builder.CreateAdd(NumBits, One);
    // Now, convert Lo and Hi to ValTy bit width
    Lo = Builder.CreateIntCast(Lo_pn, ValTy, /* isSigned */ false);
    // Determine if the replacement bits are larger than the number of bits we
    // are replacing and deal with it.
    Value* is_large = Builder.CreateICmpULT(NumBits, RepBitWidth);
    Builder.CreateCondBr(is_large, large, small);

    // BASIC BLOCK: large
    Builder.SetInsertPoint(large);
    Value* MaskBits = Builder.CreateSub(RepBitWidth, NumBits);
    MaskBits = Builder.CreateIntCast(MaskBits, RepMask->getType(),
                                     /* isSigned */ false);
    Value* Mask1 = Builder.CreateLShr(RepMask, MaskBits);
    Value* Rep2 = Builder.CreateAnd(Mask1, Rep);
    Builder.CreateBr(small);

    // BASIC BLOCK: small
    Builder.SetInsertPoint(small);
    PHINode* Rep3 = Builder.CreatePHI(RepTy);
    Rep3->reserveOperandSpace(2);
    Rep3->addIncoming(Rep2, large);
    Rep3->addIncoming(Rep, entry);
    Value* Rep4 = Builder.CreateIntCast(Rep3, ValTy, /* isSigned */ false);
    Builder.CreateCondBr(is_forward, result, reverse);

    // BASIC BLOCK: reverse (reverses the bits of the replacement)
    Builder.SetInsertPoint(reverse);
    // Set up our loop counter as a PHI so we can decrement on each iteration.
    // We will loop for the number of bits in the replacement value.
    PHINode *Count = Builder.CreatePHI(Type::Int32Ty, "count");
    Count->reserveOperandSpace(2);
    Count->addIncoming(NumBits, small);

    // Get the value that we are shifting bits out of as a PHI because
    // we'll change this with each iteration.
    PHINode *BitsToShift = Builder.CreatePHI(Val->getType(), "val");
    BitsToShift->reserveOperandSpace(2);
    BitsToShift->addIncoming(Rep4, small);

    // Get the result of the last computation or zero on first iteration
    PHINode *RRes = Builder.CreatePHI(Val->getType(), "rres");
    RRes->reserveOperandSpace(2);
    RRes->addIncoming(ValZero, small);

    // Decrement the loop counter by one
    Value *Decr = Builder.CreateSub(Count, One);
    Count->addIncoming(Decr, reverse);

    // Get the bit that we want to move into the result
    Value *Bit = Builder.CreateAnd(BitsToShift, ValOne);

    // Compute the new value of the bits to shift for the next iteration.
    Value *NewVal = Builder.CreateLShr(BitsToShift, ValOne);
    BitsToShift->addIncoming(NewVal, reverse);

    // Shift the bit we extracted into the low bit of the result.
    Value *NewRes = Builder.CreateShl(RRes, ValOne);
    NewRes = Builder.CreateOr(NewRes, Bit);
    RRes->addIncoming(NewRes, reverse);
    
    // Terminate loop if we've moved all the bits.
    Value *Cond = Builder.CreateICmpEQ(Decr, Zero);
    Builder.CreateCondBr(Cond, result, reverse);

    // BASIC BLOCK: result
    Builder.SetInsertPoint(result);
    PHINode *Rplcmnt = Builder.CreatePHI(Val->getType());
    Rplcmnt->reserveOperandSpace(2);
    Rplcmnt->addIncoming(NewRes, reverse);
    Rplcmnt->addIncoming(Rep4, small);
    Value* t0   = Builder.CreateIntCast(NumBits, ValTy, /* isSigned */ false);
    Value* t1   = Builder.CreateShl(ValMask, Lo);
    Value* t2   = Builder.CreateNot(t1);
    Value* t3   = Builder.CreateShl(t1, t0);
    Value* t4   = Builder.CreateOr(t2, t3);
    Value* t5   = Builder.CreateAnd(t4, Val);
    Value* t6   = Builder.CreateShl(Rplcmnt, Lo);
    Value* Rslt = Builder.CreateOr(t5, t6, "part_set");
    Builder.CreateRet(Rslt);
  }

  // Return a call to the implementation function
  Builder.SetInsertPoint(CI->getParent(), CI);
  CallInst *NewCI = Builder.CreateCall4(F, CI->getOperand(1),
                                        CI->getOperand(2), CI->getOperand(3),
                                        CI->getOperand(4));
  NewCI->setName(CI->getName());
  return NewCI;
}

static void ReplaceFPIntrinsicWithCall(CallInst *CI, Constant *FCache,
                                       Constant *DCache, Constant *LDCache,
                                       const char *Fname, const char *Dname,
                                       const char *LDname) {
  switch (CI->getOperand(1)->getType()->getTypeID()) {
  default: assert(0 && "Invalid type in intrinsic"); abort();
  case Type::FloatTyID:
    ReplaceCallWith(Fname, CI, CI->op_begin() + 1, CI->op_end(),
                  Type::FloatTy, FCache);
    break;
  case Type::DoubleTyID:
    ReplaceCallWith(Dname, CI, CI->op_begin() + 1, CI->op_end(),
                  Type::DoubleTy, DCache);
    break;
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
    ReplaceCallWith(LDname, CI, CI->op_begin() + 1, CI->op_end(),
                  CI->getOperand(1)->getType(), LDCache);
    break;
  }
}

void IntrinsicLowering::LowerIntrinsicCall(CallInst *CI) {
  IRBuilder<> Builder(CI->getParent(), CI);

  Function *Callee = CI->getCalledFunction();
  assert(Callee && "Cannot lower an indirect call!");

  switch (Callee->getIntrinsicID()) {
  case Intrinsic::not_intrinsic:
    cerr << "Cannot lower a call to a non-intrinsic function '"
         << Callee->getName() << "'!\n";
    abort();
  default:
    cerr << "Error: Code generator does not support intrinsic function '"
         << Callee->getName() << "'!\n";
    abort();

    // The setjmp/longjmp intrinsics should only exist in the code if it was
    // never optimized (ie, right out of the CFE), or if it has been hacked on
    // by the lowerinvoke pass.  In both cases, the right thing to do is to
    // convert the call to an explicit setjmp or longjmp call.
  case Intrinsic::setjmp: {
    static Constant *SetjmpFCache = 0;
    Value *V = ReplaceCallWith("setjmp", CI, CI->op_begin() + 1, CI->op_end(),
                               Type::Int32Ty, SetjmpFCache);
    if (CI->getType() != Type::VoidTy)
      CI->replaceAllUsesWith(V);
    break;
  }
  case Intrinsic::sigsetjmp:
     if (CI->getType() != Type::VoidTy)
       CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
     break;

  case Intrinsic::longjmp: {
    static Constant *LongjmpFCache = 0;
    ReplaceCallWith("longjmp", CI, CI->op_begin() + 1, CI->op_end(),
                    Type::VoidTy, LongjmpFCache);
    break;
  }

  case Intrinsic::siglongjmp: {
    // Insert the call to abort
    static Constant *AbortFCache = 0;
    ReplaceCallWith("abort", CI, CI->op_end(), CI->op_end(), 
                    Type::VoidTy, AbortFCache);
    break;
  }
  case Intrinsic::ctpop:
    CI->replaceAllUsesWith(LowerCTPOP(CI->getOperand(1), CI));
    break;

  case Intrinsic::bswap:
    CI->replaceAllUsesWith(LowerBSWAP(CI->getOperand(1), CI));
    break;
    
  case Intrinsic::ctlz:
    CI->replaceAllUsesWith(LowerCTLZ(CI->getOperand(1), CI));
    break;

  case Intrinsic::cttz: {
    // cttz(x) -> ctpop(~X & (X-1))
    Value *Src = CI->getOperand(1);
    Value *NotSrc = Builder.CreateNot(Src);
    NotSrc->setName(Src->getName() + ".not");
    Value *SrcM1 = ConstantInt::get(Src->getType(), 1);
    SrcM1 = Builder.CreateSub(Src, SrcM1);
    Src = LowerCTPOP(Builder.CreateAnd(NotSrc, SrcM1), CI);
    CI->replaceAllUsesWith(Src);
    break;
  }

  case Intrinsic::part_select:
    CI->replaceAllUsesWith(LowerPartSelect(CI));
    break;

  case Intrinsic::part_set:
    CI->replaceAllUsesWith(LowerPartSet(CI));
    break;

  case Intrinsic::stacksave:
  case Intrinsic::stackrestore: {
    static bool Warned = false;
    if (!Warned)
      cerr << "WARNING: this target does not support the llvm.stack"
           << (Callee->getIntrinsicID() == Intrinsic::stacksave ?
               "save" : "restore") << " intrinsic.\n";
    Warned = true;
    if (Callee->getIntrinsicID() == Intrinsic::stacksave)
      CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    break;
  }
    
  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
    cerr << "WARNING: this target does not support the llvm."
         << (Callee->getIntrinsicID() == Intrinsic::returnaddress ?
             "return" : "frame") << "address intrinsic.\n";
    CI->replaceAllUsesWith(ConstantPointerNull::get(
                                            cast<PointerType>(CI->getType())));
    break;

  case Intrinsic::prefetch:
    break;    // Simply strip out prefetches on unsupported architectures

  case Intrinsic::pcmarker:
    break;    // Simply strip out pcmarker on unsupported architectures
  case Intrinsic::readcyclecounter: {
    cerr << "WARNING: this target does not support the llvm.readcyclecoun"
         << "ter intrinsic.  It is being lowered to a constant 0\n";
    CI->replaceAllUsesWith(ConstantInt::get(Type::Int64Ty, 0));
    break;
  }

  case Intrinsic::dbg_stoppoint:
  case Intrinsic::dbg_region_start:
  case Intrinsic::dbg_region_end:
  case Intrinsic::dbg_func_start:
  case Intrinsic::dbg_declare:
    break;    // Simply strip out debugging intrinsics

  case Intrinsic::eh_exception:
  case Intrinsic::eh_selector_i32:
  case Intrinsic::eh_selector_i64:
    CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    break;

  case Intrinsic::eh_typeid_for_i32:
  case Intrinsic::eh_typeid_for_i64:
    // Return something different to eh_selector.
    CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 1));
    break;

  case Intrinsic::var_annotation:
    break;   // Strip out annotate intrinsic
    
  case Intrinsic::memcpy: {
    static Constant *MemcpyFCache = 0;
    const IntegerType *IntPtr = TD.getIntPtrType();
    Value *Size = Builder.CreateIntCast(CI->getOperand(3), IntPtr,
                                        /* isSigned */ false);
    Value *Ops[3];
    Ops[0] = CI->getOperand(1);
    Ops[1] = CI->getOperand(2);
    Ops[2] = Size;
    ReplaceCallWith("memcpy", CI, Ops, Ops+3, CI->getOperand(1)->getType(),
                    MemcpyFCache);
    break;
  }
  case Intrinsic::memmove: {
    static Constant *MemmoveFCache = 0;
    const IntegerType *IntPtr = TD.getIntPtrType();
    Value *Size = Builder.CreateIntCast(CI->getOperand(3), IntPtr,
                                        /* isSigned */ false);
    Value *Ops[3];
    Ops[0] = CI->getOperand(1);
    Ops[1] = CI->getOperand(2);
    Ops[2] = Size;
    ReplaceCallWith("memmove", CI, Ops, Ops+3, CI->getOperand(1)->getType(),
                    MemmoveFCache);
    break;
  }
  case Intrinsic::memset: {
    static Constant *MemsetFCache = 0;
    const IntegerType *IntPtr = TD.getIntPtrType();
    Value *Size = Builder.CreateIntCast(CI->getOperand(3), IntPtr,
                                        /* isSigned */ false);
    Value *Ops[3];
    Ops[0] = CI->getOperand(1);
    // Extend the amount to i32.
    Ops[1] = Builder.CreateIntCast(CI->getOperand(2), Type::Int32Ty,
                                   /* isSigned */ false);
    Ops[2] = Size;
    ReplaceCallWith("memset", CI, Ops, Ops+3, CI->getOperand(1)->getType(),
                    MemsetFCache);
    break;
  }
  case Intrinsic::sqrt: {
    static Constant *sqrtFCache = 0;
    static Constant *sqrtDCache = 0;
    static Constant *sqrtLDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, sqrtFCache, sqrtDCache, sqrtLDCache,
                               "sqrtf", "sqrt", "sqrtl");
    break;
  }
  case Intrinsic::log: {
    static Constant *logFCache = 0;
    static Constant *logDCache = 0;
    static Constant *logLDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, logFCache, logDCache, logLDCache,
                               "logf", "log", "logl");
    break;
  }
  case Intrinsic::log2: {
    static Constant *log2FCache = 0;
    static Constant *log2DCache = 0;
    static Constant *log2LDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, log2FCache, log2DCache, log2LDCache,
                               "log2f", "log2", "log2l");
    break;
  }
  case Intrinsic::log10: {
    static Constant *log10FCache = 0;
    static Constant *log10DCache = 0;
    static Constant *log10LDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, log10FCache, log10DCache, log10LDCache,
                               "log10f", "log10", "log10l");
    break;
  }
  case Intrinsic::exp: {
    static Constant *expFCache = 0;
    static Constant *expDCache = 0;
    static Constant *expLDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, expFCache, expDCache, expLDCache,
                               "expf", "exp", "expl");
    break;
  }
  case Intrinsic::exp2: {
    static Constant *exp2FCache = 0;
    static Constant *exp2DCache = 0;
    static Constant *exp2LDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, exp2FCache, exp2DCache, exp2LDCache,
                               "exp2f", "exp2", "exp2l");
    break;
  }
  case Intrinsic::pow: {
    static Constant *powFCache = 0;
    static Constant *powDCache = 0;
    static Constant *powLDCache = 0;
    ReplaceFPIntrinsicWithCall(CI, powFCache, powDCache, powLDCache,
                               "powf", "pow", "powl");
    break;
  }
  case Intrinsic::flt_rounds:
     // Lower to "round to the nearest"
     if (CI->getType() != Type::VoidTy)
       CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 1));
     break;
  }

  assert(CI->use_empty() &&
         "Lowering should have eliminated any uses of the intrinsic call!");
  CI->eraseFromParent();
}

//===-- IntrinsicLowering.cpp - Intrinsic Lowering default implementation -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IntrinsicLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Support/Streams.h"
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

  SmallVector<Value*, 8> Operands(ArgBegin, ArgEnd);
  CallInst *NewCI = new CallInst(FCache, &Operands[0], Operands.size(),
                                 CI->getName(), CI);
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
      case Intrinsic::memcpy_i32:
      case Intrinsic::memcpy_i64:
        M.getOrInsertFunction("memcpy", PointerType::get(Type::Int8Ty),
                              PointerType::get(Type::Int8Ty), 
                              PointerType::get(Type::Int8Ty), 
                              TD.getIntPtrType(), (Type *)0);
        break;
      case Intrinsic::memmove_i32:
      case Intrinsic::memmove_i64:
        M.getOrInsertFunction("memmove", PointerType::get(Type::Int8Ty),
                              PointerType::get(Type::Int8Ty), 
                              PointerType::get(Type::Int8Ty), 
                              TD.getIntPtrType(), (Type *)0);
        break;
      case Intrinsic::memset_i32:
      case Intrinsic::memset_i64:
        M.getOrInsertFunction("memset", PointerType::get(Type::Int8Ty),
                              PointerType::get(Type::Int8Ty), Type::Int32Ty, 
                              TD.getIntPtrType(), (Type *)0);
        break;
      case Intrinsic::sqrt_f32:
      case Intrinsic::sqrt_f64:
        if(I->arg_begin()->getType() == Type::FloatTy)
          EnsureFunctionExists(M, "sqrtf", I->arg_begin(), I->arg_end(),
                               Type::FloatTy);
        else
          EnsureFunctionExists(M, "sqrt", I->arg_begin(), I->arg_end(),
                               Type::DoubleTy);
        break;
      }
}

/// LowerBSWAP - Emit the code to lower bswap of V before the specified
/// instruction IP.
static Value *LowerBSWAP(Value *V, Instruction *IP) {
  assert(V->getType()->isInteger() && "Can't bswap a non-integer type!");

  unsigned BitSize = V->getType()->getPrimitiveSizeInBits();
  
  switch(BitSize) {
  default: assert(0 && "Unhandled type size of value to byteswap!");
  case 16: {
    Value *Tmp1 = BinaryOperator::createShl(V,
                                ConstantInt::get(V->getType(),8),"bswap.2",IP);
    Value *Tmp2 = BinaryOperator::createLShr(V,
                                ConstantInt::get(V->getType(),8),"bswap.1",IP);
    V = BinaryOperator::createOr(Tmp1, Tmp2, "bswap.i16", IP);
    break;
  }
  case 32: {
    Value *Tmp4 = BinaryOperator::createShl(V,
                              ConstantInt::get(V->getType(),24),"bswap.4", IP);
    Value *Tmp3 = BinaryOperator::createShl(V,
                              ConstantInt::get(V->getType(),8),"bswap.3",IP);
    Value *Tmp2 = BinaryOperator::createLShr(V,
                              ConstantInt::get(V->getType(),8),"bswap.2",IP);
    Value *Tmp1 = BinaryOperator::createLShr(V,
                              ConstantInt::get(V->getType(),24),"bswap.1", IP);
    Tmp3 = BinaryOperator::createAnd(Tmp3, 
                                     ConstantInt::get(Type::Int32Ty, 0xFF0000),
                                     "bswap.and3", IP);
    Tmp2 = BinaryOperator::createAnd(Tmp2, 
                                     ConstantInt::get(Type::Int32Ty, 0xFF00),
                                     "bswap.and2", IP);
    Tmp4 = BinaryOperator::createOr(Tmp4, Tmp3, "bswap.or1", IP);
    Tmp2 = BinaryOperator::createOr(Tmp2, Tmp1, "bswap.or2", IP);
    V = BinaryOperator::createOr(Tmp4, Tmp3, "bswap.i32", IP);
    break;
  }
  case 64: {
    Value *Tmp8 = BinaryOperator::createShl(V,
                              ConstantInt::get(V->getType(),56),"bswap.8", IP);
    Value *Tmp7 = BinaryOperator::createShl(V,
                              ConstantInt::get(V->getType(),40),"bswap.7", IP);
    Value *Tmp6 = BinaryOperator::createShl(V,
                              ConstantInt::get(V->getType(),24),"bswap.6", IP);
    Value *Tmp5 = BinaryOperator::createShl(V,
                              ConstantInt::get(V->getType(),8),"bswap.5", IP);
    Value* Tmp4 = BinaryOperator::createLShr(V,
                              ConstantInt::get(V->getType(),8),"bswap.4", IP);
    Value* Tmp3 = BinaryOperator::createLShr(V,
                              ConstantInt::get(V->getType(),24),"bswap.3", IP);
    Value* Tmp2 = BinaryOperator::createLShr(V,
                              ConstantInt::get(V->getType(),40),"bswap.2", IP);
    Value* Tmp1 = BinaryOperator::createLShr(V,
                              ConstantInt::get(V->getType(),56),"bswap.1", IP);
    Tmp7 = BinaryOperator::createAnd(Tmp7,
                             ConstantInt::get(Type::Int64Ty, 
                               0xFF000000000000ULL),
                             "bswap.and7", IP);
    Tmp6 = BinaryOperator::createAnd(Tmp6,
                             ConstantInt::get(Type::Int64Ty, 0xFF0000000000ULL),
                             "bswap.and6", IP);
    Tmp5 = BinaryOperator::createAnd(Tmp5,
                             ConstantInt::get(Type::Int64Ty, 0xFF00000000ULL),
                             "bswap.and5", IP);
    Tmp4 = BinaryOperator::createAnd(Tmp4,
                             ConstantInt::get(Type::Int64Ty, 0xFF000000ULL),
                             "bswap.and4", IP);
    Tmp3 = BinaryOperator::createAnd(Tmp3,
                             ConstantInt::get(Type::Int64Ty, 0xFF0000ULL),
                             "bswap.and3", IP);
    Tmp2 = BinaryOperator::createAnd(Tmp2,
                             ConstantInt::get(Type::Int64Ty, 0xFF00ULL),
                             "bswap.and2", IP);
    Tmp8 = BinaryOperator::createOr(Tmp8, Tmp7, "bswap.or1", IP);
    Tmp6 = BinaryOperator::createOr(Tmp6, Tmp5, "bswap.or2", IP);
    Tmp4 = BinaryOperator::createOr(Tmp4, Tmp3, "bswap.or3", IP);
    Tmp2 = BinaryOperator::createOr(Tmp2, Tmp1, "bswap.or4", IP);
    Tmp8 = BinaryOperator::createOr(Tmp8, Tmp6, "bswap.or5", IP);
    Tmp4 = BinaryOperator::createOr(Tmp4, Tmp2, "bswap.or6", IP);
    V = BinaryOperator::createOr(Tmp8, Tmp4, "bswap.i64", IP);
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

  unsigned BitSize = V->getType()->getPrimitiveSizeInBits();

  for (unsigned i = 1, ct = 0; i != BitSize; i <<= 1, ++ct) {
    Value *MaskCst = ConstantInt::get(V->getType(), MaskValues[ct]);
    Value *LHS = BinaryOperator::createAnd(V, MaskCst, "cppop.and1", IP);
    Value *VShift = BinaryOperator::createLShr(V,
                      ConstantInt::get(V->getType(), i), "ctpop.sh", IP);
    Value *RHS = BinaryOperator::createAnd(VShift, MaskCst, "cppop.and2", IP);
    V = BinaryOperator::createAdd(LHS, RHS, "ctpop.step", IP);
  }

  return CastInst::createIntegerCast(V, Type::Int32Ty, false, "ctpop", IP);
}

/// LowerCTLZ - Emit the code to lower ctlz of V before the specified
/// instruction IP.
static Value *LowerCTLZ(Value *V, Instruction *IP) {

  unsigned BitSize = V->getType()->getPrimitiveSizeInBits();
  for (unsigned i = 1; i != BitSize; i <<= 1) {
    Value *ShVal = ConstantInt::get(V->getType(), i);
    ShVal = BinaryOperator::createLShr(V, ShVal, "ctlz.sh", IP);
    V = BinaryOperator::createOr(V, ShVal, "ctlz.step", IP);
  }

  V = BinaryOperator::createNot(V, "", IP);
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
  F->setLinkage(GlobalValue::WeakLinkage);

  // If we haven't defined the impl function yet, do so now
  if (F->isDeclaration()) {

    // Get the arguments to the function
    Function::arg_iterator args = F->arg_begin();
    Value* Val = args++; Val->setName("Val");
    Value* Lo = args++; Lo->setName("Lo");
    Value* Hi  = args++; Hi->setName("High");

    // We want to select a range of bits here such that [Hi, Lo] is shifted
    // down to the low bits. However, it is quite possible that Hi is smaller
    // than Lo in which case the bits have to be reversed. 
    
    // Create the blocks we will need for the two cases (forward, reverse)
    BasicBlock* CurBB   = new BasicBlock("entry", F);
    BasicBlock *RevSize = new BasicBlock("revsize", CurBB->getParent());
    BasicBlock *FwdSize = new BasicBlock("fwdsize", CurBB->getParent());
    BasicBlock *Compute = new BasicBlock("compute", CurBB->getParent());
    BasicBlock *Reverse = new BasicBlock("reverse", CurBB->getParent());
    BasicBlock *RsltBlk = new BasicBlock("result",  CurBB->getParent());

    // Cast Hi and Lo to the size of Val so the widths are all the same
    if (Hi->getType() != Val->getType())
      Hi = CastInst::createIntegerCast(Hi, Val->getType(), false, 
                                         "tmp", CurBB);
    if (Lo->getType() != Val->getType())
      Lo = CastInst::createIntegerCast(Lo, Val->getType(), false, 
                                          "tmp", CurBB);

    // Compute a few things that both cases will need, up front.
    Constant* Zero = ConstantInt::get(Val->getType(), 0);
    Constant* One = ConstantInt::get(Val->getType(), 1);
    Constant* AllOnes = ConstantInt::getAllOnesValue(Val->getType());

    // Compare the Hi and Lo bit positions. This is used to determine 
    // which case we have (forward or reverse)
    ICmpInst *Cmp = new ICmpInst(ICmpInst::ICMP_ULT, Hi, Lo, "less",CurBB);
    new BranchInst(RevSize, FwdSize, Cmp, CurBB);

    // First, copmute the number of bits in the forward case.
    Instruction* FBitSize = 
      BinaryOperator::createSub(Hi, Lo,"fbits", FwdSize);
    new BranchInst(Compute, FwdSize);

    // Second, compute the number of bits in the reverse case.
    Instruction* RBitSize = 
      BinaryOperator::createSub(Lo, Hi, "rbits", RevSize);
    new BranchInst(Compute, RevSize);

    // Now, compute the bit range. Start by getting the bitsize and the shift
    // amount (either Hi or Lo) from PHI nodes. Then we compute a mask for 
    // the number of bits we want in the range. We shift the bits down to the 
    // least significant bits, apply the mask to zero out unwanted high bits, 
    // and we have computed the "forward" result. It may still need to be 
    // reversed.

    // Get the BitSize from one of the two subtractions
    PHINode *BitSize = new PHINode(Val->getType(), "bits", Compute);
    BitSize->reserveOperandSpace(2);
    BitSize->addIncoming(FBitSize, FwdSize);
    BitSize->addIncoming(RBitSize, RevSize);

    // Get the ShiftAmount as the smaller of Hi/Lo
    PHINode *ShiftAmt = new PHINode(Val->getType(), "shiftamt", Compute);
    ShiftAmt->reserveOperandSpace(2);
    ShiftAmt->addIncoming(Lo, FwdSize);
    ShiftAmt->addIncoming(Hi, RevSize);

    // Increment the bit size
    Instruction *BitSizePlusOne = 
      BinaryOperator::createAdd(BitSize, One, "bits", Compute);

    // Create a Mask to zero out the high order bits.
    Instruction* Mask = 
      BinaryOperator::createShl(AllOnes, BitSizePlusOne, "mask", Compute);
    Mask = BinaryOperator::createNot(Mask, "mask", Compute);

    // Shift the bits down and apply the mask
    Instruction* FRes = 
      BinaryOperator::createLShr(Val, ShiftAmt, "fres", Compute);
    FRes = BinaryOperator::createAnd(FRes, Mask, "fres", Compute);
    new BranchInst(Reverse, RsltBlk, Cmp, Compute);

    // In the Reverse block we have the mask already in FRes but we must reverse
    // it by shifting FRes bits right and putting them in RRes by shifting them 
    // in from left.

    // First set up our loop counters
    PHINode *Count = new PHINode(Val->getType(), "count", Reverse);
    Count->reserveOperandSpace(2);
    Count->addIncoming(BitSizePlusOne, Compute);

    // Next, get the value that we are shifting.
    PHINode *BitsToShift   = new PHINode(Val->getType(), "val", Reverse);
    BitsToShift->reserveOperandSpace(2);
    BitsToShift->addIncoming(FRes, Compute);

    // Finally, get the result of the last computation
    PHINode *RRes  = new PHINode(Val->getType(), "rres", Reverse);
    RRes->reserveOperandSpace(2);
    RRes->addIncoming(Zero, Compute);

    // Decrement the counter
    Instruction *Decr = BinaryOperator::createSub(Count, One, "decr", Reverse);
    Count->addIncoming(Decr, Reverse);

    // Compute the Bit that we want to move
    Instruction *Bit = 
      BinaryOperator::createAnd(BitsToShift, One, "bit", Reverse);

    // Compute the new value for next iteration.
    Instruction *NewVal = 
      BinaryOperator::createLShr(BitsToShift, One, "rshift", Reverse);
    BitsToShift->addIncoming(NewVal, Reverse);

    // Shift the bit into the low bits of the result.
    Instruction *NewRes = 
      BinaryOperator::createShl(RRes, One, "lshift", Reverse);
    NewRes = BinaryOperator::createOr(NewRes, Bit, "addbit", Reverse);
    RRes->addIncoming(NewRes, Reverse);
    
    // Terminate loop if we've moved all the bits.
    ICmpInst *Cond = 
      new ICmpInst(ICmpInst::ICMP_EQ, Decr, Zero, "cond", Reverse);
    new BranchInst(RsltBlk, Reverse, Cond, Reverse);

    // Finally, in the result block, select one of the two results with a PHI
    // node and return the result;
    CurBB = RsltBlk;
    PHINode *BitSelect = new PHINode(Val->getType(), "part_select", CurBB);
    BitSelect->reserveOperandSpace(2);
    BitSelect->addIncoming(FRes, Compute);
    BitSelect->addIncoming(NewRes, Reverse);
    new ReturnInst(BitSelect, CurBB);
  }

  // Return a call to the implementation function
  Value *Args[3];
  Args[0] = CI->getOperand(1);
  Args[1] = CI->getOperand(2);
  Args[2] = CI->getOperand(3);
  return new CallInst(F, Args, 3, CI->getName(), CI);
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
  F->setLinkage(GlobalValue::WeakLinkage);

  // If we haven't defined the impl function yet, do so now
  if (F->isDeclaration()) {
    // Note: the following code is based on code generated by llvm2cpp with 
    // the following input. This is just *one* example of a generated function.
    // The functions vary by bit width of result and first two arguments.
    // The generated code has been changed to deal with any bit width not just
    // the 32/64 bitwidths used in the above sample.
    //
    // define i64 @part_set(i64 %Val, i32 %Rep, i32 %Lo, i32 %Hi) {
    // entry:
    //   %is_forward = icmp ult i32 %Lo, %Hi
    //   %Lo.pn = select i1 %is_forward, i32 %Hi, i32 %Lo
    //   %Hi.pn = select i1 %is_forward, i32 %Lo, i32 %Hi
    //   %iftmp.16.0 = sub i32 %Lo.pn, %Hi.pn
    //   icmp ult i32 %iftmp.16.0, 32
    //   br i1 %1, label %cond_true11, label %cond_next19
    // cond_true11:
    //   %tmp13 = sub i32 32, %iftmp.16.0
    //   %tmp14 = lshr i32 -1, %tmp13
    //   %tmp16 = and i32 %tmp14, %Rep
    //   br label %cond_next19
    // cond_next19:
    //   %iftmp.17.0 = phi i32 [ %tmp16, %cond_true11 ], [ %Rep, %entry ]
    //   %tmp2021 = zext i32 %iftmp.17.0 to i64
    //   icmp ugt i32 %Lo, %Hi
    //   br i1 %2, label %cond_next60, label %cond_true24
    // cond_true24:
    //   %tmp25.cast = zext i32 %Hi to i64
    //   %tmp26 = lshr i64 -1, %tmp25.cast
    //   %tmp27.cast = zext i32 %Lo to i64
    //   %tmp28 = shl i64 %tmp26, %tmp27.cast
    //   %tmp28not = xor i64 %tmp28, -1
    //   %tmp31 = shl i64 %tmp2021, %tmp27.cast
    //   %tmp34 = and i64 %tmp28not, %Val
    //   %Val_addr.064 = or i64 %tmp31, %tmp34
    //   ret i64 %Val_addr.064
    // cond_next60:
    //   %tmp39.cast = zext i32 %Lo to i64
    //   %tmp40 = shl i64 -1, %tmp39.cast
    //   %tmp41.cast = zext i32 %Hi to i64
    //   %tmp42 = shl i64 -1, %tmp41.cast
    //   %tmp45.demorgan = or i64 %tmp42, %tmp40
    //   %tmp45 = xor i64 %tmp45.demorgan, -1
    //   %tmp47 = and i64 %tmp45, %Val
    //   %tmp50 = shl i64 %tmp2021, %tmp39.cast
    //   %tmp52 = sub i32 32, %Hi
    //   %tmp52.cast = zext i32 %tmp52 to i64
    //   %tmp54 = lshr i64 %tmp2021, %tmp52.cast
    //   %tmp57 = or i64 %tmp50, %tmp47
    //   %Val_addr.0 = or i64 %tmp57, %tmp54
    //   ret i64 %Val_addr.0
    // }

    // Get the arguments for the function.
    Function::arg_iterator args = F->arg_begin();
    Value* Val = args++; Val->setName("Val");
    Value* Rep = args++; Rep->setName("Rep");
    Value* Lo  = args++; Lo->setName("Lo");
    Value* Hi  = args++; Hi->setName("Hi");

    // Get some types we need
    const IntegerType* ValTy = cast<IntegerType>(Val->getType());
    const IntegerType* RepTy = cast<IntegerType>(Rep->getType());
    uint32_t ValBits = ValTy->getBitWidth();
    uint32_t RepBits = RepTy->getBitWidth();

    // Constant Definitions
    ConstantInt* RepBitWidth = ConstantInt::get(Type::Int32Ty, RepBits);
    ConstantInt* RepMask = ConstantInt::getAllOnesValue(RepTy);
    ConstantInt* ValMask = ConstantInt::getAllOnesValue(ValTy);

    BasicBlock* entry = new BasicBlock("entry",F,0);
    BasicBlock* large = new BasicBlock("large",F,0);
    BasicBlock* small = new BasicBlock("small",F,0);
    BasicBlock* forward = new BasicBlock("forward",F,0);
    BasicBlock* reverse = new BasicBlock("reverse",F,0);

    // Block entry (entry)
    // First, get the number of bits that we're placing as an i32
    ICmpInst* is_forward = 
      new ICmpInst(ICmpInst::ICMP_ULT, Lo, Hi, "", entry);
    SelectInst* Lo_pn = new SelectInst(is_forward, Hi, Lo, "", entry);
    SelectInst* Hi_pn = new SelectInst(is_forward, Lo, Hi, "", entry);
    BinaryOperator* NumBits = BinaryOperator::createSub(Lo_pn, Hi_pn, "",entry);
    // Now, convert Lo and Hi to ValTy bit width
    if (ValBits > 32) {
      Hi = new ZExtInst(Hi, ValTy, "", entry);
      Lo = new ZExtInst(Lo, ValTy, "", entry);
    } else if (ValBits < 32) {
      Hi = new TruncInst(Hi, ValTy, "", entry);
      Lo = new TruncInst(Lo, ValTy, "", entry);
    }
    // Determine if the replacement bits are larger than the number of bits we
    // are replacing and deal with it.
    ICmpInst* is_large = 
      new ICmpInst(ICmpInst::ICMP_ULT, NumBits, RepBitWidth, "", entry);
    new BranchInst(large, small, is_large, entry);

    // Block "large"
    Instruction* MaskBits = 
      BinaryOperator::createSub(RepBitWidth, NumBits, "", large);
    MaskBits = CastInst::createIntegerCast(MaskBits, RepMask->getType(), 
                                           false, "", large);
    BinaryOperator* Mask1 = 
      BinaryOperator::createLShr(RepMask, MaskBits, "", large);
    BinaryOperator* Rep2 = BinaryOperator::createAnd(Mask1, Rep, "", large);
    new BranchInst(small, large);

    // Block "small"
    PHINode* Rep3 = new PHINode(RepTy, "", small);
    Rep3->reserveOperandSpace(2);
    Rep3->addIncoming(Rep2, large);
    Rep3->addIncoming(Rep, entry);
    Value* Rep4 = Rep3;
    if (ValBits > RepBits)
      Rep4 = new ZExtInst(Rep3, ValTy, "", small);
    else if (ValBits < RepBits)
      Rep4 = new TruncInst(Rep3, ValTy, "", small);
    ICmpInst* is_reverse = 
      new ICmpInst(ICmpInst::ICMP_UGT, Lo, Hi, "", small);
    new BranchInst(reverse, forward, is_reverse, small);

    // Block "forward"
    Value* t1    = BinaryOperator::createLShr(ValMask, Hi, "", forward);
    Value* t2    = BinaryOperator::createShl(t1, Lo, "", forward);
    Value* nott2 = BinaryOperator::createXor(t2, ValMask, "", forward);
    Value* t3    = BinaryOperator::createShl(Rep4, Lo, "", forward);
    Value* t4    = BinaryOperator::createAnd(nott2, Val, "", forward);
    Value* FRslt = BinaryOperator::createOr(t3, t4, "", forward);
    new ReturnInst(FRslt, forward);

    // Block "reverse"
    Value* t5    = BinaryOperator::createShl(ValMask, Lo, "", reverse);
    Value* t6    = BinaryOperator::createShl(ValMask, Hi, "", reverse);
    Value* t7    = BinaryOperator::createOr(t6, t5, "", reverse);
    Value* t8    = BinaryOperator::createXor(t7, ValMask, "", reverse);
    Value* t9    = BinaryOperator::createAnd(t8, Val, "", reverse);
    Value* t10   = BinaryOperator::createShl(Rep4, Lo, "", reverse);
    if (32 < ValBits)
      RepBitWidth = 
        cast<ConstantInt>(ConstantExpr::getZExt(RepBitWidth, ValTy));
    else if (32 > ValBits)
      RepBitWidth = 
        cast<ConstantInt>(ConstantExpr::getTrunc(RepBitWidth, ValTy));
    Value* t11   = BinaryOperator::createSub(RepBitWidth, Hi, "", reverse);
    Value* t13   = BinaryOperator::createLShr(Rep4, t11, "",reverse);
    Value* t14   = BinaryOperator::createOr(t10, t9, "", reverse);
    Value* RRslt = BinaryOperator::createOr(t14, t13, "", reverse);
    new ReturnInst(RRslt, reverse);
  }

  // Return a call to the implementation function
  Value *Args[3];
  Args[0] = CI->getOperand(1);
  Args[1] = CI->getOperand(2);
  Args[2] = CI->getOperand(3);
  Args[3] = CI->getOperand(4);
  return new CallInst(F, Args, 4, CI->getName(), CI);
}


void IntrinsicLowering::LowerIntrinsicCall(CallInst *CI) {
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
    Value *V = ReplaceCallWith("setjmp", CI, CI->op_begin()+1, CI->op_end(),
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
    ReplaceCallWith("longjmp", CI, CI->op_begin()+1, CI->op_end(),
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
    Value *NotSrc = BinaryOperator::createNot(Src, Src->getName()+".not", CI);
    Value *SrcM1  = ConstantInt::get(Src->getType(), 1);
    SrcM1 = BinaryOperator::createSub(Src, SrcM1, "", CI);
    Src = LowerCTPOP(BinaryOperator::createAnd(NotSrc, SrcM1, "", CI), CI);
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
  case Intrinsic::eh_exception:
  case Intrinsic::eh_selector:
  case Intrinsic::eh_filter:
    break;    // Simply strip out debugging and eh intrinsics

  case Intrinsic::memcpy_i32:
  case Intrinsic::memcpy_i64: {
    static Constant *MemcpyFCache = 0;
    Value *Size = CI->getOperand(3);
    const Type *IntPtr = TD.getIntPtrType();
    if (Size->getType()->getPrimitiveSizeInBits() <
        IntPtr->getPrimitiveSizeInBits())
      Size = new ZExtInst(Size, IntPtr, "", CI);
    else if (Size->getType()->getPrimitiveSizeInBits() >
             IntPtr->getPrimitiveSizeInBits())
      Size = new TruncInst(Size, IntPtr, "", CI);
    Value *Ops[3];
    Ops[0] = CI->getOperand(1);
    Ops[1] = CI->getOperand(2);
    Ops[2] = Size;
    ReplaceCallWith("memcpy", CI, Ops, Ops+3, CI->getOperand(1)->getType(),
                    MemcpyFCache);
    break;
  }
  case Intrinsic::memmove_i32: 
  case Intrinsic::memmove_i64: {
    static Constant *MemmoveFCache = 0;
    Value *Size = CI->getOperand(3);
    const Type *IntPtr = TD.getIntPtrType();
    if (Size->getType()->getPrimitiveSizeInBits() <
        IntPtr->getPrimitiveSizeInBits())
      Size = new ZExtInst(Size, IntPtr, "", CI);
    else if (Size->getType()->getPrimitiveSizeInBits() >
             IntPtr->getPrimitiveSizeInBits())
      Size = new TruncInst(Size, IntPtr, "", CI);
    Value *Ops[3];
    Ops[0] = CI->getOperand(1);
    Ops[1] = CI->getOperand(2);
    Ops[2] = Size;
    ReplaceCallWith("memmove", CI, Ops, Ops+3, CI->getOperand(1)->getType(),
                    MemmoveFCache);
    break;
  }
  case Intrinsic::memset_i32:
  case Intrinsic::memset_i64: {
    static Constant *MemsetFCache = 0;
    Value *Size = CI->getOperand(3);
    const Type *IntPtr = TD.getIntPtrType();
    if (Size->getType()->getPrimitiveSizeInBits() <
        IntPtr->getPrimitiveSizeInBits())
      Size = new ZExtInst(Size, IntPtr, "", CI);
    else if (Size->getType()->getPrimitiveSizeInBits() >
             IntPtr->getPrimitiveSizeInBits())
      Size = new TruncInst(Size, IntPtr, "", CI);
    Value *Ops[3];
    Ops[0] = CI->getOperand(1);
    // Extend the amount to i32.
    Ops[1] = new ZExtInst(CI->getOperand(2), Type::Int32Ty, "", CI);
    Ops[2] = Size;
    ReplaceCallWith("memset", CI, Ops, Ops+3, CI->getOperand(1)->getType(),
                    MemsetFCache);
    break;
  }
  case Intrinsic::sqrt_f32: {
    static Constant *sqrtfFCache = 0;
    ReplaceCallWith("sqrtf", CI, CI->op_begin()+1, CI->op_end(),
                    Type::FloatTy, sqrtfFCache);
    break;
  }
  case Intrinsic::sqrt_f64: {
    static Constant *sqrtFCache = 0;
    ReplaceCallWith("sqrt", CI, CI->op_begin()+1, CI->op_end(),
                    Type::DoubleTy, sqrtFCache);
    break;
  }
  }

  assert(CI->use_empty() &&
         "Lowering should have eliminated any uses of the intrinsic call!");
  CI->eraseFromParent();
}

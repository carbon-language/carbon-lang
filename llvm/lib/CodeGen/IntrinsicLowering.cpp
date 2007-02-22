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

  return V;
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
  case Intrinsic::ctpop_i8:
  case Intrinsic::ctpop_i16:
  case Intrinsic::ctpop_i32:
  case Intrinsic::ctpop_i64:
    CI->replaceAllUsesWith(LowerCTPOP(CI->getOperand(1), CI));
    break;

  case Intrinsic::bswap_i16:
  case Intrinsic::bswap_i32:
  case Intrinsic::bswap_i64:
    CI->replaceAllUsesWith(LowerBSWAP(CI->getOperand(1), CI));
    break;
    
  case Intrinsic::ctlz_i8:
  case Intrinsic::ctlz_i16:
  case Intrinsic::ctlz_i32:
  case Intrinsic::ctlz_i64:
    CI->replaceAllUsesWith(LowerCTLZ(CI->getOperand(1), CI));
    break;

  case Intrinsic::cttz_i8:
  case Intrinsic::cttz_i16:
  case Intrinsic::cttz_i32:
  case Intrinsic::cttz_i64: {
    // cttz(x) -> ctpop(~X & (X-1))
    Value *Src = CI->getOperand(1);
    Value *NotSrc = BinaryOperator::createNot(Src, Src->getName()+".not", CI);
    Value *SrcM1  = ConstantInt::get(Src->getType(), 1);
    SrcM1 = BinaryOperator::createSub(Src, SrcM1, "", CI);
    Src = LowerCTPOP(BinaryOperator::createAnd(NotSrc, SrcM1, "", CI), CI);
    CI->replaceAllUsesWith(Src);
    break;
  }

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
  case Intrinsic::eh_handlers:
    break;    // Simply strip out debugging intrinsics

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

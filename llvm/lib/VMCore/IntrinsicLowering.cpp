//===-- IntrinsicLowering.cpp - Intrinsic Lowering default implementation -===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the default intrinsic lowering implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/IntrinsicLowering.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/iOther.h"
using namespace llvm;

template <class ArgIt>
static Function *EnsureFunctionExists(Module &M, const char *Name,
                                      ArgIt ArgBegin, ArgIt ArgEnd,
                                      const Type *RetTy) {
  if (Function *F = M.getNamedFunction(Name)) return F;
  // It doesn't already exist in the program, insert a new definition now.
  std::vector<const Type *> ParamTys;
  for (ArgIt I = ArgBegin; I != ArgEnd; ++I)
    ParamTys.push_back(I->getType());
  return M.getOrInsertFunction(Name, FunctionType::get(RetTy, ParamTys, false));
}

/// ReplaceCallWith - This function is used when we want to lower an intrinsic
/// call to a call of an external function.  This handles hard cases such as
/// when there was already a prototype for the external function, and if that
/// prototype doesn't match the arguments we expect to pass in.
template <class ArgIt>
static CallInst *ReplaceCallWith(const char *NewFn, CallInst *CI,
                                 ArgIt ArgBegin, ArgIt ArgEnd,
                                 const Type *RetTy, Function *&FCache) {
  if (!FCache) {
    // If we haven't already looked up this function, check to see if the
    // program already contains a function with this name.
    Module *M = CI->getParent()->getParent()->getParent();
    FCache = M->getNamedFunction(NewFn);
    if (!FCache) {
      // It doesn't already exist in the program, insert a new definition now.
      std::vector<const Type *> ParamTys;
      for (ArgIt I = ArgBegin; I != ArgEnd; ++I)
        ParamTys.push_back((*I)->getType());
      FCache = M->getOrInsertFunction(NewFn,
                                     FunctionType::get(RetTy, ParamTys, false));
    }
   }

  const FunctionType *FT = FCache->getFunctionType();
  std::vector<Value*> Operands;
  unsigned ArgNo = 0;
  for (ArgIt I = ArgBegin; I != ArgEnd && ArgNo != FT->getNumParams();
       ++I, ++ArgNo) {
    Value *Arg = *I;
    if (Arg->getType() != FT->getParamType(ArgNo))
      Arg = new CastInst(Arg, FT->getParamType(ArgNo), Arg->getName(), CI);
    Operands.push_back(Arg);
  }
  // Pass nulls into any additional arguments...
  for (; ArgNo != FT->getNumParams(); ++ArgNo)
    Operands.push_back(Constant::getNullValue(FT->getParamType(ArgNo)));

  std::string Name = CI->getName(); CI->setName("");
  if (FT->getReturnType() == Type::VoidTy) Name.clear();
  CallInst *NewCI = new CallInst(FCache, Operands, Name, CI);
  if (!CI->use_empty()) {
    Value *V = NewCI;
    if (CI->getType() != NewCI->getType())
      V = new CastInst(NewCI, CI->getType(), Name, CI);
    CI->replaceAllUsesWith(V);
  }
  return NewCI;
}

void DefaultIntrinsicLowering::AddPrototypes(Module &M) {
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->isExternal() && !I->use_empty())
      switch (I->getIntrinsicID()) {
      default: break;
      case Intrinsic::setjmp:
        EnsureFunctionExists(M, "setjmp", I->abegin(), I->aend(), Type::IntTy);
        break;
      case Intrinsic::longjmp:
        EnsureFunctionExists(M, "longjmp", I->abegin(), I->aend(),Type::VoidTy);
        break;
      case Intrinsic::siglongjmp:
        EnsureFunctionExists(M, "abort", I->aend(), I->aend(), Type::VoidTy);
        break;
      case Intrinsic::memcpy:
        EnsureFunctionExists(M, "memcpy", I->abegin(), --I->aend(),
                             I->abegin()->getType());
        break;
      case Intrinsic::memmove:
        EnsureFunctionExists(M, "memmove", I->abegin(), --I->aend(),
                             I->abegin()->getType());
        break;
      case Intrinsic::memset:
        EnsureFunctionExists(M, "memset", I->abegin(), --I->aend(),
                             I->abegin()->getType());
        break;
      case Intrinsic::isnan:
        EnsureFunctionExists(M, "isnan", I->abegin(), I->aend(), Type::BoolTy);
        break;
      }

}

void DefaultIntrinsicLowering::LowerIntrinsicCall(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  assert(Callee && "Cannot lower an indirect call!");
  
  switch (Callee->getIntrinsicID()) {
  case Intrinsic::not_intrinsic:
    std::cerr << "Cannot lower a call to a non-intrinsic function '"
              << Callee->getName() << "'!\n";
    abort();
  default:
    std::cerr << "Error: Code generator does not support intrinsic function '"
              << Callee->getName() << "'!\n";
    abort();

    // The setjmp/longjmp intrinsics should only exist in the code if it was
    // never optimized (ie, right out of the CFE), or if it has been hacked on
    // by the lowerinvoke pass.  In both cases, the right thing to do is to
    // convert the call to an explicit setjmp or longjmp call.
  case Intrinsic::setjmp: {
    static Function *SetjmpFCache = 0;
    Value *V = ReplaceCallWith("setjmp", CI, CI->op_begin()+1, CI->op_end(),
                               Type::IntTy, SetjmpFCache);
    if (CI->getType() != Type::VoidTy)
      CI->replaceAllUsesWith(V);
    break;
  }
  case Intrinsic::sigsetjmp: 
     if (CI->getType() != Type::VoidTy)
       CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
     break;

  case Intrinsic::longjmp: {
    static Function *LongjmpFCache = 0;
    ReplaceCallWith("longjmp", CI, CI->op_begin()+1, CI->op_end(),
                    Type::VoidTy, LongjmpFCache);
    break;
  }

  case Intrinsic::siglongjmp: {
    // Insert the call to abort
    static Function *AbortFCache = 0;
    ReplaceCallWith("abort", CI, CI->op_end(), CI->op_end(), Type::VoidTy,
                    AbortFCache);
    break;
  }

  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
    std::cerr << "WARNING: this target does not support the llvm."
              << (Callee->getIntrinsicID() == Intrinsic::returnaddress ? 
                  "return" : "frame") << "address intrinsic.\n";
    CI->replaceAllUsesWith(ConstantPointerNull::get(
                                            cast<PointerType>(CI->getType())));
    break;

  case Intrinsic::dbg_stoppoint:
  case Intrinsic::dbg_region_start:
  case Intrinsic::dbg_region_end:
  case Intrinsic::dbg_declare:
  case Intrinsic::dbg_func_start:
    if (CI->getType() != Type::VoidTy)
      CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    break;    // Simply strip out debugging intrinsics

  case Intrinsic::memcpy: {
    // The memcpy intrinsic take an extra alignment argument that the memcpy
    // libc function does not.
    static Function *MemcpyFCache = 0;
    ReplaceCallWith("memcpy", CI, CI->op_begin()+1, CI->op_end()-1,
                    (*(CI->op_begin()+1))->getType(), MemcpyFCache);
    break;
  }
  case Intrinsic::memmove: {
    // The memmove intrinsic take an extra alignment argument that the memmove
    // libc function does not.
    static Function *MemmoveFCache = 0;
    ReplaceCallWith("memmove", CI, CI->op_begin()+1, CI->op_end()-1,
                    (*(CI->op_begin()+1))->getType(), MemmoveFCache);
    break;
  }
  case Intrinsic::memset: {
    // The memset intrinsic take an extra alignment argument that the memset
    // libc function does not.
    static Function *MemsetFCache = 0;
    ReplaceCallWith("memset", CI, CI->op_begin()+1, CI->op_end()-1,
                    (*(CI->op_begin()+1))->getType(), MemsetFCache);
    break;
  }
  case Intrinsic::isnan: {
    // FIXME: This should force the argument to be a double.  There may be
    // multiple isnans for different FP arguments.
    static Function *isnanFCache = 0;
    ReplaceCallWith("isnan", CI, CI->op_begin()+1, CI->op_end(),
                    Type::BoolTy, isnanFCache);
    break;
  }
  case Intrinsic::isunordered: {
    static Function *isunorderedFCache = 0;
    ReplaceCallWith("isunordered", CI, CI->op_begin()+1, CI->op_end(),
                    Type::BoolTy, isunorderedFCache);
    break;
  }
  }
  
  assert(CI->use_empty() &&
         "Lowering should have eliminated any uses of the intrinsic call!");
  CI->getParent()->getInstList().erase(CI);
}

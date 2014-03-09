//===------ SimplifyLibCalls.cpp - Library calls simplifier ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility pass used for testing the InstructionSimplify analysis.
// The analysis is applied to every instruction, and if it simplifies then the
// instruction is replaced by the simplification.  If you are looking for a pass
// that performs serious instruction folding, use the instcombine pass instead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

using namespace llvm;

static cl::opt<bool>
ColdErrorCalls("error-reporting-is-cold",  cl::init(true),
  cl::Hidden, cl::desc("Treat error-reporting calls as cold"));

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
namespace {
class LibCallOptimization {
protected:
  Function *Caller;
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  const LibCallSimplifier *LCS;
  LLVMContext* Context;
public:
  LibCallOptimization() { }
  virtual ~LibCallOptimization() {}

  /// callOptimizer - This pure virtual method is implemented by base classes to
  /// do various optimizations.  If this returns null then no transformation was
  /// performed.  If it returns CI, then it transformed the call and CI is to be
  /// deleted.  If it returns something else, replace CI with the new value and
  /// delete CI.
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B)
    =0;

  /// ignoreCallingConv - Returns false if this transformation could possibly
  /// change the calling convention.
  virtual bool ignoreCallingConv() { return false; }

  Value *optimizeCall(CallInst *CI, const DataLayout *DL,
                      const TargetLibraryInfo *TLI,
                      const LibCallSimplifier *LCS, IRBuilder<> &B) {
    Caller = CI->getParent()->getParent();
    this->DL = DL;
    this->TLI = TLI;
    this->LCS = LCS;
    if (CI->getCalledFunction())
      Context = &CI->getCalledFunction()->getContext();

    // We never change the calling convention.
    if (!ignoreCallingConv() && CI->getCallingConv() != llvm::CallingConv::C)
      return NULL;

    return callOptimizer(CI->getCalledFunction(), CI, B);
  }
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// isOnlyUsedInZeroEqualityComparison - Return true if it only matters that the
/// value is equal or not-equal to zero.
static bool isOnlyUsedInZeroEqualityComparison(Value *V) {
  for (User *U : V->users()) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(U))
      if (IC->isEquality())
        if (Constant *C = dyn_cast<Constant>(IC->getOperand(1)))
          if (C->isNullValue())
            continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

/// isOnlyUsedInEqualityComparison - Return true if it is only used in equality
/// comparisons with With.
static bool isOnlyUsedInEqualityComparison(Value *V, Value *With) {
  for (User *U : V->users()) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(U))
      if (IC->isEquality() && IC->getOperand(1) == With)
        continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

static bool callHasFloatingPointArgument(const CallInst *CI) {
  for (CallInst::const_op_iterator it = CI->op_begin(), e = CI->op_end();
       it != e; ++it) {
    if ((*it)->getType()->isFloatingPointTy())
      return true;
  }
  return false;
}

/// \brief Check whether the overloaded unary floating point function
/// corresponing to \a Ty is available.
static bool hasUnaryFloatFn(const TargetLibraryInfo *TLI, Type *Ty,
                            LibFunc::Func DoubleFn, LibFunc::Func FloatFn,
                            LibFunc::Func LongDoubleFn) {
  switch (Ty->getTypeID()) {
  case Type::FloatTyID:
    return TLI->has(FloatFn);
  case Type::DoubleTyID:
    return TLI->has(DoubleFn);
  default:
    return TLI->has(LongDoubleFn);
  }
}

//===----------------------------------------------------------------------===//
// Fortified Library Call Optimizations
//===----------------------------------------------------------------------===//

struct FortifiedLibCallOptimization : public LibCallOptimization {
protected:
  virtual bool isFoldable(unsigned SizeCIOp, unsigned SizeArgOp,
			  bool isString) const = 0;
};

struct InstFortifiedLibCallOptimization : public FortifiedLibCallOptimization {
  CallInst *CI;

  bool isFoldable(unsigned SizeCIOp, unsigned SizeArgOp,
                  bool isString) const override {
    if (CI->getArgOperand(SizeCIOp) == CI->getArgOperand(SizeArgOp))
      return true;
    if (ConstantInt *SizeCI =
                           dyn_cast<ConstantInt>(CI->getArgOperand(SizeCIOp))) {
      if (SizeCI->isAllOnesValue())
        return true;
      if (isString) {
        uint64_t Len = GetStringLength(CI->getArgOperand(SizeArgOp));
        // If the length is 0 we don't know how long it is and so we can't
        // remove the check.
        if (Len == 0) return false;
        return SizeCI->getZExtValue() >= Len;
      }
      if (ConstantInt *Arg = dyn_cast<ConstantInt>(
                                                  CI->getArgOperand(SizeArgOp)))
        return SizeCI->getZExtValue() >= Arg->getZExtValue();
    }
    return false;
  }
};

struct MemCpyChkOpt : public InstFortifiedLibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    this->CI = CI;
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != DL->getIntPtrType(Context) ||
        FT->getParamType(3) != DL->getIntPtrType(Context))
      return 0;

    if (isFoldable(3, 2, false)) {
      B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                     CI->getArgOperand(2), 1);
      return CI->getArgOperand(0);
    }
    return 0;
  }
};

struct MemMoveChkOpt : public InstFortifiedLibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    this->CI = CI;
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != DL->getIntPtrType(Context) ||
        FT->getParamType(3) != DL->getIntPtrType(Context))
      return 0;

    if (isFoldable(3, 2, false)) {
      B.CreateMemMove(CI->getArgOperand(0), CI->getArgOperand(1),
                      CI->getArgOperand(2), 1);
      return CI->getArgOperand(0);
    }
    return 0;
  }
};

struct MemSetChkOpt : public InstFortifiedLibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    this->CI = CI;
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isIntegerTy() ||
        FT->getParamType(2) != DL->getIntPtrType(Context) ||
        FT->getParamType(3) != DL->getIntPtrType(Context))
      return 0;

    if (isFoldable(3, 2, false)) {
      Value *Val = B.CreateIntCast(CI->getArgOperand(1), B.getInt8Ty(),
                                   false);
      B.CreateMemSet(CI->getArgOperand(0), Val, CI->getArgOperand(2), 1);
      return CI->getArgOperand(0);
    }
    return 0;
  }
};

struct StrCpyChkOpt : public InstFortifiedLibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    this->CI = CI;
    StringRef Name = Callee->getName();
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 3 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        FT->getParamType(2) != DL->getIntPtrType(Context))
      return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src)      // __strcpy_chk(x,x)  -> x
      return Src;

    // If a) we don't have any length information, or b) we know this will
    // fit then just lower to a plain strcpy. Otherwise we'll keep our
    // strcpy_chk call which may fail at runtime if the size is too long.
    // TODO: It might be nice to get a maximum length out of the possible
    // string lengths for varying.
    if (isFoldable(2, 1, true)) {
      Value *Ret = EmitStrCpy(Dst, Src, B, DL, TLI, Name.substr(2, 6));
      return Ret;
    } else {
      // Maybe we can stil fold __strcpy_chk to __memcpy_chk.
      uint64_t Len = GetStringLength(Src);
      if (Len == 0) return 0;

      // This optimization require DataLayout.
      if (!DL) return 0;

      Value *Ret =
	EmitMemCpyChk(Dst, Src,
                      ConstantInt::get(DL->getIntPtrType(Context), Len),
                      CI->getArgOperand(2), B, DL, TLI);
      return Ret;
    }
    return 0;
  }
};

struct StpCpyChkOpt : public InstFortifiedLibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    this->CI = CI;
    StringRef Name = Callee->getName();
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 3 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        FT->getParamType(2) != DL->getIntPtrType(FT->getParamType(0)))
      return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src) {  // stpcpy(x,x)  -> x+strlen(x)
      Value *StrLen = EmitStrLen(Src, B, DL, TLI);
      return StrLen ? B.CreateInBoundsGEP(Dst, StrLen) : 0;
    }

    // If a) we don't have any length information, or b) we know this will
    // fit then just lower to a plain stpcpy. Otherwise we'll keep our
    // stpcpy_chk call which may fail at runtime if the size is too long.
    // TODO: It might be nice to get a maximum length out of the possible
    // string lengths for varying.
    if (isFoldable(2, 1, true)) {
      Value *Ret = EmitStrCpy(Dst, Src, B, DL, TLI, Name.substr(2, 6));
      return Ret;
    } else {
      // Maybe we can stil fold __stpcpy_chk to __memcpy_chk.
      uint64_t Len = GetStringLength(Src);
      if (Len == 0) return 0;

      // This optimization require DataLayout.
      if (!DL) return 0;

      Type *PT = FT->getParamType(0);
      Value *LenV = ConstantInt::get(DL->getIntPtrType(PT), Len);
      Value *DstEnd = B.CreateGEP(Dst,
                                  ConstantInt::get(DL->getIntPtrType(PT),
                                                   Len - 1));
      if (!EmitMemCpyChk(Dst, Src, LenV, CI->getArgOperand(2), B, DL, TLI))
        return 0;
      return DstEnd;
    }
    return 0;
  }
};

struct StrNCpyChkOpt : public InstFortifiedLibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    this->CI = CI;
    StringRef Name = Callee->getName();
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        !FT->getParamType(2)->isIntegerTy() ||
        FT->getParamType(3) != DL->getIntPtrType(Context))
      return 0;

    if (isFoldable(3, 2, false)) {
      Value *Ret = EmitStrNCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                               CI->getArgOperand(2), B, DL, TLI,
                               Name.substr(2, 7));
      return Ret;
    }
    return 0;
  }
};

//===----------------------------------------------------------------------===//
// String and Memory Library Call Optimizations
//===----------------------------------------------------------------------===//

struct StrCatOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Verify the "strcat" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != B.getInt8PtrTy() ||
        FT->getParamType(0) != FT->getReturnType() ||
        FT->getParamType(1) != FT->getReturnType())
      return 0;

    // Extract some information from the instruction
    Value *Dst = CI->getArgOperand(0);
    Value *Src = CI->getArgOperand(1);

    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;
    --Len;  // Unbias length.

    // Handle the simple, do-nothing case: strcat(x, "") -> x
    if (Len == 0)
      return Dst;

    // These optimizations require DataLayout.
    if (!DL) return 0;

    return emitStrLenMemCpy(Src, Dst, Len, B);
  }

  Value *emitStrLenMemCpy(Value *Src, Value *Dst, uint64_t Len,
                          IRBuilder<> &B) {
    // We need to find the end of the destination string.  That's where the
    // memory is to be moved to. We just generate a call to strlen.
    Value *DstLen = EmitStrLen(Dst, B, DL, TLI);
    if (!DstLen)
      return 0;

    // Now that we have the destination's length, we must index into the
    // destination's pointer to get the actual memcpy destination (end of
    // the string .. we're concatenating).
    Value *CpyDst = B.CreateGEP(Dst, DstLen, "endptr");

    // We have enough information to now generate the memcpy call to do the
    // concatenation for us.  Make a memcpy to copy the nul byte with align = 1.
    B.CreateMemCpy(CpyDst, Src,
                   ConstantInt::get(DL->getIntPtrType(*Context), Len + 1), 1);
    return Dst;
  }
};

struct StrNCatOpt : public StrCatOpt {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Verify the "strncat" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 ||
        FT->getReturnType() != B.getInt8PtrTy() ||
        FT->getParamType(0) != FT->getReturnType() ||
        FT->getParamType(1) != FT->getReturnType() ||
        !FT->getParamType(2)->isIntegerTy())
      return 0;

    // Extract some information from the instruction
    Value *Dst = CI->getArgOperand(0);
    Value *Src = CI->getArgOperand(1);
    uint64_t Len;

    // We don't do anything if length is not constant
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
      Len = LengthArg->getZExtValue();
    else
      return 0;

    // See if we can get the length of the input string.
    uint64_t SrcLen = GetStringLength(Src);
    if (SrcLen == 0) return 0;
    --SrcLen;  // Unbias length.

    // Handle the simple, do-nothing cases:
    // strncat(x, "", c) -> x
    // strncat(x,  c, 0) -> x
    if (SrcLen == 0 || Len == 0) return Dst;

    // These optimizations require DataLayout.
    if (!DL) return 0;

    // We don't optimize this case
    if (Len < SrcLen) return 0;

    // strncat(x, s, c) -> strcat(x, s)
    // s is constant so the strcat can be optimized further
    return emitStrLenMemCpy(Src, Dst, SrcLen, B);
  }
};

struct StrChrOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Verify the "strchr" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != B.getInt8PtrTy() ||
        FT->getParamType(0) != FT->getReturnType() ||
        !FT->getParamType(1)->isIntegerTy(32))
      return 0;

    Value *SrcStr = CI->getArgOperand(0);

    // If the second operand is non-constant, see if we can compute the length
    // of the input string and turn this into memchr.
    ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (CharC == 0) {
      // These optimizations require DataLayout.
      if (!DL) return 0;

      uint64_t Len = GetStringLength(SrcStr);
      if (Len == 0 || !FT->getParamType(1)->isIntegerTy(32))// memchr needs i32.
        return 0;

      return EmitMemChr(SrcStr, CI->getArgOperand(1), // include nul.
                        ConstantInt::get(DL->getIntPtrType(*Context), Len),
                        B, DL, TLI);
    }

    // Otherwise, the character is a constant, see if the first argument is
    // a string literal.  If so, we can constant fold.
    StringRef Str;
    if (!getConstantStringInfo(SrcStr, Str)) {
      if (DL && CharC->isZero()) // strchr(p, 0) -> p + strlen(p)
        return B.CreateGEP(SrcStr, EmitStrLen(SrcStr, B, DL, TLI), "strchr");
      return 0;
    }

    // Compute the offset, make sure to handle the case when we're searching for
    // zero (a weird way to spell strlen).
    size_t I = (0xFF & CharC->getSExtValue()) == 0 ?
        Str.size() : Str.find(CharC->getSExtValue());
    if (I == StringRef::npos) // Didn't find the char.  strchr returns null.
      return Constant::getNullValue(CI->getType());

    // strchr(s+n,c)  -> gep(s+n+i,c)
    return B.CreateGEP(SrcStr, B.getInt64(I), "strchr");
  }
};

struct StrRChrOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Verify the "strrchr" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != B.getInt8PtrTy() ||
        FT->getParamType(0) != FT->getReturnType() ||
        !FT->getParamType(1)->isIntegerTy(32))
      return 0;

    Value *SrcStr = CI->getArgOperand(0);
    ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));

    // Cannot fold anything if we're not looking for a constant.
    if (!CharC)
      return 0;

    StringRef Str;
    if (!getConstantStringInfo(SrcStr, Str)) {
      // strrchr(s, 0) -> strchr(s, 0)
      if (DL && CharC->isZero())
        return EmitStrChr(SrcStr, '\0', B, DL, TLI);
      return 0;
    }

    // Compute the offset.
    size_t I = (0xFF & CharC->getSExtValue()) == 0 ?
        Str.size() : Str.rfind(CharC->getSExtValue());
    if (I == StringRef::npos) // Didn't find the char. Return null.
      return Constant::getNullValue(CI->getType());

    // strrchr(s+n,c) -> gep(s+n+i,c)
    return B.CreateGEP(SrcStr, B.getInt64(I), "strrchr");
  }
};

struct StrCmpOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Verify the "strcmp" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        !FT->getReturnType()->isIntegerTy(32) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy())
      return 0;

    Value *Str1P = CI->getArgOperand(0), *Str2P = CI->getArgOperand(1);
    if (Str1P == Str2P)      // strcmp(x,x)  -> 0
      return ConstantInt::get(CI->getType(), 0);

    StringRef Str1, Str2;
    bool HasStr1 = getConstantStringInfo(Str1P, Str1);
    bool HasStr2 = getConstantStringInfo(Str2P, Str2);

    // strcmp(x, y)  -> cnst  (if both x and y are constant strings)
    if (HasStr1 && HasStr2)
      return ConstantInt::get(CI->getType(), Str1.compare(Str2));

    if (HasStr1 && Str1.empty()) // strcmp("", x) -> -*x
      return B.CreateNeg(B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"),
                                      CI->getType()));

    if (HasStr2 && Str2.empty()) // strcmp(x,"") -> *x
      return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());

    // strcmp(P, "x") -> memcmp(P, "x", 2)
    uint64_t Len1 = GetStringLength(Str1P);
    uint64_t Len2 = GetStringLength(Str2P);
    if (Len1 && Len2) {
      // These optimizations require DataLayout.
      if (!DL) return 0;

      return EmitMemCmp(Str1P, Str2P,
                        ConstantInt::get(DL->getIntPtrType(*Context),
                        std::min(Len1, Len2)), B, DL, TLI);
    }

    return 0;
  }
};

struct StrNCmpOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Verify the "strncmp" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 ||
        !FT->getReturnType()->isIntegerTy(32) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        !FT->getParamType(2)->isIntegerTy())
      return 0;

    Value *Str1P = CI->getArgOperand(0), *Str2P = CI->getArgOperand(1);
    if (Str1P == Str2P)      // strncmp(x,x,n)  -> 0
      return ConstantInt::get(CI->getType(), 0);

    // Get the length argument if it is constant.
    uint64_t Length;
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
      Length = LengthArg->getZExtValue();
    else
      return 0;

    if (Length == 0) // strncmp(x,y,0)   -> 0
      return ConstantInt::get(CI->getType(), 0);

    if (DL && Length == 1) // strncmp(x,y,1) -> memcmp(x,y,1)
      return EmitMemCmp(Str1P, Str2P, CI->getArgOperand(2), B, DL, TLI);

    StringRef Str1, Str2;
    bool HasStr1 = getConstantStringInfo(Str1P, Str1);
    bool HasStr2 = getConstantStringInfo(Str2P, Str2);

    // strncmp(x, y)  -> cnst  (if both x and y are constant strings)
    if (HasStr1 && HasStr2) {
      StringRef SubStr1 = Str1.substr(0, Length);
      StringRef SubStr2 = Str2.substr(0, Length);
      return ConstantInt::get(CI->getType(), SubStr1.compare(SubStr2));
    }

    if (HasStr1 && Str1.empty())  // strncmp("", x, n) -> -*x
      return B.CreateNeg(B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"),
                                      CI->getType()));

    if (HasStr2 && Str2.empty())  // strncmp(x, "", n) -> *x
      return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());

    return 0;
  }
};

struct StrCpyOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Verify the "strcpy" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy())
      return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src)      // strcpy(x,x)  -> x
      return Src;

    // These optimizations require DataLayout.
    if (!DL) return 0;

    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;

    // We have enough information to now generate the memcpy call to do the
    // copy for us.  Make a memcpy to copy the nul byte with align = 1.
    B.CreateMemCpy(Dst, Src,
		   ConstantInt::get(DL->getIntPtrType(*Context), Len), 1);
    return Dst;
  }
};

struct StpCpyOpt: public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Verify the "stpcpy" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy())
      return 0;

    // These optimizations require DataLayout.
    if (!DL) return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src) {  // stpcpy(x,x)  -> x+strlen(x)
      Value *StrLen = EmitStrLen(Src, B, DL, TLI);
      return StrLen ? B.CreateInBoundsGEP(Dst, StrLen) : 0;
    }

    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;

    Type *PT = FT->getParamType(0);
    Value *LenV = ConstantInt::get(DL->getIntPtrType(PT), Len);
    Value *DstEnd = B.CreateGEP(Dst,
                                ConstantInt::get(DL->getIntPtrType(PT),
                                                 Len - 1));

    // We have enough information to now generate the memcpy call to do the
    // copy for us.  Make a memcpy to copy the nul byte with align = 1.
    B.CreateMemCpy(Dst, Src, LenV, 1);
    return DstEnd;
  }
};

struct StrNCpyOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        !FT->getParamType(2)->isIntegerTy())
      return 0;

    Value *Dst = CI->getArgOperand(0);
    Value *Src = CI->getArgOperand(1);
    Value *LenOp = CI->getArgOperand(2);

    // See if we can get the length of the input string.
    uint64_t SrcLen = GetStringLength(Src);
    if (SrcLen == 0) return 0;
    --SrcLen;

    if (SrcLen == 0) {
      // strncpy(x, "", y) -> memset(x, '\0', y, 1)
      B.CreateMemSet(Dst, B.getInt8('\0'), LenOp, 1);
      return Dst;
    }

    uint64_t Len;
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(LenOp))
      Len = LengthArg->getZExtValue();
    else
      return 0;

    if (Len == 0) return Dst; // strncpy(x, y, 0) -> x

    // These optimizations require DataLayout.
    if (!DL) return 0;

    // Let strncpy handle the zero padding
    if (Len > SrcLen+1) return 0;

    Type *PT = FT->getParamType(0);
    // strncpy(x, s, c) -> memcpy(x, s, c, 1) [s and c are constant]
    B.CreateMemCpy(Dst, Src,
                   ConstantInt::get(DL->getIntPtrType(PT), Len), 1);

    return Dst;
  }
};

struct StrLenOpt : public LibCallOptimization {
  bool ignoreCallingConv() override { return true; }
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 1 ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    Value *Src = CI->getArgOperand(0);

    // Constant folding: strlen("xyz") -> 3
    if (uint64_t Len = GetStringLength(Src))
      return ConstantInt::get(CI->getType(), Len-1);

    // strlen(x) != 0 --> *x != 0
    // strlen(x) == 0 --> *x == 0
    if (isOnlyUsedInZeroEqualityComparison(CI))
      return B.CreateZExt(B.CreateLoad(Src, "strlenfirst"), CI->getType());
    return 0;
  }
};

struct StrPBrkOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        FT->getParamType(1) != FT->getParamType(0) ||
        FT->getReturnType() != FT->getParamType(0))
      return 0;

    StringRef S1, S2;
    bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
    bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

    // strpbrk(s, "") -> NULL
    // strpbrk("", s) -> NULL
    if ((HasS1 && S1.empty()) || (HasS2 && S2.empty()))
      return Constant::getNullValue(CI->getType());

    // Constant folding.
    if (HasS1 && HasS2) {
      size_t I = S1.find_first_of(S2);
      if (I == StringRef::npos) // No match.
        return Constant::getNullValue(CI->getType());

      return B.CreateGEP(CI->getArgOperand(0), B.getInt64(I), "strpbrk");
    }

    // strpbrk(s, "a") -> strchr(s, 'a')
    if (DL && HasS2 && S2.size() == 1)
      return EmitStrChr(CI->getArgOperand(0), S2[0], B, DL, TLI);

    return 0;
  }
};

struct StrToOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    if ((FT->getNumParams() != 2 && FT->getNumParams() != 3) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy())
      return 0;

    Value *EndPtr = CI->getArgOperand(1);
    if (isa<ConstantPointerNull>(EndPtr)) {
      // With a null EndPtr, this function won't capture the main argument.
      // It would be readonly too, except that it still may write to errno.
      CI->addAttribute(1, Attribute::NoCapture);
    }

    return 0;
  }
};

struct StrSpnOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        FT->getParamType(1) != FT->getParamType(0) ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    StringRef S1, S2;
    bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
    bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

    // strspn(s, "") -> 0
    // strspn("", s) -> 0
    if ((HasS1 && S1.empty()) || (HasS2 && S2.empty()))
      return Constant::getNullValue(CI->getType());

    // Constant folding.
    if (HasS1 && HasS2) {
      size_t Pos = S1.find_first_not_of(S2);
      if (Pos == StringRef::npos) Pos = S1.size();
      return ConstantInt::get(CI->getType(), Pos);
    }

    return 0;
  }
};

struct StrCSpnOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getParamType(0) != B.getInt8PtrTy() ||
        FT->getParamType(1) != FT->getParamType(0) ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    StringRef S1, S2;
    bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
    bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

    // strcspn("", s) -> 0
    if (HasS1 && S1.empty())
      return Constant::getNullValue(CI->getType());

    // Constant folding.
    if (HasS1 && HasS2) {
      size_t Pos = S1.find_first_of(S2);
      if (Pos == StringRef::npos) Pos = S1.size();
      return ConstantInt::get(CI->getType(), Pos);
    }

    // strcspn(s, "") -> strlen(s)
    if (DL && HasS2 && S2.empty())
      return EmitStrLen(CI->getArgOperand(0), B, DL, TLI);

    return 0;
  }
};

struct StrStrOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !FT->getReturnType()->isPointerTy())
      return 0;

    // fold strstr(x, x) -> x.
    if (CI->getArgOperand(0) == CI->getArgOperand(1))
      return B.CreateBitCast(CI->getArgOperand(0), CI->getType());

    // fold strstr(a, b) == a -> strncmp(a, b, strlen(b)) == 0
    if (DL && isOnlyUsedInEqualityComparison(CI, CI->getArgOperand(0))) {
      Value *StrLen = EmitStrLen(CI->getArgOperand(1), B, DL, TLI);
      if (!StrLen)
        return 0;
      Value *StrNCmp = EmitStrNCmp(CI->getArgOperand(0), CI->getArgOperand(1),
                                   StrLen, B, DL, TLI);
      if (!StrNCmp)
        return 0;
      for (auto UI = CI->user_begin(), UE = CI->user_end(); UI != UE;) {
        ICmpInst *Old = cast<ICmpInst>(*UI++);
        Value *Cmp = B.CreateICmp(Old->getPredicate(), StrNCmp,
                                  ConstantInt::getNullValue(StrNCmp->getType()),
                                  "cmp");
        LCS->replaceAllUsesWith(Old, Cmp);
      }
      return CI;
    }

    // See if either input string is a constant string.
    StringRef SearchStr, ToFindStr;
    bool HasStr1 = getConstantStringInfo(CI->getArgOperand(0), SearchStr);
    bool HasStr2 = getConstantStringInfo(CI->getArgOperand(1), ToFindStr);

    // fold strstr(x, "") -> x.
    if (HasStr2 && ToFindStr.empty())
      return B.CreateBitCast(CI->getArgOperand(0), CI->getType());

    // If both strings are known, constant fold it.
    if (HasStr1 && HasStr2) {
      size_t Offset = SearchStr.find(ToFindStr);

      if (Offset == StringRef::npos) // strstr("foo", "bar") -> null
        return Constant::getNullValue(CI->getType());

      // strstr("abcd", "bc") -> gep((char*)"abcd", 1)
      Value *Result = CastToCStr(CI->getArgOperand(0), B);
      Result = B.CreateConstInBoundsGEP1_64(Result, Offset, "strstr");
      return B.CreateBitCast(Result, CI->getType());
    }

    // fold strstr(x, "y") -> strchr(x, 'y').
    if (HasStr2 && ToFindStr.size() == 1) {
      Value *StrChr= EmitStrChr(CI->getArgOperand(0), ToFindStr[0], B, DL, TLI);
      return StrChr ? B.CreateBitCast(StrChr, CI->getType()) : 0;
    }
    return 0;
  }
};

struct MemCmpOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !FT->getReturnType()->isIntegerTy(32))
      return 0;

    Value *LHS = CI->getArgOperand(0), *RHS = CI->getArgOperand(1);

    if (LHS == RHS)  // memcmp(s,s,x) -> 0
      return Constant::getNullValue(CI->getType());

    // Make sure we have a constant length.
    ConstantInt *LenC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!LenC) return 0;
    uint64_t Len = LenC->getZExtValue();

    if (Len == 0) // memcmp(s1,s2,0) -> 0
      return Constant::getNullValue(CI->getType());

    // memcmp(S1,S2,1) -> *(unsigned char*)LHS - *(unsigned char*)RHS
    if (Len == 1) {
      Value *LHSV = B.CreateZExt(B.CreateLoad(CastToCStr(LHS, B), "lhsc"),
                                 CI->getType(), "lhsv");
      Value *RHSV = B.CreateZExt(B.CreateLoad(CastToCStr(RHS, B), "rhsc"),
                                 CI->getType(), "rhsv");
      return B.CreateSub(LHSV, RHSV, "chardiff");
    }

    // Constant folding: memcmp(x, y, l) -> cnst (all arguments are constant)
    StringRef LHSStr, RHSStr;
    if (getConstantStringInfo(LHS, LHSStr) &&
        getConstantStringInfo(RHS, RHSStr)) {
      // Make sure we're not reading out-of-bounds memory.
      if (Len > LHSStr.size() || Len > RHSStr.size())
        return 0;
      // Fold the memcmp and normalize the result.  This way we get consistent
      // results across multiple platforms.
      uint64_t Ret = 0;
      int Cmp = memcmp(LHSStr.data(), RHSStr.data(), Len);
      if (Cmp < 0)
        Ret = -1;
      else if (Cmp > 0)
        Ret = 1;
      return ConstantInt::get(CI->getType(), Ret);
    }

    return 0;
  }
};

struct MemCpyOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // These optimizations require DataLayout.
    if (!DL) return 0;

    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != DL->getIntPtrType(*Context))
      return 0;

    // memcpy(x, y, n) -> llvm.memcpy(x, y, n, 1)
    B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                   CI->getArgOperand(2), 1);
    return CI->getArgOperand(0);
  }
};

struct MemMoveOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // These optimizations require DataLayout.
    if (!DL) return 0;

    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != DL->getIntPtrType(*Context))
      return 0;

    // memmove(x, y, n) -> llvm.memmove(x, y, n, 1)
    B.CreateMemMove(CI->getArgOperand(0), CI->getArgOperand(1),
                    CI->getArgOperand(2), 1);
    return CI->getArgOperand(0);
  }
};

struct MemSetOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // These optimizations require DataLayout.
    if (!DL) return 0;

    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isIntegerTy() ||
        FT->getParamType(2) != DL->getIntPtrType(FT->getParamType(0)))
      return 0;

    // memset(p, v, n) -> llvm.memset(p, v, n, 1)
    Value *Val = B.CreateIntCast(CI->getArgOperand(1), B.getInt8Ty(), false);
    B.CreateMemSet(CI->getArgOperand(0), Val, CI->getArgOperand(2), 1);
    return CI->getArgOperand(0);
  }
};

//===----------------------------------------------------------------------===//
// Math Library Optimizations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Double -> Float Shrinking Optimizations for Unary Functions like 'floor'

struct UnaryDoubleFPOpt : public LibCallOptimization {
  bool CheckRetType;
  UnaryDoubleFPOpt(bool CheckReturnType): CheckRetType(CheckReturnType) {}
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 1 || !FT->getReturnType()->isDoubleTy() ||
        !FT->getParamType(0)->isDoubleTy())
      return 0;

    if (CheckRetType) {
      // Check if all the uses for function like 'sin' are converted to float.
      for (User *U : CI->users()) {
        FPTruncInst *Cast = dyn_cast<FPTruncInst>(U);
        if (Cast == 0 || !Cast->getType()->isFloatTy())
          return 0;
      }
    }

    // If this is something like 'floor((double)floatval)', convert to floorf.
    FPExtInst *Cast = dyn_cast<FPExtInst>(CI->getArgOperand(0));
    if (Cast == 0 || !Cast->getOperand(0)->getType()->isFloatTy())
      return 0;

    // floor((double)floatval) -> (double)floorf(floatval)
    Value *V = Cast->getOperand(0);
    V = EmitUnaryFloatFnCall(V, Callee->getName(), B, Callee->getAttributes());
    return B.CreateFPExt(V, B.getDoubleTy());
  }
};

// Double -> Float Shrinking Optimizations for Binary Functions like 'fmin/fmax'
struct BinaryDoubleFPOpt : public LibCallOptimization {
  bool CheckRetType;
  BinaryDoubleFPOpt(bool CheckReturnType): CheckRetType(CheckReturnType) {}
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    if (FT->getNumParams() != 2 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        !FT->getParamType(0)->isFloatingPointTy())
      return 0;

    if (CheckRetType) {
      // Check if all the uses for function like 'fmin/fmax' are converted to
      // float.
      for (User *U : CI->users()) {
        FPTruncInst *Cast = dyn_cast<FPTruncInst>(U);
        if (Cast == 0 || !Cast->getType()->isFloatTy())
          return 0;
      }
    }

    // If this is something like 'fmin((double)floatval1, (double)floatval2)',
    // we convert it to fminf.
    FPExtInst *Cast1 = dyn_cast<FPExtInst>(CI->getArgOperand(0));
    FPExtInst *Cast2 = dyn_cast<FPExtInst>(CI->getArgOperand(1));
    if (Cast1 == 0 || !Cast1->getOperand(0)->getType()->isFloatTy() ||
        Cast2 == 0 || !Cast2->getOperand(0)->getType()->isFloatTy())
      return 0;

    // fmin((double)floatval1, (double)floatval2)
    //                      -> (double)fmin(floatval1, floatval2)
    Value *V = NULL;
    Value *V1 = Cast1->getOperand(0);
    Value *V2 = Cast2->getOperand(0);
    V = EmitBinaryFloatFnCall(V1, V2, Callee->getName(), B,
                              Callee->getAttributes());
    return B.CreateFPExt(V, B.getDoubleTy());
  }
};

struct UnsafeFPLibCallOptimization : public LibCallOptimization {
  bool UnsafeFPShrink;
  UnsafeFPLibCallOptimization(bool UnsafeFPShrink) {
    this->UnsafeFPShrink = UnsafeFPShrink;
  }
};

struct CosOpt : public UnsafeFPLibCallOptimization {
  CosOpt(bool UnsafeFPShrink) : UnsafeFPLibCallOptimization(UnsafeFPShrink) {}
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    Value *Ret = NULL;
    if (UnsafeFPShrink && Callee->getName() == "cos" &&
        TLI->has(LibFunc::cosf)) {
      UnaryDoubleFPOpt UnsafeUnaryDoubleFP(true);
      Ret = UnsafeUnaryDoubleFP.callOptimizer(Callee, CI, B);
    }

    FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 1 argument of FP type, which matches the
    // result type.
    if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isFloatingPointTy())
      return Ret;

    // cos(-x) -> cos(x)
    Value *Op1 = CI->getArgOperand(0);
    if (BinaryOperator::isFNeg(Op1)) {
      BinaryOperator *BinExpr = cast<BinaryOperator>(Op1);
      return B.CreateCall(Callee, BinExpr->getOperand(1), "cos");
    }
    return Ret;
  }
};

struct PowOpt : public UnsafeFPLibCallOptimization {
  PowOpt(bool UnsafeFPShrink) : UnsafeFPLibCallOptimization(UnsafeFPShrink) {}
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    Value *Ret = NULL;
    if (UnsafeFPShrink && Callee->getName() == "pow" &&
        TLI->has(LibFunc::powf)) {
      UnaryDoubleFPOpt UnsafeUnaryDoubleFP(true);
      Ret = UnsafeUnaryDoubleFP.callOptimizer(Callee, CI, B);
    }

    FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    if (FT->getNumParams() != 2 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        !FT->getParamType(0)->isFloatingPointTy())
      return Ret;

    Value *Op1 = CI->getArgOperand(0), *Op2 = CI->getArgOperand(1);
    if (ConstantFP *Op1C = dyn_cast<ConstantFP>(Op1)) {
      // pow(1.0, x) -> 1.0
      if (Op1C->isExactlyValue(1.0))
        return Op1C;
      // pow(2.0, x) -> exp2(x)
      if (Op1C->isExactlyValue(2.0) &&
          hasUnaryFloatFn(TLI, Op1->getType(), LibFunc::exp2, LibFunc::exp2f,
                          LibFunc::exp2l))
        return EmitUnaryFloatFnCall(Op2, "exp2", B, Callee->getAttributes());
      // pow(10.0, x) -> exp10(x)
      if (Op1C->isExactlyValue(10.0) &&
          hasUnaryFloatFn(TLI, Op1->getType(), LibFunc::exp10, LibFunc::exp10f,
                          LibFunc::exp10l))
        return EmitUnaryFloatFnCall(Op2, TLI->getName(LibFunc::exp10), B,
                                    Callee->getAttributes());
    }

    ConstantFP *Op2C = dyn_cast<ConstantFP>(Op2);
    if (Op2C == 0) return Ret;

    if (Op2C->getValueAPF().isZero())  // pow(x, 0.0) -> 1.0
      return ConstantFP::get(CI->getType(), 1.0);

    if (Op2C->isExactlyValue(0.5) &&
        hasUnaryFloatFn(TLI, Op2->getType(), LibFunc::sqrt, LibFunc::sqrtf,
                        LibFunc::sqrtl) &&
        hasUnaryFloatFn(TLI, Op2->getType(), LibFunc::fabs, LibFunc::fabsf,
                        LibFunc::fabsl)) {
      // Expand pow(x, 0.5) to (x == -infinity ? +infinity : fabs(sqrt(x))).
      // This is faster than calling pow, and still handles negative zero
      // and negative infinity correctly.
      // TODO: In fast-math mode, this could be just sqrt(x).
      // TODO: In finite-only mode, this could be just fabs(sqrt(x)).
      Value *Inf = ConstantFP::getInfinity(CI->getType());
      Value *NegInf = ConstantFP::getInfinity(CI->getType(), true);
      Value *Sqrt = EmitUnaryFloatFnCall(Op1, "sqrt", B,
                                         Callee->getAttributes());
      Value *FAbs = EmitUnaryFloatFnCall(Sqrt, "fabs", B,
                                         Callee->getAttributes());
      Value *FCmp = B.CreateFCmpOEQ(Op1, NegInf);
      Value *Sel = B.CreateSelect(FCmp, Inf, FAbs);
      return Sel;
    }

    if (Op2C->isExactlyValue(1.0))  // pow(x, 1.0) -> x
      return Op1;
    if (Op2C->isExactlyValue(2.0))  // pow(x, 2.0) -> x*x
      return B.CreateFMul(Op1, Op1, "pow2");
    if (Op2C->isExactlyValue(-1.0)) // pow(x, -1.0) -> 1.0/x
      return B.CreateFDiv(ConstantFP::get(CI->getType(), 1.0),
                          Op1, "powrecip");
    return 0;
  }
};

struct Exp2Opt : public UnsafeFPLibCallOptimization {
  Exp2Opt(bool UnsafeFPShrink) : UnsafeFPLibCallOptimization(UnsafeFPShrink) {}
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    Value *Ret = NULL;
    if (UnsafeFPShrink && Callee->getName() == "exp2" &&
        TLI->has(LibFunc::exp2f)) {
      UnaryDoubleFPOpt UnsafeUnaryDoubleFP(true);
      Ret = UnsafeUnaryDoubleFP.callOptimizer(Callee, CI, B);
    }

    FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 1 argument of FP type, which matches the
    // result type.
    if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isFloatingPointTy())
      return Ret;

    Value *Op = CI->getArgOperand(0);
    // Turn exp2(sitofp(x)) -> ldexp(1.0, sext(x))  if sizeof(x) <= 32
    // Turn exp2(uitofp(x)) -> ldexp(1.0, zext(x))  if sizeof(x) < 32
    LibFunc::Func LdExp = LibFunc::ldexpl;
    if (Op->getType()->isFloatTy())
      LdExp = LibFunc::ldexpf;
    else if (Op->getType()->isDoubleTy())
      LdExp = LibFunc::ldexp;

    if (TLI->has(LdExp)) {
      Value *LdExpArg = 0;
      if (SIToFPInst *OpC = dyn_cast<SIToFPInst>(Op)) {
        if (OpC->getOperand(0)->getType()->getPrimitiveSizeInBits() <= 32)
          LdExpArg = B.CreateSExt(OpC->getOperand(0), B.getInt32Ty());
      } else if (UIToFPInst *OpC = dyn_cast<UIToFPInst>(Op)) {
        if (OpC->getOperand(0)->getType()->getPrimitiveSizeInBits() < 32)
          LdExpArg = B.CreateZExt(OpC->getOperand(0), B.getInt32Ty());
      }

      if (LdExpArg) {
        Constant *One = ConstantFP::get(*Context, APFloat(1.0f));
        if (!Op->getType()->isFloatTy())
          One = ConstantExpr::getFPExtend(One, Op->getType());

        Module *M = Caller->getParent();
        Value *Callee =
            M->getOrInsertFunction(TLI->getName(LdExp), Op->getType(),
                                   Op->getType(), B.getInt32Ty(), NULL);
        CallInst *CI = B.CreateCall2(Callee, One, LdExpArg);
        if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
          CI->setCallingConv(F->getCallingConv());

        return CI;
      }
    }
    return Ret;
  }
};

struct SinCosPiOpt : public LibCallOptimization {
  SinCosPiOpt() {}

  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Make sure the prototype is as expected, otherwise the rest of the
    // function is probably invalid and likely to abort.
    if (!isTrigLibCall(CI))
      return 0;

    Value *Arg = CI->getArgOperand(0);
    SmallVector<CallInst *, 1> SinCalls;
    SmallVector<CallInst *, 1> CosCalls;
    SmallVector<CallInst *, 1> SinCosCalls;

    bool IsFloat = Arg->getType()->isFloatTy();

    // Look for all compatible sinpi, cospi and sincospi calls with the same
    // argument. If there are enough (in some sense) we can make the
    // substitution.
    for (User *U : Arg->users())
      classifyArgUse(U, CI->getParent(), IsFloat, SinCalls, CosCalls,
                     SinCosCalls);

    // It's only worthwhile if both sinpi and cospi are actually used.
    if (SinCosCalls.empty() && (SinCalls.empty() || CosCalls.empty()))
      return 0;

    Value *Sin, *Cos, *SinCos;
    insertSinCosCall(B, CI->getCalledFunction(), Arg, IsFloat, Sin, Cos,
                     SinCos);

    replaceTrigInsts(SinCalls, Sin);
    replaceTrigInsts(CosCalls, Cos);
    replaceTrigInsts(SinCosCalls, SinCos);

    return 0;
  }

  bool isTrigLibCall(CallInst *CI) {
    Function *Callee = CI->getCalledFunction();
    FunctionType *FT = Callee->getFunctionType();

    // We can only hope to do anything useful if we can ignore things like errno
    // and floating-point exceptions.
    bool AttributesSafe = CI->hasFnAttr(Attribute::NoUnwind) &&
                          CI->hasFnAttr(Attribute::ReadNone);

    // Other than that we need float(float) or double(double)
    return AttributesSafe && FT->getNumParams() == 1 &&
           FT->getReturnType() == FT->getParamType(0) &&
           (FT->getParamType(0)->isFloatTy() ||
            FT->getParamType(0)->isDoubleTy());
  }

  void classifyArgUse(Value *Val, BasicBlock *BB, bool IsFloat,
                      SmallVectorImpl<CallInst *> &SinCalls,
                      SmallVectorImpl<CallInst *> &CosCalls,
                      SmallVectorImpl<CallInst *> &SinCosCalls) {
    CallInst *CI = dyn_cast<CallInst>(Val);

    if (!CI)
      return;

    Function *Callee = CI->getCalledFunction();
    StringRef FuncName = Callee->getName();
    LibFunc::Func Func;
    if (!TLI->getLibFunc(FuncName, Func) || !TLI->has(Func) ||
        !isTrigLibCall(CI))
      return;

    if (IsFloat) {
      if (Func == LibFunc::sinpif)
        SinCalls.push_back(CI);
      else if (Func == LibFunc::cospif)
        CosCalls.push_back(CI);
      else if (Func == LibFunc::sincospif_stret)
        SinCosCalls.push_back(CI);
    } else {
      if (Func == LibFunc::sinpi)
        SinCalls.push_back(CI);
      else if (Func == LibFunc::cospi)
        CosCalls.push_back(CI);
      else if (Func == LibFunc::sincospi_stret)
        SinCosCalls.push_back(CI);
    }
  }

  void replaceTrigInsts(SmallVectorImpl<CallInst*> &Calls, Value *Res) {
    for (SmallVectorImpl<CallInst*>::iterator I = Calls.begin(),
           E = Calls.end();
         I != E; ++I) {
      LCS->replaceAllUsesWith(*I, Res);
    }
  }

  void insertSinCosCall(IRBuilder<> &B, Function *OrigCallee, Value *Arg,
                        bool UseFloat, Value *&Sin, Value *&Cos,
                        Value *&SinCos) {
    Type *ArgTy = Arg->getType();
    Type *ResTy;
    StringRef Name;

    Triple T(OrigCallee->getParent()->getTargetTriple());
    if (UseFloat) {
      Name = "__sincospif_stret";

      assert(T.getArch() != Triple::x86 && "x86 messy and unsupported for now");
      // x86_64 can't use {float, float} since that would be returned in both
      // xmm0 and xmm1, which isn't what a real struct would do.
      ResTy = T.getArch() == Triple::x86_64
                  ? static_cast<Type *>(VectorType::get(ArgTy, 2))
                  : static_cast<Type *>(StructType::get(ArgTy, ArgTy, NULL));
    } else {
      Name = "__sincospi_stret";
      ResTy = StructType::get(ArgTy, ArgTy, NULL);
    }

    Module *M = OrigCallee->getParent();
    Value *Callee = M->getOrInsertFunction(Name, OrigCallee->getAttributes(),
                                           ResTy, ArgTy, NULL);

    if (Instruction *ArgInst = dyn_cast<Instruction>(Arg)) {
      // If the argument is an instruction, it must dominate all uses so put our
      // sincos call there.
      BasicBlock::iterator Loc = ArgInst;
      B.SetInsertPoint(ArgInst->getParent(), ++Loc);
    } else {
      // Otherwise (e.g. for a constant) the beginning of the function is as
      // good a place as any.
      BasicBlock &EntryBB = B.GetInsertBlock()->getParent()->getEntryBlock();
      B.SetInsertPoint(&EntryBB, EntryBB.begin());
    }

    SinCos = B.CreateCall(Callee, Arg, "sincospi");

    if (SinCos->getType()->isStructTy()) {
      Sin = B.CreateExtractValue(SinCos, 0, "sinpi");
      Cos = B.CreateExtractValue(SinCos, 1, "cospi");
    } else {
      Sin = B.CreateExtractElement(SinCos, ConstantInt::get(B.getInt32Ty(), 0),
                                   "sinpi");
      Cos = B.CreateExtractElement(SinCos, ConstantInt::get(B.getInt32Ty(), 1),
                                   "cospi");
    }
  }

};

//===----------------------------------------------------------------------===//
// Integer Library Call Optimizations
//===----------------------------------------------------------------------===//

struct FFSOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    if (FT->getNumParams() != 1 ||
        !FT->getReturnType()->isIntegerTy(32) ||
        !FT->getParamType(0)->isIntegerTy())
      return 0;

    Value *Op = CI->getArgOperand(0);

    // Constant fold.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op)) {
      if (CI->isZero()) // ffs(0) -> 0.
        return B.getInt32(0);
      // ffs(c) -> cttz(c)+1
      return B.getInt32(CI->getValue().countTrailingZeros() + 1);
    }

    // ffs(x) -> x != 0 ? (i32)llvm.cttz(x)+1 : 0
    Type *ArgType = Op->getType();
    Value *F = Intrinsic::getDeclaration(Callee->getParent(),
                                         Intrinsic::cttz, ArgType);
    Value *V = B.CreateCall2(F, Op, B.getFalse(), "cttz");
    V = B.CreateAdd(V, ConstantInt::get(V->getType(), 1));
    V = B.CreateIntCast(V, B.getInt32Ty(), false);

    Value *Cond = B.CreateICmpNE(Op, Constant::getNullValue(ArgType));
    return B.CreateSelect(Cond, V, B.getInt32(0));
  }
};

struct AbsOpt : public LibCallOptimization {
  bool ignoreCallingConv() override { return true; }
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    // We require integer(integer) where the types agree.
    if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy() ||
        FT->getParamType(0) != FT->getReturnType())
      return 0;

    // abs(x) -> x >s -1 ? x : -x
    Value *Op = CI->getArgOperand(0);
    Value *Pos = B.CreateICmpSGT(Op, Constant::getAllOnesValue(Op->getType()),
                                 "ispos");
    Value *Neg = B.CreateNeg(Op, "neg");
    return B.CreateSelect(Pos, Op, Neg);
  }
};

struct IsDigitOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    // We require integer(i32)
    if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy() ||
        !FT->getParamType(0)->isIntegerTy(32))
      return 0;

    // isdigit(c) -> (c-'0') <u 10
    Value *Op = CI->getArgOperand(0);
    Op = B.CreateSub(Op, B.getInt32('0'), "isdigittmp");
    Op = B.CreateICmpULT(Op, B.getInt32(10), "isdigit");
    return B.CreateZExt(Op, CI->getType());
  }
};

struct IsAsciiOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    // We require integer(i32)
    if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy() ||
        !FT->getParamType(0)->isIntegerTy(32))
      return 0;

    // isascii(c) -> c <u 128
    Value *Op = CI->getArgOperand(0);
    Op = B.CreateICmpULT(Op, B.getInt32(128), "isascii");
    return B.CreateZExt(Op, CI->getType());
  }
};

struct ToAsciiOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    FunctionType *FT = Callee->getFunctionType();
    // We require i32(i32)
    if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isIntegerTy(32))
      return 0;

    // toascii(c) -> c & 0x7f
    return B.CreateAnd(CI->getArgOperand(0),
                       ConstantInt::get(CI->getType(),0x7F));
  }
};

//===----------------------------------------------------------------------===//
// Formatting and IO Library Call Optimizations
//===----------------------------------------------------------------------===//

struct ErrorReportingOpt : public LibCallOptimization {
  ErrorReportingOpt(int S = -1) : StreamArg(S) {}

  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &) override {
    // Error reporting calls should be cold, mark them as such.
    // This applies even to non-builtin calls: it is only a hint and applies to
    // functions that the frontend might not understand as builtins.

    // This heuristic was suggested in:
    // Improving Static Branch Prediction in a Compiler
    // Brian L. Deitrich, Ben-Chung Cheng, Wen-mei W. Hwu
    // Proceedings of PACT'98, Oct. 1998, IEEE

    if (!CI->hasFnAttr(Attribute::Cold) && isReportingError(Callee, CI)) {
      CI->addAttribute(AttributeSet::FunctionIndex, Attribute::Cold);
    }

    return 0;
  }

protected:
  bool isReportingError(Function *Callee, CallInst *CI) {
    if (!ColdErrorCalls)
      return false;
 
    if (!Callee || !Callee->isDeclaration())
      return false;

    if (StreamArg < 0)
      return true;

    // These functions might be considered cold, but only if their stream
    // argument is stderr.

    if (StreamArg >= (int) CI->getNumArgOperands())
      return false;
    LoadInst *LI = dyn_cast<LoadInst>(CI->getArgOperand(StreamArg));
    if (!LI)
      return false;
    GlobalVariable *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand());
    if (!GV || !GV->isDeclaration())
      return false;
    return GV->getName() == "stderr";
  }

  int StreamArg;
};

struct PrintFOpt : public LibCallOptimization {
  Value *optimizeFixedFormatString(Function *Callee, CallInst *CI,
                                   IRBuilder<> &B) {
    // Check for a fixed format string.
    StringRef FormatStr;
    if (!getConstantStringInfo(CI->getArgOperand(0), FormatStr))
      return 0;

    // Empty format string -> noop.
    if (FormatStr.empty())  // Tolerate printf's declared void.
      return CI->use_empty() ? (Value*)CI :
                               ConstantInt::get(CI->getType(), 0);

    // Do not do any of the following transformations if the printf return value
    // is used, in general the printf return value is not compatible with either
    // putchar() or puts().
    if (!CI->use_empty())
      return 0;

    // printf("x") -> putchar('x'), even for '%'.
    if (FormatStr.size() == 1) {
      Value *Res = EmitPutChar(B.getInt32(FormatStr[0]), B, DL, TLI);
      if (CI->use_empty() || !Res) return Res;
      return B.CreateIntCast(Res, CI->getType(), true);
    }

    // printf("foo\n") --> puts("foo")
    if (FormatStr[FormatStr.size()-1] == '\n' &&
        FormatStr.find('%') == StringRef::npos) { // No format characters.
      // Create a string literal with no \n on it.  We expect the constant merge
      // pass to be run after this pass, to merge duplicate strings.
      FormatStr = FormatStr.drop_back();
      Value *GV = B.CreateGlobalString(FormatStr, "str");
      Value *NewCI = EmitPutS(GV, B, DL, TLI);
      return (CI->use_empty() || !NewCI) ?
              NewCI :
              ConstantInt::get(CI->getType(), FormatStr.size()+1);
    }

    // Optimize specific format strings.
    // printf("%c", chr) --> putchar(chr)
    if (FormatStr == "%c" && CI->getNumArgOperands() > 1 &&
        CI->getArgOperand(1)->getType()->isIntegerTy()) {
      Value *Res = EmitPutChar(CI->getArgOperand(1), B, DL, TLI);

      if (CI->use_empty() || !Res) return Res;
      return B.CreateIntCast(Res, CI->getType(), true);
    }

    // printf("%s\n", str) --> puts(str)
    if (FormatStr == "%s\n" && CI->getNumArgOperands() > 1 &&
        CI->getArgOperand(1)->getType()->isPointerTy()) {
      return EmitPutS(CI->getArgOperand(1), B, DL, TLI);
    }
    return 0;
  }

  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Require one fixed pointer argument and an integer/void result.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() < 1 || !FT->getParamType(0)->isPointerTy() ||
        !(FT->getReturnType()->isIntegerTy() ||
          FT->getReturnType()->isVoidTy()))
      return 0;

    if (Value *V = optimizeFixedFormatString(Callee, CI, B)) {
      return V;
    }

    // printf(format, ...) -> iprintf(format, ...) if no floating point
    // arguments.
    if (TLI->has(LibFunc::iprintf) && !callHasFloatingPointArgument(CI)) {
      Module *M = B.GetInsertBlock()->getParent()->getParent();
      Constant *IPrintFFn =
        M->getOrInsertFunction("iprintf", FT, Callee->getAttributes());
      CallInst *New = cast<CallInst>(CI->clone());
      New->setCalledFunction(IPrintFFn);
      B.Insert(New);
      return New;
    }
    return 0;
  }
};

struct SPrintFOpt : public LibCallOptimization {
  Value *OptimizeFixedFormatString(Function *Callee, CallInst *CI,
                                   IRBuilder<> &B) {
    // Check for a fixed format string.
    StringRef FormatStr;
    if (!getConstantStringInfo(CI->getArgOperand(1), FormatStr))
      return 0;

    // If we just have a format string (nothing else crazy) transform it.
    if (CI->getNumArgOperands() == 2) {
      // Make sure there's no % in the constant array.  We could try to handle
      // %% -> % in the future if we cared.
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')
          return 0; // we found a format specifier, bail out.

      // These optimizations require DataLayout.
      if (!DL) return 0;

      // sprintf(str, fmt) -> llvm.memcpy(str, fmt, strlen(fmt)+1, 1)
      B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                     ConstantInt::get(DL->getIntPtrType(*Context), // Copy the
                                      FormatStr.size() + 1), 1);   // nul byte.
      return ConstantInt::get(CI->getType(), FormatStr.size());
    }

    // The remaining optimizations require the format string to be "%s" or "%c"
    // and have an extra operand.
    if (FormatStr.size() != 2 || FormatStr[0] != '%' ||
        CI->getNumArgOperands() < 3)
      return 0;

    // Decode the second character of the format string.
    if (FormatStr[1] == 'c') {
      // sprintf(dst, "%c", chr) --> *(i8*)dst = chr; *((i8*)dst+1) = 0
      if (!CI->getArgOperand(2)->getType()->isIntegerTy()) return 0;
      Value *V = B.CreateTrunc(CI->getArgOperand(2), B.getInt8Ty(), "char");
      Value *Ptr = CastToCStr(CI->getArgOperand(0), B);
      B.CreateStore(V, Ptr);
      Ptr = B.CreateGEP(Ptr, B.getInt32(1), "nul");
      B.CreateStore(B.getInt8(0), Ptr);

      return ConstantInt::get(CI->getType(), 1);
    }

    if (FormatStr[1] == 's') {
      // These optimizations require DataLayout.
      if (!DL) return 0;

      // sprintf(dest, "%s", str) -> llvm.memcpy(dest, str, strlen(str)+1, 1)
      if (!CI->getArgOperand(2)->getType()->isPointerTy()) return 0;

      Value *Len = EmitStrLen(CI->getArgOperand(2), B, DL, TLI);
      if (!Len)
        return 0;
      Value *IncLen = B.CreateAdd(Len,
                                  ConstantInt::get(Len->getType(), 1),
                                  "leninc");
      B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(2), IncLen, 1);

      // The sprintf result is the unincremented number of bytes in the string.
      return B.CreateIntCast(Len, CI->getType(), false);
    }
    return 0;
  }

  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Require two fixed pointer arguments and an integer result.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    if (Value *V = OptimizeFixedFormatString(Callee, CI, B)) {
      return V;
    }

    // sprintf(str, format, ...) -> siprintf(str, format, ...) if no floating
    // point arguments.
    if (TLI->has(LibFunc::siprintf) && !callHasFloatingPointArgument(CI)) {
      Module *M = B.GetInsertBlock()->getParent()->getParent();
      Constant *SIPrintFFn =
        M->getOrInsertFunction("siprintf", FT, Callee->getAttributes());
      CallInst *New = cast<CallInst>(CI->clone());
      New->setCalledFunction(SIPrintFFn);
      B.Insert(New);
      return New;
    }
    return 0;
  }
};

struct FPrintFOpt : public LibCallOptimization {
  Value *optimizeFixedFormatString(Function *Callee, CallInst *CI,
                                   IRBuilder<> &B) {
    ErrorReportingOpt ER(/* StreamArg = */ 0);
    (void) ER.callOptimizer(Callee, CI, B);

    // All the optimizations depend on the format string.
    StringRef FormatStr;
    if (!getConstantStringInfo(CI->getArgOperand(1), FormatStr))
      return 0;

    // Do not do any of the following transformations if the fprintf return
    // value is used, in general the fprintf return value is not compatible
    // with fwrite(), fputc() or fputs().
    if (!CI->use_empty())
      return 0;

    // fprintf(F, "foo") --> fwrite("foo", 3, 1, F)
    if (CI->getNumArgOperands() == 2) {
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')  // Could handle %% -> % if we cared.
          return 0; // We found a format specifier.

      // These optimizations require DataLayout.
      if (!DL) return 0;

      return EmitFWrite(CI->getArgOperand(1),
                        ConstantInt::get(DL->getIntPtrType(*Context),
                                         FormatStr.size()),
                        CI->getArgOperand(0), B, DL, TLI);
    }

    // The remaining optimizations require the format string to be "%s" or "%c"
    // and have an extra operand.
    if (FormatStr.size() != 2 || FormatStr[0] != '%' ||
        CI->getNumArgOperands() < 3)
      return 0;

    // Decode the second character of the format string.
    if (FormatStr[1] == 'c') {
      // fprintf(F, "%c", chr) --> fputc(chr, F)
      if (!CI->getArgOperand(2)->getType()->isIntegerTy()) return 0;
      return EmitFPutC(CI->getArgOperand(2), CI->getArgOperand(0), B, DL, TLI);
    }

    if (FormatStr[1] == 's') {
      // fprintf(F, "%s", str) --> fputs(str, F)
      if (!CI->getArgOperand(2)->getType()->isPointerTy())
        return 0;
      return EmitFPutS(CI->getArgOperand(2), CI->getArgOperand(0), B, DL, TLI);
    }
    return 0;
  }

  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Require two fixed paramters as pointers and integer result.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    if (Value *V = optimizeFixedFormatString(Callee, CI, B)) {
      return V;
    }

    // fprintf(stream, format, ...) -> fiprintf(stream, format, ...) if no
    // floating point arguments.
    if (TLI->has(LibFunc::fiprintf) && !callHasFloatingPointArgument(CI)) {
      Module *M = B.GetInsertBlock()->getParent()->getParent();
      Constant *FIPrintFFn =
        M->getOrInsertFunction("fiprintf", FT, Callee->getAttributes());
      CallInst *New = cast<CallInst>(CI->clone());
      New->setCalledFunction(FIPrintFFn);
      B.Insert(New);
      return New;
    }
    return 0;
  }
};

struct FWriteOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    ErrorReportingOpt ER(/* StreamArg = */ 3);
    (void) ER.callOptimizer(Callee, CI, B);

    // Require a pointer, an integer, an integer, a pointer, returning integer.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 4 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isIntegerTy() ||
        !FT->getParamType(2)->isIntegerTy() ||
        !FT->getParamType(3)->isPointerTy() ||
        !FT->getReturnType()->isIntegerTy())
      return 0;

    // Get the element size and count.
    ConstantInt *SizeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    ConstantInt *CountC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!SizeC || !CountC) return 0;
    uint64_t Bytes = SizeC->getZExtValue()*CountC->getZExtValue();

    // If this is writing zero records, remove the call (it's a noop).
    if (Bytes == 0)
      return ConstantInt::get(CI->getType(), 0);

    // If this is writing one byte, turn it into fputc.
    // This optimisation is only valid, if the return value is unused.
    if (Bytes == 1 && CI->use_empty()) {  // fwrite(S,1,1,F) -> fputc(S[0],F)
      Value *Char = B.CreateLoad(CastToCStr(CI->getArgOperand(0), B), "char");
      Value *NewCI = EmitFPutC(Char, CI->getArgOperand(3), B, DL, TLI);
      return NewCI ? ConstantInt::get(CI->getType(), 1) : 0;
    }

    return 0;
  }
};

struct FPutsOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    ErrorReportingOpt ER(/* StreamArg = */ 1);
    (void) ER.callOptimizer(Callee, CI, B);

    // These optimizations require DataLayout.
    if (!DL) return 0;

    // Require two pointers.  Also, we can't optimize if return value is used.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        !CI->use_empty())
      return 0;

    // fputs(s,F) --> fwrite(s,1,strlen(s),F)
    uint64_t Len = GetStringLength(CI->getArgOperand(0));
    if (!Len) return 0;
    // Known to have no uses (see above).
    return EmitFWrite(CI->getArgOperand(0),
                      ConstantInt::get(DL->getIntPtrType(*Context), Len-1),
                      CI->getArgOperand(1), B, DL, TLI);
  }
};

struct PutsOpt : public LibCallOptimization {
  Value *callOptimizer(Function *Callee, CallInst *CI,
                       IRBuilder<> &B) override {
    // Require one fixed pointer argument and an integer/void result.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() < 1 || !FT->getParamType(0)->isPointerTy() ||
        !(FT->getReturnType()->isIntegerTy() ||
          FT->getReturnType()->isVoidTy()))
      return 0;

    // Check for a constant string.
    StringRef Str;
    if (!getConstantStringInfo(CI->getArgOperand(0), Str))
      return 0;

    if (Str.empty() && CI->use_empty()) {
      // puts("") -> putchar('\n')
      Value *Res = EmitPutChar(B.getInt32('\n'), B, DL, TLI);
      if (CI->use_empty() || !Res) return Res;
      return B.CreateIntCast(Res, CI->getType(), true);
    }

    return 0;
  }
};

} // End anonymous namespace.

namespace llvm {

class LibCallSimplifierImpl {
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  const LibCallSimplifier *LCS;
  bool UnsafeFPShrink;

  // Math library call optimizations.
  CosOpt Cos;
  PowOpt Pow;
  Exp2Opt Exp2;
public:
  LibCallSimplifierImpl(const DataLayout *DL, const TargetLibraryInfo *TLI,
                        const LibCallSimplifier *LCS,
                        bool UnsafeFPShrink = false)
    : Cos(UnsafeFPShrink), Pow(UnsafeFPShrink), Exp2(UnsafeFPShrink) {
    this->DL = DL;
    this->TLI = TLI;
    this->LCS = LCS;
    this->UnsafeFPShrink = UnsafeFPShrink;
  }

  Value *optimizeCall(CallInst *CI);
  LibCallOptimization *lookupOptimization(CallInst *CI);
  bool hasFloatVersion(StringRef FuncName);
};

bool LibCallSimplifierImpl::hasFloatVersion(StringRef FuncName) {
  LibFunc::Func Func;
  SmallString<20> FloatFuncName = FuncName;
  FloatFuncName += 'f';
  if (TLI->getLibFunc(FloatFuncName, Func))
    return TLI->has(Func);
  return false;
}

// Fortified library call optimizations.
static MemCpyChkOpt MemCpyChk;
static MemMoveChkOpt MemMoveChk;
static MemSetChkOpt MemSetChk;
static StrCpyChkOpt StrCpyChk;
static StpCpyChkOpt StpCpyChk;
static StrNCpyChkOpt StrNCpyChk;

// String library call optimizations.
static StrCatOpt StrCat;
static StrNCatOpt StrNCat;
static StrChrOpt StrChr;
static StrRChrOpt StrRChr;
static StrCmpOpt StrCmp;
static StrNCmpOpt StrNCmp;
static StrCpyOpt StrCpy;
static StpCpyOpt StpCpy;
static StrNCpyOpt StrNCpy;
static StrLenOpt StrLen;
static StrPBrkOpt StrPBrk;
static StrToOpt StrTo;
static StrSpnOpt StrSpn;
static StrCSpnOpt StrCSpn;
static StrStrOpt StrStr;

// Memory library call optimizations.
static MemCmpOpt MemCmp;
static MemCpyOpt MemCpy;
static MemMoveOpt MemMove;
static MemSetOpt MemSet;

// Math library call optimizations.
static UnaryDoubleFPOpt UnaryDoubleFP(false);
static BinaryDoubleFPOpt BinaryDoubleFP(false);
static UnaryDoubleFPOpt UnsafeUnaryDoubleFP(true);
static SinCosPiOpt SinCosPi;

  // Integer library call optimizations.
static FFSOpt FFS;
static AbsOpt Abs;
static IsDigitOpt IsDigit;
static IsAsciiOpt IsAscii;
static ToAsciiOpt ToAscii;

// Formatting and IO library call optimizations.
static ErrorReportingOpt ErrorReporting;
static ErrorReportingOpt ErrorReporting0(0);
static ErrorReportingOpt ErrorReporting1(1);
static PrintFOpt PrintF;
static SPrintFOpt SPrintF;
static FPrintFOpt FPrintF;
static FWriteOpt FWrite;
static FPutsOpt FPuts;
static PutsOpt Puts;

LibCallOptimization *LibCallSimplifierImpl::lookupOptimization(CallInst *CI) {
  LibFunc::Func Func;
  Function *Callee = CI->getCalledFunction();
  StringRef FuncName = Callee->getName();

  // Next check for intrinsics.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::pow:
       return &Pow;
    case Intrinsic::exp2:
       return &Exp2;
    default:
       return 0;
    }
  }

  // Then check for known library functions.
  if (TLI->getLibFunc(FuncName, Func) && TLI->has(Func)) {
    switch (Func) {
      case LibFunc::strcat:
        return &StrCat;
      case LibFunc::strncat:
        return &StrNCat;
      case LibFunc::strchr:
        return &StrChr;
      case LibFunc::strrchr:
        return &StrRChr;
      case LibFunc::strcmp:
        return &StrCmp;
      case LibFunc::strncmp:
        return &StrNCmp;
      case LibFunc::strcpy:
        return &StrCpy;
      case LibFunc::stpcpy:
        return &StpCpy;
      case LibFunc::strncpy:
        return &StrNCpy;
      case LibFunc::strlen:
        return &StrLen;
      case LibFunc::strpbrk:
        return &StrPBrk;
      case LibFunc::strtol:
      case LibFunc::strtod:
      case LibFunc::strtof:
      case LibFunc::strtoul:
      case LibFunc::strtoll:
      case LibFunc::strtold:
      case LibFunc::strtoull:
        return &StrTo;
      case LibFunc::strspn:
        return &StrSpn;
      case LibFunc::strcspn:
        return &StrCSpn;
      case LibFunc::strstr:
        return &StrStr;
      case LibFunc::memcmp:
        return &MemCmp;
      case LibFunc::memcpy:
        return &MemCpy;
      case LibFunc::memmove:
        return &MemMove;
      case LibFunc::memset:
        return &MemSet;
      case LibFunc::cosf:
      case LibFunc::cos:
      case LibFunc::cosl:
        return &Cos;
      case LibFunc::sinpif:
      case LibFunc::sinpi:
      case LibFunc::cospif:
      case LibFunc::cospi:
        return &SinCosPi;
      case LibFunc::powf:
      case LibFunc::pow:
      case LibFunc::powl:
        return &Pow;
      case LibFunc::exp2l:
      case LibFunc::exp2:
      case LibFunc::exp2f:
        return &Exp2;
      case LibFunc::ffs:
      case LibFunc::ffsl:
      case LibFunc::ffsll:
        return &FFS;
      case LibFunc::abs:
      case LibFunc::labs:
      case LibFunc::llabs:
        return &Abs;
      case LibFunc::isdigit:
        return &IsDigit;
      case LibFunc::isascii:
        return &IsAscii;
      case LibFunc::toascii:
        return &ToAscii;
      case LibFunc::printf:
        return &PrintF;
      case LibFunc::sprintf:
        return &SPrintF;
      case LibFunc::fprintf:
        return &FPrintF;
      case LibFunc::fwrite:
        return &FWrite;
      case LibFunc::fputs:
        return &FPuts;
      case LibFunc::puts:
        return &Puts;
      case LibFunc::perror:
        return &ErrorReporting;
      case LibFunc::vfprintf:
      case LibFunc::fiprintf:
        return &ErrorReporting0;
      case LibFunc::fputc:
        return &ErrorReporting1;
      case LibFunc::ceil:
      case LibFunc::fabs:
      case LibFunc::floor:
      case LibFunc::rint:
      case LibFunc::round:
      case LibFunc::nearbyint:
      case LibFunc::trunc:
        if (hasFloatVersion(FuncName))
          return &UnaryDoubleFP;
        return 0;
      case LibFunc::acos:
      case LibFunc::acosh:
      case LibFunc::asin:
      case LibFunc::asinh:
      case LibFunc::atan:
      case LibFunc::atanh:
      case LibFunc::cbrt:
      case LibFunc::cosh:
      case LibFunc::exp:
      case LibFunc::exp10:
      case LibFunc::expm1:
      case LibFunc::log:
      case LibFunc::log10:
      case LibFunc::log1p:
      case LibFunc::log2:
      case LibFunc::logb:
      case LibFunc::sin:
      case LibFunc::sinh:
      case LibFunc::sqrt:
      case LibFunc::tan:
      case LibFunc::tanh:
        if (UnsafeFPShrink && hasFloatVersion(FuncName))
         return &UnsafeUnaryDoubleFP;
        return 0;
      case LibFunc::fmin:
      case LibFunc::fmax:
        if (hasFloatVersion(FuncName))
          return &BinaryDoubleFP;
        return 0;
      case LibFunc::memcpy_chk:
        return &MemCpyChk;
      default:
        return 0;
      }
  }

  // Finally check for fortified library calls.
  if (FuncName.endswith("_chk")) {
    if (FuncName == "__memmove_chk")
      return &MemMoveChk;
    else if (FuncName == "__memset_chk")
      return &MemSetChk;
    else if (FuncName == "__strcpy_chk")
      return &StrCpyChk;
    else if (FuncName == "__stpcpy_chk")
      return &StpCpyChk;
    else if (FuncName == "__strncpy_chk")
      return &StrNCpyChk;
    else if (FuncName == "__stpncpy_chk")
      return &StrNCpyChk;
  }

  return 0;

}

Value *LibCallSimplifierImpl::optimizeCall(CallInst *CI) {
  LibCallOptimization *LCO = lookupOptimization(CI);
  if (LCO) {
    IRBuilder<> Builder(CI);
    return LCO->optimizeCall(CI, DL, TLI, LCS, Builder);
  }
  return 0;
}

LibCallSimplifier::LibCallSimplifier(const DataLayout *DL,
                                     const TargetLibraryInfo *TLI,
                                     bool UnsafeFPShrink) {
  Impl = new LibCallSimplifierImpl(DL, TLI, this, UnsafeFPShrink);
}

LibCallSimplifier::~LibCallSimplifier() {
  delete Impl;
}

Value *LibCallSimplifier::optimizeCall(CallInst *CI) {
  if (CI->isNoBuiltin()) return 0;
  return Impl->optimizeCall(CI);
}

void LibCallSimplifier::replaceAllUsesWith(Instruction *I, Value *With) const {
  I->replaceAllUsesWith(With);
  I->eraseFromParent();
}

}

// TODO:
//   Additional cases that we need to add to this file:
//
// cbrt:
//   * cbrt(expN(X))  -> expN(x/3)
//   * cbrt(sqrt(x))  -> pow(x,1/6)
//   * cbrt(sqrt(x))  -> pow(x,1/9)
//
// exp, expf, expl:
//   * exp(log(x))  -> x
//
// log, logf, logl:
//   * log(exp(x))   -> x
//   * log(x**y)     -> y*log(x)
//   * log(exp(y))   -> y*log(e)
//   * log(exp2(y))  -> y*log(2)
//   * log(exp10(y)) -> y*log(10)
//   * log(sqrt(x))  -> 0.5*log(x)
//   * log(pow(x,y)) -> y*log(x)
//
// lround, lroundf, lroundl:
//   * lround(cnst) -> cnst'
//
// pow, powf, powl:
//   * pow(exp(x),y)  -> exp(x*y)
//   * pow(sqrt(x),y) -> pow(x,y*0.5)
//   * pow(pow(x,y),z)-> pow(x,y*z)
//
// round, roundf, roundl:
//   * round(cnst) -> cnst'
//
// signbit:
//   * signbit(cnst) -> cnst'
//   * signbit(nncst) -> 0 (if pstv is a non-negative constant)
//
// sqrt, sqrtf, sqrtl:
//   * sqrt(expN(x))  -> expN(x*0.5)
//   * sqrt(Nroot(x)) -> pow(x,1/(2*N))
//   * sqrt(pow(x,y)) -> pow(|x|,y*0.5)
//
// tan, tanf, tanl:
//   * tan(atan(x)) -> x
//
// trunc, truncf, truncl:
//   * trunc(cnst) -> cnst'
//
//

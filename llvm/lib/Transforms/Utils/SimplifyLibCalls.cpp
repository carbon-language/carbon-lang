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
#include "llvm/DataLayout.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/LLVMContext.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

using namespace llvm;

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
namespace {
class LibCallOptimization {
protected:
  Function *Caller;
  const DataLayout *TD;
  const TargetLibraryInfo *TLI;
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

  Value *optimizeCall(CallInst *CI, const DataLayout *TD,
                      const TargetLibraryInfo *TLI, IRBuilder<> &B) {
    Caller = CI->getParent()->getParent();
    this->TD = TD;
    this->TLI = TLI;
    if (CI->getCalledFunction())
      Context = &CI->getCalledFunction()->getContext();

    // We never change the calling convention.
    if (CI->getCallingConv() != llvm::CallingConv::C)
      return NULL;

    return callOptimizer(CI->getCalledFunction(), CI, B);
  }
};

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

  bool isFoldable(unsigned SizeCIOp, unsigned SizeArgOp, bool isString) const {
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
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    FunctionType *FT = Callee->getFunctionType();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(FT->getParamType(0)) ||
        FT->getParamType(3) != TD->getIntPtrType(FT->getParamType(1)))
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
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    FunctionType *FT = Callee->getFunctionType();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(FT->getParamType(0)) ||
        FT->getParamType(3) != TD->getIntPtrType(FT->getParamType(1)))
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
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    FunctionType *FT = Callee->getFunctionType();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isIntegerTy() ||
        FT->getParamType(2) != TD->getIntPtrType(FT->getParamType(0)) ||
        FT->getParamType(3) != TD->getIntPtrType(FT->getParamType(0)))
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
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    StringRef Name = Callee->getName();
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 3 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        FT->getParamType(2) != TD->getIntPtrType(FT->getParamType(0)))
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
      Value *Ret = EmitStrCpy(Dst, Src, B, TD, TLI, Name.substr(2, 6));
      return Ret;
    } else {
      // Maybe we can stil fold __strcpy_chk to __memcpy_chk.
      uint64_t Len = GetStringLength(Src);
      if (Len == 0) return 0;

      // This optimization require DataLayout.
      if (!TD) return 0;

      Value *Ret =
	EmitMemCpyChk(Dst, Src,
                      ConstantInt::get(TD->getIntPtrType(Dst->getType()),
                      Len), CI->getArgOperand(2), B, TD, TLI);
      return Ret;
    }
    return 0;
  }
};

struct StpCpyChkOpt : public InstFortifiedLibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    StringRef Name = Callee->getName();
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 3 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        FT->getParamType(2) != TD->getIntPtrType(FT->getParamType(0)))
      return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src) {  // stpcpy(x,x)  -> x+strlen(x)
      Value *StrLen = EmitStrLen(Src, B, TD, TLI);
      return StrLen ? B.CreateInBoundsGEP(Dst, StrLen) : 0;
    }

    // If a) we don't have any length information, or b) we know this will
    // fit then just lower to a plain stpcpy. Otherwise we'll keep our
    // stpcpy_chk call which may fail at runtime if the size is too long.
    // TODO: It might be nice to get a maximum length out of the possible
    // string lengths for varying.
    if (isFoldable(2, 1, true)) {
      Value *Ret = EmitStrCpy(Dst, Src, B, TD, TLI, Name.substr(2, 6));
      return Ret;
    } else {
      // Maybe we can stil fold __stpcpy_chk to __memcpy_chk.
      uint64_t Len = GetStringLength(Src);
      if (Len == 0) return 0;

      // This optimization require DataLayout.
      if (!TD) return 0;

      Type *PT = FT->getParamType(0);
      Value *LenV = ConstantInt::get(TD->getIntPtrType(PT), Len);
      Value *DstEnd = B.CreateGEP(Dst,
                                  ConstantInt::get(TD->getIntPtrType(PT),
                                                   Len - 1));
      if (!EmitMemCpyChk(Dst, Src, LenV, CI->getArgOperand(2), B, TD, TLI))
        return 0;
      return DstEnd;
    }
    return 0;
  }
};

struct StrNCpyChkOpt : public InstFortifiedLibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    this->CI = CI;
    StringRef Name = Callee->getName();
    FunctionType *FT = Callee->getFunctionType();
    LLVMContext &Context = CI->getParent()->getContext();

    // Check if this has the right signature.
    if (FT->getNumParams() != 4 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != Type::getInt8PtrTy(Context) ||
        !FT->getParamType(2)->isIntegerTy() ||
        FT->getParamType(3) != TD->getIntPtrType(FT->getParamType(0)))
      return 0;

    if (isFoldable(3, 2, false)) {
      Value *Ret = EmitStrNCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                               CI->getArgOperand(2), B, TD, TLI,
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
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
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
    if (!TD) return 0;

    return emitStrLenMemCpy(Src, Dst, Len, B);
  }

  Value *emitStrLenMemCpy(Value *Src, Value *Dst, uint64_t Len,
                          IRBuilder<> &B) {
    // We need to find the end of the destination string.  That's where the
    // memory is to be moved to. We just generate a call to strlen.
    Value *DstLen = EmitStrLen(Dst, B, TD, TLI);
    if (!DstLen)
      return 0;

    // Now that we have the destination's length, we must index into the
    // destination's pointer to get the actual memcpy destination (end of
    // the string .. we're concatenating).
    Value *CpyDst = B.CreateGEP(Dst, DstLen, "endptr");

    // We have enough information to now generate the memcpy call to do the
    // concatenation for us.  Make a memcpy to copy the nul byte with align = 1.
    B.CreateMemCpy(CpyDst, Src,
                   ConstantInt::get(TD->getIntPtrType(Src->getType()),
                   Len + 1), 1);
    return Dst;
  }
};

struct StrNCatOpt : public StrCatOpt {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
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
    if (!TD) return 0;

    // We don't optimize this case
    if (Len < SrcLen) return 0;

    // strncat(x, s, c) -> strcat(x, s)
    // s is constant so the strcat can be optimized further
    return emitStrLenMemCpy(Src, Dst, SrcLen, B);
  }
};

struct StrChrOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
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
      if (!TD) return 0;

      uint64_t Len = GetStringLength(SrcStr);
      if (Len == 0 || !FT->getParamType(1)->isIntegerTy(32))// memchr needs i32.
        return 0;

      Type *PT = FT->getParamType(0);
      return EmitMemChr(SrcStr, CI->getArgOperand(1), // include nul.
                        ConstantInt::get(TD->getIntPtrType(PT), Len),
                        B, TD, TLI);
    }

    // Otherwise, the character is a constant, see if the first argument is
    // a string literal.  If so, we can constant fold.
    StringRef Str;
    if (!getConstantStringInfo(SrcStr, Str))
      return 0;

    // Compute the offset, make sure to handle the case when we're searching for
    // zero (a weird way to spell strlen).
    size_t I = CharC->getSExtValue() == 0 ?
        Str.size() : Str.find(CharC->getSExtValue());
    if (I == StringRef::npos) // Didn't find the char.  strchr returns null.
      return Constant::getNullValue(CI->getType());

    // strchr(s+n,c)  -> gep(s+n+i,c)
    return B.CreateGEP(SrcStr, B.getInt64(I), "strchr");
  }
};

struct StrRChrOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
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
      if (TD && CharC->isZero())
        return EmitStrChr(SrcStr, '\0', B, TD, TLI);
      return 0;
    }

    // Compute the offset.
    size_t I = CharC->getSExtValue() == 0 ?
        Str.size() : Str.rfind(CharC->getSExtValue());
    if (I == StringRef::npos) // Didn't find the char. Return null.
      return Constant::getNullValue(CI->getType());

    // strrchr(s+n,c) -> gep(s+n+i,c)
    return B.CreateGEP(SrcStr, B.getInt64(I), "strrchr");
  }
};

struct StrCmpOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
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
      if (!TD) return 0;

      Type *PT = FT->getParamType(0);
      return EmitMemCmp(Str1P, Str2P,
                        ConstantInt::get(TD->getIntPtrType(PT),
                        std::min(Len1, Len2)), B, TD, TLI);
    }

    return 0;
  }
};

struct StrNCmpOpt : public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
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

    if (TD && Length == 1) // strncmp(x,y,1) -> memcmp(x,y,1)
      return EmitMemCmp(Str1P, Str2P, CI->getArgOperand(2), B, TD, TLI);

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
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
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
    if (!TD) return 0;

    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;

    // We have enough information to now generate the memcpy call to do the
    // copy for us.  Make a memcpy to copy the nul byte with align = 1.
    B.CreateMemCpy(Dst, Src,
		   ConstantInt::get(TD->getIntPtrType(Dst->getType()), Len), 1);
    return Dst;
  }
};

struct StpCpyOpt: public LibCallOptimization {
  virtual Value *callOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B) {
    // Verify the "stpcpy" function prototype.
    FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != B.getInt8PtrTy())
      return 0;

    // These optimizations require DataLayout.
    if (!TD) return 0;

    Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
    if (Dst == Src) {  // stpcpy(x,x)  -> x+strlen(x)
      Value *StrLen = EmitStrLen(Src, B, TD, TLI);
      return StrLen ? B.CreateInBoundsGEP(Dst, StrLen) : 0;
    }

    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return 0;

    Type *PT = FT->getParamType(0);
    Value *LenV = ConstantInt::get(TD->getIntPtrType(PT), Len);
    Value *DstEnd = B.CreateGEP(Dst,
                                ConstantInt::get(TD->getIntPtrType(PT),
                                                 Len - 1));

    // We have enough information to now generate the memcpy call to do the
    // copy for us.  Make a memcpy to copy the nul byte with align = 1.
    B.CreateMemCpy(Dst, Src, LenV, 1);
    return DstEnd;
  }
};

} // End anonymous namespace.

namespace llvm {

class LibCallSimplifierImpl {
  const DataLayout *TD;
  const TargetLibraryInfo *TLI;
  StringMap<LibCallOptimization*> Optimizations;

  // Fortified library call optimizations.
  MemCpyChkOpt MemCpyChk;
  MemMoveChkOpt MemMoveChk;
  MemSetChkOpt MemSetChk;
  StrCpyChkOpt StrCpyChk;
  StpCpyChkOpt StpCpyChk;
  StrNCpyChkOpt StrNCpyChk;

  // String and memory library call optimizations.
  StrCatOpt StrCat;
  StrNCatOpt StrNCat;
  StrChrOpt StrChr;
  StrRChrOpt StrRChr;
  StrCmpOpt StrCmp;
  StrNCmpOpt StrNCmp;
  StrCpyOpt StrCpy;
  StpCpyOpt StpCpy;

  void initOptimizations();
public:
  LibCallSimplifierImpl(const DataLayout *TD, const TargetLibraryInfo *TLI) {
    this->TD = TD;
    this->TLI = TLI;
  }

  Value *optimizeCall(CallInst *CI);
};

void LibCallSimplifierImpl::initOptimizations() {
  // Fortified library call optimizations.
  Optimizations["__memcpy_chk"] = &MemCpyChk;
  Optimizations["__memmove_chk"] = &MemMoveChk;
  Optimizations["__memset_chk"] = &MemSetChk;
  Optimizations["__strcpy_chk"] = &StrCpyChk;
  Optimizations["__stpcpy_chk"] = &StpCpyChk;
  Optimizations["__strncpy_chk"] = &StrNCpyChk;
  Optimizations["__stpncpy_chk"] = &StrNCpyChk;

  // String and memory library call optimizations.
  Optimizations["strcat"] = &StrCat;
  Optimizations["strncat"] = &StrNCat;
  Optimizations["strchr"] = &StrChr;
  Optimizations["strrchr"] = &StrRChr;
  Optimizations["strcmp"] = &StrCmp;
  Optimizations["strncmp"] = &StrNCmp;
  Optimizations["strcpy"] = &StrCpy;
  Optimizations["stpcpy"] = &StpCpy;
}

Value *LibCallSimplifierImpl::optimizeCall(CallInst *CI) {
  if (Optimizations.empty())
    initOptimizations();

  Function *Callee = CI->getCalledFunction();
  LibCallOptimization *LCO = Optimizations.lookup(Callee->getName());
  if (LCO) {
    IRBuilder<> Builder(CI);
    return LCO->optimizeCall(CI, TD, TLI, Builder);
  }
  return 0;
}

LibCallSimplifier::LibCallSimplifier(const DataLayout *TD,
                                     const TargetLibraryInfo *TLI) {
  Impl = new LibCallSimplifierImpl(TD, TLI);
}

LibCallSimplifier::~LibCallSimplifier() {
  delete Impl;
}

Value *LibCallSimplifier::optimizeCall(CallInst *CI) {
  return Impl->optimizeCall(CI);
}

}

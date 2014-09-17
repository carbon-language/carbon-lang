//===- SimplifyLibCalls.h - Library call simplifier -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an interface to build some C language libcalls for
// optimization passes that need to call the various functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SIMPLIFYLIBCALLS_H
#define LLVM_TRANSFORMS_UTILS_SIMPLIFYLIBCALLS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"

namespace llvm {
class Value;
class CallInst;
class DataLayout;
class Instruction;
class TargetLibraryInfo;
class BasicBlock;
class Function;

/// LibCallSimplifier - This class implements a collection of optimizations
/// that replace well formed calls to library functions with a more optimal
/// form.  For example, replacing 'printf("Hello!")' with 'puts("Hello!")'.
class LibCallSimplifier {
private:
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  bool UnsafeFPShrink;

protected:
  ~LibCallSimplifier() {}

public:
  LibCallSimplifier(const DataLayout *TD, const TargetLibraryInfo *TLI,
                    bool UnsafeFPShrink);

  /// optimizeCall - Take the given call instruction and return a more
  /// optimal value to replace the instruction with or 0 if a more
  /// optimal form can't be found.  Note that the returned value may
  /// be equal to the instruction being optimized.  In this case all
  /// other instructions that use the given instruction were modified
  /// and the given instruction is dead.
  Value *optimizeCall(CallInst *CI);

  /// replaceAllUsesWith - This method is used when the library call
  /// simplifier needs to replace instructions other than the library
  /// call being modified.
  virtual void replaceAllUsesWith(Instruction *I, Value *With) const;

private:
  // Fortified Library Call Optimizations
  Value *optimizeMemCpyChk(CallInst *CI, IRBuilder<> &B);
  Value *optimizeMemMoveChk(CallInst *CI, IRBuilder<> &B);
  Value *optimizeMemSetChk(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrCpyChk(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStpCpyChk(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrNCpyChk(CallInst *CI, IRBuilder<> &B);

  // String and Memory Library Call Optimizations
  Value *optimizeStrCat(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrNCat(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrChr(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrRChr(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrCmp(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrNCmp(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrCpy(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStpCpy(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrNCpy(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrLen(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrPBrk(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrTo(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrSpn(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrCSpn(CallInst *CI, IRBuilder<> &B);
  Value *optimizeStrStr(CallInst *CI, IRBuilder<> &B);
  Value *optimizeMemCmp(CallInst *CI, IRBuilder<> &B);
  Value *optimizeMemCpy(CallInst *CI, IRBuilder<> &B);
  Value *optimizeMemMove(CallInst *CI, IRBuilder<> &B);
  Value *optimizeMemSet(CallInst *CI, IRBuilder<> &B);

  // Math Library Optimizations
  Value *optimizeUnaryDoubleFP(CallInst *CI, IRBuilder<> &B, bool CheckRetType);
  Value *optimizeBinaryDoubleFP(CallInst *CI, IRBuilder<> &B);
  Value *optimizeCos(CallInst *CI, IRBuilder<> &B);
  Value *optimizePow(CallInst *CI, IRBuilder<> &B);
  Value *optimizeExp2(CallInst *CI, IRBuilder<> &B);
  Value *optimizeSinCosPi(CallInst *CI, IRBuilder<> &B);

  // Integer Library Call Optimizations
  Value *optimizeFFS(CallInst *CI, IRBuilder<> &B);
  Value *optimizeAbs(CallInst *CI, IRBuilder<> &B);
  Value *optimizeIsDigit(CallInst *CI, IRBuilder<> &B);
  Value *optimizeIsAscii(CallInst *CI, IRBuilder<> &B);
  Value *optimizeToAscii(CallInst *CI, IRBuilder<> &B);

  // Formatting and IO Library Call Optimizations
  Value *optimizeErrorReporting(CallInst *CI, IRBuilder<> &B,
                                int StreamArg = -1);
  Value *optimizePrintF(CallInst *CI, IRBuilder<> &B);
  Value *optimizeSPrintF(CallInst *CI, IRBuilder<> &B);
  Value *optimizeFPrintF(CallInst *CI, IRBuilder<> &B);
  Value *optimizeFWrite(CallInst *CI, IRBuilder<> &B);
  Value *optimizeFPuts(CallInst *CI, IRBuilder<> &B);
  Value *optimizePuts(CallInst *CI, IRBuilder<> &B);

  // Helper methods
  Value *emitStrLenMemCpy(Value *Src, Value *Dst, uint64_t Len, IRBuilder<> &B);
  void classifyArgUse(Value *Val, BasicBlock *BB, bool IsFloat,
                      SmallVectorImpl<CallInst *> &SinCalls,
                      SmallVectorImpl<CallInst *> &CosCalls,
                      SmallVectorImpl<CallInst *> &SinCosCalls);
  void replaceTrigInsts(SmallVectorImpl<CallInst *> &Calls, Value *Res);
  Value *optimizePrintFString(CallInst *CI, IRBuilder<> &B);
  Value *optimizeSPrintFString(CallInst *CI, IRBuilder<> &B);
  Value *optimizeFPrintFString(CallInst *CI, IRBuilder<> &B);

  /// hasFloatVersion - Checks if there is a float version of the specified
  /// function by checking for an existing function with name FuncName + f
  bool hasFloatVersion(StringRef FuncName);
};
} // End llvm namespace

#endif

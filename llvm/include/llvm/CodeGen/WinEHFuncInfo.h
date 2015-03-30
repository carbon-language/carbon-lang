//===-- llvm/CodeGen/WinEHFuncInfo.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Data structures and associated state for Windows exception handling schemes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_WINEHFUNCINFO_H
#define LLVM_CODEGEN_WINEHFUNCINFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class BasicBlock;
class Constant;
class Function;
class GlobalValue;
class LandingPadInst;
class MCSymbol;
class Value;

enum ActionType { Catch, Cleanup };

class ActionHandler {
public:
  ActionHandler(BasicBlock *BB, ActionType Type)
      : StartBB(BB), Type(Type), EHState(-1), HandlerBlockOrFunc(nullptr) {}

  ActionType getType() const { return Type; }
  BasicBlock *getStartBlock() const { return StartBB; }

  bool hasBeenProcessed() { return HandlerBlockOrFunc != nullptr; }

  void setHandlerBlockOrFunc(Constant *F) { HandlerBlockOrFunc = F; }
  Constant *getHandlerBlockOrFunc() { return HandlerBlockOrFunc; }

  void setEHState(int State) { EHState = State; }
  int getEHState() const { return EHState; }

private:
  BasicBlock *StartBB;
  ActionType Type;
  int EHState;

  // Can be either a BlockAddress or a Function depending on the EH personality.
  Constant *HandlerBlockOrFunc;
};

class CatchHandler : public ActionHandler {
public:
  CatchHandler(BasicBlock *BB, Constant *Selector, BasicBlock *NextBB)
      : ActionHandler(BB, ActionType::Catch), Selector(Selector),
        NextBB(NextBB), ExceptionObjectVar(nullptr) {}

  // Method for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ActionHandler *H) {
    return H->getType() == ActionType::Catch;
  }

  Constant *getSelector() const { return Selector; }
  BasicBlock *getNextBB() const { return NextBB; }

  const Value *getExceptionVar() { return ExceptionObjectVar; }
  TinyPtrVector<BasicBlock *> &getReturnTargets() { return ReturnTargets; }

  void setExceptionVar(const Value *Val) { ExceptionObjectVar = Val; }
  void setReturnTargets(TinyPtrVector<BasicBlock *> &Targets) {
    ReturnTargets = Targets;
  }

private:
  Constant *Selector;
  BasicBlock *NextBB;
  const Value *ExceptionObjectVar;
  TinyPtrVector<BasicBlock *> ReturnTargets;
};

class CleanupHandler : public ActionHandler {
public:
  CleanupHandler(BasicBlock *BB) : ActionHandler(BB, ActionType::Cleanup) {}

  // Method for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ActionHandler *H) {
    return H->getType() == ActionType::Cleanup;
  }
};

// The following structs respresent the .xdata for functions using C++
// exceptions on Windows.

struct WinEHUnwindMapEntry {
  int ToState;
  Function *Cleanup;
};

struct WinEHHandlerType {
  int Adjectives;
  GlobalValue *TypeDescriptor;
  Function *Handler;
};

struct WinEHTryBlockMapEntry {
  int TryLow;
  int TryHigh;
  int CatchHigh;
  SmallVector<WinEHHandlerType, 4> HandlerArray;
};

struct WinEHFuncInfo {
  DenseMap<const LandingPadInst *, int> LandingPadStateMap;
  DenseMap<const Function *, int> CatchHandlerParentFrameObjIdx;
  DenseMap<const Function *, int> CatchHandlerParentFrameObjOffset;
  SmallVector<WinEHUnwindMapEntry, 4> UnwindMap;
  SmallVector<WinEHTryBlockMapEntry, 4> TryBlockMap;
  SmallVector<std::pair<MCSymbol *, int>, 4> IPToStateList;
  int UnwindHelpFrameIdx;
  int UnwindHelpFrameOffset;

  WinEHFuncInfo() : UnwindHelpFrameIdx(INT_MAX), UnwindHelpFrameOffset(-1) {}
};

}
#endif // LLVM_CODEGEN_WINEHFUNCINFO_H

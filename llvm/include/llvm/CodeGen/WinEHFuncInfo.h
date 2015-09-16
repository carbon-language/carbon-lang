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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"

namespace llvm {
class AllocaInst;
class BasicBlock;
class Constant;
class Function;
class GlobalVariable;
class InvokeInst;
class IntrinsicInst;
class LandingPadInst;
class MCSymbol;
class MachineBasicBlock;
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
      NextBB(NextBB), ExceptionObjectVar(nullptr),
      ExceptionObjectIndex(-1) {}

  // Method for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ActionHandler *H) {
    return H->getType() == ActionType::Catch;
  }

  Constant *getSelector() const { return Selector; }
  BasicBlock *getNextBB() const { return NextBB; }

  const Value *getExceptionVar() { return ExceptionObjectVar; }
  TinyPtrVector<BasicBlock *> &getReturnTargets() { return ReturnTargets; }

  void setExceptionVar(const Value *Val) { ExceptionObjectVar = Val; }
  void setExceptionVarIndex(int Index) { ExceptionObjectIndex = Index;  }
  int getExceptionVarIndex() const { return ExceptionObjectIndex; }
  void setReturnTargets(TinyPtrVector<BasicBlock *> &Targets) {
    ReturnTargets = Targets;
  }

private:
  Constant *Selector;
  BasicBlock *NextBB;
  // While catch handlers are being outlined the ExceptionObjectVar field will
  // be populated with the instruction in the parent frame that corresponds
  // to the exception object (or nullptr if the catch does not use an
  // exception object) and the ExceptionObjectIndex field will be -1.
  // When the parseEHActions function is called to populate a vector of
  // instances of this class, the ExceptionObjectVar field will be nullptr
  // and the ExceptionObjectIndex will be the index of the exception object in
  // the parent function's localescape block.
  const Value *ExceptionObjectVar;
  int ExceptionObjectIndex;
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

void parseEHActions(const IntrinsicInst *II,
                    SmallVectorImpl<std::unique_ptr<ActionHandler>> &Actions);

// The following structs respresent the .xdata for functions using C++
// exceptions on Windows.

typedef PointerUnion<const BasicBlock *, MachineBasicBlock *> MBBOrBasicBlock;
typedef PointerUnion<const Value *, MachineBasicBlock *> ValueOrMBB;

struct WinEHUnwindMapEntry {
  int ToState;
  ValueOrMBB Cleanup;
};

/// Similar to WinEHUnwindMapEntry, but supports SEH filters.
struct SEHUnwindMapEntry {
  /// If unwinding continues through this handler, transition to the handler at
  /// this state. This indexes into SEHUnwindMap.
  int ToState = -1;

  /// Holds the filter expression function.
  const Function *Filter = nullptr;

  /// Holds the __except or __finally basic block.
  MBBOrBasicBlock Handler;
};

struct WinEHHandlerType {
  int Adjectives;
  int CatchObjRecoverIdx;
  /// The CatchObj starts out life as an LLVM alloca, is turned into a frame
  /// index, and after PEI, becomes a raw offset.
  union {
    const AllocaInst *Alloca;
    int FrameOffset;
    int FrameIndex;
  } CatchObj = {};
  GlobalVariable *TypeDescriptor;
  ValueOrMBB Handler;
};

struct WinEHTryBlockMapEntry {
  int TryLow = -1;
  int TryHigh = -1;
  int CatchHigh = -1;
  SmallVector<WinEHHandlerType, 1> HandlerArray;
};

struct WinEHFuncInfo {
  DenseMap<const Instruction *, int> EHPadStateMap;
  SmallVector<WinEHUnwindMapEntry, 4> UnwindMap;
  SmallVector<WinEHTryBlockMapEntry, 4> TryBlockMap;
  SmallVector<SEHUnwindMapEntry, 4> SEHUnwindMap;
  SmallVector<std::pair<MCSymbol *, int>, 4> IPToStateList;
  int UnwindHelpFrameIdx = INT_MAX;
  int UnwindHelpFrameOffset = -1;

  int getLastStateNumber() const { return UnwindMap.size() - 1; }

  /// localescape index of the 32-bit EH registration node. Set by
  /// WinEHStatePass and used indirectly by SEH filter functions of the parent.
  int EHRegNodeEscapeIndex = INT_MAX;
  const AllocaInst *EHRegNode = nullptr;
  int EHRegNodeFrameIndex = INT_MAX;
  int EHRegNodeEndOffset = INT_MAX;

  WinEHFuncInfo() {}
};

/// Analyze the IR in ParentFn and it's handlers to build WinEHFuncInfo, which
/// describes the state numbers and tables used by __CxxFrameHandler3. This
/// analysis assumes that WinEHPrepare has already been run.
void calculateWinCXXEHStateNumbers(const Function *ParentFn,
                                   WinEHFuncInfo &FuncInfo);

void calculateSEHStateNumbers(const Function *ParentFn,
                              WinEHFuncInfo &FuncInfo);
}
#endif // LLVM_CODEGEN_WINEHFUNCINFO_H

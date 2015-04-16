//===-- WinEHPrepare - Prepare exception handling for code generation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers LLVM IR exception handling into something closer to what the
// backend wants. It snifs the personality function to see which kind of
// preparation is necessary. If the personality function uses the Itanium LSDA,
// this pass delegates to the DWARF EH preparation pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/LibCallSemantics.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <memory>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "winehprepare"

namespace {

// This map is used to model frame variable usage during outlining, to
// construct a structure type to hold the frame variables in a frame
// allocation block, and to remap the frame variable allocas (including
// spill locations as needed) to GEPs that get the variable from the
// frame allocation structure.
typedef MapVector<Value *, TinyPtrVector<AllocaInst *>> FrameVarInfoMap;

// TinyPtrVector cannot hold nullptr, so we need our own sentinel that isn't
// quite null.
AllocaInst *getCatchObjectSentinel() {
  return static_cast<AllocaInst *>(nullptr) + 1;
}

typedef SmallSet<BasicBlock *, 4> VisitedBlockSet;

class LandingPadActions;
class LandingPadMap;

typedef DenseMap<const BasicBlock *, CatchHandler *> CatchHandlerMapTy;
typedef DenseMap<const BasicBlock *, CleanupHandler *> CleanupHandlerMapTy;

class WinEHPrepare : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid.
  WinEHPrepare(const TargetMachine *TM = nullptr)
      : FunctionPass(ID), DT(nullptr) {}

  bool runOnFunction(Function &Fn) override;

  bool doFinalization(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  const char *getPassName() const override {
    return "Windows exception handling preparation";
  }

private:
  bool prepareExceptionHandlers(Function &F,
                                SmallVectorImpl<LandingPadInst *> &LPads);
  void promoteLandingPadValues(LandingPadInst *LPad);
  void completeNestedLandingPad(Function *ParentFn,
                                LandingPadInst *OutlinedLPad,
                                const LandingPadInst *OriginalLPad,
                                FrameVarInfoMap &VarInfo);
  bool outlineHandler(ActionHandler *Action, Function *SrcFn,
                      LandingPadInst *LPad, BasicBlock *StartBB,
                      FrameVarInfoMap &VarInfo);
  void addStubInvokeToHandlerIfNeeded(Function *Handler, Value *PersonalityFn);

  void mapLandingPadBlocks(LandingPadInst *LPad, LandingPadActions &Actions);
  CatchHandler *findCatchHandler(BasicBlock *BB, BasicBlock *&NextBB,
                                 VisitedBlockSet &VisitedBlocks);
  void findCleanupHandlers(LandingPadActions &Actions, BasicBlock *StartBB,
                           BasicBlock *EndBB);

  void processSEHCatchHandler(CatchHandler *Handler, BasicBlock *StartBB);

  // All fields are reset by runOnFunction.
  DominatorTree *DT;
  EHPersonality Personality;
  CatchHandlerMapTy CatchHandlerMap;
  CleanupHandlerMapTy CleanupHandlerMap;
  DenseMap<const LandingPadInst *, LandingPadMap> LPadMaps;

  // This maps landing pad instructions found in outlined handlers to
  // the landing pad instruction in the parent function from which they
  // were cloned.  The cloned/nested landing pad is used as the key
  // because the landing pad may be cloned into multiple handlers.
  // This map will be used to add the llvm.eh.actions call to the nested
  // landing pads after all handlers have been outlined.
  DenseMap<LandingPadInst *, const LandingPadInst *> NestedLPtoOriginalLP;

  // This maps blocks in the parent function which are destinations of
  // catch handlers to cloned blocks in (other) outlined handlers. This
  // handles the case where a nested landing pads has a catch handler that
  // returns to a handler function rather than the parent function.
  // The original block is used as the key here because there should only
  // ever be one handler function from which the cloned block is not pruned.
  // The original block will be pruned from the parent function after all
  // handlers have been outlined.  This map will be used to adjust the
  // return instructions of handlers which return to the block that was
  // outlined into a handler.  This is done after all handlers have been
  // outlined but before the outlined code is pruned from the parent function.
  DenseMap<const BasicBlock *, BasicBlock *> LPadTargetBlocks;
};

class WinEHFrameVariableMaterializer : public ValueMaterializer {
public:
  WinEHFrameVariableMaterializer(Function *OutlinedFn,
                                 FrameVarInfoMap &FrameVarInfo);
  ~WinEHFrameVariableMaterializer() override {}

  Value *materializeValueFor(Value *V) override;

  void escapeCatchObject(Value *V);

private:
  FrameVarInfoMap &FrameVarInfo;
  IRBuilder<> Builder;
};

class LandingPadMap {
public:
  LandingPadMap() : OriginLPad(nullptr) {}
  void mapLandingPad(const LandingPadInst *LPad);

  bool isInitialized() { return OriginLPad != nullptr; }

  bool isOriginLandingPadBlock(const BasicBlock *BB) const;
  bool isLandingPadSpecificInst(const Instruction *Inst) const;

  void remapEHValues(ValueToValueMapTy &VMap, Value *EHPtrValue,
                     Value *SelectorValue) const;

private:
  const LandingPadInst *OriginLPad;
  // We will normally only see one of each of these instructions, but
  // if more than one occurs for some reason we can handle that.
  TinyPtrVector<const ExtractValueInst *> ExtractedEHPtrs;
  TinyPtrVector<const ExtractValueInst *> ExtractedSelectors;
};

class WinEHCloningDirectorBase : public CloningDirector {
public:
  WinEHCloningDirectorBase(Function *HandlerFn, FrameVarInfoMap &VarInfo,
                           LandingPadMap &LPadMap)
      : Materializer(HandlerFn, VarInfo),
        SelectorIDType(Type::getInt32Ty(HandlerFn->getContext())),
        Int8PtrType(Type::getInt8PtrTy(HandlerFn->getContext())),
        LPadMap(LPadMap) {}

  CloningAction handleInstruction(ValueToValueMapTy &VMap,
                                  const Instruction *Inst,
                                  BasicBlock *NewBB) override;

  virtual CloningAction handleBeginCatch(ValueToValueMapTy &VMap,
                                         const Instruction *Inst,
                                         BasicBlock *NewBB) = 0;
  virtual CloningAction handleEndCatch(ValueToValueMapTy &VMap,
                                       const Instruction *Inst,
                                       BasicBlock *NewBB) = 0;
  virtual CloningAction handleTypeIdFor(ValueToValueMapTy &VMap,
                                        const Instruction *Inst,
                                        BasicBlock *NewBB) = 0;
  virtual CloningAction handleInvoke(ValueToValueMapTy &VMap,
                                     const InvokeInst *Invoke,
                                     BasicBlock *NewBB) = 0;
  virtual CloningAction handleResume(ValueToValueMapTy &VMap,
                                     const ResumeInst *Resume,
                                     BasicBlock *NewBB) = 0;
  virtual CloningAction handleLandingPad(ValueToValueMapTy &VMap,
                                         const LandingPadInst *LPad,
                                         BasicBlock *NewBB) = 0;

  ValueMaterializer *getValueMaterializer() override { return &Materializer; }

protected:
  WinEHFrameVariableMaterializer Materializer;
  Type *SelectorIDType;
  Type *Int8PtrType;
  LandingPadMap &LPadMap;
};

class WinEHCatchDirector : public WinEHCloningDirectorBase {
public:
  WinEHCatchDirector(
      Function *CatchFn, Value *Selector, FrameVarInfoMap &VarInfo,
      LandingPadMap &LPadMap,
      DenseMap<LandingPadInst *, const LandingPadInst *> &NestedLPads)
      : WinEHCloningDirectorBase(CatchFn, VarInfo, LPadMap),
        CurrentSelector(Selector->stripPointerCasts()),
        ExceptionObjectVar(nullptr), NestedLPtoOriginalLP(NestedLPads) {}

  CloningAction handleBeginCatch(ValueToValueMapTy &VMap,
                                 const Instruction *Inst,
                                 BasicBlock *NewBB) override;
  CloningAction handleEndCatch(ValueToValueMapTy &VMap, const Instruction *Inst,
                               BasicBlock *NewBB) override;
  CloningAction handleTypeIdFor(ValueToValueMapTy &VMap,
                                const Instruction *Inst,
                                BasicBlock *NewBB) override;
  CloningAction handleInvoke(ValueToValueMapTy &VMap, const InvokeInst *Invoke,
                             BasicBlock *NewBB) override;
  CloningAction handleResume(ValueToValueMapTy &VMap, const ResumeInst *Resume,
                             BasicBlock *NewBB) override;
  CloningAction handleLandingPad(ValueToValueMapTy &VMap,
                                 const LandingPadInst *LPad,
                                 BasicBlock *NewBB) override;

  Value *getExceptionVar() { return ExceptionObjectVar; }
  TinyPtrVector<BasicBlock *> &getReturnTargets() { return ReturnTargets; }

private:
  Value *CurrentSelector;

  Value *ExceptionObjectVar;
  TinyPtrVector<BasicBlock *> ReturnTargets;

  // This will be a reference to the field of the same name in the WinEHPrepare
  // object which instantiates this WinEHCatchDirector object.
  DenseMap<LandingPadInst *, const LandingPadInst *> &NestedLPtoOriginalLP;
};

class WinEHCleanupDirector : public WinEHCloningDirectorBase {
public:
  WinEHCleanupDirector(Function *CleanupFn, FrameVarInfoMap &VarInfo,
                       LandingPadMap &LPadMap)
      : WinEHCloningDirectorBase(CleanupFn, VarInfo, LPadMap) {}

  CloningAction handleBeginCatch(ValueToValueMapTy &VMap,
                                 const Instruction *Inst,
                                 BasicBlock *NewBB) override;
  CloningAction handleEndCatch(ValueToValueMapTy &VMap, const Instruction *Inst,
                               BasicBlock *NewBB) override;
  CloningAction handleTypeIdFor(ValueToValueMapTy &VMap,
                                const Instruction *Inst,
                                BasicBlock *NewBB) override;
  CloningAction handleInvoke(ValueToValueMapTy &VMap, const InvokeInst *Invoke,
                             BasicBlock *NewBB) override;
  CloningAction handleResume(ValueToValueMapTy &VMap, const ResumeInst *Resume,
                             BasicBlock *NewBB) override;
  CloningAction handleLandingPad(ValueToValueMapTy &VMap,
                                 const LandingPadInst *LPad,
                                 BasicBlock *NewBB) override;
};

class LandingPadActions {
public:
  LandingPadActions() : HasCleanupHandlers(false) {}

  void insertCatchHandler(CatchHandler *Action) { Actions.push_back(Action); }
  void insertCleanupHandler(CleanupHandler *Action) {
    Actions.push_back(Action);
    HasCleanupHandlers = true;
  }

  bool includesCleanup() const { return HasCleanupHandlers; }

  SmallVectorImpl<ActionHandler *> &actions() { return Actions; }
  SmallVectorImpl<ActionHandler *>::iterator begin() { return Actions.begin(); }
  SmallVectorImpl<ActionHandler *>::iterator end() { return Actions.end(); }

private:
  // Note that this class does not own the ActionHandler objects in this vector.
  // The ActionHandlers are owned by the CatchHandlerMap and CleanupHandlerMap
  // in the WinEHPrepare class.
  SmallVector<ActionHandler *, 4> Actions;
  bool HasCleanupHandlers;
};

} // end anonymous namespace

char WinEHPrepare::ID = 0;
INITIALIZE_TM_PASS(WinEHPrepare, "winehprepare", "Prepare Windows exceptions",
                   false, false)

FunctionPass *llvm::createWinEHPass(const TargetMachine *TM) {
  return new WinEHPrepare(TM);
}

// FIXME: Remove this once the backend can handle the prepared IR.
static cl::opt<bool>
    SEHPrepare("sehprepare", cl::Hidden,
               cl::desc("Prepare functions with SEH personalities"));

bool WinEHPrepare::runOnFunction(Function &Fn) {
  SmallVector<LandingPadInst *, 4> LPads;
  SmallVector<ResumeInst *, 4> Resumes;
  for (BasicBlock &BB : Fn) {
    if (auto *LP = BB.getLandingPadInst())
      LPads.push_back(LP);
    if (auto *Resume = dyn_cast<ResumeInst>(BB.getTerminator()))
      Resumes.push_back(Resume);
  }

  // No need to prepare functions that lack landing pads.
  if (LPads.empty())
    return false;

  // Classify the personality to see what kind of preparation we need.
  Personality = classifyEHPersonality(LPads.back()->getPersonalityFn());

  // Do nothing if this is not an MSVC personality.
  if (!isMSVCEHPersonality(Personality))
    return false;

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  if (isAsynchronousEHPersonality(Personality) && !SEHPrepare) {
    // Replace all resume instructions with unreachable.
    // FIXME: Remove this once the backend can handle the prepared IR.
    for (ResumeInst *Resume : Resumes) {
      IRBuilder<>(Resume).CreateUnreachable();
      Resume->eraseFromParent();
    }
    return true;
  }

  // If there were any landing pads, prepareExceptionHandlers will make changes.
  prepareExceptionHandlers(Fn, LPads);
  return true;
}

bool WinEHPrepare::doFinalization(Module &M) { return false; }

void WinEHPrepare::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
}

bool WinEHPrepare::prepareExceptionHandlers(
    Function &F, SmallVectorImpl<LandingPadInst *> &LPads) {
  // These containers are used to re-map frame variables that are used in
  // outlined catch and cleanup handlers.  They will be populated as the
  // handlers are outlined.
  FrameVarInfoMap FrameVarInfo;

  bool HandlersOutlined = false;

  Module *M = F.getParent();
  LLVMContext &Context = M->getContext();

  // Create a new function to receive the handler contents.
  PointerType *Int8PtrType = Type::getInt8PtrTy(Context);
  Type *Int32Type = Type::getInt32Ty(Context);
  Function *ActionIntrin = Intrinsic::getDeclaration(M, Intrinsic::eh_actions);

  for (LandingPadInst *LPad : LPads) {
    // Look for evidence that this landingpad has already been processed.
    bool LPadHasActionList = false;
    BasicBlock *LPadBB = LPad->getParent();
    for (Instruction &Inst : *LPadBB) {
      if (auto *IntrinCall = dyn_cast<IntrinsicInst>(&Inst)) {
        if (IntrinCall->getIntrinsicID() == Intrinsic::eh_actions) {
          LPadHasActionList = true;
          break;
        }
      }
      // FIXME: This is here to help with the development of nested landing pad
      //        outlining.  It should be removed when that is finished.
      if (isa<UnreachableInst>(Inst)) {
        LPadHasActionList = true;
        break;
      }
    }

    // If we've already outlined the handlers for this landingpad,
    // there's nothing more to do here.
    if (LPadHasActionList)
      continue;

    // If either of the values in the aggregate returned by the landing pad is
    // extracted and stored to memory, promote the stored value to a register.
    promoteLandingPadValues(LPad);

    LandingPadActions Actions;
    mapLandingPadBlocks(LPad, Actions);

    HandlersOutlined |= !Actions.actions().empty();
    for (ActionHandler *Action : Actions) {
      if (Action->hasBeenProcessed())
        continue;
      BasicBlock *StartBB = Action->getStartBlock();

      // SEH doesn't do any outlining for catches. Instead, pass the handler
      // basic block addr to llvm.eh.actions and list the block as a return
      // target.
      if (isAsynchronousEHPersonality(Personality)) {
        if (auto *CatchAction = dyn_cast<CatchHandler>(Action)) {
          processSEHCatchHandler(CatchAction, StartBB);
          continue;
        }
      }

      outlineHandler(Action, &F, LPad, StartBB, FrameVarInfo);
    }

    // Replace the landing pad with a new llvm.eh.action based landing pad.
    BasicBlock *NewLPadBB = BasicBlock::Create(Context, "lpad", &F, LPadBB);
    assert(!isa<PHINode>(LPadBB->begin()));
    auto *NewLPad = cast<LandingPadInst>(LPad->clone());
    NewLPadBB->getInstList().push_back(NewLPad);
    while (!pred_empty(LPadBB)) {
      auto *pred = *pred_begin(LPadBB);
      InvokeInst *Invoke = cast<InvokeInst>(pred->getTerminator());
      Invoke->setUnwindDest(NewLPadBB);
    }

    // If anyone is still using the old landingpad value, just give them undef
    // instead. The eh pointer and selector values are not real.
    LPad->replaceAllUsesWith(UndefValue::get(LPad->getType()));

    // Replace the mapping of any nested landing pad that previously mapped
    // to this landing pad with a referenced to the cloned version.
    for (auto &LPadPair : NestedLPtoOriginalLP) {
      const LandingPadInst *OriginalLPad = LPadPair.second;
      if (OriginalLPad == LPad) {
        LPadPair.second = NewLPad;
      }
    }

    // Replace uses of the old lpad in phis with this block and delete the old
    // block.
    LPadBB->replaceSuccessorsPhiUsesWith(NewLPadBB);
    LPadBB->getTerminator()->eraseFromParent();
    new UnreachableInst(LPadBB->getContext(), LPadBB);

    // Add a call to describe the actions for this landing pad.
    std::vector<Value *> ActionArgs;
    for (ActionHandler *Action : Actions) {
      // Action codes from docs are: 0 cleanup, 1 catch.
      if (auto *CatchAction = dyn_cast<CatchHandler>(Action)) {
        ActionArgs.push_back(ConstantInt::get(Int32Type, 1));
        ActionArgs.push_back(CatchAction->getSelector());
        // Find the frame escape index of the exception object alloca in the
        // parent.
        int FrameEscapeIdx = -1;
        Value *EHObj = const_cast<Value *>(CatchAction->getExceptionVar());
        if (EHObj && !isa<ConstantPointerNull>(EHObj)) {
          auto I = FrameVarInfo.find(EHObj);
          assert(I != FrameVarInfo.end() &&
                 "failed to map llvm.eh.begincatch var");
          FrameEscapeIdx = std::distance(FrameVarInfo.begin(), I);
        }
        ActionArgs.push_back(ConstantInt::get(Int32Type, FrameEscapeIdx));
      } else {
        ActionArgs.push_back(ConstantInt::get(Int32Type, 0));
      }
      ActionArgs.push_back(Action->getHandlerBlockOrFunc());
    }
    CallInst *Recover =
        CallInst::Create(ActionIntrin, ActionArgs, "recover", NewLPadBB);

    // Add an indirect branch listing possible successors of the catch handlers.
    IndirectBrInst *Branch = IndirectBrInst::Create(Recover, 0, NewLPadBB);
    for (ActionHandler *Action : Actions) {
      if (auto *CatchAction = dyn_cast<CatchHandler>(Action)) {
        for (auto *Target : CatchAction->getReturnTargets()) {
          Branch->addDestination(Target);
        }
      }
    }
  } // End for each landingpad

  // If nothing got outlined, there is no more processing to be done.
  if (!HandlersOutlined)
    return false;

  // Replace any nested landing pad stubs with the correct action handler.
  // This must be done before we remove unreachable blocks because it
  // cleans up references to outlined blocks that will be deleted.
  for (auto &LPadPair : NestedLPtoOriginalLP)
    completeNestedLandingPad(&F, LPadPair.first, LPadPair.second, FrameVarInfo);
  NestedLPtoOriginalLP.clear();

  F.addFnAttr("wineh-parent", F.getName());

  // Delete any blocks that were only used by handlers that were outlined above.
  removeUnreachableBlocks(F);

  BasicBlock *Entry = &F.getEntryBlock();
  IRBuilder<> Builder(F.getParent()->getContext());
  Builder.SetInsertPoint(Entry->getFirstInsertionPt());

  Function *FrameEscapeFn =
      Intrinsic::getDeclaration(M, Intrinsic::frameescape);
  Function *RecoverFrameFn =
      Intrinsic::getDeclaration(M, Intrinsic::framerecover);

  // Finally, replace all of the temporary allocas for frame variables used in
  // the outlined handlers with calls to llvm.framerecover.
  BasicBlock::iterator II = Entry->getFirstInsertionPt();
  Instruction *AllocaInsertPt = II;
  SmallVector<Value *, 8> AllocasToEscape;
  for (auto &VarInfoEntry : FrameVarInfo) {
    Value *ParentVal = VarInfoEntry.first;
    TinyPtrVector<AllocaInst *> &Allocas = VarInfoEntry.second;

    // If the mapped value isn't already an alloca, we need to spill it if it
    // is a computed value or copy it if it is an argument.
    AllocaInst *ParentAlloca = dyn_cast<AllocaInst>(ParentVal);
    if (!ParentAlloca) {
      if (auto *Arg = dyn_cast<Argument>(ParentVal)) {
        // Lower this argument to a copy and then demote that to the stack.
        // We can't just use the argument location because the handler needs
        // it to be in the frame allocation block.
        // Use 'select i8 true, %arg, undef' to simulate a 'no-op' instruction.
        Value *TrueValue = ConstantInt::getTrue(Context);
        Value *UndefValue = UndefValue::get(Arg->getType());
        Instruction *SI =
            SelectInst::Create(TrueValue, Arg, UndefValue,
                               Arg->getName() + ".tmp", AllocaInsertPt);
        Arg->replaceAllUsesWith(SI);
        // Reset the select operand, because it was clobbered by the RAUW above.
        SI->setOperand(1, Arg);
        ParentAlloca = DemoteRegToStack(*SI, true, SI);
      } else if (auto *PN = dyn_cast<PHINode>(ParentVal)) {
        ParentAlloca = DemotePHIToStack(PN, AllocaInsertPt);
      } else {
        Instruction *ParentInst = cast<Instruction>(ParentVal);
        // FIXME: This is a work-around to temporarily handle the case where an
        //        instruction that is only used in handlers is not sunk.
        //        Without uses, DemoteRegToStack would just eliminate the value.
        //        This will fail if ParentInst is an invoke.
        if (ParentInst->getNumUses() == 0) {
          BasicBlock::iterator InsertPt = ParentInst;
          ++InsertPt;
          ParentAlloca =
              new AllocaInst(ParentInst->getType(), nullptr,
                             ParentInst->getName() + ".reg2mem",
                             AllocaInsertPt);
          new StoreInst(ParentInst, ParentAlloca, InsertPt);
        } else {
          ParentAlloca = DemoteRegToStack(*ParentInst, true, AllocaInsertPt);
        }
      }
    }

    // FIXME: We should try to sink unescaped allocas from the parent frame into
    // the child frame. If the alloca is escaped, we have to use the lifetime
    // markers to ensure that the alloca is only live within the child frame.

    // Add this alloca to the list of things to escape.
    AllocasToEscape.push_back(ParentAlloca);

    // Next replace all outlined allocas that are mapped to it.
    for (AllocaInst *TempAlloca : Allocas) {
      if (TempAlloca == getCatchObjectSentinel())
        continue; // Skip catch parameter sentinels.
      Function *HandlerFn = TempAlloca->getParent()->getParent();
      // FIXME: Sink this GEP into the blocks where it is used.
      Builder.SetInsertPoint(TempAlloca);
      Builder.SetCurrentDebugLocation(TempAlloca->getDebugLoc());
      Value *RecoverArgs[] = {
          Builder.CreateBitCast(&F, Int8PtrType, ""),
          &(HandlerFn->getArgumentList().back()),
          llvm::ConstantInt::get(Int32Type, AllocasToEscape.size() - 1)};
      Value *RecoveredAlloca = Builder.CreateCall(RecoverFrameFn, RecoverArgs);
      // Add a pointer bitcast if the alloca wasn't an i8.
      if (RecoveredAlloca->getType() != TempAlloca->getType()) {
        RecoveredAlloca->setName(Twine(TempAlloca->getName()) + ".i8");
        RecoveredAlloca =
            Builder.CreateBitCast(RecoveredAlloca, TempAlloca->getType());
      }
      TempAlloca->replaceAllUsesWith(RecoveredAlloca);
      TempAlloca->removeFromParent();
      RecoveredAlloca->takeName(TempAlloca);
      delete TempAlloca;
    }
  } // End for each FrameVarInfo entry.

  // Insert 'call void (...)* @llvm.frameescape(...)' at the end of the entry
  // block.
  Builder.SetInsertPoint(&F.getEntryBlock().back());
  Builder.CreateCall(FrameEscapeFn, AllocasToEscape);

  // Clean up the handler action maps we created for this function
  DeleteContainerSeconds(CatchHandlerMap);
  CatchHandlerMap.clear();
  DeleteContainerSeconds(CleanupHandlerMap);
  CleanupHandlerMap.clear();

  return HandlersOutlined;
}

void WinEHPrepare::promoteLandingPadValues(LandingPadInst *LPad) {
  // If the return values of the landing pad instruction are extracted and
  // stored to memory, we want to promote the store locations to reg values.
  SmallVector<AllocaInst *, 2> EHAllocas;

  // The landingpad instruction returns an aggregate value.  Typically, its
  // value will be passed to a pair of extract value instructions and the
  // results of those extracts are often passed to store instructions.
  // In unoptimized code the stored value will often be loaded and then stored
  // again.
  for (auto *U : LPad->users()) {
    ExtractValueInst *Extract = dyn_cast<ExtractValueInst>(U);
    if (!Extract)
      continue;

    for (auto *EU : Extract->users()) {
      if (auto *Store = dyn_cast<StoreInst>(EU)) {
        auto *AV = cast<AllocaInst>(Store->getPointerOperand());
        EHAllocas.push_back(AV);
      }
    }
  }

  // We can't do this without a dominator tree.
  assert(DT);

  if (!EHAllocas.empty()) {
    PromoteMemToReg(EHAllocas, *DT);
    EHAllocas.clear();
  }

  // After promotion, some extracts may be trivially dead. Remove them.
  SmallVector<Value *, 4> Users(LPad->user_begin(), LPad->user_end());
  for (auto *U : Users)
    RecursivelyDeleteTriviallyDeadInstructions(U);
}

void WinEHPrepare::completeNestedLandingPad(Function *ParentFn,
                                            LandingPadInst *OutlinedLPad,
                                            const LandingPadInst *OriginalLPad,
                                            FrameVarInfoMap &FrameVarInfo) {
  // Get the nested block and erase the unreachable instruction that was
  // temporarily inserted as its terminator.
  LLVMContext &Context = ParentFn->getContext();
  BasicBlock *OutlinedBB = OutlinedLPad->getParent();
  assert(isa<UnreachableInst>(OutlinedBB->getTerminator()));
  OutlinedBB->getTerminator()->eraseFromParent();
  // That should leave OutlinedLPad as the last instruction in its block.
  assert(&OutlinedBB->back() == OutlinedLPad);

  // The original landing pad will have already had its action intrinsic
  // built by the outlining loop.  We need to clone that into the outlined
  // location.  It may also be necessary to add references to the exception
  // variables to the outlined handler in which this landing pad is nested
  // and remap return instructions in the nested handlers that should return
  // to an address in the outlined handler.
  Function *OutlinedHandlerFn = OutlinedBB->getParent();
  BasicBlock::const_iterator II = OriginalLPad;
  ++II;
  // The instruction after the landing pad should now be a call to eh.actions.
  const Instruction *Recover = II;
  assert(match(Recover, m_Intrinsic<Intrinsic::eh_actions>()));
  IntrinsicInst *EHActions = cast<IntrinsicInst>(Recover->clone());

  // Remap the exception variables into the outlined function.
  WinEHFrameVariableMaterializer Materializer(OutlinedHandlerFn, FrameVarInfo);
  SmallVector<BlockAddress *, 4> ActionTargets;
  SmallVector<ActionHandler *, 4> ActionList;
  parseEHActions(EHActions, ActionList);
  for (auto *Action : ActionList) {
    auto *Catch = dyn_cast<CatchHandler>(Action);
    if (!Catch)
      continue;
    // The dyn_cast to function here selects C++ catch handlers and skips
    // SEH catch handlers.
    auto *Handler = dyn_cast<Function>(Catch->getHandlerBlockOrFunc());
    if (!Handler)
      continue;
    // Visit all the return instructions, looking for places that return
    // to a location within OutlinedHandlerFn.
    for (BasicBlock &NestedHandlerBB : *Handler) {
      auto *Ret = dyn_cast<ReturnInst>(NestedHandlerBB.getTerminator());
      if (!Ret)
        continue;

      // Handler functions must always return a block address.
      BlockAddress *BA = cast<BlockAddress>(Ret->getReturnValue());
      // The original target will have been in the main parent function,
      // but if it is the address of a block that has been outlined, it
      // should be a block that was outlined into OutlinedHandlerFn.
      assert(BA->getFunction() == ParentFn);

      // Ignore targets that aren't part of OutlinedHandlerFn.
      if (!LPadTargetBlocks.count(BA->getBasicBlock()))
        continue;

      // If the return value is the address ofF a block that we
      // previously outlined into the parent handler function, replace
      // the return instruction and add the mapped target to the list
      // of possible return addresses.
      BasicBlock *MappedBB = LPadTargetBlocks[BA->getBasicBlock()];
      assert(MappedBB->getParent() == OutlinedHandlerFn);
      BlockAddress *NewBA = BlockAddress::get(OutlinedHandlerFn, MappedBB);
      Ret->eraseFromParent();
      ReturnInst::Create(Context, NewBA, &NestedHandlerBB);
      ActionTargets.push_back(NewBA);
    }
  }
  DeleteContainerPointers(ActionList);
  ActionList.clear();
  OutlinedBB->getInstList().push_back(EHActions);

  // Insert an indirect branch into the outlined landing pad BB.
  IndirectBrInst *IBr = IndirectBrInst::Create(EHActions, 0, OutlinedBB);
  // Add the previously collected action targets.
  for (auto *Target : ActionTargets)
    IBr->addDestination(Target->getBasicBlock());
}

// This function examines a block to determine whether the block ends with a
// conditional branch to a catch handler based on a selector comparison.
// This function is used both by the WinEHPrepare::findSelectorComparison() and
// WinEHCleanupDirector::handleTypeIdFor().
static bool isSelectorDispatch(BasicBlock *BB, BasicBlock *&CatchHandler,
                               Constant *&Selector, BasicBlock *&NextBB) {
  ICmpInst::Predicate Pred;
  BasicBlock *TBB, *FBB;
  Value *LHS, *RHS;

  if (!match(BB->getTerminator(),
             m_Br(m_ICmp(Pred, m_Value(LHS), m_Value(RHS)), TBB, FBB)))
    return false;

  if (!match(LHS,
             m_Intrinsic<Intrinsic::eh_typeid_for>(m_Constant(Selector))) &&
      !match(RHS, m_Intrinsic<Intrinsic::eh_typeid_for>(m_Constant(Selector))))
    return false;

  if (Pred == CmpInst::ICMP_EQ) {
    CatchHandler = TBB;
    NextBB = FBB;
    return true;
  }

  if (Pred == CmpInst::ICMP_NE) {
    CatchHandler = FBB;
    NextBB = TBB;
    return true;
  }

  return false;
}

static BasicBlock *createStubLandingPad(Function *Handler,
                                        Value *PersonalityFn) {
  // FIXME: Finish this!
  LLVMContext &Context = Handler->getContext();
  BasicBlock *StubBB = BasicBlock::Create(Context, "stub");
  Handler->getBasicBlockList().push_back(StubBB);
  IRBuilder<> Builder(StubBB);
  LandingPadInst *LPad = Builder.CreateLandingPad(
      llvm::StructType::get(Type::getInt8PtrTy(Context),
                            Type::getInt32Ty(Context), nullptr),
      PersonalityFn, 0);
  LPad->setCleanup(true);
  Builder.CreateUnreachable();
  return StubBB;
}

// Cycles through the blocks in an outlined handler function looking for an
// invoke instruction and inserts an invoke of llvm.donothing with an empty
// landing pad if none is found.  The code that generates the .xdata tables for
// the handler needs at least one landing pad to identify the parent function's
// personality.
void WinEHPrepare::addStubInvokeToHandlerIfNeeded(Function *Handler,
                                                  Value *PersonalityFn) {
  ReturnInst *Ret = nullptr;
  for (BasicBlock &BB : *Handler) {
    TerminatorInst *Terminator = BB.getTerminator();
    // If we find an invoke, there is nothing to be done.
    auto *II = dyn_cast<InvokeInst>(Terminator);
    if (II)
      return;
    // If we've already recorded a return instruction, keep looking for invokes.
    if (Ret)
      continue;
    // If we haven't recorded a return instruction yet, try this terminator.
    Ret = dyn_cast<ReturnInst>(Terminator);
  }

  // If we got this far, the handler contains no invokes.  We should have seen
  // at least one return.  We'll insert an invoke of llvm.donothing ahead of
  // that return.
  assert(Ret);
  BasicBlock *OldRetBB = Ret->getParent();
  BasicBlock *NewRetBB = SplitBlock(OldRetBB, Ret);
  // SplitBlock adds an unconditional branch instruction at the end of the
  // parent block.  We want to replace that with an invoke call, so we can
  // erase it now.
  OldRetBB->getTerminator()->eraseFromParent();
  BasicBlock *StubLandingPad = createStubLandingPad(Handler, PersonalityFn);
  Function *F =
      Intrinsic::getDeclaration(Handler->getParent(), Intrinsic::donothing);
  InvokeInst::Create(F, NewRetBB, StubLandingPad, None, "", OldRetBB);
}

bool WinEHPrepare::outlineHandler(ActionHandler *Action, Function *SrcFn,
                                  LandingPadInst *LPad, BasicBlock *StartBB,
                                  FrameVarInfoMap &VarInfo) {
  Module *M = SrcFn->getParent();
  LLVMContext &Context = M->getContext();

  // Create a new function to receive the handler contents.
  Type *Int8PtrType = Type::getInt8PtrTy(Context);
  std::vector<Type *> ArgTys;
  ArgTys.push_back(Int8PtrType);
  ArgTys.push_back(Int8PtrType);
  Function *Handler;
  if (Action->getType() == Catch) {
    FunctionType *FnType = FunctionType::get(Int8PtrType, ArgTys, false);
    Handler = Function::Create(FnType, GlobalVariable::InternalLinkage,
                               SrcFn->getName() + ".catch", M);
  } else {
    FunctionType *FnType =
        FunctionType::get(Type::getVoidTy(Context), ArgTys, false);
    Handler = Function::Create(FnType, GlobalVariable::InternalLinkage,
                               SrcFn->getName() + ".cleanup", M);
  }

  Handler->addFnAttr("wineh-parent", SrcFn->getName());

  // Generate a standard prolog to setup the frame recovery structure.
  IRBuilder<> Builder(Context);
  BasicBlock *Entry = BasicBlock::Create(Context, "entry");
  Handler->getBasicBlockList().push_front(Entry);
  Builder.SetInsertPoint(Entry);
  Builder.SetCurrentDebugLocation(LPad->getDebugLoc());

  std::unique_ptr<WinEHCloningDirectorBase> Director;

  ValueToValueMapTy VMap;

  LandingPadMap &LPadMap = LPadMaps[LPad];
  if (!LPadMap.isInitialized())
    LPadMap.mapLandingPad(LPad);
  if (auto *CatchAction = dyn_cast<CatchHandler>(Action)) {
    Constant *Sel = CatchAction->getSelector();
    Director.reset(new WinEHCatchDirector(Handler, Sel, VarInfo, LPadMap,
                                          NestedLPtoOriginalLP));
    LPadMap.remapEHValues(VMap, UndefValue::get(Int8PtrType),
                          ConstantInt::get(Type::getInt32Ty(Context), 1));
  } else {
    Director.reset(new WinEHCleanupDirector(Handler, VarInfo, LPadMap));
    LPadMap.remapEHValues(VMap, UndefValue::get(Int8PtrType),
                          UndefValue::get(Type::getInt32Ty(Context)));
  }

  SmallVector<ReturnInst *, 8> Returns;
  ClonedCodeInfo OutlinedFunctionInfo;

  // If the start block contains PHI nodes, we need to map them.
  BasicBlock::iterator II = StartBB->begin();
  while (auto *PN = dyn_cast<PHINode>(II)) {
    bool Mapped = false;
    // Look for PHI values that we have already mapped (such as the selector).
    for (Value *Val : PN->incoming_values()) {
      if (VMap.count(Val)) {
        VMap[PN] = VMap[Val];
        Mapped = true;
      }
    }
    // If we didn't find a match for this value, map it as an undef.
    if (!Mapped) {
      VMap[PN] = UndefValue::get(PN->getType());
    }
    ++II;
  }

  // Skip over PHIs and, if applicable, landingpad instructions.
  II = StartBB->getFirstInsertionPt();

  CloneAndPruneIntoFromInst(Handler, SrcFn, II, VMap,
                            /*ModuleLevelChanges=*/false, Returns, "",
                            &OutlinedFunctionInfo, Director.get());

  // Move all the instructions in the first cloned block into our entry block.
  BasicBlock *FirstClonedBB = std::next(Function::iterator(Entry));
  Entry->getInstList().splice(Entry->end(), FirstClonedBB->getInstList());
  FirstClonedBB->eraseFromParent();

  // Make sure we can identify the handler's personality later.
  addStubInvokeToHandlerIfNeeded(Handler, LPad->getPersonalityFn());

  if (auto *CatchAction = dyn_cast<CatchHandler>(Action)) {
    WinEHCatchDirector *CatchDirector =
        reinterpret_cast<WinEHCatchDirector *>(Director.get());
    CatchAction->setExceptionVar(CatchDirector->getExceptionVar());
    CatchAction->setReturnTargets(CatchDirector->getReturnTargets());

    // Look for blocks that are not part of the landing pad that we just
    // outlined but terminate with a call to llvm.eh.endcatch and a
    // branch to a block that is in the handler we just outlined.
    // These blocks will be part of a nested landing pad that intends to
    // return to an address in this handler.  This case is best handled
    // after both landing pads have been outlined, so for now we'll just
    // save the association of the blocks in LPadTargetBlocks.  The
    // return instructions which are created from these branches will be
    // replaced after all landing pads have been outlined.
    for (const auto MapEntry : VMap) {
      // VMap maps all values and blocks that were just cloned, but dead
      // blocks which were pruned will map to nullptr.
      if (!isa<BasicBlock>(MapEntry.first) || MapEntry.second == nullptr)
        continue;
      const BasicBlock *MappedBB = cast<BasicBlock>(MapEntry.first);
      for (auto *Pred : predecessors(const_cast<BasicBlock *>(MappedBB))) {
        auto *Branch = dyn_cast<BranchInst>(Pred->getTerminator());
        if (!Branch || !Branch->isUnconditional() || Pred->size() <= 1)
          continue;
        BasicBlock::iterator II = const_cast<BranchInst *>(Branch);
        --II;
        if (match(cast<Value>(II), m_Intrinsic<Intrinsic::eh_endcatch>())) {
          // This would indicate that a nested landing pad wants to return
          // to a block that is outlined into two different handlers.
          assert(!LPadTargetBlocks.count(MappedBB));
          LPadTargetBlocks[MappedBB] = cast<BasicBlock>(MapEntry.second);
        }
      }
    }
  } // End if (CatchAction)

  Action->setHandlerBlockOrFunc(Handler);

  return true;
}

/// This BB must end in a selector dispatch. All we need to do is pass the
/// handler block to llvm.eh.actions and list it as a possible indirectbr
/// target.
void WinEHPrepare::processSEHCatchHandler(CatchHandler *CatchAction,
                                          BasicBlock *StartBB) {
  BasicBlock *HandlerBB;
  BasicBlock *NextBB;
  Constant *Selector;
  bool Res = isSelectorDispatch(StartBB, HandlerBB, Selector, NextBB);
  if (Res) {
    // If this was EH dispatch, this must be a conditional branch to the handler
    // block.
    // FIXME: Handle instructions in the dispatch block. Currently we drop them,
    // leading to crashes if some optimization hoists stuff here.
    assert(CatchAction->getSelector() && HandlerBB &&
           "expected catch EH dispatch");
  } else {
    // This must be a catch-all. Split the block after the landingpad.
    assert(CatchAction->getSelector()->isNullValue() && "expected catch-all");
    HandlerBB =
        StartBB->splitBasicBlock(StartBB->getFirstInsertionPt(), "catch.all");
  }
  CatchAction->setHandlerBlockOrFunc(BlockAddress::get(HandlerBB));
  TinyPtrVector<BasicBlock *> Targets(HandlerBB);
  CatchAction->setReturnTargets(Targets);
}

void LandingPadMap::mapLandingPad(const LandingPadInst *LPad) {
  // Each instance of this class should only ever be used to map a single
  // landing pad.
  assert(OriginLPad == nullptr || OriginLPad == LPad);

  // If the landing pad has already been mapped, there's nothing more to do.
  if (OriginLPad == LPad)
    return;

  OriginLPad = LPad;

  // The landingpad instruction returns an aggregate value.  Typically, its
  // value will be passed to a pair of extract value instructions and the
  // results of those extracts will have been promoted to reg values before
  // this routine is called.
  for (auto *U : LPad->users()) {
    const ExtractValueInst *Extract = dyn_cast<ExtractValueInst>(U);
    if (!Extract)
      continue;
    assert(Extract->getNumIndices() == 1 &&
           "Unexpected operation: extracting both landing pad values");
    unsigned int Idx = *(Extract->idx_begin());
    assert((Idx == 0 || Idx == 1) &&
           "Unexpected operation: extracting an unknown landing pad element");
    if (Idx == 0) {
      ExtractedEHPtrs.push_back(Extract);
    } else if (Idx == 1) {
      ExtractedSelectors.push_back(Extract);
    }
  }
}

bool LandingPadMap::isOriginLandingPadBlock(const BasicBlock *BB) const {
  return BB->getLandingPadInst() == OriginLPad;
}

bool LandingPadMap::isLandingPadSpecificInst(const Instruction *Inst) const {
  if (Inst == OriginLPad)
    return true;
  for (auto *Extract : ExtractedEHPtrs) {
    if (Inst == Extract)
      return true;
  }
  for (auto *Extract : ExtractedSelectors) {
    if (Inst == Extract)
      return true;
  }
  return false;
}

void LandingPadMap::remapEHValues(ValueToValueMapTy &VMap, Value *EHPtrValue,
                                  Value *SelectorValue) const {
  // Remap all landing pad extract instructions to the specified values.
  for (auto *Extract : ExtractedEHPtrs)
    VMap[Extract] = EHPtrValue;
  for (auto *Extract : ExtractedSelectors)
    VMap[Extract] = SelectorValue;
}

CloningDirector::CloningAction WinEHCloningDirectorBase::handleInstruction(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  // If this is one of the boilerplate landing pad instructions, skip it.
  // The instruction will have already been remapped in VMap.
  if (LPadMap.isLandingPadSpecificInst(Inst))
    return CloningDirector::SkipInstruction;

  // Nested landing pads will be cloned as stubs, with just the
  // landingpad instruction and an unreachable instruction. When
  // all landingpads have been outlined, we'll replace this with the
  // llvm.eh.actions call and indirect branch created when the
  // landing pad was outlined.
  if (auto *LPad = dyn_cast<LandingPadInst>(Inst)) {
    return handleLandingPad(VMap, LPad, NewBB);
  }

  if (auto *Invoke = dyn_cast<InvokeInst>(Inst))
    return handleInvoke(VMap, Invoke, NewBB);

  if (auto *Resume = dyn_cast<ResumeInst>(Inst))
    return handleResume(VMap, Resume, NewBB);

  if (match(Inst, m_Intrinsic<Intrinsic::eh_begincatch>()))
    return handleBeginCatch(VMap, Inst, NewBB);
  if (match(Inst, m_Intrinsic<Intrinsic::eh_endcatch>()))
    return handleEndCatch(VMap, Inst, NewBB);
  if (match(Inst, m_Intrinsic<Intrinsic::eh_typeid_for>()))
    return handleTypeIdFor(VMap, Inst, NewBB);

  // Continue with the default cloning behavior.
  return CloningDirector::CloneInstruction;
}

CloningDirector::CloningAction WinEHCatchDirector::handleLandingPad(
    ValueToValueMapTy &VMap, const LandingPadInst *LPad, BasicBlock *NewBB) {
  Instruction *NewInst = LPad->clone();
  if (LPad->hasName())
    NewInst->setName(LPad->getName());
  // Save this correlation for later processing.
  NestedLPtoOriginalLP[cast<LandingPadInst>(NewInst)] = LPad;
  VMap[LPad] = NewInst;
  BasicBlock::InstListType &InstList = NewBB->getInstList();
  InstList.push_back(NewInst);
  InstList.push_back(new UnreachableInst(NewBB->getContext()));
  return CloningDirector::StopCloningBB;
}

CloningDirector::CloningAction WinEHCatchDirector::handleBeginCatch(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  // The argument to the call is some form of the first element of the
  // landingpad aggregate value, but that doesn't matter.  It isn't used
  // here.
  // The second argument is an outparameter where the exception object will be
  // stored. Typically the exception object is a scalar, but it can be an
  // aggregate when catching by value.
  // FIXME: Leave something behind to indicate where the exception object lives
  // for this handler. Should it be part of llvm.eh.actions?
  assert(ExceptionObjectVar == nullptr && "Multiple calls to "
                                          "llvm.eh.begincatch found while "
                                          "outlining catch handler.");
  ExceptionObjectVar = Inst->getOperand(1)->stripPointerCasts();
  if (isa<ConstantPointerNull>(ExceptionObjectVar))
    return CloningDirector::SkipInstruction;
  assert(cast<AllocaInst>(ExceptionObjectVar)->isStaticAlloca() &&
         "catch parameter is not static alloca");
  Materializer.escapeCatchObject(ExceptionObjectVar);
  return CloningDirector::SkipInstruction;
}

CloningDirector::CloningAction
WinEHCatchDirector::handleEndCatch(ValueToValueMapTy &VMap,
                                   const Instruction *Inst, BasicBlock *NewBB) {
  auto *IntrinCall = dyn_cast<IntrinsicInst>(Inst);
  // It might be interesting to track whether or not we are inside a catch
  // function, but that might make the algorithm more brittle than it needs
  // to be.

  // The end catch call can occur in one of two places: either in a
  // landingpad block that is part of the catch handlers exception mechanism,
  // or at the end of the catch block.  However, a catch-all handler may call
  // end catch from the original landing pad.  If the call occurs in a nested
  // landing pad block, we must skip it and continue so that the landing pad
  // gets cloned.
  auto *ParentBB = IntrinCall->getParent();
  if (ParentBB->isLandingPad() && !LPadMap.isOriginLandingPadBlock(ParentBB))
    return CloningDirector::SkipInstruction;

  // If an end catch occurs anywhere else we want to terminate the handler
  // with a return to the code that follows the endcatch call.  If the
  // next instruction is not an unconditional branch, we need to split the
  // block to provide a clear target for the return instruction.
  BasicBlock *ContinueBB;
  auto Next = std::next(BasicBlock::const_iterator(IntrinCall));
  const BranchInst *Branch = dyn_cast<BranchInst>(Next);
  if (!Branch || !Branch->isUnconditional()) {
    // We're interrupting the cloning process at this location, so the
    // const_cast we're doing here will not cause a problem.
    ContinueBB = SplitBlock(const_cast<BasicBlock *>(ParentBB),
                            const_cast<Instruction *>(cast<Instruction>(Next)));
  } else {
    ContinueBB = Branch->getSuccessor(0);
  }

  ReturnInst::Create(NewBB->getContext(), BlockAddress::get(ContinueBB), NewBB);
  ReturnTargets.push_back(ContinueBB);

  // We just added a terminator to the cloned block.
  // Tell the caller to stop processing the current basic block so that
  // the branch instruction will be skipped.
  return CloningDirector::StopCloningBB;
}

CloningDirector::CloningAction WinEHCatchDirector::handleTypeIdFor(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  auto *IntrinCall = dyn_cast<IntrinsicInst>(Inst);
  Value *Selector = IntrinCall->getArgOperand(0)->stripPointerCasts();
  // This causes a replacement that will collapse the landing pad CFG based
  // on the filter function we intend to match.
  if (Selector == CurrentSelector)
    VMap[Inst] = ConstantInt::get(SelectorIDType, 1);
  else
    VMap[Inst] = ConstantInt::get(SelectorIDType, 0);
  // Tell the caller not to clone this instruction.
  return CloningDirector::SkipInstruction;
}

CloningDirector::CloningAction
WinEHCatchDirector::handleInvoke(ValueToValueMapTy &VMap,
                                 const InvokeInst *Invoke, BasicBlock *NewBB) {
  return CloningDirector::CloneInstruction;
}

CloningDirector::CloningAction
WinEHCatchDirector::handleResume(ValueToValueMapTy &VMap,
                                 const ResumeInst *Resume, BasicBlock *NewBB) {
  // Resume instructions shouldn't be reachable from catch handlers.
  // We still need to handle it, but it will be pruned.
  BasicBlock::InstListType &InstList = NewBB->getInstList();
  InstList.push_back(new UnreachableInst(NewBB->getContext()));
  return CloningDirector::StopCloningBB;
}

CloningDirector::CloningAction WinEHCleanupDirector::handleLandingPad(
    ValueToValueMapTy &VMap, const LandingPadInst *LPad, BasicBlock *NewBB) {
  // The MS runtime will terminate the process if an exception occurs in a
  // cleanup handler, so we shouldn't encounter landing pads in the actual
  // cleanup code, but they may appear in catch blocks.  Depending on where
  // we started cloning we may see one, but it will get dropped during dead
  // block pruning.
  Instruction *NewInst = new UnreachableInst(NewBB->getContext());
  VMap[LPad] = NewInst;
  BasicBlock::InstListType &InstList = NewBB->getInstList();
  InstList.push_back(NewInst);
  return CloningDirector::StopCloningBB;
}

CloningDirector::CloningAction WinEHCleanupDirector::handleBeginCatch(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  // Catch blocks within cleanup handlers will always be unreachable.
  // We'll insert an unreachable instruction now, but it will be pruned
  // before the cloning process is complete.
  BasicBlock::InstListType &InstList = NewBB->getInstList();
  InstList.push_back(new UnreachableInst(NewBB->getContext()));
  return CloningDirector::StopCloningBB;
}

CloningDirector::CloningAction WinEHCleanupDirector::handleEndCatch(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  // Cleanup handlers nested within catch handlers may begin with a call to
  // eh.endcatch.  We can just ignore that instruction.
  return CloningDirector::SkipInstruction;
}

CloningDirector::CloningAction WinEHCleanupDirector::handleTypeIdFor(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  // If we encounter a selector comparison while cloning a cleanup handler,
  // we want to stop cloning immediately.  Anything after the dispatch
  // will be outlined into a different handler.
  BasicBlock *CatchHandler;
  Constant *Selector;
  BasicBlock *NextBB;
  if (isSelectorDispatch(const_cast<BasicBlock *>(Inst->getParent()),
                         CatchHandler, Selector, NextBB)) {
    ReturnInst::Create(NewBB->getContext(), nullptr, NewBB);
    return CloningDirector::StopCloningBB;
  }
  // If eg.typeid.for is called for any other reason, it can be ignored.
  VMap[Inst] = ConstantInt::get(SelectorIDType, 0);
  return CloningDirector::SkipInstruction;
}

CloningDirector::CloningAction WinEHCleanupDirector::handleInvoke(
    ValueToValueMapTy &VMap, const InvokeInst *Invoke, BasicBlock *NewBB) {
  // All invokes in cleanup handlers can be replaced with calls.
  SmallVector<Value *, 16> CallArgs(Invoke->op_begin(), Invoke->op_end() - 3);
  // Insert a normal call instruction...
  CallInst *NewCall =
      CallInst::Create(const_cast<Value *>(Invoke->getCalledValue()), CallArgs,
                       Invoke->getName(), NewBB);
  NewCall->setCallingConv(Invoke->getCallingConv());
  NewCall->setAttributes(Invoke->getAttributes());
  NewCall->setDebugLoc(Invoke->getDebugLoc());
  VMap[Invoke] = NewCall;

  // Remap the operands.
  llvm::RemapInstruction(NewCall, VMap, RF_None, nullptr, &Materializer);

  // Insert an unconditional branch to the normal destination.
  BranchInst::Create(Invoke->getNormalDest(), NewBB);

  // The unwind destination won't be cloned into the new function, so
  // we don't need to clean up its phi nodes.

  // We just added a terminator to the cloned block.
  // Tell the caller to stop processing the current basic block.
  return CloningDirector::CloneSuccessors;
}

CloningDirector::CloningAction WinEHCleanupDirector::handleResume(
    ValueToValueMapTy &VMap, const ResumeInst *Resume, BasicBlock *NewBB) {
  ReturnInst::Create(NewBB->getContext(), nullptr, NewBB);

  // We just added a terminator to the cloned block.
  // Tell the caller to stop processing the current basic block so that
  // the branch instruction will be skipped.
  return CloningDirector::StopCloningBB;
}

WinEHFrameVariableMaterializer::WinEHFrameVariableMaterializer(
    Function *OutlinedFn, FrameVarInfoMap &FrameVarInfo)
    : FrameVarInfo(FrameVarInfo), Builder(OutlinedFn->getContext()) {
  BasicBlock *EntryBB = &OutlinedFn->getEntryBlock();
  Builder.SetInsertPoint(EntryBB, EntryBB->getFirstInsertionPt());
}

Value *WinEHFrameVariableMaterializer::materializeValueFor(Value *V) {
  // If we're asked to materialize a value that is an instruction, we
  // temporarily create an alloca in the outlined function and add this
  // to the FrameVarInfo map.  When all the outlining is complete, we'll
  // collect these into a structure, spilling non-alloca values in the
  // parent frame as necessary, and replace these temporary allocas with
  // GEPs referencing the frame allocation block.

  // If the value is an alloca, the mapping is direct.
  if (auto *AV = dyn_cast<AllocaInst>(V)) {
    AllocaInst *NewAlloca = dyn_cast<AllocaInst>(AV->clone());
    Builder.Insert(NewAlloca, AV->getName());
    FrameVarInfo[AV].push_back(NewAlloca);
    return NewAlloca;
  }

  // For other types of instructions or arguments, we need an alloca based on
  // the value's type and a load of the alloca.  The alloca will be replaced
  // by a GEP, but the load will stay.  In the parent function, the value will
  // be spilled to a location in the frame allocation block.
  if (isa<Instruction>(V) || isa<Argument>(V)) {
    AllocaInst *NewAlloca =
        Builder.CreateAlloca(V->getType(), nullptr, "eh.temp.alloca");
    FrameVarInfo[V].push_back(NewAlloca);
    LoadInst *NewLoad = Builder.CreateLoad(NewAlloca, V->getName() + ".reload");
    return NewLoad;
  }

  // Don't materialize other values.
  return nullptr;
}

void WinEHFrameVariableMaterializer::escapeCatchObject(Value *V) {
  // Catch parameter objects have to live in the parent frame. When we see a use
  // of a catch parameter, add a sentinel to the multimap to indicate that it's
  // used from another handler. This will prevent us from trying to sink the
  // alloca into the handler and ensure that the catch parameter is present in
  // the call to llvm.frameescape.
  FrameVarInfo[V].push_back(getCatchObjectSentinel());
}

// This function maps the catch and cleanup handlers that are reachable from the
// specified landing pad. The landing pad sequence will have this basic shape:
//
//  <cleanup handler>
//  <selector comparison>
//  <catch handler>
//  <cleanup handler>
//  <selector comparison>
//  <catch handler>
//  <cleanup handler>
//  ...
//
// Any of the cleanup slots may be absent.  The cleanup slots may be occupied by
// any arbitrary control flow, but all paths through the cleanup code must
// eventually reach the next selector comparison and no path can skip to a
// different selector comparisons, though some paths may terminate abnormally.
// Therefore, we will use a depth first search from the start of any given
// cleanup block and stop searching when we find the next selector comparison.
//
// If the landingpad instruction does not have a catch clause, we will assume
// that any instructions other than selector comparisons and catch handlers can
// be ignored.  In practice, these will only be the boilerplate instructions.
//
// The catch handlers may also have any control structure, but we are only
// interested in the start of the catch handlers, so we don't need to actually
// follow the flow of the catch handlers.  The start of the catch handlers can
// be located from the compare instructions, but they can be skipped in the
// flow by following the contrary branch.
void WinEHPrepare::mapLandingPadBlocks(LandingPadInst *LPad,
                                       LandingPadActions &Actions) {
  unsigned int NumClauses = LPad->getNumClauses();
  unsigned int HandlersFound = 0;
  BasicBlock *BB = LPad->getParent();

  DEBUG(dbgs() << "Mapping landing pad: " << BB->getName() << "\n");

  if (NumClauses == 0) {
    findCleanupHandlers(Actions, BB, nullptr);
    return;
  }

  VisitedBlockSet VisitedBlocks;

  while (HandlersFound != NumClauses) {
    BasicBlock *NextBB = nullptr;

    // See if the clause we're looking for is a catch-all.
    // If so, the catch begins immediately.
    if (isa<ConstantPointerNull>(LPad->getClause(HandlersFound))) {
      // The catch all must occur last.
      assert(HandlersFound == NumClauses - 1);

      // For C++ EH, check if there is any interesting cleanup code before we
      // begin the catch. This is important because cleanups cannot rethrow
      // exceptions but code called from catches can. For SEH, it isn't
      // important if some finally code before a catch-all is executed out of
      // line or after recovering from the exception.
      if (Personality == EHPersonality::MSVC_CXX)
        findCleanupHandlers(Actions, BB, BB);

      // Add the catch handler to the action list.
      CatchHandler *Action =
          new CatchHandler(BB, LPad->getClause(HandlersFound), nullptr);
      CatchHandlerMap[BB] = Action;
      Actions.insertCatchHandler(Action);
      DEBUG(dbgs() << "  Catch all handler at block " << BB->getName() << "\n");
      ++HandlersFound;

      // Once we reach a catch-all, don't expect to hit a resume instruction.
      BB = nullptr;
      break;
    }

    CatchHandler *CatchAction = findCatchHandler(BB, NextBB, VisitedBlocks);
    // See if there is any interesting code executed before the dispatch.
    findCleanupHandlers(Actions, BB, CatchAction->getStartBlock());

    assert(CatchAction);
    ++HandlersFound;

    // Add the catch handler to the action list.
    Actions.insertCatchHandler(CatchAction);
    DEBUG(dbgs() << "  Found catch dispatch in block "
                 << CatchAction->getStartBlock()->getName() << "\n");

    // Move on to the block after the catch handler.
    BB = NextBB;
  }

  // If we didn't wind up in a catch-all, see if there is any interesting code
  // executed before the resume.
  findCleanupHandlers(Actions, BB, BB);

  // It's possible that some optimization moved code into a landingpad that
  // wasn't
  // previously being used for cleanup.  If that happens, we need to execute
  // that
  // extra code from a cleanup handler.
  if (Actions.includesCleanup() && !LPad->isCleanup())
    LPad->setCleanup(true);
}

// This function searches starting with the input block for the next
// block that terminates with a branch whose condition is based on a selector
// comparison.  This may be the input block.  See the mapLandingPadBlocks
// comments for a discussion of control flow assumptions.
//
CatchHandler *WinEHPrepare::findCatchHandler(BasicBlock *BB,
                                             BasicBlock *&NextBB,
                                             VisitedBlockSet &VisitedBlocks) {
  // See if we've already found a catch handler use it.
  // Call count() first to avoid creating a null entry for blocks
  // we haven't seen before.
  if (CatchHandlerMap.count(BB) && CatchHandlerMap[BB] != nullptr) {
    CatchHandler *Action = cast<CatchHandler>(CatchHandlerMap[BB]);
    NextBB = Action->getNextBB();
    return Action;
  }

  // VisitedBlocks applies only to the current search.  We still
  // need to consider blocks that we've visited while mapping other
  // landing pads.
  VisitedBlocks.insert(BB);

  BasicBlock *CatchBlock = nullptr;
  Constant *Selector = nullptr;

  // If this is the first time we've visited this block from any landing pad
  // look to see if it is a selector dispatch block.
  if (!CatchHandlerMap.count(BB)) {
    if (isSelectorDispatch(BB, CatchBlock, Selector, NextBB)) {
      CatchHandler *Action = new CatchHandler(BB, Selector, NextBB);
      CatchHandlerMap[BB] = Action;
      return Action;
    }
  }

  // Visit each successor, looking for the dispatch.
  // FIXME: We expect to find the dispatch quickly, so this will probably
  //        work better as a breadth first search.
  for (BasicBlock *Succ : successors(BB)) {
    if (VisitedBlocks.count(Succ))
      continue;

    CatchHandler *Action = findCatchHandler(Succ, NextBB, VisitedBlocks);
    if (Action)
      return Action;
  }
  return nullptr;
}

// These are helper functions to combine repeated code from findCleanupHandlers.
static void createCleanupHandler(LandingPadActions &Actions,
                                 CleanupHandlerMapTy &CleanupHandlerMap,
                                 BasicBlock *BB) {
  CleanupHandler *Action = new CleanupHandler(BB);
  CleanupHandlerMap[BB] = Action;
  Actions.insertCleanupHandler(Action);
  DEBUG(dbgs() << "  Found cleanup code in block "
               << Action->getStartBlock()->getName() << "\n");
}

static bool isFrameAddressCall(Value *V) {
  return match(V, m_Intrinsic<Intrinsic::frameaddress>(m_SpecificInt(0)));
}

static CallSite matchOutlinedFinallyCall(BasicBlock *BB,
                                         Instruction *MaybeCall) {
  // Look for finally blocks that Clang has already outlined for us.
  //   %fp = call i8* @llvm.frameaddress(i32 0)
  //   call void @"fin$parent"(iN 1, i8* %fp)
  if (isFrameAddressCall(MaybeCall) && MaybeCall != BB->getTerminator())
    MaybeCall = MaybeCall->getNextNode();
  CallSite FinallyCall(MaybeCall);
  if (!FinallyCall || FinallyCall.arg_size() != 2)
    return CallSite();
  if (!match(FinallyCall.getArgument(0), m_SpecificInt(1)))
    return CallSite();
  if (!isFrameAddressCall(FinallyCall.getArgument(1)))
    return CallSite();
  return FinallyCall;
}

static BasicBlock *followSingleUnconditionalBranches(BasicBlock *BB) {
  // Skip single ubr blocks.
  while (BB->getFirstNonPHIOrDbg() == BB->getTerminator()) {
    auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
    if (Br && Br->isUnconditional())
      BB = Br->getSuccessor(0);
    else
      return BB;
  }
  return BB;
}

// This function searches starting with the input block for the next block that
// contains code that is not part of a catch handler and would not be eliminated
// during handler outlining.
//
void WinEHPrepare::findCleanupHandlers(LandingPadActions &Actions,
                                       BasicBlock *StartBB, BasicBlock *EndBB) {
  // Here we will skip over the following:
  //
  // landing pad prolog:
  //
  // Unconditional branches
  //
  // Selector dispatch
  //
  // Resume pattern
  //
  // Anything else marks the start of an interesting block

  BasicBlock *BB = StartBB;
  // Anything other than an unconditional branch will kick us out of this loop
  // one way or another.
  while (BB) {
    BB = followSingleUnconditionalBranches(BB);
    // If we've already scanned this block, don't scan it again.  If it is
    // a cleanup block, there will be an action in the CleanupHandlerMap.
    // If we've scanned it and it is not a cleanup block, there will be a
    // nullptr in the CleanupHandlerMap.  If we have not scanned it, there will
    // be no entry in the CleanupHandlerMap.  We must call count() first to
    // avoid creating a null entry for blocks we haven't scanned.
    if (CleanupHandlerMap.count(BB)) {
      if (auto *Action = CleanupHandlerMap[BB]) {
        Actions.insertCleanupHandler(Action);
        DEBUG(dbgs() << "  Found cleanup code in block "
              << Action->getStartBlock()->getName() << "\n");
        // FIXME: This cleanup might chain into another, and we need to discover
        // that.
        return;
      } else {
        // Here we handle the case where the cleanup handler map contains a
        // value for this block but the value is a nullptr.  This means that
        // we have previously analyzed the block and determined that it did
        // not contain any cleanup code.  Based on the earlier analysis, we
        // know the the block must end in either an unconditional branch, a
        // resume or a conditional branch that is predicated on a comparison
        // with a selector.  Either the resume or the selector dispatch
        // would terminate the search for cleanup code, so the unconditional
        // branch is the only case for which we might need to continue
        // searching.
        BasicBlock *SuccBB = followSingleUnconditionalBranches(BB);
        if (SuccBB == BB || SuccBB == EndBB)
          return;
        BB = SuccBB;
        continue;
      }
    }

    // Create an entry in the cleanup handler map for this block.  Initially
    // we create an entry that says this isn't a cleanup block.  If we find
    // cleanup code, the caller will replace this entry.
    CleanupHandlerMap[BB] = nullptr;

    TerminatorInst *Terminator = BB->getTerminator();

    // Landing pad blocks have extra instructions we need to accept.
    LandingPadMap *LPadMap = nullptr;
    if (BB->isLandingPad()) {
      LandingPadInst *LPad = BB->getLandingPadInst();
      LPadMap = &LPadMaps[LPad];
      if (!LPadMap->isInitialized())
        LPadMap->mapLandingPad(LPad);
    }

    // Look for the bare resume pattern:
    //   %lpad.val1 = insertvalue { i8*, i32 } undef, i8* %exn, 0
    //   %lpad.val2 = insertvalue { i8*, i32 } %lpad.val1, i32 %sel, 1
    //   resume { i8*, i32 } %lpad.val2
    if (auto *Resume = dyn_cast<ResumeInst>(Terminator)) {
      InsertValueInst *Insert1 = nullptr;
      InsertValueInst *Insert2 = nullptr;
      Value *ResumeVal = Resume->getOperand(0);
      // If the resume value isn't a phi or landingpad value, it should be a
      // series of insertions. Identify them so we can avoid them when scanning
      // for cleanups.
      if (!isa<PHINode>(ResumeVal) && !isa<LandingPadInst>(ResumeVal)) {
        Insert2 = dyn_cast<InsertValueInst>(ResumeVal);
        if (!Insert2)
          return createCleanupHandler(Actions, CleanupHandlerMap, BB);
        Insert1 = dyn_cast<InsertValueInst>(Insert2->getAggregateOperand());
        if (!Insert1)
          return createCleanupHandler(Actions, CleanupHandlerMap, BB);
      }
      for (BasicBlock::iterator II = BB->getFirstNonPHIOrDbg(), IE = BB->end();
           II != IE; ++II) {
        Instruction *Inst = II;
        if (LPadMap && LPadMap->isLandingPadSpecificInst(Inst))
          continue;
        if (Inst == Insert1 || Inst == Insert2 || Inst == Resume)
          continue;
        if (!Inst->hasOneUse() ||
            (Inst->user_back() != Insert1 && Inst->user_back() != Insert2)) {
          return createCleanupHandler(Actions, CleanupHandlerMap, BB);
        }
      }
      return;
    }

    BranchInst *Branch = dyn_cast<BranchInst>(Terminator);
    if (Branch && Branch->isConditional()) {
      // Look for the selector dispatch.
      //   %2 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIf to i8*))
      //   %matches = icmp eq i32 %sel, %2
      //   br i1 %matches, label %catch14, label %eh.resume
      CmpInst *Compare = dyn_cast<CmpInst>(Branch->getCondition());
      if (!Compare || !Compare->isEquality())
        return createCleanupHandler(Actions, CleanupHandlerMap, BB);
      for (BasicBlock::iterator II = BB->getFirstNonPHIOrDbg(), IE = BB->end();
           II != IE; ++II) {
        Instruction *Inst = II;
        if (LPadMap && LPadMap->isLandingPadSpecificInst(Inst))
          continue;
        if (Inst == Compare || Inst == Branch)
          continue;
        if (match(Inst, m_Intrinsic<Intrinsic::eh_typeid_for>()))
          continue;
        return createCleanupHandler(Actions, CleanupHandlerMap, BB);
      }
      // The selector dispatch block should always terminate our search.
      assert(BB == EndBB);
      return;
    }

    if (isAsynchronousEHPersonality(Personality)) {
      // If this is a landingpad block, split the block at the first non-landing
      // pad instruction.
      Instruction *MaybeCall = BB->getFirstNonPHIOrDbg();
      if (LPadMap) {
        while (MaybeCall != BB->getTerminator() &&
               LPadMap->isLandingPadSpecificInst(MaybeCall))
          MaybeCall = MaybeCall->getNextNode();
      }

      // Look for outlined finally calls.
      if (CallSite FinallyCall = matchOutlinedFinallyCall(BB, MaybeCall)) {
        Function *Fin = FinallyCall.getCalledFunction();
        assert(Fin && "outlined finally call should be direct");
        auto *Action = new CleanupHandler(BB);
        Action->setHandlerBlockOrFunc(Fin);
        Actions.insertCleanupHandler(Action);
        CleanupHandlerMap[BB] = Action;
        DEBUG(dbgs() << "  Found frontend-outlined finally call to "
                     << Fin->getName() << " in block "
                     << Action->getStartBlock()->getName() << "\n");

        // Split the block if there were more interesting instructions and look
        // for finally calls in the normal successor block.
        BasicBlock *SuccBB = BB;
        if (FinallyCall.getInstruction() != BB->getTerminator() &&
            FinallyCall.getInstruction()->getNextNode() != BB->getTerminator()) {
          SuccBB = BB->splitBasicBlock(FinallyCall.getInstruction()->getNextNode());
        } else {
          if (FinallyCall.isInvoke()) {
            SuccBB = cast<InvokeInst>(FinallyCall.getInstruction())->getNormalDest();
          } else {
            SuccBB = BB->getUniqueSuccessor();
            assert(SuccBB && "splitOutlinedFinallyCalls didn't insert a branch");
          }
        }
        BB = SuccBB;
        if (BB == EndBB)
          return;
        continue;
      }
    }

    // Anything else is either a catch block or interesting cleanup code.
    for (BasicBlock::iterator II = BB->getFirstNonPHIOrDbg(), IE = BB->end();
         II != IE; ++II) {
      Instruction *Inst = II;
      if (LPadMap && LPadMap->isLandingPadSpecificInst(Inst))
        continue;
      // Unconditional branches fall through to this loop.
      if (Inst == Branch)
        continue;
      // If this is a catch block, there is no cleanup code to be found.
      if (match(Inst, m_Intrinsic<Intrinsic::eh_begincatch>()))
        return;
      // If this a nested landing pad, it may contain an endcatch call.
      if (match(Inst, m_Intrinsic<Intrinsic::eh_endcatch>()))
        return;
      // Anything else makes this interesting cleanup code.
      return createCleanupHandler(Actions, CleanupHandlerMap, BB);
    }

    // Only unconditional branches in empty blocks should get this far.
    assert(Branch && Branch->isUnconditional());
    if (BB == EndBB)
      return;
    BB = Branch->getSuccessor(0);
  }
}

// This is a public function, declared in WinEHFuncInfo.h and is also
// referenced by WinEHNumbering in FunctionLoweringInfo.cpp.
void llvm::parseEHActions(const IntrinsicInst *II,
                          SmallVectorImpl<ActionHandler *> &Actions) {
  for (unsigned I = 0, E = II->getNumArgOperands(); I != E;) {
    uint64_t ActionKind =
        cast<ConstantInt>(II->getArgOperand(I))->getZExtValue();
    if (ActionKind == /*catch=*/1) {
      auto *Selector = cast<Constant>(II->getArgOperand(I + 1));
      ConstantInt *EHObjIndex = cast<ConstantInt>(II->getArgOperand(I + 2));
      int64_t EHObjIndexVal = EHObjIndex->getSExtValue();
      Constant *Handler = cast<Constant>(II->getArgOperand(I + 3));
      I += 4;
      auto *CH = new CatchHandler(/*BB=*/nullptr, Selector, /*NextBB=*/nullptr);
      CH->setHandlerBlockOrFunc(Handler);
      CH->setExceptionVarIndex(EHObjIndexVal);
      Actions.push_back(CH);
    } else if (ActionKind == 0) {
      Constant *Handler = cast<Constant>(II->getArgOperand(I + 1));
      I += 2;
      auto *CH = new CleanupHandler(/*BB=*/nullptr);
      CH->setHandlerBlockOrFunc(Handler);
      Actions.push_back(CH);
    } else {
      llvm_unreachable("Expected either a catch or cleanup handler!");
    }
  }
  std::reverse(Actions.begin(), Actions.end());
}

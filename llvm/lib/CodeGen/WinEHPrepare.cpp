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
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/LibCallSemantics.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
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

class WinEHPrepare : public FunctionPass {
  std::unique_ptr<FunctionPass> DwarfPrepare;

  enum HandlerType { Catch, Cleanup };

public:
  static char ID; // Pass identification, replacement for typeid.
  WinEHPrepare(const TargetMachine *TM = nullptr)
      : FunctionPass(ID), DwarfPrepare(createDwarfEHPass(TM)) {}

  bool runOnFunction(Function &Fn) override;

  bool doFinalization(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  const char *getPassName() const override {
    return "Windows exception handling preparation";
  }

private:
  bool prepareCPPEHHandlers(Function &F,
                            SmallVectorImpl<LandingPadInst *> &LPads);
  bool outlineHandler(HandlerType CatchOrCleanup, Function *SrcFn,
                      Constant *SelectorType, LandingPadInst *LPad,
                      FrameVarInfoMap &VarInfo);
};

class WinEHFrameVariableMaterializer : public ValueMaterializer {
public:
  WinEHFrameVariableMaterializer(Function *OutlinedFn,
                                 FrameVarInfoMap &FrameVarInfo);
  ~WinEHFrameVariableMaterializer() {}

  virtual Value *materializeValueFor(Value *V) override;

private:
  FrameVarInfoMap &FrameVarInfo;
  IRBuilder<> Builder;
};

class WinEHCloningDirectorBase : public CloningDirector {
public:
  WinEHCloningDirectorBase(LandingPadInst *LPI, Function *HandlerFn,
                           FrameVarInfoMap &VarInfo)
      : LPI(LPI), Materializer(HandlerFn, VarInfo),
        SelectorIDType(Type::getInt32Ty(LPI->getContext())),
        Int8PtrType(Type::getInt8PtrTy(LPI->getContext())),
        ExtractedEHPtr(nullptr), ExtractedSelector(nullptr),
        EHPtrStoreAddr(nullptr), SelectorStoreAddr(nullptr) {}

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
  virtual CloningAction handleResume(ValueToValueMapTy &VMap,
                                     const ResumeInst *Resume,
                                     BasicBlock *NewBB) = 0;

  ValueMaterializer *getValueMaterializer() override { return &Materializer; }

protected:
  LandingPadInst *LPI;
  WinEHFrameVariableMaterializer Materializer;
  Type *SelectorIDType;
  Type *Int8PtrType;

  const Value *ExtractedEHPtr;
  const Value *ExtractedSelector;
  const Value *EHPtrStoreAddr;
  const Value *SelectorStoreAddr;
};

class WinEHCatchDirector : public WinEHCloningDirectorBase {
public:
  WinEHCatchDirector(LandingPadInst *LPI, Function *CatchFn, Value *Selector,
                     FrameVarInfoMap &VarInfo)
      : WinEHCloningDirectorBase(LPI, CatchFn, VarInfo),
        CurrentSelector(Selector->stripPointerCasts()) {}

  CloningAction handleBeginCatch(ValueToValueMapTy &VMap,
                                 const Instruction *Inst,
                                 BasicBlock *NewBB) override;
  CloningAction handleEndCatch(ValueToValueMapTy &VMap, const Instruction *Inst,
                               BasicBlock *NewBB) override;
  CloningAction handleTypeIdFor(ValueToValueMapTy &VMap,
                                const Instruction *Inst,
                                BasicBlock *NewBB) override;
  CloningAction handleResume(ValueToValueMapTy &VMap, const ResumeInst *Resume,
                             BasicBlock *NewBB) override;

private:
  Value *CurrentSelector;
};

class WinEHCleanupDirector : public WinEHCloningDirectorBase {
public:
  WinEHCleanupDirector(LandingPadInst *LPI, Function *CleanupFn,
                       FrameVarInfoMap &VarInfo)
      : WinEHCloningDirectorBase(LPI, CleanupFn, VarInfo) {}

  CloningAction handleBeginCatch(ValueToValueMapTy &VMap,
                                 const Instruction *Inst,
                                 BasicBlock *NewBB) override;
  CloningAction handleEndCatch(ValueToValueMapTy &VMap, const Instruction *Inst,
                               BasicBlock *NewBB) override;
  CloningAction handleTypeIdFor(ValueToValueMapTy &VMap,
                                const Instruction *Inst,
                                BasicBlock *NewBB) override;
  CloningAction handleResume(ValueToValueMapTy &VMap, const ResumeInst *Resume,
                             BasicBlock *NewBB) override;
};

} // end anonymous namespace

char WinEHPrepare::ID = 0;
INITIALIZE_TM_PASS_BEGIN(WinEHPrepare, "winehprepare",
                         "Prepare Windows exceptions", false, false)
INITIALIZE_PASS_DEPENDENCY(DwarfEHPrepare)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_TM_PASS_END(WinEHPrepare, "winehprepare",
                       "Prepare Windows exceptions", false, false)

FunctionPass *llvm::createWinEHPass(const TargetMachine *TM) {
  return new WinEHPrepare(TM);
}

static bool isMSVCPersonality(EHPersonality Pers) {
  return Pers == EHPersonality::MSVC_Win64SEH ||
         Pers == EHPersonality::MSVC_CXX;
}

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
  EHPersonality Pers = classifyEHPersonality(LPads.back()->getPersonalityFn());

  // Delegate through to the DWARF pass if this is unrecognized.
  if (!isMSVCPersonality(Pers)) {
    if (!DwarfPrepare->getResolver()) {
      // Build an AnalysisResolver with the analyses needed by DwarfEHPrepare.
      // It will take ownership of the AnalysisResolver.
      assert(getResolver());
      auto *AR = new AnalysisResolver(getResolver()->getPMDataManager());
      AR->addAnalysisImplsPair(
          &TargetTransformInfoWrapperPass::ID,
          getResolver()->findImplPass(&TargetTransformInfoWrapperPass::ID));
      AR->addAnalysisImplsPair(
          &DominatorTreeWrapperPass::ID,
          getResolver()->findImplPass(&DominatorTreeWrapperPass::ID));
      DwarfPrepare->setResolver(AR);
    }

    return DwarfPrepare->runOnFunction(Fn);
  }

  // FIXME: This only returns true if the C++ EH handlers were outlined.
  //        When that code is complete, it should always return whatever
  //        prepareCPPEHHandlers returns.
  if (Pers == EHPersonality::MSVC_CXX && prepareCPPEHHandlers(Fn, LPads))
    return true;

  // FIXME: SEH Cleanups are unimplemented. Replace them with unreachable.
  if (Resumes.empty())
    return false;

  for (ResumeInst *Resume : Resumes) {
    IRBuilder<>(Resume).CreateUnreachable();
    Resume->eraseFromParent();
  }

  return true;
}

bool WinEHPrepare::doFinalization(Module &M) {
  return DwarfPrepare->doFinalization(M);
}

void WinEHPrepare::getAnalysisUsage(AnalysisUsage &AU) const {
  DwarfPrepare->getAnalysisUsage(AU);
}

bool WinEHPrepare::prepareCPPEHHandlers(
    Function &F, SmallVectorImpl<LandingPadInst *> &LPads) {
  // These containers are used to re-map frame variables that are used in
  // outlined catch and cleanup handlers.  They will be populated as the
  // handlers are outlined.
  FrameVarInfoMap FrameVarInfo;

  bool HandlersOutlined = false;

  for (LandingPadInst *LPad : LPads) {
    // Look for evidence that this landingpad has already been processed.
    bool LPadHasActionList = false;
    BasicBlock *LPadBB = LPad->getParent();
    for (Instruction &Inst : LPadBB->getInstList()) {
      // FIXME: Make this an intrinsic.
      if (auto *Call = dyn_cast<CallInst>(&Inst))
        if (Call->getCalledFunction()->getName() == "llvm.eh.actions") {
          LPadHasActionList = true;
          break;
        }
    }

    // If we've already outlined the handlers for this landingpad,
    // there's nothing more to do here.
    if (LPadHasActionList)
      continue;

    for (unsigned Idx = 0, NumClauses = LPad->getNumClauses(); Idx < NumClauses;
         ++Idx) {
      if (LPad->isCatch(Idx)) {
        // Create a new instance of the handler data structure in the
        // HandlerData vector.
        bool Outlined = outlineHandler(Catch, &F, LPad->getClause(Idx), LPad,
                                       FrameVarInfo);
        if (Outlined) {
          HandlersOutlined = true;
        }
      } // End if (isCatch)
    }   // End for each clause

    // FIXME: This only handles the simple case where there is a 1:1
    //        correspondence between landing pad and cleanup blocks.
    //        It does not handle cases where there are catch blocks between
    //        cleanup blocks or the case where a cleanup block is shared by
    //        multiple landing pads.  Those cases will be supported later
    //        when landing pad block analysis is added.
    if (LPad->isCleanup()) {
      bool Outlined =
          outlineHandler(Cleanup, &F, nullptr, LPad, FrameVarInfo);
      if (Outlined) {
        HandlersOutlined = true;
      }
    }
  } // End for each landingpad

  // If nothing got outlined, there is no more processing to be done.
  if (!HandlersOutlined)
    return false;

  // FIXME: We will replace the landingpad bodies with llvm.eh.actions
  //        calls and indirect branches here and then delete blocks
  //        which are no longer reachable.  That will get rid of the
  //        handlers that we have outlined.  There is code below
  //        that looks for allocas with no uses in the parent function.
  //        That will only happen after the pruning is implemented.

  Module *M = F.getParent();
  LLVMContext &Context = M->getContext();
  BasicBlock *Entry = &F.getEntryBlock();
  IRBuilder<> Builder(F.getParent()->getContext());
  Builder.SetInsertPoint(Entry->getFirstInsertionPt());

  Function *FrameEscapeFn =
      Intrinsic::getDeclaration(M, Intrinsic::frameescape);
  Function *RecoverFrameFn =
      Intrinsic::getDeclaration(M, Intrinsic::framerecover);
  Type *Int8PtrType = Type::getInt8PtrTy(Context);
  Type *Int32Type = Type::getInt32Ty(Context);

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
        ParentAlloca = DemoteRegToStack(*ParentInst, true, ParentInst);
      }
    }

    // If the parent alloca is no longer used and only one of the handlers used
    // it, erase the parent and leave the copy in the outlined handler.
    if (ParentAlloca->getNumUses() == 0 && Allocas.size() == 1) {
      ParentAlloca->eraseFromParent();
      continue;
    }

    // Add this alloca to the list of things to escape.
    AllocasToEscape.push_back(ParentAlloca);

    // Next replace all outlined allocas that are mapped to it.
    for (AllocaInst *TempAlloca : Allocas) {
      Function *HandlerFn = TempAlloca->getParent()->getParent();
      // FIXME: Sink this GEP into the blocks where it is used.
      Builder.SetInsertPoint(TempAlloca);
      Builder.SetCurrentDebugLocation(TempAlloca->getDebugLoc());
      Value *RecoverArgs[] = {
          Builder.CreateBitCast(&F, Int8PtrType, ""),
          &(HandlerFn->getArgumentList().back()),
          llvm::ConstantInt::get(Int32Type, AllocasToEscape.size() - 1)};
      Value *RecoveredAlloca =
          Builder.CreateCall(RecoverFrameFn, RecoverArgs);
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
  }   // End for each FrameVarInfo entry.

  // Insert 'call void (...)* @llvm.frameescape(...)' at the end of the entry
  // block.
  Builder.SetInsertPoint(&F.getEntryBlock().back());
  Builder.CreateCall(FrameEscapeFn, AllocasToEscape);

  return HandlersOutlined;
}

bool WinEHPrepare::outlineHandler(HandlerType CatchOrCleanup, Function *SrcFn,
                                  Constant *SelectorType, LandingPadInst *LPad,
                                  FrameVarInfoMap &VarInfo) {
  Module *M = SrcFn->getParent();
  LLVMContext &Context = M->getContext();

  // Create a new function to receive the handler contents.
  Type *Int8PtrType = Type::getInt8PtrTy(Context);
  std::vector<Type *> ArgTys;
  ArgTys.push_back(Int8PtrType);
  ArgTys.push_back(Int8PtrType);
  Function *Handler;
  if (CatchOrCleanup == Catch) {
    FunctionType *FnType = FunctionType::get(Int8PtrType, ArgTys, false);
    Handler = Function::Create(FnType, GlobalVariable::InternalLinkage,
                               SrcFn->getName() + ".catch", M);
  } else {
    FunctionType *FnType =
        FunctionType::get(Type::getVoidTy(Context), ArgTys, false);
    Handler = Function::Create(FnType, GlobalVariable::InternalLinkage,
                               SrcFn->getName() + ".cleanup", M);
  }

  // Generate a standard prolog to setup the frame recovery structure.
  IRBuilder<> Builder(Context);
  BasicBlock *Entry = BasicBlock::Create(Context, "entry");
  Handler->getBasicBlockList().push_front(Entry);
  Builder.SetInsertPoint(Entry);
  Builder.SetCurrentDebugLocation(LPad->getDebugLoc());

  std::unique_ptr<WinEHCloningDirectorBase> Director;

  if (CatchOrCleanup == Catch) {
    Director.reset(
        new WinEHCatchDirector(LPad, Handler, SelectorType, VarInfo));
  } else {
    Director.reset(new WinEHCleanupDirector(LPad, Handler, VarInfo));
  }

  ValueToValueMapTy VMap;

  // FIXME: Map other values referenced in the filter handler.

  SmallVector<ReturnInst *, 8> Returns;
  ClonedCodeInfo InlinedFunctionInfo;

  BasicBlock::iterator II = LPad;

  CloneAndPruneIntoFromInst(
      Handler, SrcFn, ++II, VMap,
      /*ModuleLevelChanges=*/false, Returns, "", &InlinedFunctionInfo,
      &SrcFn->getParent()->getDataLayout(), Director.get());

  // Move all the instructions in the first cloned block into our entry block.
  BasicBlock *FirstClonedBB = std::next(Function::iterator(Entry));
  Entry->getInstList().splice(Entry->end(), FirstClonedBB->getInstList());
  FirstClonedBB->eraseFromParent();

  return true;
}

CloningDirector::CloningAction WinEHCloningDirectorBase::handleInstruction(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  // Intercept instructions which extract values from the landing pad aggregate.
  if (auto *Extract = dyn_cast<ExtractValueInst>(Inst)) {
    if (Extract->getAggregateOperand() == LPI) {
      assert(Extract->getNumIndices() == 1 &&
             "Unexpected operation: extracting both landing pad values");
      assert((*(Extract->idx_begin()) == 0 || *(Extract->idx_begin()) == 1) &&
             "Unexpected operation: extracting an unknown landing pad element");

      if (*(Extract->idx_begin()) == 0) {
        // Element 0 doesn't directly corresponds to anything in the WinEH
        // scheme.
        // It will be stored to a memory location, then later loaded and finally
        // the loaded value will be used as the argument to an
        // llvm.eh.begincatch
        // call.  We're tracking it here so that we can skip the store and load.
        ExtractedEHPtr = Inst;
      } else {
        // Element 1 corresponds to the filter selector.  We'll map it to 1 for
        // matching purposes, but it will also probably be stored to memory and
        // reloaded, so we need to track the instuction so that we can map the
        // loaded value too.
        VMap[Inst] = ConstantInt::get(SelectorIDType, 1);
        ExtractedSelector = Inst;
      }

      // Tell the caller not to clone this instruction.
      return CloningDirector::SkipInstruction;
    }
    // Other extract value instructions just get cloned.
    return CloningDirector::CloneInstruction;
  }

  if (auto *Store = dyn_cast<StoreInst>(Inst)) {
    // Look for and suppress stores of the extracted landingpad values.
    const Value *StoredValue = Store->getValueOperand();
    if (StoredValue == ExtractedEHPtr) {
      EHPtrStoreAddr = Store->getPointerOperand();
      return CloningDirector::SkipInstruction;
    }
    if (StoredValue == ExtractedSelector) {
      SelectorStoreAddr = Store->getPointerOperand();
      return CloningDirector::SkipInstruction;
    }

    // Any other store just gets cloned.
    return CloningDirector::CloneInstruction;
  }

  if (auto *Load = dyn_cast<LoadInst>(Inst)) {
    // Look for loads of (previously suppressed) landingpad values.
    // The EHPtr load can be ignored (it should only be used as
    // an argument to llvm.eh.begincatch), but the selector value
    // needs to be mapped to a constant value of 1 to be used to
    // simplify the branching to always flow to the current handler.
    const Value *LoadAddr = Load->getPointerOperand();
    if (LoadAddr == EHPtrStoreAddr) {
      VMap[Inst] = UndefValue::get(Int8PtrType);
      return CloningDirector::SkipInstruction;
    }
    if (LoadAddr == SelectorStoreAddr) {
      VMap[Inst] = ConstantInt::get(SelectorIDType, 1);
      return CloningDirector::SkipInstruction;
    }

    // Any other loads just get cloned.
    return CloningDirector::CloneInstruction;
  }

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
  // landingpad
  // block that is part of the catch handlers exception mechanism, or at the
  // end of the catch block.  If it occurs in a landing pad, we must skip it
  // and continue so that the landing pad gets cloned.
  // FIXME: This case isn't fully supported yet and shouldn't turn up in any
  //        of the test cases until it is.
  if (IntrinCall->getParent()->isLandingPad())
    return CloningDirector::SkipInstruction;

  // If an end catch occurs anywhere else the next instruction should be an
  // unconditional branch instruction that we want to replace with a return
  // to the the address of the branch target.
  const BasicBlock *EndCatchBB = IntrinCall->getParent();
  const TerminatorInst *Terminator = EndCatchBB->getTerminator();
  const BranchInst *Branch = dyn_cast<BranchInst>(Terminator);
  assert(Branch && Branch->isUnconditional());
  assert(std::next(BasicBlock::const_iterator(IntrinCall)) ==
         BasicBlock::const_iterator(Branch));

  ReturnInst::Create(NewBB->getContext(),
                     BlockAddress::get(Branch->getSuccessor(0)), NewBB);

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
WinEHCatchDirector::handleResume(ValueToValueMapTy &VMap,
                                 const ResumeInst *Resume, BasicBlock *NewBB) {
  // Resume instructions shouldn't be reachable from catch handlers.
  // We still need to handle it, but it will be pruned.
  BasicBlock::InstListType &InstList = NewBB->getInstList();
  InstList.push_back(new UnreachableInst(NewBB->getContext()));
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
  // Catch blocks within cleanup handlers will always be unreachable.
  // We'll insert an unreachable instruction now, but it will be pruned
  // before the cloning process is complete.
  BasicBlock::InstListType &InstList = NewBB->getInstList();
  InstList.push_back(new UnreachableInst(NewBB->getContext()));
  return CloningDirector::StopCloningBB;
}

CloningDirector::CloningAction WinEHCleanupDirector::handleTypeIdFor(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  // This causes a replacement that will collapse the landing pad CFG
  // to just the cleanup code.
  VMap[Inst] = ConstantInt::get(SelectorIDType, 0);
  // Tell the caller not to clone this instruction.
  return CloningDirector::SkipInstruction;
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
  Builder.SetInsertPoint(&OutlinedFn->getEntryBlock());
  // FIXME: Do something with the FrameVarMapped so that it is shared across the
  // function.
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

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
#include "llvm/Analysis/LibCallSemantics.h"
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
class WinEHPrepare : public FunctionPass {
  std::unique_ptr<FunctionPass> DwarfPrepare;

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
  bool outlineCatchHandler(Function *SrcFn, Constant *SelectorType,
                           LandingPadInst *LPad, StructType *EHDataStructTy);
};

class WinEHCatchDirector : public CloningDirector {
public:
  WinEHCatchDirector(LandingPadInst *LPI, Value *Selector, Value *EHObj)
      : LPI(LPI), CurrentSelector(Selector->stripPointerCasts()), EHObj(EHObj),
        SelectorIDType(Type::getInt32Ty(LPI->getContext())),
        Int8PtrType(Type::getInt8PtrTy(LPI->getContext())) {}
  virtual ~WinEHCatchDirector() {}

  CloningAction handleInstruction(ValueToValueMapTy &VMap,
                                  const Instruction *Inst,
                                  BasicBlock *NewBB) override;

private:
  LandingPadInst *LPI;
  Value *CurrentSelector;
  Value *EHObj;
  Type *SelectorIDType;
  Type *Int8PtrType;

  const Value *ExtractedEHPtr;
  const Value *ExtractedSelector;
  const Value *EHPtrStoreAddr;
  const Value *SelectorStoreAddr;
};
} // end anonymous namespace

char WinEHPrepare::ID = 0;
INITIALIZE_TM_PASS(WinEHPrepare, "winehprepare", "Prepare Windows exceptions",
                   false, false)

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
  if (!isMSVCPersonality(Pers))
    return DwarfPrepare->runOnFunction(Fn);

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
  // FIXME: Find all frame variable references in the handlers
  //        to populate the structure elements.
  SmallVector<Type *, 2> AllocStructTys;
  AllocStructTys.push_back(Type::getInt32Ty(F.getContext()));   // EH state
  AllocStructTys.push_back(Type::getInt8PtrTy(F.getContext())); // EH object
  StructType *EHDataStructTy =
      StructType::create(F.getContext(), AllocStructTys, 
                         "struct." + F.getName().str() + ".ehdata");
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
      if (LPad->isCatch(Idx))
        HandlersOutlined =
            outlineCatchHandler(&F, LPad->getClause(Idx), LPad, EHDataStructTy);
    } // End for each clause
  }   // End for each landingpad

  return HandlersOutlined;
}

bool WinEHPrepare::outlineCatchHandler(Function *SrcFn, Constant *SelectorType,
                                       LandingPadInst *LPad,
                                       StructType *EHDataStructTy) {
  Module *M = SrcFn->getParent();
  LLVMContext &Context = M->getContext();

  // Create a new function to receive the handler contents.
  Type *Int8PtrType = Type::getInt8PtrTy(Context);
  std::vector<Type *> ArgTys;
  ArgTys.push_back(Int8PtrType);
  ArgTys.push_back(Int8PtrType);
  FunctionType *FnType = FunctionType::get(Int8PtrType, ArgTys, false);
  Function *CatchHandler = Function::Create(
      FnType, GlobalVariable::ExternalLinkage, SrcFn->getName() + ".catch", M);

  // Generate a standard prolog to setup the frame recovery structure.
  IRBuilder<> Builder(Context);
  BasicBlock *Entry = BasicBlock::Create(Context, "catch.entry");
  CatchHandler->getBasicBlockList().push_front(Entry);
  Builder.SetInsertPoint(Entry);
  Builder.SetCurrentDebugLocation(LPad->getDebugLoc());

  // The outlined handler will be called with the parent's frame pointer as
  // its second argument. To enable the handler to access variables from
  // the parent frame, we use that pointer to get locate a special block
  // of memory that was allocated using llvm.eh.allocateframe for this
  // purpose.  During the outlining process we will determine which frame
  // variables are used in handlers and create a structure that maps these
  // variables into the frame allocation block.
  //
  // The frame allocation block also contains an exception state variable
  // used by the runtime and a pointer to the exception object pointer
  // which will be filled in by the runtime for use in the handler.
  Function *RecoverFrameFn =
      Intrinsic::getDeclaration(M, Intrinsic::framerecover);
  Value *RecoverArgs[] = {Builder.CreateBitCast(SrcFn, Int8PtrType, ""),
                          &(CatchHandler->getArgumentList().back())};
  CallInst *EHAlloc =
      Builder.CreateCall(RecoverFrameFn, RecoverArgs, "eh.alloc");
  Value *EHData =
      Builder.CreateBitCast(EHAlloc, EHDataStructTy->getPointerTo(), "ehdata");
  Value *EHObjPtr =
      Builder.CreateConstInBoundsGEP2_32(EHData, 0, 1, "eh.obj.ptr");

  // This will give us a raw pointer to the exception object, which
  // corresponds to the formal parameter of the catch statement.  If the
  // handler uses this object, we will generate code during the outlining
  // process to cast the pointer to the appropriate type and deference it
  // as necessary.  The un-outlined landing pad code represents the
  // exception object as the result of the llvm.eh.begincatch call.
  Value *EHObj = Builder.CreateLoad(EHObjPtr, false, "eh.obj");

  ValueToValueMapTy VMap;

  // FIXME: Map other values referenced in the filter handler.

  WinEHCatchDirector Director(LPad, SelectorType, EHObj);

  SmallVector<ReturnInst *, 8> Returns;
  ClonedCodeInfo InlinedFunctionInfo;

  BasicBlock::iterator II = LPad;

  CloneAndPruneIntoFromInst(CatchHandler, SrcFn, ++II, VMap,
                            /*ModuleLevelChanges=*/false, Returns, "",
                            &InlinedFunctionInfo,
                            SrcFn->getParent()->getDataLayout(), &Director);

  // Move all the instructions in the first cloned block into our entry block.
  BasicBlock *FirstClonedBB = std::next(Function::iterator(Entry));
  Entry->getInstList().splice(Entry->end(), FirstClonedBB->getInstList());
  FirstClonedBB->eraseFromParent();

  return true;
}

CloningDirector::CloningAction WinEHCatchDirector::handleInstruction(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  // Intercept instructions which extract values from the landing pad aggregate.
  if (auto *Extract = dyn_cast<ExtractValueInst>(Inst)) {
    if (Extract->getAggregateOperand() == LPI) {
      assert(Extract->getNumIndices() == 1 &&
             "Unexpected operation: extracting both landing pad values");
      assert((*(Extract->idx_begin()) == 0 || *(Extract->idx_begin()) == 1) &&
             "Unexpected operation: extracting an unknown landing pad element");

      if (*(Extract->idx_begin()) == 0) {
        // Element 0 doesn't directly corresponds to anything in the WinEH scheme.
        // It will be stored to a memory location, then later loaded and finally
        // the loaded value will be used as the argument to an llvm.eh.begincatch
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

  if (match(Inst, m_Intrinsic<Intrinsic::eh_begincatch>())) {
    // The argument to the call is some form of the first element of the
    // landingpad aggregate value, but that doesn't matter.  It isn't used
    // here.
    // The return value of this instruction, however, is used to access the
    // EH object pointer.  We have generated an instruction to get that value
    // from the EH alloc block, so we can just map to that here.
    VMap[Inst] = EHObj;
    return CloningDirector::SkipInstruction;
  }
  if (match(Inst, m_Intrinsic<Intrinsic::eh_endcatch>())) {
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
  if (match(Inst, m_Intrinsic<Intrinsic::eh_typeid_for>())) {
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

  // Continue with the default cloning behavior.
  return CloningDirector::CloneInstruction;
}

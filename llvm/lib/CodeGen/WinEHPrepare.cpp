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
// backend wants for functions using a personality function from a runtime
// provided by MSVC. Functions with other personality functions are left alone
// and may be prepared by other passes. In particular, all supported MSVC
// personality functions require cleanup code to be outlined, and the C++
// personality requires catch handler code to be outlined.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LibCallSemantics.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <memory>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "winehprepare"

static cl::opt<bool> DisableDemotion(
    "disable-demotion", cl::Hidden,
    cl::desc(
        "Clone multicolor basic blocks but do not demote cross funclet values"),
    cl::init(false));

static cl::opt<bool> DisableCleanups(
    "disable-cleanups", cl::Hidden,
    cl::desc("Do not remove implausible terminators or other similar cleanups"),
    cl::init(false));

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
      : FunctionPass(ID) {
    if (TM)
      TheTriple = TM->getTargetTriple();
  }

  bool runOnFunction(Function &Fn) override;

  bool doFinalization(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  const char *getPassName() const override {
    return "Windows exception handling preparation";
  }

private:
  bool prepareExceptionHandlers(Function &F,
                                SmallVectorImpl<LandingPadInst *> &LPads);
  void identifyEHBlocks(Function &F, SmallVectorImpl<LandingPadInst *> &LPads);
  void promoteLandingPadValues(LandingPadInst *LPad);
  void demoteValuesLiveAcrossHandlers(Function &F,
                                      SmallVectorImpl<LandingPadInst *> &LPads);
  void findSEHEHReturnPoints(Function &F,
                             SetVector<BasicBlock *> &EHReturnBlocks);
  void findCXXEHReturnPoints(Function &F,
                             SetVector<BasicBlock *> &EHReturnBlocks);
  void getPossibleReturnTargets(Function *ParentF, Function *HandlerF,
                                SetVector<BasicBlock*> &Targets);
  void completeNestedLandingPad(Function *ParentFn,
                                LandingPadInst *OutlinedLPad,
                                const LandingPadInst *OriginalLPad,
                                FrameVarInfoMap &VarInfo);
  Function *createHandlerFunc(Function *ParentFn, Type *RetTy,
                              const Twine &Name, Module *M, Value *&ParentFP);
  bool outlineHandler(ActionHandler *Action, Function *SrcFn,
                      LandingPadInst *LPad, BasicBlock *StartBB,
                      FrameVarInfoMap &VarInfo);
  void addStubInvokeToHandlerIfNeeded(Function *Handler);

  void mapLandingPadBlocks(LandingPadInst *LPad, LandingPadActions &Actions);
  CatchHandler *findCatchHandler(BasicBlock *BB, BasicBlock *&NextBB,
                                 VisitedBlockSet &VisitedBlocks);
  void findCleanupHandlers(LandingPadActions &Actions, BasicBlock *StartBB,
                           BasicBlock *EndBB);

  void processSEHCatchHandler(CatchHandler *Handler, BasicBlock *StartBB);
  void insertPHIStores(PHINode *OriginalPHI, AllocaInst *SpillSlot);
  void
  insertPHIStore(BasicBlock *PredBlock, Value *PredVal, AllocaInst *SpillSlot,
                 SmallVectorImpl<std::pair<BasicBlock *, Value *>> &Worklist);
  AllocaInst *insertPHILoads(PHINode *PN, Function &F);
  void replaceUseWithLoad(Value *V, Use &U, AllocaInst *&SpillSlot,
                          DenseMap<BasicBlock *, Value *> &Loads, Function &F);
  void demoteNonlocalUses(Value *V, std::set<BasicBlock *> &ColorsForBB,
                          Function &F);
  bool prepareExplicitEH(Function &F,
                         SmallVectorImpl<BasicBlock *> &EntryBlocks);
  void replaceTerminatePadWithCleanup(Function &F);
  void colorFunclets(Function &F, SmallVectorImpl<BasicBlock *> &EntryBlocks);
  void demotePHIsOnFunclets(Function &F);
  void demoteUsesBetweenFunclets(Function &F);
  void demoteArgumentUses(Function &F);
  void cloneCommonBlocks(Function &F,
                         SmallVectorImpl<BasicBlock *> &EntryBlocks);
  void removeImplausibleTerminators(Function &F);
  void cleanupPreparedFunclets(Function &F);
  void verifyPreparedFunclets(Function &F);

  Triple TheTriple;

  // All fields are reset by runOnFunction.
  DominatorTree *DT = nullptr;
  const TargetLibraryInfo *LibInfo = nullptr;
  EHPersonality Personality = EHPersonality::Unknown;
  CatchHandlerMapTy CatchHandlerMap;
  CleanupHandlerMapTy CleanupHandlerMap;
  DenseMap<const LandingPadInst *, LandingPadMap> LPadMaps;
  SmallPtrSet<BasicBlock *, 4> NormalBlocks;
  SmallPtrSet<BasicBlock *, 4> EHBlocks;
  SetVector<BasicBlock *> EHReturnBlocks;

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

  // Map from outlined handler to call to parent local address. Only used for
  // 32-bit EH.
  DenseMap<Function *, Value *> HandlerToParentFP;

  AllocaInst *SEHExceptionCodeSlot = nullptr;

  std::map<BasicBlock *, std::set<BasicBlock *>> BlockColors;
  std::map<BasicBlock *, std::set<BasicBlock *>> FuncletBlocks;
  std::map<BasicBlock *, std::set<BasicBlock *>> FuncletChildren;
};

class WinEHFrameVariableMaterializer : public ValueMaterializer {
public:
  WinEHFrameVariableMaterializer(Function *OutlinedFn, Value *ParentFP,
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
  WinEHCloningDirectorBase(Function *HandlerFn, Value *ParentFP,
                           FrameVarInfoMap &VarInfo, LandingPadMap &LPadMap)
      : Materializer(HandlerFn, ParentFP, VarInfo),
        SelectorIDType(Type::getInt32Ty(HandlerFn->getContext())),
        Int8PtrType(Type::getInt8PtrTy(HandlerFn->getContext())),
        LPadMap(LPadMap), ParentFP(ParentFP) {}

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
  virtual CloningAction handleIndirectBr(ValueToValueMapTy &VMap,
                                         const IndirectBrInst *IBr,
                                         BasicBlock *NewBB) = 0;
  virtual CloningAction handleInvoke(ValueToValueMapTy &VMap,
                                     const InvokeInst *Invoke,
                                     BasicBlock *NewBB) = 0;
  virtual CloningAction handleResume(ValueToValueMapTy &VMap,
                                     const ResumeInst *Resume,
                                     BasicBlock *NewBB) = 0;
  virtual CloningAction handleCompare(ValueToValueMapTy &VMap,
                                      const CmpInst *Compare,
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

  /// The value representing the parent frame pointer.
  Value *ParentFP;
};

class WinEHCatchDirector : public WinEHCloningDirectorBase {
public:
  WinEHCatchDirector(
      Function *CatchFn, Value *ParentFP, Value *Selector,
      FrameVarInfoMap &VarInfo, LandingPadMap &LPadMap,
      DenseMap<LandingPadInst *, const LandingPadInst *> &NestedLPads,
      DominatorTree *DT, SmallPtrSetImpl<BasicBlock *> &EHBlocks)
      : WinEHCloningDirectorBase(CatchFn, ParentFP, VarInfo, LPadMap),
        CurrentSelector(Selector->stripPointerCasts()),
        ExceptionObjectVar(nullptr), NestedLPtoOriginalLP(NestedLPads),
        DT(DT), EHBlocks(EHBlocks) {}

  CloningAction handleBeginCatch(ValueToValueMapTy &VMap,
                                 const Instruction *Inst,
                                 BasicBlock *NewBB) override;
  CloningAction handleEndCatch(ValueToValueMapTy &VMap, const Instruction *Inst,
                               BasicBlock *NewBB) override;
  CloningAction handleTypeIdFor(ValueToValueMapTy &VMap,
                                const Instruction *Inst,
                                BasicBlock *NewBB) override;
  CloningAction handleIndirectBr(ValueToValueMapTy &VMap,
                                 const IndirectBrInst *IBr,
                                 BasicBlock *NewBB) override;
  CloningAction handleInvoke(ValueToValueMapTy &VMap, const InvokeInst *Invoke,
                             BasicBlock *NewBB) override;
  CloningAction handleResume(ValueToValueMapTy &VMap, const ResumeInst *Resume,
                             BasicBlock *NewBB) override;
  CloningAction handleCompare(ValueToValueMapTy &VMap, const CmpInst *Compare,
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
  DominatorTree *DT;
  SmallPtrSetImpl<BasicBlock *> &EHBlocks;
};

class WinEHCleanupDirector : public WinEHCloningDirectorBase {
public:
  WinEHCleanupDirector(Function *CleanupFn, Value *ParentFP,
                       FrameVarInfoMap &VarInfo, LandingPadMap &LPadMap)
      : WinEHCloningDirectorBase(CleanupFn, ParentFP, VarInfo,
                                 LPadMap) {}

  CloningAction handleBeginCatch(ValueToValueMapTy &VMap,
                                 const Instruction *Inst,
                                 BasicBlock *NewBB) override;
  CloningAction handleEndCatch(ValueToValueMapTy &VMap, const Instruction *Inst,
                               BasicBlock *NewBB) override;
  CloningAction handleTypeIdFor(ValueToValueMapTy &VMap,
                                const Instruction *Inst,
                                BasicBlock *NewBB) override;
  CloningAction handleIndirectBr(ValueToValueMapTy &VMap,
                                 const IndirectBrInst *IBr,
                                 BasicBlock *NewBB) override;
  CloningAction handleInvoke(ValueToValueMapTy &VMap, const InvokeInst *Invoke,
                             BasicBlock *NewBB) override;
  CloningAction handleResume(ValueToValueMapTy &VMap, const ResumeInst *Resume,
                             BasicBlock *NewBB) override;
  CloningAction handleCompare(ValueToValueMapTy &VMap, const CmpInst *Compare,
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

bool WinEHPrepare::runOnFunction(Function &Fn) {
  if (!Fn.hasPersonalityFn())
    return false;

  // No need to prepare outlined handlers.
  if (Fn.hasFnAttribute("wineh-parent"))
    return false;

  // Classify the personality to see what kind of preparation we need.
  Personality = classifyEHPersonality(Fn.getPersonalityFn());

  // Do nothing if this is not an MSVC personality.
  if (!isMSVCEHPersonality(Personality))
    return false;

  SmallVector<LandingPadInst *, 4> LPads;
  SmallVector<ResumeInst *, 4> Resumes;
  SmallVector<BasicBlock *, 4> EntryBlocks;
  bool ForExplicitEH = false;
  for (BasicBlock &BB : Fn) {
    Instruction *First = BB.getFirstNonPHI();
    if (auto *LP = dyn_cast<LandingPadInst>(First)) {
      LPads.push_back(LP);
    } else if (First->isEHPad()) {
      if (!ForExplicitEH)
        EntryBlocks.push_back(&Fn.getEntryBlock());
      if (!isa<CatchEndPadInst>(First) && !isa<CleanupEndPadInst>(First))
        EntryBlocks.push_back(&BB);
      ForExplicitEH = true;
    }
    if (auto *Resume = dyn_cast<ResumeInst>(BB.getTerminator()))
      Resumes.push_back(Resume);
  }

  if (ForExplicitEH)
    return prepareExplicitEH(Fn, EntryBlocks);

  // No need to prepare functions that lack landing pads.
  if (LPads.empty())
    return false;

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LibInfo = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

  // If there were any landing pads, prepareExceptionHandlers will make changes.
  prepareExceptionHandlers(Fn, LPads);
  return true;
}

bool WinEHPrepare::doFinalization(Module &M) { return false; }

void WinEHPrepare::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}

static bool isSelectorDispatch(BasicBlock *BB, BasicBlock *&CatchHandler,
                               Constant *&Selector, BasicBlock *&NextBB);

// Finds blocks reachable from the starting set Worklist. Does not follow unwind
// edges or blocks listed in StopPoints.
static void findReachableBlocks(SmallPtrSetImpl<BasicBlock *> &ReachableBBs,
                                SetVector<BasicBlock *> &Worklist,
                                const SetVector<BasicBlock *> *StopPoints) {
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();

    // Don't cross blocks that we should stop at.
    if (StopPoints && StopPoints->count(BB))
      continue;

    if (!ReachableBBs.insert(BB).second)
      continue; // Already visited.

    // Don't follow unwind edges of invokes.
    if (auto *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      Worklist.insert(II->getNormalDest());
      continue;
    }

    // Otherwise, follow all successors.
    Worklist.insert(succ_begin(BB), succ_end(BB));
  }
}

// Attempt to find an instruction where a block can be split before
// a call to llvm.eh.begincatch and its operands.  If the block
// begins with the begincatch call or one of its adjacent operands
// the block will not be split.
static Instruction *findBeginCatchSplitPoint(BasicBlock *BB,
                                             IntrinsicInst *II) {
  // If the begincatch call is already the first instruction in the block,
  // don't split.
  Instruction *FirstNonPHI = BB->getFirstNonPHI();
  if (II == FirstNonPHI)
    return nullptr;

  // If either operand is in the same basic block as the instruction and
  // isn't used by another instruction before the begincatch call, include it
  // in the split block.
  auto *Op0 = dyn_cast<Instruction>(II->getOperand(0));
  auto *Op1 = dyn_cast<Instruction>(II->getOperand(1));

  Instruction *I = II->getPrevNode();
  Instruction *LastI = II;

  while (I == Op0 || I == Op1) {
    // If the block begins with one of the operands and there are no other
    // instructions between the operand and the begincatch call, don't split.
    if (I == FirstNonPHI)
      return nullptr;

    LastI = I;
    I = I->getPrevNode();
  }

  // If there is at least one instruction in the block before the begincatch
  // call and its operands, split the block at either the begincatch or
  // its operand.
  return LastI;
}

/// Find all points where exceptional control rejoins normal control flow via
/// llvm.eh.endcatch. Add them to the normal bb reachability worklist.
void WinEHPrepare::findCXXEHReturnPoints(
    Function &F, SetVector<BasicBlock *> &EHReturnBlocks) {
  for (auto BBI = F.begin(), BBE = F.end(); BBI != BBE; ++BBI) {
    BasicBlock *BB = BBI;
    for (Instruction &I : *BB) {
      if (match(&I, m_Intrinsic<Intrinsic::eh_begincatch>())) {
        Instruction *SplitPt =
            findBeginCatchSplitPoint(BB, cast<IntrinsicInst>(&I));
        if (SplitPt) {
          // Split the block before the llvm.eh.begincatch call to allow
          // cleanup and catch code to be distinguished later.
          // Do not update BBI because we still need to process the
          // portion of the block that we are splitting off.
          SplitBlock(BB, SplitPt, DT);
          break;
        }
      }
      if (match(&I, m_Intrinsic<Intrinsic::eh_endcatch>())) {
        // Split the block after the call to llvm.eh.endcatch if there is
        // anything other than an unconditional branch, or if the successor
        // starts with a phi.
        auto *Br = dyn_cast<BranchInst>(I.getNextNode());
        if (!Br || !Br->isUnconditional() ||
            isa<PHINode>(Br->getSuccessor(0)->begin())) {
          DEBUG(dbgs() << "splitting block " << BB->getName()
                       << " with llvm.eh.endcatch\n");
          BBI = SplitBlock(BB, I.getNextNode(), DT);
        }
        // The next BB is normal control flow.
        EHReturnBlocks.insert(BB->getTerminator()->getSuccessor(0));
        break;
      }
    }
  }
}

static bool isCatchAllLandingPad(const BasicBlock *BB) {
  const LandingPadInst *LP = BB->getLandingPadInst();
  if (!LP)
    return false;
  unsigned N = LP->getNumClauses();
  return (N > 0 && LP->isCatch(N - 1) &&
          isa<ConstantPointerNull>(LP->getClause(N - 1)));
}

/// Find all points where exceptions control rejoins normal control flow via
/// selector dispatch.
void WinEHPrepare::findSEHEHReturnPoints(
    Function &F, SetVector<BasicBlock *> &EHReturnBlocks) {
  for (auto BBI = F.begin(), BBE = F.end(); BBI != BBE; ++BBI) {
    BasicBlock *BB = BBI;
    // If the landingpad is a catch-all, treat the whole lpad as if it is
    // reachable from normal control flow.
    // FIXME: This is imprecise. We need a better way of identifying where a
    // catch-all starts and cleanups stop. As far as LLVM is concerned, there
    // is no difference.
    if (isCatchAllLandingPad(BB)) {
      EHReturnBlocks.insert(BB);
      continue;
    }

    BasicBlock *CatchHandler;
    BasicBlock *NextBB;
    Constant *Selector;
    if (isSelectorDispatch(BB, CatchHandler, Selector, NextBB)) {
      // Split the edge if there are multiple predecessors. This creates a place
      // where we can insert EH recovery code.
      if (!CatchHandler->getSinglePredecessor()) {
        DEBUG(dbgs() << "splitting EH return edge from " << BB->getName()
                     << " to " << CatchHandler->getName() << '\n');
        BBI = CatchHandler = SplitCriticalEdge(
            BB, std::find(succ_begin(BB), succ_end(BB), CatchHandler));
      }
      EHReturnBlocks.insert(CatchHandler);
    }
  }
}

void WinEHPrepare::identifyEHBlocks(Function &F, 
                                    SmallVectorImpl<LandingPadInst *> &LPads) {
  DEBUG(dbgs() << "Demoting values live across exception handlers in function "
               << F.getName() << '\n');

  // Build a set of all non-exceptional blocks and exceptional blocks.
  // - Non-exceptional blocks are blocks reachable from the entry block while
  //   not following invoke unwind edges.
  // - Exceptional blocks are blocks reachable from landingpads. Analysis does
  //   not follow llvm.eh.endcatch blocks, which mark a transition from
  //   exceptional to normal control.

  if (Personality == EHPersonality::MSVC_CXX)
    findCXXEHReturnPoints(F, EHReturnBlocks);
  else
    findSEHEHReturnPoints(F, EHReturnBlocks);

  DEBUG({
    dbgs() << "identified the following blocks as EH return points:\n";
    for (BasicBlock *BB : EHReturnBlocks)
      dbgs() << "  " << BB->getName() << '\n';
  });

// Join points should not have phis at this point, unless they are a
// landingpad, in which case we will demote their phis later.
#ifndef NDEBUG
  for (BasicBlock *BB : EHReturnBlocks)
    assert((BB->isLandingPad() || !isa<PHINode>(BB->begin())) &&
           "non-lpad EH return block has phi");
#endif

  // Normal blocks are the blocks reachable from the entry block and all EH
  // return points.
  SetVector<BasicBlock *> Worklist;
  Worklist = EHReturnBlocks;
  Worklist.insert(&F.getEntryBlock());
  findReachableBlocks(NormalBlocks, Worklist, nullptr);
  DEBUG({
    dbgs() << "marked the following blocks as normal:\n";
    for (BasicBlock *BB : NormalBlocks)
      dbgs() << "  " << BB->getName() << '\n';
  });

  // Exceptional blocks are the blocks reachable from landingpads that don't
  // cross EH return points.
  Worklist.clear();
  for (auto *LPI : LPads)
    Worklist.insert(LPI->getParent());
  findReachableBlocks(EHBlocks, Worklist, &EHReturnBlocks);
  DEBUG({
    dbgs() << "marked the following blocks as exceptional:\n";
    for (BasicBlock *BB : EHBlocks)
      dbgs() << "  " << BB->getName() << '\n';
  });

}

/// Ensure that all values live into and out of exception handlers are stored
/// in memory.
/// FIXME: This falls down when values are defined in one handler and live into
/// another handler. For example, a cleanup defines a value used only by a
/// catch handler.
void WinEHPrepare::demoteValuesLiveAcrossHandlers(
    Function &F, SmallVectorImpl<LandingPadInst *> &LPads) {
  DEBUG(dbgs() << "Demoting values live across exception handlers in function "
               << F.getName() << '\n');

  // identifyEHBlocks() should have been called before this function.
  assert(!NormalBlocks.empty());

  // Try to avoid demoting EH pointer and selector values. They get in the way
  // of our pattern matching.
  SmallPtrSet<Instruction *, 10> EHVals;
  for (BasicBlock &BB : F) {
    LandingPadInst *LP = BB.getLandingPadInst();
    if (!LP)
      continue;
    EHVals.insert(LP);
    for (User *U : LP->users()) {
      auto *EI = dyn_cast<ExtractValueInst>(U);
      if (!EI)
        continue;
      EHVals.insert(EI);
      for (User *U2 : EI->users()) {
        if (auto *PN = dyn_cast<PHINode>(U2))
          EHVals.insert(PN);
      }
    }
  }

  SetVector<Argument *> ArgsToDemote;
  SetVector<Instruction *> InstrsToDemote;
  for (BasicBlock &BB : F) {
    bool IsNormalBB = NormalBlocks.count(&BB);
    bool IsEHBB = EHBlocks.count(&BB);
    if (!IsNormalBB && !IsEHBB)
      continue; // Blocks that are neither normal nor EH are unreachable.
    for (Instruction &I : BB) {
      for (Value *Op : I.operands()) {
        // Don't demote static allocas, constants, and labels.
        if (isa<Constant>(Op) || isa<BasicBlock>(Op) || isa<InlineAsm>(Op))
          continue;
        auto *AI = dyn_cast<AllocaInst>(Op);
        if (AI && AI->isStaticAlloca())
          continue;

        if (auto *Arg = dyn_cast<Argument>(Op)) {
          if (IsEHBB) {
            DEBUG(dbgs() << "Demoting argument " << *Arg
                         << " used by EH instr: " << I << "\n");
            ArgsToDemote.insert(Arg);
          }
          continue;
        }

        // Don't demote EH values.
        auto *OpI = cast<Instruction>(Op);
        if (EHVals.count(OpI))
          continue;

        BasicBlock *OpBB = OpI->getParent();
        // If a value is produced and consumed in the same BB, we don't need to
        // demote it.
        if (OpBB == &BB)
          continue;
        bool IsOpNormalBB = NormalBlocks.count(OpBB);
        bool IsOpEHBB = EHBlocks.count(OpBB);
        if (IsNormalBB != IsOpNormalBB || IsEHBB != IsOpEHBB) {
          DEBUG({
            dbgs() << "Demoting instruction live in-out from EH:\n";
            dbgs() << "Instr: " << *OpI << '\n';
            dbgs() << "User: " << I << '\n';
          });
          InstrsToDemote.insert(OpI);
        }
      }
    }
  }

  // Demote values live into and out of handlers.
  // FIXME: This demotion is inefficient. We should insert spills at the point
  // of definition, insert one reload in each handler that uses the value, and
  // insert reloads in the BB used to rejoin normal control flow.
  Instruction *AllocaInsertPt = F.getEntryBlock().getFirstInsertionPt();
  for (Instruction *I : InstrsToDemote)
    DemoteRegToStack(*I, false, AllocaInsertPt);

  // Demote arguments separately, and only for uses in EH blocks.
  for (Argument *Arg : ArgsToDemote) {
    auto *Slot = new AllocaInst(Arg->getType(), nullptr,
                                Arg->getName() + ".reg2mem", AllocaInsertPt);
    SmallVector<User *, 4> Users(Arg->user_begin(), Arg->user_end());
    for (User *U : Users) {
      auto *I = dyn_cast<Instruction>(U);
      if (I && EHBlocks.count(I->getParent())) {
        auto *Reload = new LoadInst(Slot, Arg->getName() + ".reload", false, I);
        U->replaceUsesOfWith(Arg, Reload);
      }
    }
    new StoreInst(Arg, Slot, AllocaInsertPt);
  }

  // Demote landingpad phis, as the landingpad will be removed from the machine
  // CFG.
  for (LandingPadInst *LPI : LPads) {
    BasicBlock *BB = LPI->getParent();
    while (auto *Phi = dyn_cast<PHINode>(BB->begin()))
      DemotePHIToStack(Phi, AllocaInsertPt);
  }

  DEBUG(dbgs() << "Demoted " << InstrsToDemote.size() << " instructions and "
               << ArgsToDemote.size() << " arguments for WinEHPrepare\n\n");
}

bool WinEHPrepare::prepareExceptionHandlers(
    Function &F, SmallVectorImpl<LandingPadInst *> &LPads) {
  // Don't run on functions that are already prepared.
  for (LandingPadInst *LPad : LPads) {
    BasicBlock *LPadBB = LPad->getParent();
    for (Instruction &Inst : *LPadBB)
      if (match(&Inst, m_Intrinsic<Intrinsic::eh_actions>()))
        return false;
  }

  identifyEHBlocks(F, LPads);
  demoteValuesLiveAcrossHandlers(F, LPads);

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

  if (isAsynchronousEHPersonality(Personality)) {
    // FIXME: Switch the ehptr type to i32 and then switch this.
    SEHExceptionCodeSlot =
        new AllocaInst(Int8PtrType, nullptr, "seh_exception_code",
                       F.getEntryBlock().getFirstInsertionPt());
  }

  // In order to handle the case where one outlined catch handler returns
  // to a block within another outlined catch handler that would otherwise
  // be unreachable, we need to outline the nested landing pad before we
  // outline the landing pad which encloses it.
  if (!isAsynchronousEHPersonality(Personality))
    std::sort(LPads.begin(), LPads.end(),
              [this](LandingPadInst *const &L, LandingPadInst *const &R) {
                return DT->properlyDominates(R->getParent(), L->getParent());
              });

  // This container stores the llvm.eh.recover and IndirectBr instructions
  // that make up the body of each landing pad after it has been outlined.
  // We need to defer the population of the target list for the indirectbr
  // until all landing pads have been outlined so that we can handle the
  // case of blocks in the target that are reached only from nested
  // landing pads.
  SmallVector<std::pair<CallInst*, IndirectBrInst *>, 4> LPadImpls;

  for (LandingPadInst *LPad : LPads) {
    // Look for evidence that this landingpad has already been processed.
    bool LPadHasActionList = false;
    BasicBlock *LPadBB = LPad->getParent();
    for (Instruction &Inst : *LPadBB) {
      if (match(&Inst, m_Intrinsic<Intrinsic::eh_actions>())) {
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

    // Split the block after the landingpad instruction so that it is just a
    // call to llvm.eh.actions followed by indirectbr.
    assert(!isa<PHINode>(LPadBB->begin()) && "lpad phi not removed");
    SplitBlock(LPadBB, LPad->getNextNode(), DT);
    // Erase the branch inserted by the split so we can insert indirectbr.
    LPadBB->getTerminator()->eraseFromParent();

    // Replace all extracted values with undef and ultimately replace the
    // landingpad with undef.
    SmallVector<Instruction *, 4> SEHCodeUses;
    SmallVector<Instruction *, 4> EHUndefs;
    for (User *U : LPad->users()) {
      auto *E = dyn_cast<ExtractValueInst>(U);
      if (!E)
        continue;
      assert(E->getNumIndices() == 1 &&
             "Unexpected operation: extracting both landing pad values");
      unsigned Idx = *E->idx_begin();
      assert((Idx == 0 || Idx == 1) && "unexpected index");
      if (Idx == 0 && isAsynchronousEHPersonality(Personality))
        SEHCodeUses.push_back(E);
      else
        EHUndefs.push_back(E);
    }
    for (Instruction *E : EHUndefs) {
      E->replaceAllUsesWith(UndefValue::get(E->getType()));
      E->eraseFromParent();
    }
    LPad->replaceAllUsesWith(UndefValue::get(LPad->getType()));

    // Rewrite uses of the exception pointer to loads of an alloca.
    while (!SEHCodeUses.empty()) {
      Instruction *E = SEHCodeUses.pop_back_val();
      SmallVector<Use *, 4> Uses;
      for (Use &U : E->uses())
        Uses.push_back(&U);
      for (Use *U : Uses) {
        auto *I = cast<Instruction>(U->getUser());
        if (isa<ResumeInst>(I))
          continue;
        if (auto *Phi = dyn_cast<PHINode>(I))
          SEHCodeUses.push_back(Phi);
        else
          U->set(new LoadInst(SEHExceptionCodeSlot, "sehcode", false, I));
      }
      E->replaceAllUsesWith(UndefValue::get(E->getType()));
      E->eraseFromParent();
    }

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
        CallInst::Create(ActionIntrin, ActionArgs, "recover", LPadBB);

    SetVector<BasicBlock *> ReturnTargets;
    for (ActionHandler *Action : Actions) {
      if (auto *CatchAction = dyn_cast<CatchHandler>(Action)) {
        const auto &CatchTargets = CatchAction->getReturnTargets();
        ReturnTargets.insert(CatchTargets.begin(), CatchTargets.end());
      }
    }
    IndirectBrInst *Branch =
        IndirectBrInst::Create(Recover, ReturnTargets.size(), LPadBB);
    for (BasicBlock *Target : ReturnTargets)
      Branch->addDestination(Target);

    if (!isAsynchronousEHPersonality(Personality)) {
      // C++ EH must repopulate the targets later to handle the case of
      // targets that are reached indirectly through nested landing pads.
      LPadImpls.push_back(std::make_pair(Recover, Branch));
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

  // Update the indirectbr instructions' target lists if necessary.
  SetVector<BasicBlock*> CheckedTargets;
  SmallVector<std::unique_ptr<ActionHandler>, 4> ActionList;
  for (auto &LPadImplPair : LPadImpls) {
    IntrinsicInst *Recover = cast<IntrinsicInst>(LPadImplPair.first);
    IndirectBrInst *Branch = LPadImplPair.second;

    // Get a list of handlers called by 
    parseEHActions(Recover, ActionList);

    // Add an indirect branch listing possible successors of the catch handlers.
    SetVector<BasicBlock *> ReturnTargets;
    for (const auto &Action : ActionList) {
      if (auto *CA = dyn_cast<CatchHandler>(Action.get())) {
        Function *Handler = cast<Function>(CA->getHandlerBlockOrFunc());
        getPossibleReturnTargets(&F, Handler, ReturnTargets);
      }
    }
    ActionList.clear();
    // Clear any targets we already knew about.
    for (unsigned int I = 0, E = Branch->getNumDestinations(); I < E; ++I) {
      BasicBlock *KnownTarget = Branch->getDestination(I);
      if (ReturnTargets.count(KnownTarget))
        ReturnTargets.remove(KnownTarget);
    }
    for (BasicBlock *Target : ReturnTargets) {
      Branch->addDestination(Target);
      // The target may be a block that we excepted to get pruned.
      // If it is, it may contain a call to llvm.eh.endcatch.
      if (CheckedTargets.insert(Target)) {
        // Earlier preparations guarantee that all calls to llvm.eh.endcatch
        // will be followed by an unconditional branch.
        auto *Br = dyn_cast<BranchInst>(Target->getTerminator());
        if (Br && Br->isUnconditional() &&
            Br != Target->getFirstNonPHIOrDbgOrLifetime()) {
          Instruction *Prev = Br->getPrevNode();
          if (match(cast<Value>(Prev), m_Intrinsic<Intrinsic::eh_endcatch>()))
            Prev->eraseFromParent();
        }
      }
    }
  }
  LPadImpls.clear();

  F.addFnAttr("wineh-parent", F.getName());

  // Delete any blocks that were only used by handlers that were outlined above.
  removeUnreachableBlocks(F);

  BasicBlock *Entry = &F.getEntryBlock();
  IRBuilder<> Builder(F.getParent()->getContext());
  Builder.SetInsertPoint(Entry->getFirstInsertionPt());

  Function *FrameEscapeFn =
      Intrinsic::getDeclaration(M, Intrinsic::localescape);
  Function *RecoverFrameFn =
      Intrinsic::getDeclaration(M, Intrinsic::localrecover);
  SmallVector<Value *, 8> AllocasToEscape;

  // Scan the entry block for an existing call to llvm.localescape. We need to
  // keep escaping those objects.
  for (Instruction &I : F.front()) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (II && II->getIntrinsicID() == Intrinsic::localescape) {
      auto Args = II->arg_operands();
      AllocasToEscape.append(Args.begin(), Args.end());
      II->eraseFromParent();
      break;
    }
  }

  // Finally, replace all of the temporary allocas for frame variables used in
  // the outlined handlers with calls to llvm.localrecover.
  for (auto &VarInfoEntry : FrameVarInfo) {
    Value *ParentVal = VarInfoEntry.first;
    TinyPtrVector<AllocaInst *> &Allocas = VarInfoEntry.second;
    AllocaInst *ParentAlloca = cast<AllocaInst>(ParentVal);

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
      llvm::Value *FP = HandlerToParentFP[HandlerFn];
      assert(FP);

      // FIXME: Sink this localrecover into the blocks where it is used.
      Builder.SetInsertPoint(TempAlloca);
      Builder.SetCurrentDebugLocation(TempAlloca->getDebugLoc());
      Value *RecoverArgs[] = {
          Builder.CreateBitCast(&F, Int8PtrType, ""), FP,
          llvm::ConstantInt::get(Int32Type, AllocasToEscape.size() - 1)};
      Instruction *RecoveredAlloca =
          Builder.CreateCall(RecoverFrameFn, RecoverArgs);

      // Add a pointer bitcast if the alloca wasn't an i8.
      if (RecoveredAlloca->getType() != TempAlloca->getType()) {
        RecoveredAlloca->setName(Twine(TempAlloca->getName()) + ".i8");
        RecoveredAlloca = cast<Instruction>(
            Builder.CreateBitCast(RecoveredAlloca, TempAlloca->getType()));
      }
      TempAlloca->replaceAllUsesWith(RecoveredAlloca);
      TempAlloca->removeFromParent();
      RecoveredAlloca->takeName(TempAlloca);
      delete TempAlloca;
    }
  } // End for each FrameVarInfo entry.

  // Insert 'call void (...)* @llvm.localescape(...)' at the end of the entry
  // block.
  Builder.SetInsertPoint(&F.getEntryBlock().back());
  Builder.CreateCall(FrameEscapeFn, AllocasToEscape);

  if (SEHExceptionCodeSlot) {
    if (isAllocaPromotable(SEHExceptionCodeSlot)) {
      SmallPtrSet<BasicBlock *, 4> UserBlocks;
      for (User *U : SEHExceptionCodeSlot->users()) {
        if (auto *Inst = dyn_cast<Instruction>(U))
          UserBlocks.insert(Inst->getParent());
      }
      PromoteMemToReg(SEHExceptionCodeSlot, *DT);
      // After the promotion, kill off dead instructions.
      for (BasicBlock *BB : UserBlocks)
        SimplifyInstructionsInBlock(BB, LibInfo);
    }
  }

  // Clean up the handler action maps we created for this function
  DeleteContainerSeconds(CatchHandlerMap);
  CatchHandlerMap.clear();
  DeleteContainerSeconds(CleanupHandlerMap);
  CleanupHandlerMap.clear();
  HandlerToParentFP.clear();
  DT = nullptr;
  LibInfo = nullptr;
  SEHExceptionCodeSlot = nullptr;
  EHBlocks.clear();
  NormalBlocks.clear();
  EHReturnBlocks.clear();

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

void WinEHPrepare::getPossibleReturnTargets(Function *ParentF,
                                            Function *HandlerF,
                                            SetVector<BasicBlock*> &Targets) {
  for (BasicBlock &BB : *HandlerF) {
    // If the handler contains landing pads, check for any
    // handlers that may return directly to a block in the
    // parent function.
    if (auto *LPI = BB.getLandingPadInst()) {
      IntrinsicInst *Recover = cast<IntrinsicInst>(LPI->getNextNode());
      SmallVector<std::unique_ptr<ActionHandler>, 4> ActionList;
      parseEHActions(Recover, ActionList);
      for (const auto &Action : ActionList) {
        if (auto *CH = dyn_cast<CatchHandler>(Action.get())) {
          Function *NestedF = cast<Function>(CH->getHandlerBlockOrFunc());
          getPossibleReturnTargets(ParentF, NestedF, Targets);
        }
      }
    }

    auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
    if (!Ret)
      continue;

    // Handler functions must always return a block address.
    BlockAddress *BA = cast<BlockAddress>(Ret->getReturnValue());

    // If this is the handler for a nested landing pad, the
    // return address may have been remapped to a block in the
    // parent handler.  We're not interested in those.
    if (BA->getFunction() != ParentF)
      continue;

    Targets.insert(BA->getBasicBlock());
  }
}

void WinEHPrepare::completeNestedLandingPad(Function *ParentFn,
                                            LandingPadInst *OutlinedLPad,
                                            const LandingPadInst *OriginalLPad,
                                            FrameVarInfoMap &FrameVarInfo) {
  // Get the nested block and erase the unreachable instruction that was
  // temporarily inserted as its terminator.
  LLVMContext &Context = ParentFn->getContext();
  BasicBlock *OutlinedBB = OutlinedLPad->getParent();
  // If the nested landing pad was outlined before the landing pad that enclosed
  // it, it will already be in outlined form.  In that case, we just need to see
  // if the returns and the enclosing branch instruction need to be updated.
  IndirectBrInst *Branch =
      dyn_cast<IndirectBrInst>(OutlinedBB->getTerminator());
  if (!Branch) {
    // If the landing pad wasn't in outlined form, it should be a stub with
    // an unreachable terminator.
    assert(isa<UnreachableInst>(OutlinedBB->getTerminator()));
    OutlinedBB->getTerminator()->eraseFromParent();
    // That should leave OutlinedLPad as the last instruction in its block.
    assert(&OutlinedBB->back() == OutlinedLPad);
  }

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
  const IntrinsicInst *EHActions = cast<IntrinsicInst>(Recover);

  // Remap the return target in the nested handler.
  SmallVector<BlockAddress *, 4> ActionTargets;
  SmallVector<std::unique_ptr<ActionHandler>, 4> ActionList;
  parseEHActions(EHActions, ActionList);
  for (const auto &Action : ActionList) {
    auto *Catch = dyn_cast<CatchHandler>(Action.get());
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

      // Ignore targets that aren't part of an outlined handler function.
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
  ActionList.clear();

  if (Branch) {
    // If the landing pad was already in outlined form, just update its targets.
    for (unsigned int I = Branch->getNumDestinations(); I > 0; --I)
      Branch->removeDestination(I);
    // Add the previously collected action targets.
    for (auto *Target : ActionTargets)
      Branch->addDestination(Target->getBasicBlock());
  } else {
    // If the landing pad was previously stubbed out, fill in its outlined form.
    IntrinsicInst *NewEHActions = cast<IntrinsicInst>(EHActions->clone());
    OutlinedBB->getInstList().push_back(NewEHActions);

    // Insert an indirect branch into the outlined landing pad BB.
    IndirectBrInst *IBr = IndirectBrInst::Create(NewEHActions, 0, OutlinedBB);
    // Add the previously collected action targets.
    for (auto *Target : ActionTargets)
      IBr->addDestination(Target->getBasicBlock());
  }
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

static bool isCatchBlock(BasicBlock *BB) {
  for (BasicBlock::iterator II = BB->getFirstNonPHIOrDbg(), IE = BB->end();
       II != IE; ++II) {
    if (match(cast<Value>(II), m_Intrinsic<Intrinsic::eh_begincatch>()))
      return true;
  }
  return false;
}

static BasicBlock *createStubLandingPad(Function *Handler) {
  // FIXME: Finish this!
  LLVMContext &Context = Handler->getContext();
  BasicBlock *StubBB = BasicBlock::Create(Context, "stub");
  Handler->getBasicBlockList().push_back(StubBB);
  IRBuilder<> Builder(StubBB);
  LandingPadInst *LPad = Builder.CreateLandingPad(
      llvm::StructType::get(Type::getInt8PtrTy(Context),
                            Type::getInt32Ty(Context), nullptr),
      0);
  // Insert a call to llvm.eh.actions so that we don't try to outline this lpad.
  Function *ActionIntrin =
      Intrinsic::getDeclaration(Handler->getParent(), Intrinsic::eh_actions);
  Builder.CreateCall(ActionIntrin, {}, "recover");
  LPad->setCleanup(true);
  Builder.CreateUnreachable();
  return StubBB;
}

// Cycles through the blocks in an outlined handler function looking for an
// invoke instruction and inserts an invoke of llvm.donothing with an empty
// landing pad if none is found.  The code that generates the .xdata tables for
// the handler needs at least one landing pad to identify the parent function's
// personality.
void WinEHPrepare::addStubInvokeToHandlerIfNeeded(Function *Handler) {
  ReturnInst *Ret = nullptr;
  UnreachableInst *Unreached = nullptr;
  for (BasicBlock &BB : *Handler) {
    TerminatorInst *Terminator = BB.getTerminator();
    // If we find an invoke, there is nothing to be done.
    auto *II = dyn_cast<InvokeInst>(Terminator);
    if (II)
      return;
    // If we've already recorded a return instruction, keep looking for invokes.
    if (!Ret)
      Ret = dyn_cast<ReturnInst>(Terminator);
    // If we haven't recorded an unreachable instruction, try this terminator.
    if (!Unreached)
      Unreached = dyn_cast<UnreachableInst>(Terminator);
  }

  // If we got this far, the handler contains no invokes.  We should have seen
  // at least one return or unreachable instruction.  We'll insert an invoke of
  // llvm.donothing ahead of that instruction.
  assert(Ret || Unreached);
  TerminatorInst *Term;
  if (Ret)
    Term = Ret;
  else
    Term = Unreached;
  BasicBlock *OldRetBB = Term->getParent();
  BasicBlock *NewRetBB = SplitBlock(OldRetBB, Term, DT);
  // SplitBlock adds an unconditional branch instruction at the end of the
  // parent block.  We want to replace that with an invoke call, so we can
  // erase it now.
  OldRetBB->getTerminator()->eraseFromParent();
  BasicBlock *StubLandingPad = createStubLandingPad(Handler);
  Function *F =
      Intrinsic::getDeclaration(Handler->getParent(), Intrinsic::donothing);
  InvokeInst::Create(F, NewRetBB, StubLandingPad, None, "", OldRetBB);
}

// FIXME: Consider sinking this into lib/Target/X86 somehow. TargetLowering
// usually doesn't build LLVM IR, so that's probably the wrong place.
Function *WinEHPrepare::createHandlerFunc(Function *ParentFn, Type *RetTy,
                                          const Twine &Name, Module *M,
                                          Value *&ParentFP) {
  // x64 uses a two-argument prototype where the parent FP is the second
  // argument. x86 uses no arguments, just the incoming EBP value.
  LLVMContext &Context = M->getContext();
  Type *Int8PtrType = Type::getInt8PtrTy(Context);
  FunctionType *FnType;
  if (TheTriple.getArch() == Triple::x86_64) {
    Type *ArgTys[2] = {Int8PtrType, Int8PtrType};
    FnType = FunctionType::get(RetTy, ArgTys, false);
  } else {
    FnType = FunctionType::get(RetTy, None, false);
  }

  Function *Handler =
      Function::Create(FnType, GlobalVariable::InternalLinkage, Name, M);
  BasicBlock *Entry = BasicBlock::Create(Context, "entry");
  Handler->getBasicBlockList().push_front(Entry);
  if (TheTriple.getArch() == Triple::x86_64) {
    ParentFP = &(Handler->getArgumentList().back());
  } else {
    assert(M);
    Function *FrameAddressFn =
        Intrinsic::getDeclaration(M, Intrinsic::frameaddress);
    Function *RecoverFPFn =
        Intrinsic::getDeclaration(M, Intrinsic::x86_seh_recoverfp);
    IRBuilder<> Builder(&Handler->getEntryBlock());
    Value *EBP =
        Builder.CreateCall(FrameAddressFn, {Builder.getInt32(1)}, "ebp");
    Value *ParentI8Fn = Builder.CreateBitCast(ParentFn, Int8PtrType);
    ParentFP = Builder.CreateCall(RecoverFPFn, {ParentI8Fn, EBP});
  }
  return Handler;
}

bool WinEHPrepare::outlineHandler(ActionHandler *Action, Function *SrcFn,
                                  LandingPadInst *LPad, BasicBlock *StartBB,
                                  FrameVarInfoMap &VarInfo) {
  Module *M = SrcFn->getParent();
  LLVMContext &Context = M->getContext();
  Type *Int8PtrType = Type::getInt8PtrTy(Context);

  // Create a new function to receive the handler contents.
  Value *ParentFP;
  Function *Handler;
  if (Action->getType() == Catch) {
    Handler = createHandlerFunc(SrcFn, Int8PtrType, SrcFn->getName() + ".catch", M,
                                ParentFP);
  } else {
    Handler = createHandlerFunc(SrcFn, Type::getVoidTy(Context),
                                SrcFn->getName() + ".cleanup", M, ParentFP);
  }
  Handler->setPersonalityFn(SrcFn->getPersonalityFn());
  HandlerToParentFP[Handler] = ParentFP;
  Handler->addFnAttr("wineh-parent", SrcFn->getName());
  BasicBlock *Entry = &Handler->getEntryBlock();

  // Generate a standard prolog to setup the frame recovery structure.
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(Entry);
  Builder.SetCurrentDebugLocation(LPad->getDebugLoc());

  std::unique_ptr<WinEHCloningDirectorBase> Director;

  ValueToValueMapTy VMap;

  LandingPadMap &LPadMap = LPadMaps[LPad];
  if (!LPadMap.isInitialized())
    LPadMap.mapLandingPad(LPad);
  if (auto *CatchAction = dyn_cast<CatchHandler>(Action)) {
    Constant *Sel = CatchAction->getSelector();
    Director.reset(new WinEHCatchDirector(Handler, ParentFP, Sel, VarInfo,
                                          LPadMap, NestedLPtoOriginalLP, DT,
                                          EHBlocks));
    LPadMap.remapEHValues(VMap, UndefValue::get(Int8PtrType),
                          ConstantInt::get(Type::getInt32Ty(Context), 1));
  } else {
    Director.reset(
        new WinEHCleanupDirector(Handler, ParentFP, VarInfo, LPadMap));
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

  // The landing pad value may be used by PHI nodes.  It will ultimately be
  // eliminated, but we need it in the map for intermediate handling.
  VMap[LPad] = UndefValue::get(LPad->getType());

  // Skip over PHIs and, if applicable, landingpad instructions.
  II = StartBB->getFirstInsertionPt();

  CloneAndPruneIntoFromInst(Handler, SrcFn, II, VMap,
                            /*ModuleLevelChanges=*/false, Returns, "",
                            &OutlinedFunctionInfo, Director.get());

  // Move all the instructions in the cloned "entry" block into our entry block.
  // Depending on how the parent function was laid out, the block that will
  // correspond to the outlined entry block may not be the first block in the
  // list.  We can recognize it, however, as the cloned block which has no
  // predecessors.  Any other block wouldn't have been cloned if it didn't
  // have a predecessor which was also cloned.
  Function::iterator ClonedIt = std::next(Function::iterator(Entry));
  while (!pred_empty(ClonedIt))
    ++ClonedIt;
  BasicBlock *ClonedEntryBB = ClonedIt;
  assert(ClonedEntryBB);
  Entry->getInstList().splice(Entry->end(), ClonedEntryBB->getInstList());
  ClonedEntryBB->eraseFromParent();

  // Make sure we can identify the handler's personality later.
  addStubInvokeToHandlerIfNeeded(Handler);

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
    HandlerBB = SplitBlock(StartBB, StartBB->getFirstInsertionPt(), DT);
  }
  IRBuilder<> Builder(HandlerBB->getFirstInsertionPt());
  Function *EHCodeFn = Intrinsic::getDeclaration(
      StartBB->getParent()->getParent(), Intrinsic::eh_exceptioncode);
  Value *Code = Builder.CreateCall(EHCodeFn, {}, "sehcode");
  Code = Builder.CreateIntToPtr(Code, SEHExceptionCodeSlot->getAllocatedType());
  Builder.CreateStore(Code, SEHExceptionCodeSlot);
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

static bool isLocalAddressCall(const Value *V) {
  return match(const_cast<Value *>(V), m_Intrinsic<Intrinsic::localaddress>());
}

CloningDirector::CloningAction WinEHCloningDirectorBase::handleInstruction(
    ValueToValueMapTy &VMap, const Instruction *Inst, BasicBlock *NewBB) {
  // If this is one of the boilerplate landing pad instructions, skip it.
  // The instruction will have already been remapped in VMap.
  if (LPadMap.isLandingPadSpecificInst(Inst))
    return CloningDirector::SkipInstruction;

  // Nested landing pads that have not already been outlined will be cloned as
  // stubs, with just the landingpad instruction and an unreachable instruction.
  // When all landingpads have been outlined, we'll replace this with the
  // llvm.eh.actions call and indirect branch created when the landing pad was
  // outlined.
  if (auto *LPad = dyn_cast<LandingPadInst>(Inst)) {
    return handleLandingPad(VMap, LPad, NewBB);
  }

  // Nested landing pads that have already been outlined will be cloned in their
  // outlined form, but we need to intercept the ibr instruction to filter out
  // targets that do not return to the handler we are outlining.
  if (auto *IBr = dyn_cast<IndirectBrInst>(Inst)) {
    return handleIndirectBr(VMap, IBr, NewBB);
  }

  if (auto *Invoke = dyn_cast<InvokeInst>(Inst))
    return handleInvoke(VMap, Invoke, NewBB);

  if (auto *Resume = dyn_cast<ResumeInst>(Inst))
    return handleResume(VMap, Resume, NewBB);

  if (auto *Cmp = dyn_cast<CmpInst>(Inst))
    return handleCompare(VMap, Cmp, NewBB);

  if (match(Inst, m_Intrinsic<Intrinsic::eh_begincatch>()))
    return handleBeginCatch(VMap, Inst, NewBB);
  if (match(Inst, m_Intrinsic<Intrinsic::eh_endcatch>()))
    return handleEndCatch(VMap, Inst, NewBB);
  if (match(Inst, m_Intrinsic<Intrinsic::eh_typeid_for>()))
    return handleTypeIdFor(VMap, Inst, NewBB);

  // When outlining llvm.localaddress(), remap that to the second argument,
  // which is the FP of the parent.
  if (isLocalAddressCall(Inst)) {
    VMap[Inst] = ParentFP;
    return CloningDirector::SkipInstruction;
  }

  // Continue with the default cloning behavior.
  return CloningDirector::CloneInstruction;
}

CloningDirector::CloningAction WinEHCatchDirector::handleLandingPad(
    ValueToValueMapTy &VMap, const LandingPadInst *LPad, BasicBlock *NewBB) {
  // If the instruction after the landing pad is a call to llvm.eh.actions
  // the landing pad has already been outlined.  In this case, we should
  // clone it because it may return to a block in the handler we are
  // outlining now that would otherwise be unreachable.  The landing pads
  // are sorted before outlining begins to enable this case to work
  // properly.
  const Instruction *NextI = LPad->getNextNode();
  if (match(NextI, m_Intrinsic<Intrinsic::eh_actions>()))
    return CloningDirector::CloneInstruction;

  // If the landing pad hasn't been outlined yet, the landing pad we are
  // outlining now does not dominate it and so it cannot return to a block
  // in this handler.  In that case, we can just insert a stub landing
  // pad now and patch it up later.
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

CloningDirector::CloningAction WinEHCatchDirector::handleIndirectBr(
    ValueToValueMapTy &VMap,
    const IndirectBrInst *IBr,
    BasicBlock *NewBB) {
  // If this indirect branch is not part of a landing pad block, just clone it.
  const BasicBlock *ParentBB = IBr->getParent();
  if (!ParentBB->isLandingPad())
    return CloningDirector::CloneInstruction;

  // If it is part of a landing pad, we want to filter out target blocks
  // that are not part of the handler we are outlining.
  const LandingPadInst *LPad = ParentBB->getLandingPadInst();

  // Save this correlation for later processing.
  NestedLPtoOriginalLP[cast<LandingPadInst>(VMap[LPad])] = LPad;

  // We should only get here for landing pads that have already been outlined.
  assert(match(LPad->getNextNode(), m_Intrinsic<Intrinsic::eh_actions>()));

  // Copy the indirectbr, but only include targets that were previously
  // identified as EH blocks and are dominated by the nested landing pad.
  SetVector<const BasicBlock *> ReturnTargets;
  for (int I = 0, E = IBr->getNumDestinations(); I < E; ++I) {
    auto *TargetBB = IBr->getDestination(I);
    if (EHBlocks.count(const_cast<BasicBlock*>(TargetBB)) &&
        DT->dominates(ParentBB, TargetBB)) {
      DEBUG(dbgs() << "  Adding destination " << TargetBB->getName() << "\n");
      ReturnTargets.insert(TargetBB);
    }
  }
  IndirectBrInst *NewBranch = 
        IndirectBrInst::Create(const_cast<Value *>(IBr->getAddress()),
                               ReturnTargets.size(), NewBB);
  for (auto *Target : ReturnTargets)
    NewBranch->addDestination(const_cast<BasicBlock*>(Target));

  // The operands and targets of the branch instruction are remapped later
  // because it is a terminator.  Tell the cloning code to clone the
  // blocks we just added to the target list.
  return CloningDirector::CloneSuccessors;
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

CloningDirector::CloningAction
WinEHCatchDirector::handleCompare(ValueToValueMapTy &VMap,
                                  const CmpInst *Compare, BasicBlock *NewBB) {
  const IntrinsicInst *IntrinCall = nullptr;
  if (match(Compare->getOperand(0), m_Intrinsic<Intrinsic::eh_typeid_for>())) {
    IntrinCall = dyn_cast<IntrinsicInst>(Compare->getOperand(0));
  } else if (match(Compare->getOperand(1),
                   m_Intrinsic<Intrinsic::eh_typeid_for>())) {
    IntrinCall = dyn_cast<IntrinsicInst>(Compare->getOperand(1));
  }
  if (IntrinCall) {
    Value *Selector = IntrinCall->getArgOperand(0)->stripPointerCasts();
    // This causes a replacement that will collapse the landing pad CFG based
    // on the filter function we intend to match.
    if (Selector == CurrentSelector->stripPointerCasts()) {
      VMap[Compare] = ConstantInt::get(SelectorIDType, 1);
    } else {
      VMap[Compare] = ConstantInt::get(SelectorIDType, 0);
    }
    return CloningDirector::SkipInstruction;
  }
  return CloningDirector::CloneInstruction;
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
  // Cleanup code may flow into catch blocks or the catch block may be part
  // of a branch that will be optimized away.  We'll insert a return
  // instruction now, but it may be pruned before the cloning process is
  // complete.
  ReturnInst::Create(NewBB->getContext(), nullptr, NewBB);
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

CloningDirector::CloningAction WinEHCleanupDirector::handleIndirectBr(
    ValueToValueMapTy &VMap,
    const IndirectBrInst *IBr,
    BasicBlock *NewBB) {
  // No special handling is required for cleanup cloning.
  return CloningDirector::CloneInstruction;
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

CloningDirector::CloningAction
WinEHCleanupDirector::handleCompare(ValueToValueMapTy &VMap,
                                    const CmpInst *Compare, BasicBlock *NewBB) {
  if (match(Compare->getOperand(0), m_Intrinsic<Intrinsic::eh_typeid_for>()) ||
      match(Compare->getOperand(1), m_Intrinsic<Intrinsic::eh_typeid_for>())) {
    VMap[Compare] = ConstantInt::get(SelectorIDType, 1);
    return CloningDirector::SkipInstruction;
  }
  return CloningDirector::CloneInstruction;
}

WinEHFrameVariableMaterializer::WinEHFrameVariableMaterializer(
    Function *OutlinedFn, Value *ParentFP, FrameVarInfoMap &FrameVarInfo)
    : FrameVarInfo(FrameVarInfo), Builder(OutlinedFn->getContext()) {
  BasicBlock *EntryBB = &OutlinedFn->getEntryBlock();

  // New allocas should be inserted in the entry block, but after the parent FP
  // is established if it is an instruction.
  Instruction *InsertPoint = EntryBB->getFirstInsertionPt();
  if (auto *FPInst = dyn_cast<Instruction>(ParentFP))
    InsertPoint = FPInst->getNextNode();
  Builder.SetInsertPoint(EntryBB, InsertPoint);
}

Value *WinEHFrameVariableMaterializer::materializeValueFor(Value *V) {
  // If we're asked to materialize a static alloca, we temporarily create an
  // alloca in the outlined function and add this to the FrameVarInfo map.  When
  // all the outlining is complete, we'll replace these temporary allocas with
  // calls to llvm.localrecover.
  if (auto *AV = dyn_cast<AllocaInst>(V)) {
    assert(AV->isStaticAlloca() &&
           "cannot materialize un-demoted dynamic alloca");
    AllocaInst *NewAlloca = dyn_cast<AllocaInst>(AV->clone());
    Builder.Insert(NewAlloca, AV->getName());
    FrameVarInfo[AV].push_back(NewAlloca);
    return NewAlloca;
  }

  if (isa<Instruction>(V) || isa<Argument>(V)) {
    Function *Parent = isa<Instruction>(V)
                           ? cast<Instruction>(V)->getParent()->getParent()
                           : cast<Argument>(V)->getParent();
    errs()
        << "Failed to demote instruction used in exception handler of function "
        << GlobalValue::getRealLinkageName(Parent->getName()) << ":\n";
    errs() << "  " << *V << '\n';
    report_fatal_error("WinEHPrepare failed to demote instruction");
  }

  // Don't materialize other values.
  return nullptr;
}

void WinEHFrameVariableMaterializer::escapeCatchObject(Value *V) {
  // Catch parameter objects have to live in the parent frame. When we see a use
  // of a catch parameter, add a sentinel to the multimap to indicate that it's
  // used from another handler. This will prevent us from trying to sink the
  // alloca into the handler and ensure that the catch parameter is present in
  // the call to llvm.localescape.
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

    // Skip over filter clauses.
    if (LPad->isFilter(HandlersFound)) {
      ++HandlersFound;
      continue;
    }

    // See if the clause we're looking for is a catch-all.
    // If so, the catch begins immediately.
    Constant *ExpectedSelector =
        LPad->getClause(HandlersFound)->stripPointerCasts();
    if (isa<ConstantPointerNull>(ExpectedSelector)) {
      // The catch all must occur last.
      assert(HandlersFound == NumClauses - 1);

      // There can be additional selector dispatches in the call chain that we
      // need to ignore.
      BasicBlock *CatchBlock = nullptr;
      Constant *Selector;
      while (BB && isSelectorDispatch(BB, CatchBlock, Selector, NextBB)) {
        DEBUG(dbgs() << "  Found extra catch dispatch in block "
                     << CatchBlock->getName() << "\n");
        BB = NextBB;
      }

      // Add the catch handler to the action list.
      CatchHandler *Action = nullptr;
      if (CatchHandlerMap.count(BB) && CatchHandlerMap[BB] != nullptr) {
        // If the CatchHandlerMap already has an entry for this BB, re-use it.
        Action = CatchHandlerMap[BB];
        assert(Action->getSelector() == ExpectedSelector);
      } else {
        // We don't expect a selector dispatch, but there may be a call to
        // llvm.eh.begincatch, which separates catch handling code from
        // cleanup code in the same control flow.  This call looks for the
        // begincatch intrinsic.
        Action = findCatchHandler(BB, NextBB, VisitedBlocks);
        if (Action) {
          // For C++ EH, check if there is any interesting cleanup code before
          // we begin the catch. This is important because cleanups cannot
          // rethrow exceptions but code called from catches can. For SEH, it
          // isn't important if some finally code before a catch-all is executed
          // out of line or after recovering from the exception.
          if (Personality == EHPersonality::MSVC_CXX)
            findCleanupHandlers(Actions, BB, BB);
        } else {
          // If an action was not found, it means that the control flows
          // directly into the catch-all handler and there is no cleanup code.
          // That's an expected situation and we must create a catch action.
          // Since this is a catch-all handler, the selector won't actually
          // appear in the code anywhere.  ExpectedSelector here is the constant
          // null ptr that we got from the landing pad instruction.
          Action = new CatchHandler(BB, ExpectedSelector, nullptr);
          CatchHandlerMap[BB] = Action;
        }
      }
      Actions.insertCatchHandler(Action);
      DEBUG(dbgs() << "  Catch all handler at block " << BB->getName() << "\n");
      ++HandlersFound;

      // Once we reach a catch-all, don't expect to hit a resume instruction.
      BB = nullptr;
      break;
    }

    CatchHandler *CatchAction = findCatchHandler(BB, NextBB, VisitedBlocks);
    assert(CatchAction);

    // See if there is any interesting code executed before the dispatch.
    findCleanupHandlers(Actions, BB, CatchAction->getStartBlock());

    // When the source program contains multiple nested try blocks the catch
    // handlers can get strung together in such a way that we can encounter
    // a dispatch for a selector that we've already had a handler for.
    if (CatchAction->getSelector()->stripPointerCasts() == ExpectedSelector) {
      ++HandlersFound;

      // Add the catch handler to the action list.
      DEBUG(dbgs() << "  Found catch dispatch in block "
                   << CatchAction->getStartBlock()->getName() << "\n");
      Actions.insertCatchHandler(CatchAction);
    } else {
      // Under some circumstances optimized IR will flow unconditionally into a
      // handler block without checking the selector.  This can only happen if
      // the landing pad has a catch-all handler and the handler for the
      // preceding catch clause is identical to the catch-call handler
      // (typically an empty catch).  In this case, the handler must be shared
      // by all remaining clauses.
      if (isa<ConstantPointerNull>(
              CatchAction->getSelector()->stripPointerCasts())) {
        DEBUG(dbgs() << "  Applying early catch-all handler in block "
                     << CatchAction->getStartBlock()->getName()
                     << "  to all remaining clauses.\n");
        Actions.insertCatchHandler(CatchAction);
        return;
      }

      DEBUG(dbgs() << "  Found extra catch dispatch in block "
                   << CatchAction->getStartBlock()->getName() << "\n");
    }

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
    // If we encounter a block containing an llvm.eh.begincatch before we
    // find a selector dispatch block, the handler is assumed to be
    // reached unconditionally.  This happens for catch-all blocks, but
    // it can also happen for other catch handlers that have been combined
    // with the catch-all handler during optimization.
    if (isCatchBlock(BB)) {
      PointerType *Int8PtrTy = Type::getInt8PtrTy(BB->getContext());
      Constant *NullSelector = ConstantPointerNull::get(Int8PtrTy);
      CatchHandler *Action = new CatchHandler(BB, NullSelector, nullptr);
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

static CallSite matchOutlinedFinallyCall(BasicBlock *BB,
                                         Instruction *MaybeCall) {
  // Look for finally blocks that Clang has already outlined for us.
  //   %fp = call i8* @llvm.localaddress()
  //   call void @"fin$parent"(iN 1, i8* %fp)
  if (isLocalAddressCall(MaybeCall) && MaybeCall != BB->getTerminator())
    MaybeCall = MaybeCall->getNextNode();
  CallSite FinallyCall(MaybeCall);
  if (!FinallyCall || FinallyCall.arg_size() != 2)
    return CallSite();
  if (!match(FinallyCall.getArgument(0), m_SpecificInt(1)))
    return CallSite();
  if (!isLocalAddressCall(FinallyCall.getArgument(1)))
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
        // know the block must end in either an unconditional branch, a
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

      // Look for outlined finally calls on x64, since those happen to match the
      // prototype provided by the runtime.
      if (TheTriple.getArch() == Triple::x86_64) {
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

          // Split the block if there were more interesting instructions and
          // look for finally calls in the normal successor block.
          BasicBlock *SuccBB = BB;
          if (FinallyCall.getInstruction() != BB->getTerminator() &&
              FinallyCall.getInstruction()->getNextNode() !=
                  BB->getTerminator()) {
            SuccBB =
                SplitBlock(BB, FinallyCall.getInstruction()->getNextNode(), DT);
          } else {
            if (FinallyCall.isInvoke()) {
              SuccBB = cast<InvokeInst>(FinallyCall.getInstruction())
                           ->getNormalDest();
            } else {
              SuccBB = BB->getUniqueSuccessor();
              assert(SuccBB &&
                     "splitOutlinedFinallyCalls didn't insert a branch");
            }
          }
          BB = SuccBB;
          if (BB == EndBB)
            return;
          continue;
        }
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
void llvm::parseEHActions(
    const IntrinsicInst *II,
    SmallVectorImpl<std::unique_ptr<ActionHandler>> &Actions) {
  assert(II->getIntrinsicID() == Intrinsic::eh_actions &&
         "attempted to parse non eh.actions intrinsic");
  for (unsigned I = 0, E = II->getNumArgOperands(); I != E;) {
    uint64_t ActionKind =
        cast<ConstantInt>(II->getArgOperand(I))->getZExtValue();
    if (ActionKind == /*catch=*/1) {
      auto *Selector = cast<Constant>(II->getArgOperand(I + 1));
      ConstantInt *EHObjIndex = cast<ConstantInt>(II->getArgOperand(I + 2));
      int64_t EHObjIndexVal = EHObjIndex->getSExtValue();
      Constant *Handler = cast<Constant>(II->getArgOperand(I + 3));
      I += 4;
      auto CH = make_unique<CatchHandler>(/*BB=*/nullptr, Selector,
                                          /*NextBB=*/nullptr);
      CH->setHandlerBlockOrFunc(Handler);
      CH->setExceptionVarIndex(EHObjIndexVal);
      Actions.push_back(std::move(CH));
    } else if (ActionKind == 0) {
      Constant *Handler = cast<Constant>(II->getArgOperand(I + 1));
      I += 2;
      auto CH = make_unique<CleanupHandler>(/*BB=*/nullptr);
      CH->setHandlerBlockOrFunc(Handler);
      Actions.push_back(std::move(CH));
    } else {
      llvm_unreachable("Expected either a catch or cleanup handler!");
    }
  }
  std::reverse(Actions.begin(), Actions.end());
}

static int addUnwindMapEntry(WinEHFuncInfo &FuncInfo, int ToState,
                             const Value *V) {
  WinEHUnwindMapEntry UME;
  UME.ToState = ToState;
  UME.Cleanup = V;
  FuncInfo.UnwindMap.push_back(UME);
  return FuncInfo.getLastStateNumber();
}

static void addTryBlockMapEntry(WinEHFuncInfo &FuncInfo, int TryLow,
                                int TryHigh, int CatchHigh,
                                ArrayRef<const CatchPadInst *> Handlers) {
  WinEHTryBlockMapEntry TBME;
  TBME.TryLow = TryLow;
  TBME.TryHigh = TryHigh;
  TBME.CatchHigh = CatchHigh;
  assert(TBME.TryLow <= TBME.TryHigh);
  for (const CatchPadInst *CPI : Handlers) {
    WinEHHandlerType HT;
    Constant *TypeInfo = cast<Constant>(CPI->getArgOperand(0));
    if (TypeInfo->isNullValue())
      HT.TypeDescriptor = nullptr;
    else
      HT.TypeDescriptor = cast<GlobalVariable>(TypeInfo->stripPointerCasts());
    HT.Adjectives = cast<ConstantInt>(CPI->getArgOperand(1))->getZExtValue();
    HT.Handler = CPI->getNormalDest();
    HT.CatchObjRecoverIdx = -2;
    if (isa<ConstantPointerNull>(CPI->getArgOperand(2)))
      HT.CatchObj.Alloca = nullptr;
    else
      HT.CatchObj.Alloca = cast<AllocaInst>(CPI->getArgOperand(2));
    TBME.HandlerArray.push_back(HT);
  }
  FuncInfo.TryBlockMap.push_back(TBME);
}

static const CatchPadInst *getSingleCatchPadPredecessor(const BasicBlock *BB) {
  for (const BasicBlock *PredBlock : predecessors(BB))
    if (auto *CPI = dyn_cast<CatchPadInst>(PredBlock->getFirstNonPHI()))
      return CPI;
  return nullptr;
}

/// Find all the catchpads that feed directly into the catchendpad. Frontends
/// using this personality should ensure that each catchendpad and catchpad has
/// one or zero catchpad predecessors.
///
/// The following C++ generates the IR after it:
///   try {
///   } catch (A) {
///   } catch (B) {
///   }
///
/// IR:
///   %catchpad.A
///     catchpad [i8* A typeinfo]
///         to label %catch.A unwind label %catchpad.B
///   %catchpad.B
///     catchpad [i8* B typeinfo]
///         to label %catch.B unwind label %endcatches
///   %endcatches
///     catchendblock unwind to caller
static void
findCatchPadsForCatchEndPad(const BasicBlock *CatchEndBB,
                            SmallVectorImpl<const CatchPadInst *> &Handlers) {
  const CatchPadInst *CPI = getSingleCatchPadPredecessor(CatchEndBB);
  while (CPI) {
    Handlers.push_back(CPI);
    CPI = getSingleCatchPadPredecessor(CPI->getParent());
  }
  // We've pushed these back into reverse source order.  Reverse them to get
  // the list back into source order.
  std::reverse(Handlers.begin(), Handlers.end());
}

// Given BB which ends in an unwind edge, return the EHPad that this BB belongs
// to. If the unwind edge came from an invoke, return null.
static const BasicBlock *getEHPadFromPredecessor(const BasicBlock *BB) {
  const TerminatorInst *TI = BB->getTerminator();
  if (isa<InvokeInst>(TI))
    return nullptr;
  if (TI->isEHPad())
    return BB;
  return cast<CleanupReturnInst>(TI)->getCleanupPad()->getParent();
}

static void calculateExplicitCXXStateNumbers(WinEHFuncInfo &FuncInfo,
                                             const BasicBlock &BB,
                                             int ParentState) {
  assert(BB.isEHPad());
  const Instruction *FirstNonPHI = BB.getFirstNonPHI();
  // All catchpad instructions will be handled when we process their
  // respective catchendpad instruction.
  if (isa<CatchPadInst>(FirstNonPHI))
    return;

  if (isa<CatchEndPadInst>(FirstNonPHI)) {
    SmallVector<const CatchPadInst *, 2> Handlers;
    findCatchPadsForCatchEndPad(&BB, Handlers);
    const BasicBlock *FirstTryPad = Handlers.front()->getParent();
    int TryLow = addUnwindMapEntry(FuncInfo, ParentState, nullptr);
    FuncInfo.EHPadStateMap[Handlers.front()] = TryLow;
    for (const BasicBlock *PredBlock : predecessors(FirstTryPad))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitCXXStateNumbers(FuncInfo, *PredBlock, TryLow);
    int CatchLow = addUnwindMapEntry(FuncInfo, ParentState, nullptr);

    // catchpads are separate funclets in C++ EH due to the way rethrow works.
    // In SEH, they aren't, so no invokes will unwind to the catchendpad.
    FuncInfo.EHPadStateMap[FirstNonPHI] = CatchLow;
    int TryHigh = CatchLow - 1;
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitCXXStateNumbers(FuncInfo, *PredBlock, CatchLow);
    int CatchHigh = FuncInfo.getLastStateNumber();
    addTryBlockMapEntry(FuncInfo, TryLow, TryHigh, CatchHigh, Handlers);
    DEBUG(dbgs() << "TryLow[" << FirstTryPad->getName() << "]: " << TryLow
                 << '\n');
    DEBUG(dbgs() << "TryHigh[" << FirstTryPad->getName() << "]: " << TryHigh
                 << '\n');
    DEBUG(dbgs() << "CatchHigh[" << FirstTryPad->getName() << "]: " << CatchHigh
                 << '\n');
  } else if (isa<CleanupPadInst>(FirstNonPHI)) {
    int CleanupState = addUnwindMapEntry(FuncInfo, ParentState, &BB);
    FuncInfo.EHPadStateMap[FirstNonPHI] = CleanupState;
    DEBUG(dbgs() << "Assigning state #" << CleanupState << " to BB "
                 << BB.getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitCXXStateNumbers(FuncInfo, *PredBlock, CleanupState);
  } else if (isa<TerminatePadInst>(FirstNonPHI)) {
    report_fatal_error("Not yet implemented!");
  } else {
    llvm_unreachable("unexpected EH Pad!");
  }
}

static int addSEHHandler(WinEHFuncInfo &FuncInfo, int ParentState,
                         const Function *Filter, const BasicBlock *Handler) {
  SEHUnwindMapEntry Entry;
  Entry.ToState = ParentState;
  Entry.Filter = Filter;
  Entry.Handler = Handler;
  FuncInfo.SEHUnwindMap.push_back(Entry);
  return FuncInfo.SEHUnwindMap.size() - 1;
}

static void calculateExplicitSEHStateNumbers(WinEHFuncInfo &FuncInfo,
                                             const BasicBlock &BB,
                                             int ParentState) {
  assert(BB.isEHPad());
  const Instruction *FirstNonPHI = BB.getFirstNonPHI();
  // All catchpad instructions will be handled when we process their
  // respective catchendpad instruction.
  if (isa<CatchPadInst>(FirstNonPHI))
    return;

  if (isa<CatchEndPadInst>(FirstNonPHI)) {
    // Extract the filter function and the __except basic block and create a
    // state for them.
    SmallVector<const CatchPadInst *, 1> Handlers;
    findCatchPadsForCatchEndPad(&BB, Handlers);
    assert(Handlers.size() == 1 &&
           "SEH doesn't have multiple handlers per __try");
    const CatchPadInst *CPI = Handlers.front();
    const BasicBlock *CatchPadBB = CPI->getParent();
    const Function *Filter =
        cast<Function>(CPI->getArgOperand(0)->stripPointerCasts());
    int TryState =
        addSEHHandler(FuncInfo, ParentState, Filter, CPI->getNormalDest());

    // Everything in the __try block uses TryState as its parent state.
    FuncInfo.EHPadStateMap[CPI] = TryState;
    DEBUG(dbgs() << "Assigning state #" << TryState << " to BB "
                 << CatchPadBB->getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(CatchPadBB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitSEHStateNumbers(FuncInfo, *PredBlock, TryState);

    // Everything in the __except block unwinds to ParentState, just like code
    // outside the __try.
    FuncInfo.EHPadStateMap[FirstNonPHI] = ParentState;
    DEBUG(dbgs() << "Assigning state #" << ParentState << " to BB "
                 << BB.getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitSEHStateNumbers(FuncInfo, *PredBlock, ParentState);
  } else if (isa<CleanupPadInst>(FirstNonPHI)) {
    int CleanupState =
        addSEHHandler(FuncInfo, ParentState, /*Filter=*/nullptr, &BB);
    FuncInfo.EHPadStateMap[FirstNonPHI] = CleanupState;
    DEBUG(dbgs() << "Assigning state #" << CleanupState << " to BB "
                 << BB.getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitSEHStateNumbers(FuncInfo, *PredBlock, CleanupState);
  } else if (isa<CleanupEndPadInst>(FirstNonPHI)) {
    // Anything unwinding through CleanupEndPadInst is in ParentState.
    FuncInfo.EHPadStateMap[FirstNonPHI] = ParentState;
    DEBUG(dbgs() << "Assigning state #" << ParentState << " to BB "
                 << BB.getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitSEHStateNumbers(FuncInfo, *PredBlock, ParentState);
  } else if (isa<TerminatePadInst>(FirstNonPHI)) {
    report_fatal_error("Not yet implemented!");
  } else {
    llvm_unreachable("unexpected EH Pad!");
  }
}

/// Check if the EH Pad unwinds to caller.  Cleanups are a little bit of a
/// special case because we have to look at the cleanupret instruction that uses
/// the cleanuppad.
static bool doesEHPadUnwindToCaller(const Instruction *EHPad) {
  auto *CPI = dyn_cast<CleanupPadInst>(EHPad);
  if (!CPI)
    return EHPad->mayThrow();

  // This cleanup does not return or unwind, so we say it unwinds to caller.
  if (CPI->use_empty())
    return true;

  const Instruction *User = CPI->user_back();
  if (auto *CRI = dyn_cast<CleanupReturnInst>(User))
    return CRI->unwindsToCaller();
  return cast<CleanupEndPadInst>(User)->unwindsToCaller();
}

void llvm::calculateSEHStateNumbers(const Function *Fn,
                                    WinEHFuncInfo &FuncInfo) {
  // Don't compute state numbers twice.
  if (!FuncInfo.SEHUnwindMap.empty())
    return;

  for (const BasicBlock &BB : *Fn) {
    if (!BB.isEHPad() || !doesEHPadUnwindToCaller(BB.getFirstNonPHI()))
      continue;
    calculateExplicitSEHStateNumbers(FuncInfo, BB, -1);
  }
}

void llvm::calculateWinCXXEHStateNumbers(const Function *Fn,
                                         WinEHFuncInfo &FuncInfo) {
  // Return if it's already been done.
  if (!FuncInfo.EHPadStateMap.empty())
    return;

  for (const BasicBlock &BB : *Fn) {
    if (!BB.isEHPad())
      continue;
    if (BB.isLandingPad())
      report_fatal_error("MSVC C++ EH cannot use landingpads");
    const Instruction *FirstNonPHI = BB.getFirstNonPHI();
    // Skip cleanupendpads; they are exits, not entries.
    if (isa<CleanupEndPadInst>(FirstNonPHI))
      continue;
    if (!doesEHPadUnwindToCaller(FirstNonPHI))
      continue;
    calculateExplicitCXXStateNumbers(FuncInfo, BB, -1);
  }
}

void WinEHPrepare::replaceTerminatePadWithCleanup(Function &F) {
  if (Personality != EHPersonality::MSVC_CXX)
    return;
  for (BasicBlock &BB : F) {
    Instruction *First = BB.getFirstNonPHI();
    auto *TPI = dyn_cast<TerminatePadInst>(First);
    if (!TPI)
      continue;

    if (TPI->getNumArgOperands() != 1)
      report_fatal_error(
          "Expected a unary terminatepad for MSVC C++ personalities!");

    auto *TerminateFn = dyn_cast<Function>(TPI->getArgOperand(0));
    if (!TerminateFn)
      report_fatal_error("Function operand expected in terminatepad for MSVC "
                         "C++ personalities!");

    // Insert the cleanuppad instruction.
    auto *CPI = CleanupPadInst::Create(
        BB.getContext(), {}, Twine("terminatepad.for.", BB.getName()), &BB);

    // Insert the call to the terminate instruction.
    auto *CallTerminate = CallInst::Create(TerminateFn, {}, &BB);
    CallTerminate->setDoesNotThrow();
    CallTerminate->setDoesNotReturn();
    CallTerminate->setCallingConv(TerminateFn->getCallingConv());

    // Insert a new terminator for the cleanuppad using the same successor as
    // the terminatepad.
    CleanupReturnInst::Create(CPI, TPI->getUnwindDest(), &BB);

    // Let's remove the terminatepad now that we've inserted the new
    // instructions.
    TPI->eraseFromParent();
  }
}

void WinEHPrepare::colorFunclets(Function &F,
                                 SmallVectorImpl<BasicBlock *> &EntryBlocks) {
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 16> Worklist;
  BasicBlock *EntryBlock = &F.getEntryBlock();

  // Build up the color map, which maps each block to its set of 'colors'.
  // For any block B, the "colors" of B are the set of funclets F (possibly
  // including a root "funclet" representing the main function), such that
  // F will need to directly contain B or a copy of B (where the term "directly
  // contain" is used to distinguish from being "transitively contained" in
  // a nested funclet).
  // Use a CFG walk driven by a worklist of (block, color) pairs.  The "color"
  // sets attached during this processing to a block which is the entry of some
  // funclet F is actually the set of F's parents -- i.e. the union of colors
  // of all predecessors of F's entry.  For all other blocks, the color sets
  // are as defined above.  A post-pass fixes up the block color map to reflect
  // the same sense of "color" for funclet entries as for other blocks.

  Worklist.push_back({EntryBlock, EntryBlock});

  while (!Worklist.empty()) {
    BasicBlock *Visiting;
    BasicBlock *Color;
    std::tie(Visiting, Color) = Worklist.pop_back_val();
    Instruction *VisitingHead = Visiting->getFirstNonPHI();
    if (VisitingHead->isEHPad() && !isa<CatchEndPadInst>(VisitingHead) &&
        !isa<CleanupEndPadInst>(VisitingHead)) {
      // Mark this as a funclet head as a member of itself.
      FuncletBlocks[Visiting].insert(Visiting);
      // Queue exits with the parent color.
      for (User *Exit : VisitingHead->users()) {
        for (BasicBlock *Succ :
             successors(cast<Instruction>(Exit)->getParent())) {
          if (BlockColors[Succ].insert(Color).second) {
            Worklist.push_back({Succ, Color});
          }
        }
      }
      // Handle CatchPad specially since its successors need different colors.
      if (CatchPadInst *CatchPad = dyn_cast<CatchPadInst>(VisitingHead)) {
        // Visit the normal successor with the color of the new EH pad, and
        // visit the unwind successor with the color of the parent.
        BasicBlock *NormalSucc = CatchPad->getNormalDest();
        if (BlockColors[NormalSucc].insert(Visiting).second) {
          Worklist.push_back({NormalSucc, Visiting});
        }
        BasicBlock *UnwindSucc = CatchPad->getUnwindDest();
        if (BlockColors[UnwindSucc].insert(Color).second) {
          Worklist.push_back({UnwindSucc, Color});
        }
        continue;
      }
      // Switch color to the current node, except for terminate pads which
      // have no bodies and only unwind successors and so need their successors
      // visited with the color of the parent.
      if (!isa<TerminatePadInst>(VisitingHead))
        Color = Visiting;
    } else {
      // Note that this is a member of the given color.
      FuncletBlocks[Color].insert(Visiting);
    }

    TerminatorInst *Terminator = Visiting->getTerminator();
    if (isa<CleanupReturnInst>(Terminator) ||
        isa<CatchReturnInst>(Terminator) ||
        isa<CleanupEndPadInst>(Terminator)) {
      // These blocks' successors have already been queued with the parent
      // color.
      continue;
    }
    for (BasicBlock *Succ : successors(Visiting)) {
      if (isa<CatchEndPadInst>(Succ->getFirstNonPHI())) {
        // The catchendpad needs to be visited with the parent's color, not
        // the current color.  This will happen in the code above that visits
        // any catchpad unwind successor with the parent color, so we can
        // safely skip this successor here.
        continue;
      }
      if (BlockColors[Succ].insert(Color).second) {
        Worklist.push_back({Succ, Color});
      }
    }
  }

  // The processing above actually accumulated the parent set for this
  // funclet into the color set for its entry; use the parent set to
  // populate the children map, and reset the color set to include just
  // the funclet itself (no instruction can target a funclet entry except on
  // that transitions to the child funclet).
  for (BasicBlock *FuncletEntry : EntryBlocks) {
    std::set<BasicBlock *> &ColorMapItem = BlockColors[FuncletEntry];
    for (BasicBlock *Parent : ColorMapItem)
      FuncletChildren[Parent].insert(FuncletEntry);
    ColorMapItem.clear();
    ColorMapItem.insert(FuncletEntry);
  }
}

void WinEHPrepare::demotePHIsOnFunclets(Function &F) {
  // Strip PHI nodes off of EH pads.
  SmallVector<PHINode *, 16> PHINodes;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE;) {
    BasicBlock *BB = FI++;
    if (!BB->isEHPad())
      continue;
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
      Instruction *I = BI++;
      auto *PN = dyn_cast<PHINode>(I);
      // Stop at the first non-PHI.
      if (!PN)
        break;

      AllocaInst *SpillSlot = insertPHILoads(PN, F);
      if (SpillSlot)
        insertPHIStores(PN, SpillSlot);

      PHINodes.push_back(PN);
    }
  }

  for (auto *PN : PHINodes) {
    // There may be lingering uses on other EH PHIs being removed
    PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
    PN->eraseFromParent();
  }
}

void WinEHPrepare::demoteUsesBetweenFunclets(Function &F) {
  // Turn all inter-funclet uses of a Value into loads and stores.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE;) {
    BasicBlock *BB = FI++;
    std::set<BasicBlock *> &ColorsForBB = BlockColors[BB];
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
      Instruction *I = BI++;
      // Funclets are permitted to use static allocas.
      if (auto *AI = dyn_cast<AllocaInst>(I))
        if (AI->isStaticAlloca())
          continue;

      demoteNonlocalUses(I, ColorsForBB, F);
    }
  }
}

void WinEHPrepare::demoteArgumentUses(Function &F) {
  // Also demote function parameters used in funclets.
  std::set<BasicBlock *> &ColorsForEntry = BlockColors[&F.getEntryBlock()];
  for (Argument &Arg : F.args())
    demoteNonlocalUses(&Arg, ColorsForEntry, F);
}

void WinEHPrepare::cloneCommonBlocks(
    Function &F, SmallVectorImpl<BasicBlock *> &EntryBlocks) {
  // We need to clone all blocks which belong to multiple funclets.  Values are
  // remapped throughout the funclet to propogate both the new instructions
  // *and* the new basic blocks themselves.
  for (BasicBlock *FuncletPadBB : EntryBlocks) {
    std::set<BasicBlock *> &BlocksInFunclet = FuncletBlocks[FuncletPadBB];

    std::map<BasicBlock *, BasicBlock *> Orig2Clone;
    ValueToValueMapTy VMap;
    for (BasicBlock *BB : BlocksInFunclet) {
      std::set<BasicBlock *> &ColorsForBB = BlockColors[BB];
      // We don't need to do anything if the block is monochromatic.
      size_t NumColorsForBB = ColorsForBB.size();
      if (NumColorsForBB == 1)
        continue;

      // Create a new basic block and copy instructions into it!
      BasicBlock *CBB =
          CloneBasicBlock(BB, VMap, Twine(".for.", FuncletPadBB->getName()));
      // Insert the clone immediately after the original to ensure determinism
      // and to keep the same relative ordering of any funclet's blocks.
      CBB->insertInto(&F, BB->getNextNode());

      // Add basic block mapping.
      VMap[BB] = CBB;

      // Record delta operations that we need to perform to our color mappings.
      Orig2Clone[BB] = CBB;
    }

    // Update our color mappings to reflect that one block has lost a color and
    // another has gained a color.
    for (auto &BBMapping : Orig2Clone) {
      BasicBlock *OldBlock = BBMapping.first;
      BasicBlock *NewBlock = BBMapping.second;

      BlocksInFunclet.insert(NewBlock);
      BlockColors[NewBlock].insert(FuncletPadBB);

      BlocksInFunclet.erase(OldBlock);
      BlockColors[OldBlock].erase(FuncletPadBB);
    }

    // Loop over all of the instructions in the function, fixing up operand
    // references as we go.  This uses VMap to do all the hard work.
    for (BasicBlock *BB : BlocksInFunclet)
      // Loop over all instructions, fixing each one as we find it...
      for (Instruction &I : *BB)
        RemapInstruction(&I, VMap, RF_IgnoreMissingEntries);

    // Check to see if SuccBB has PHI nodes. If so, we need to add entries to
    // the PHI nodes for NewBB now.
    for (auto &BBMapping : Orig2Clone) {
      BasicBlock *OldBlock = BBMapping.first;
      BasicBlock *NewBlock = BBMapping.second;
      for (BasicBlock *SuccBB : successors(NewBlock)) {
        for (Instruction &SuccI : *SuccBB) {
          auto *SuccPN = dyn_cast<PHINode>(&SuccI);
          if (!SuccPN)
            break;

          // Ok, we have a PHI node.  Figure out what the incoming value was for
          // the OldBlock.
          int OldBlockIdx = SuccPN->getBasicBlockIndex(OldBlock);
          if (OldBlockIdx == -1)
            break;
          Value *IV = SuccPN->getIncomingValue(OldBlockIdx);

          // Remap the value if necessary.
          if (auto *Inst = dyn_cast<Instruction>(IV)) {
            ValueToValueMapTy::iterator I = VMap.find(Inst);
            if (I != VMap.end())
              IV = I->second;
          }

          SuccPN->addIncoming(IV, NewBlock);
        }
      }
    }

    for (ValueToValueMapTy::value_type VT : VMap) {
      // If there were values defined in BB that are used outside the funclet,
      // then we now have to update all uses of the value to use either the
      // original value, the cloned value, or some PHI derived value.  This can
      // require arbitrary PHI insertion, of which we are prepared to do, clean
      // these up now.
      SmallVector<Use *, 16> UsesToRename;

      auto *OldI = dyn_cast<Instruction>(const_cast<Value *>(VT.first));
      if (!OldI)
        continue;
      auto *NewI = cast<Instruction>(VT.second);
      // Scan all uses of this instruction to see if it is used outside of its
      // funclet, and if so, record them in UsesToRename.
      for (Use &U : OldI->uses()) {
        Instruction *UserI = cast<Instruction>(U.getUser());
        BasicBlock *UserBB = UserI->getParent();
        std::set<BasicBlock *> &ColorsForUserBB = BlockColors[UserBB];
        assert(!ColorsForUserBB.empty());
        if (ColorsForUserBB.size() > 1 ||
            *ColorsForUserBB.begin() != FuncletPadBB)
          UsesToRename.push_back(&U);
      }

      // If there are no uses outside the block, we're done with this
      // instruction.
      if (UsesToRename.empty())
        continue;

      // We found a use of OldI outside of the funclet.  Rename all uses of OldI
      // that are outside its funclet to be uses of the appropriate PHI node
      // etc.
      SSAUpdater SSAUpdate;
      SSAUpdate.Initialize(OldI->getType(), OldI->getName());
      SSAUpdate.AddAvailableValue(OldI->getParent(), OldI);
      SSAUpdate.AddAvailableValue(NewI->getParent(), NewI);

      while (!UsesToRename.empty())
        SSAUpdate.RewriteUseAfterInsertions(*UsesToRename.pop_back_val());
    }
  }
}

void WinEHPrepare::removeImplausibleTerminators(Function &F) {
  // Remove implausible terminators and replace them with UnreachableInst.
  for (auto &Funclet : FuncletBlocks) {
    BasicBlock *FuncletPadBB = Funclet.first;
    std::set<BasicBlock *> &BlocksInFunclet = Funclet.second;
    Instruction *FirstNonPHI = FuncletPadBB->getFirstNonPHI();
    auto *CatchPad = dyn_cast<CatchPadInst>(FirstNonPHI);
    auto *CleanupPad = dyn_cast<CleanupPadInst>(FirstNonPHI);

    for (BasicBlock *BB : BlocksInFunclet) {
      TerminatorInst *TI = BB->getTerminator();
      // CatchPadInst and CleanupPadInst can't transfer control to a ReturnInst.
      bool IsUnreachableRet = isa<ReturnInst>(TI) && (CatchPad || CleanupPad);
      // The token consumed by a CatchReturnInst must match the funclet token.
      bool IsUnreachableCatchret = false;
      if (auto *CRI = dyn_cast<CatchReturnInst>(TI))
        IsUnreachableCatchret = CRI->getCatchPad() != CatchPad;
      // The token consumed by a CleanupReturnInst must match the funclet token.
      bool IsUnreachableCleanupret = false;
      if (auto *CRI = dyn_cast<CleanupReturnInst>(TI))
        IsUnreachableCleanupret = CRI->getCleanupPad() != CleanupPad;
      // The token consumed by a CleanupEndPadInst must match the funclet token.
      bool IsUnreachableCleanupendpad = false;
      if (auto *CEPI = dyn_cast<CleanupEndPadInst>(TI))
        IsUnreachableCleanupendpad = CEPI->getCleanupPad() != CleanupPad;
      if (IsUnreachableRet || IsUnreachableCatchret ||
          IsUnreachableCleanupret || IsUnreachableCleanupendpad) {
        // Loop through all of our successors and make sure they know that one
        // of their predecessors is going away.
        for (BasicBlock *SuccBB : TI->successors())
          SuccBB->removePredecessor(BB);

        if (IsUnreachableCleanupendpad) {
          // We can't simply replace a cleanupendpad with unreachable, because
          // its predecessor edges are EH edges and unreachable is not an EH
          // pad.  Change all predecessors to the "unwind to caller" form.
          for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
               PI != PE;) {
            BasicBlock *Pred = *PI++;
            removeUnwindEdge(Pred);
          }
        }

        new UnreachableInst(BB->getContext(), TI);
        TI->eraseFromParent();
      }
      // FIXME: Check for invokes/cleanuprets/cleanupendpads which unwind to
      // implausible catchendpads (i.e. catchendpad not in immediate parent
      // funclet).
    }
  }
}

void WinEHPrepare::cleanupPreparedFunclets(Function &F) {
  // Clean-up some of the mess we made by removing useles PHI nodes, trivial
  // branches, etc.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE;) {
    BasicBlock *BB = FI++;
    SimplifyInstructionsInBlock(BB);
    ConstantFoldTerminator(BB, /*DeleteDeadConditions=*/true);
    MergeBlockIntoPredecessor(BB);
  }

  // We might have some unreachable blocks after cleaning up some impossible
  // control flow.
  removeUnreachableBlocks(F);
}

void WinEHPrepare::verifyPreparedFunclets(Function &F) {
  // Recolor the CFG to verify that all is well.
  for (BasicBlock &BB : F) {
    size_t NumColors = BlockColors[&BB].size();
    assert(NumColors == 1 && "Expected monochromatic BB!");
    if (NumColors == 0)
      report_fatal_error("Uncolored BB!");
    if (NumColors > 1)
      report_fatal_error("Multicolor BB!");
    if (!DisableDemotion) {
      bool EHPadHasPHI = BB.isEHPad() && isa<PHINode>(BB.begin());
      assert(!EHPadHasPHI && "EH Pad still has a PHI!");
      if (EHPadHasPHI)
        report_fatal_error("EH Pad still has a PHI!");
    }
  }
}

bool WinEHPrepare::prepareExplicitEH(
    Function &F, SmallVectorImpl<BasicBlock *> &EntryBlocks) {
  // Remove unreachable blocks.  It is not valuable to assign them a color and
  // their existence can trick us into thinking values are alive when they are
  // not.
  removeUnreachableBlocks(F);

  replaceTerminatePadWithCleanup(F);

  // Determine which blocks are reachable from which funclet entries.
  colorFunclets(F, EntryBlocks);

  if (!DisableDemotion) {
    demotePHIsOnFunclets(F);

    demoteUsesBetweenFunclets(F);

    demoteArgumentUses(F);
  }

  cloneCommonBlocks(F, EntryBlocks);

  if (!DisableCleanups) {
    removeImplausibleTerminators(F);

    cleanupPreparedFunclets(F);
  }

  verifyPreparedFunclets(F);

  BlockColors.clear();
  FuncletBlocks.clear();
  FuncletChildren.clear();

  return true;
}

// TODO: Share loads when one use dominates another, or when a catchpad exit
// dominates uses (needs dominators).
AllocaInst *WinEHPrepare::insertPHILoads(PHINode *PN, Function &F) {
  BasicBlock *PHIBlock = PN->getParent();
  AllocaInst *SpillSlot = nullptr;

  if (isa<CleanupPadInst>(PHIBlock->getFirstNonPHI())) {
    // Insert a load in place of the PHI and replace all uses.
    SpillSlot = new AllocaInst(PN->getType(), nullptr,
                               Twine(PN->getName(), ".wineh.spillslot"),
                               F.getEntryBlock().begin());
    Value *V = new LoadInst(SpillSlot, Twine(PN->getName(), ".wineh.reload"),
                            PHIBlock->getFirstInsertionPt());
    PN->replaceAllUsesWith(V);
    return SpillSlot;
  }

  DenseMap<BasicBlock *, Value *> Loads;
  for (Value::use_iterator UI = PN->use_begin(), UE = PN->use_end();
       UI != UE;) {
    Use &U = *UI++;
    auto *UsingInst = cast<Instruction>(U.getUser());
    BasicBlock *UsingBB = UsingInst->getParent();
    if (UsingBB->isEHPad()) {
      // Use is on an EH pad phi.  Leave it alone; we'll insert loads and
      // stores for it separately.
      assert(isa<PHINode>(UsingInst));
      continue;
    }
    replaceUseWithLoad(PN, U, SpillSlot, Loads, F);
  }
  return SpillSlot;
}

// TODO: improve store placement.  Inserting at def is probably good, but need
// to be careful not to introduce interfering stores (needs liveness analysis).
// TODO: identify related phi nodes that can share spill slots, and share them
// (also needs liveness).
void WinEHPrepare::insertPHIStores(PHINode *OriginalPHI,
                                   AllocaInst *SpillSlot) {
  // Use a worklist of (Block, Value) pairs -- the given Value needs to be
  // stored to the spill slot by the end of the given Block.
  SmallVector<std::pair<BasicBlock *, Value *>, 4> Worklist;

  Worklist.push_back({OriginalPHI->getParent(), OriginalPHI});

  while (!Worklist.empty()) {
    BasicBlock *EHBlock;
    Value *InVal;
    std::tie(EHBlock, InVal) = Worklist.pop_back_val();

    PHINode *PN = dyn_cast<PHINode>(InVal);
    if (PN && PN->getParent() == EHBlock) {
      // The value is defined by another PHI we need to remove, with no room to
      // insert a store after the PHI, so each predecessor needs to store its
      // incoming value.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i) {
        Value *PredVal = PN->getIncomingValue(i);

        // Undef can safely be skipped.
        if (isa<UndefValue>(PredVal))
          continue;

        insertPHIStore(PN->getIncomingBlock(i), PredVal, SpillSlot, Worklist);
      }
    } else {
      // We need to store InVal, which dominates EHBlock, but can't put a store
      // in EHBlock, so need to put stores in each predecessor.
      for (BasicBlock *PredBlock : predecessors(EHBlock)) {
        insertPHIStore(PredBlock, InVal, SpillSlot, Worklist);
      }
    }
  }
}

void WinEHPrepare::insertPHIStore(
    BasicBlock *PredBlock, Value *PredVal, AllocaInst *SpillSlot,
    SmallVectorImpl<std::pair<BasicBlock *, Value *>> &Worklist) {

  if (PredBlock->isEHPad() &&
      !isa<CleanupPadInst>(PredBlock->getFirstNonPHI())) {
    // Pred is unsplittable, so we need to queue it on the worklist.
    Worklist.push_back({PredBlock, PredVal});
    return;
  }

  // Otherwise, insert the store at the end of the basic block.
  new StoreInst(PredVal, SpillSlot, PredBlock->getTerminator());
}

// TODO: Share loads for same-funclet uses (requires dominators if funclets
// aren't properly nested).
void WinEHPrepare::demoteNonlocalUses(Value *V,
                                      std::set<BasicBlock *> &ColorsForBB,
                                      Function &F) {
  // Tokens can only be used non-locally due to control flow involving
  // unreachable edges.  Don't try to demote the token usage, we'll simply
  // delete the cloned user later.
  if (isa<CatchPadInst>(V) || isa<CleanupPadInst>(V))
    return;

  DenseMap<BasicBlock *, Value *> Loads;
  AllocaInst *SpillSlot = nullptr;
  for (Value::use_iterator UI = V->use_begin(), UE = V->use_end(); UI != UE;) {
    Use &U = *UI++;
    auto *UsingInst = cast<Instruction>(U.getUser());
    BasicBlock *UsingBB = UsingInst->getParent();

    // Is the Use inside a block which is colored the same as the Def?
    // If so, we don't need to escape the Def because we will clone
    // ourselves our own private copy.
    std::set<BasicBlock *> &ColorsForUsingBB = BlockColors[UsingBB];
    if (ColorsForUsingBB == ColorsForBB)
      continue;

    replaceUseWithLoad(V, U, SpillSlot, Loads, F);
  }
  if (SpillSlot) {
    // Insert stores of the computed value into the stack slot.
    // We have to be careful if I is an invoke instruction,
    // because we can't insert the store AFTER the terminator instruction.
    BasicBlock::iterator InsertPt;
    if (isa<Argument>(V)) {
      InsertPt = F.getEntryBlock().getTerminator();
    } else if (isa<TerminatorInst>(V)) {
      auto *II = cast<InvokeInst>(V);
      // We cannot demote invoke instructions to the stack if their normal
      // edge is critical. Therefore, split the critical edge and create a
      // basic block into which the store can be inserted.
      if (!II->getNormalDest()->getSinglePredecessor()) {
        unsigned SuccNum =
            GetSuccessorNumber(II->getParent(), II->getNormalDest());
        assert(isCriticalEdge(II, SuccNum) && "Expected a critical edge!");
        BasicBlock *NewBlock = SplitCriticalEdge(II, SuccNum);
        assert(NewBlock && "Unable to split critical edge.");
        // Update the color mapping for the newly split edge.
        std::set<BasicBlock *> &ColorsForUsingBB = BlockColors[II->getParent()];
        BlockColors[NewBlock] = ColorsForUsingBB;
        for (BasicBlock *FuncletPad : ColorsForUsingBB)
          FuncletBlocks[FuncletPad].insert(NewBlock);
      }
      InsertPt = II->getNormalDest()->getFirstInsertionPt();
    } else {
      InsertPt = cast<Instruction>(V);
      ++InsertPt;
      // Don't insert before PHI nodes or EH pad instrs.
      for (; isa<PHINode>(InsertPt) || InsertPt->isEHPad(); ++InsertPt)
        ;
    }
    new StoreInst(V, SpillSlot, InsertPt);
  }
}

void WinEHPrepare::replaceUseWithLoad(Value *V, Use &U, AllocaInst *&SpillSlot,
                                      DenseMap<BasicBlock *, Value *> &Loads,
                                      Function &F) {
  // Lazilly create the spill slot.
  if (!SpillSlot)
    SpillSlot = new AllocaInst(V->getType(), nullptr,
                               Twine(V->getName(), ".wineh.spillslot"),
                               F.getEntryBlock().begin());

  auto *UsingInst = cast<Instruction>(U.getUser());
  if (auto *UsingPHI = dyn_cast<PHINode>(UsingInst)) {
    // If this is a PHI node, we can't insert a load of the value before
    // the use.  Instead insert the load in the predecessor block
    // corresponding to the incoming value.
    //
    // Note that if there are multiple edges from a basic block to this
    // PHI node that we cannot have multiple loads.  The problem is that
    // the resulting PHI node will have multiple values (from each load)
    // coming in from the same block, which is illegal SSA form.
    // For this reason, we keep track of and reuse loads we insert.
    BasicBlock *IncomingBlock = UsingPHI->getIncomingBlock(U);
    if (auto *CatchRet =
            dyn_cast<CatchReturnInst>(IncomingBlock->getTerminator())) {
      // Putting a load above a catchret and use on the phi would still leave
      // a cross-funclet def/use.  We need to split the edge, change the
      // catchret to target the new block, and put the load there.
      BasicBlock *PHIBlock = UsingInst->getParent();
      BasicBlock *NewBlock = SplitEdge(IncomingBlock, PHIBlock);
      // SplitEdge gives us:
      //   IncomingBlock:
      //     ...
      //     br label %NewBlock
      //   NewBlock:
      //     catchret label %PHIBlock
      // But we need:
      //   IncomingBlock:
      //     ...
      //     catchret label %NewBlock
      //   NewBlock:
      //     br label %PHIBlock
      // So move the terminators to each others' blocks and swap their
      // successors.
      BranchInst *Goto = cast<BranchInst>(IncomingBlock->getTerminator());
      Goto->removeFromParent();
      CatchRet->removeFromParent();
      IncomingBlock->getInstList().push_back(CatchRet);
      NewBlock->getInstList().push_back(Goto);
      Goto->setSuccessor(0, PHIBlock);
      CatchRet->setSuccessor(NewBlock);
      // Update the color mapping for the newly split edge.
      std::set<BasicBlock *> &ColorsForPHIBlock = BlockColors[PHIBlock];
      BlockColors[NewBlock] = ColorsForPHIBlock;
      for (BasicBlock *FuncletPad : ColorsForPHIBlock)
        FuncletBlocks[FuncletPad].insert(NewBlock);
      // Treat the new block as incoming for load insertion.
      IncomingBlock = NewBlock;
    }
    Value *&Load = Loads[IncomingBlock];
    // Insert the load into the predecessor block
    if (!Load)
      Load = new LoadInst(SpillSlot, Twine(V->getName(), ".wineh.reload"),
                          /*Volatile=*/false, IncomingBlock->getTerminator());

    U.set(Load);
  } else {
    // Reload right before the old use.
    auto *Load = new LoadInst(SpillSlot, Twine(V->getName(), ".wineh.reload"),
                              /*Volatile=*/false, UsingInst);
    U.set(Load);
  }
}

void WinEHFuncInfo::addIPToStateRange(const BasicBlock *PadBB,
                                      MCSymbol *InvokeBegin,
                                      MCSymbol *InvokeEnd) {
  assert(PadBB->isEHPad() && EHPadStateMap.count(PadBB->getFirstNonPHI()) &&
         "should get EH pad BB with precomputed state");
  InvokeToStateMap[InvokeBegin] =
      std::make_pair(EHPadStateMap[PadBB->getFirstNonPHI()], InvokeEnd);
}

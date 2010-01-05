//===-- SelectionDAGISel.cpp - Implement the SelectionDAGISel class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAGISel class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "isel"
#include "ScheduleDAGSDNodes.h"
#include "SelectionDAGBuilder.h"
#include "FunctionLoweringInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Constants.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

static cl::opt<bool>
EnableFastISelVerbose("fast-isel-verbose", cl::Hidden,
          cl::desc("Enable verbose messages in the \"fast\" "
                   "instruction selector"));
static cl::opt<bool>
EnableFastISelAbort("fast-isel-abort", cl::Hidden,
          cl::desc("Enable abort calls when \"fast\" instruction fails"));
static cl::opt<bool>
SchedLiveInCopies("schedule-livein-copies", cl::Hidden,
                  cl::desc("Schedule copies of livein registers"),
                  cl::init(false));

#ifndef NDEBUG
static cl::opt<bool>
ViewDAGCombine1("view-dag-combine1-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before the first "
                   "dag combine pass"));
static cl::opt<bool>
ViewLegalizeTypesDAGs("view-legalize-types-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before legalize types"));
static cl::opt<bool>
ViewLegalizeDAGs("view-legalize-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before legalize"));
static cl::opt<bool>
ViewDAGCombine2("view-dag-combine2-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before the second "
                   "dag combine pass"));
static cl::opt<bool>
ViewDAGCombineLT("view-dag-combine-lt-dags", cl::Hidden,
          cl::desc("Pop up a window to show dags before the post legalize types"
                   " dag combine pass"));
static cl::opt<bool>
ViewISelDAGs("view-isel-dags", cl::Hidden,
          cl::desc("Pop up a window to show isel dags as they are selected"));
static cl::opt<bool>
ViewSchedDAGs("view-sched-dags", cl::Hidden,
          cl::desc("Pop up a window to show sched dags as they are processed"));
static cl::opt<bool>
ViewSUnitDAGs("view-sunit-dags", cl::Hidden,
      cl::desc("Pop up a window to show SUnit dags after they are processed"));
#else
static const bool ViewDAGCombine1 = false,
                  ViewLegalizeTypesDAGs = false, ViewLegalizeDAGs = false,
                  ViewDAGCombine2 = false,
                  ViewDAGCombineLT = false,
                  ViewISelDAGs = false, ViewSchedDAGs = false,
                  ViewSUnitDAGs = false;
#endif

//===---------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry RegisterScheduler::Registry;

//===---------------------------------------------------------------------===//
///
/// ISHeuristic command line option for instruction schedulers.
///
//===---------------------------------------------------------------------===//
static cl::opt<RegisterScheduler::FunctionPassCtor, false,
               RegisterPassParser<RegisterScheduler> >
ISHeuristic("pre-RA-sched",
            cl::init(&createDefaultScheduler),
            cl::desc("Instruction schedulers available (before register"
                     " allocation):"));

static RegisterScheduler
defaultListDAGScheduler("default", "Best scheduler for the target",
                        createDefaultScheduler);

namespace llvm {
  //===--------------------------------------------------------------------===//
  /// createDefaultScheduler - This creates an instruction scheduler appropriate
  /// for the target.
  ScheduleDAGSDNodes* createDefaultScheduler(SelectionDAGISel *IS,
                                             CodeGenOpt::Level OptLevel) {
    const TargetLowering &TLI = IS->getTargetLowering();

    if (OptLevel == CodeGenOpt::None)
      return createFastDAGScheduler(IS, OptLevel);
    if (TLI.getSchedulingPreference() == TargetLowering::SchedulingForLatency)
      return createTDListDAGScheduler(IS, OptLevel);
    assert(TLI.getSchedulingPreference() ==
         TargetLowering::SchedulingForRegPressure && "Unknown sched type!");
    return createBURRListDAGScheduler(IS, OptLevel);
  }
}

// EmitInstrWithCustomInserter - This method should be implemented by targets
// that mark instructions with the 'usesCustomInserter' flag.  These
// instructions are special in various ways, which require special support to
// insert.  The specified MachineInstr is created but not inserted into any
// basic blocks, and this method is called to expand it into a sequence of
// instructions, potentially also creating new basic blocks and control flow.
// When new basic blocks are inserted and the edges from MBB to its successors
// are modified, the method should insert pairs of <OldSucc, NewSucc> into the
// DenseMap.
MachineBasicBlock *TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                         MachineBasicBlock *MBB,
                   DenseMap<MachineBasicBlock*, MachineBasicBlock*> *EM) const {
#ifndef NDEBUG
  dbgs() << "If a target marks an instruction with "
          "'usesCustomInserter', it must implement "
          "TargetLowering::EmitInstrWithCustomInserter!";
#endif
  llvm_unreachable(0);
  return 0;
}

/// EmitLiveInCopy - Emit a copy for a live in physical register. If the
/// physical register has only a single copy use, then coalesced the copy
/// if possible.
static void EmitLiveInCopy(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator &InsertPos,
                           unsigned VirtReg, unsigned PhysReg,
                           const TargetRegisterClass *RC,
                           DenseMap<MachineInstr*, unsigned> &CopyRegMap,
                           const MachineRegisterInfo &MRI,
                           const TargetRegisterInfo &TRI,
                           const TargetInstrInfo &TII) {
  unsigned NumUses = 0;
  MachineInstr *UseMI = NULL;
  for (MachineRegisterInfo::use_iterator UI = MRI.use_begin(VirtReg),
         UE = MRI.use_end(); UI != UE; ++UI) {
    UseMI = &*UI;
    if (++NumUses > 1)
      break;
  }

  // If the number of uses is not one, or the use is not a move instruction,
  // don't coalesce. Also, only coalesce away a virtual register to virtual
  // register copy.
  bool Coalesced = false;
  unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
  if (NumUses == 1 &&
      TII.isMoveInstr(*UseMI, SrcReg, DstReg, SrcSubReg, DstSubReg) &&
      TargetRegisterInfo::isVirtualRegister(DstReg)) {
    VirtReg = DstReg;
    Coalesced = true;
  }

  // Now find an ideal location to insert the copy.
  MachineBasicBlock::iterator Pos = InsertPos;
  while (Pos != MBB->begin()) {
    MachineInstr *PrevMI = prior(Pos);
    DenseMap<MachineInstr*, unsigned>::iterator RI = CopyRegMap.find(PrevMI);
    // copyRegToReg might emit multiple instructions to do a copy.
    unsigned CopyDstReg = (RI == CopyRegMap.end()) ? 0 : RI->second;
    if (CopyDstReg && !TRI.regsOverlap(CopyDstReg, PhysReg))
      // This is what the BB looks like right now:
      // r1024 = mov r0
      // ...
      // r1    = mov r1024
      //
      // We want to insert "r1025 = mov r1". Inserting this copy below the
      // move to r1024 makes it impossible for that move to be coalesced.
      //
      // r1025 = mov r1
      // r1024 = mov r0
      // ...
      // r1    = mov 1024
      // r2    = mov 1025
      break; // Woot! Found a good location.
    --Pos;
  }

  bool Emitted = TII.copyRegToReg(*MBB, Pos, VirtReg, PhysReg, RC, RC);
  assert(Emitted && "Unable to issue a live-in copy instruction!\n");
  (void) Emitted;

  CopyRegMap.insert(std::make_pair(prior(Pos), VirtReg));
  if (Coalesced) {
    if (&*InsertPos == UseMI) ++InsertPos;
    MBB->erase(UseMI);
  }
}

/// EmitLiveInCopies - If this is the first basic block in the function,
/// and if it has live ins that need to be copied into vregs, emit the
/// copies into the block.
static void EmitLiveInCopies(MachineBasicBlock *EntryMBB,
                             const MachineRegisterInfo &MRI,
                             const TargetRegisterInfo &TRI,
                             const TargetInstrInfo &TII) {
  if (SchedLiveInCopies) {
    // Emit the copies at a heuristically-determined location in the block.
    DenseMap<MachineInstr*, unsigned> CopyRegMap;
    MachineBasicBlock::iterator InsertPos = EntryMBB->begin();
    for (MachineRegisterInfo::livein_iterator LI = MRI.livein_begin(),
           E = MRI.livein_end(); LI != E; ++LI)
      if (LI->second) {
        const TargetRegisterClass *RC = MRI.getRegClass(LI->second);
        EmitLiveInCopy(EntryMBB, InsertPos, LI->second, LI->first,
                       RC, CopyRegMap, MRI, TRI, TII);
      }
  } else {
    // Emit the copies into the top of the block.
    for (MachineRegisterInfo::livein_iterator LI = MRI.livein_begin(),
           E = MRI.livein_end(); LI != E; ++LI)
      if (LI->second) {
        const TargetRegisterClass *RC = MRI.getRegClass(LI->second);
        bool Emitted = TII.copyRegToReg(*EntryMBB, EntryMBB->begin(),
                                        LI->second, LI->first, RC, RC);
        assert(Emitted && "Unable to issue a live-in copy instruction!\n");
        (void) Emitted;
      }
  }
}

//===----------------------------------------------------------------------===//
// SelectionDAGISel code
//===----------------------------------------------------------------------===//

SelectionDAGISel::SelectionDAGISel(TargetMachine &tm, CodeGenOpt::Level OL) :
  MachineFunctionPass(&ID), TM(tm), TLI(*tm.getTargetLowering()),
  FuncInfo(new FunctionLoweringInfo(TLI)),
  CurDAG(new SelectionDAG(TLI, *FuncInfo)),
  SDB(new SelectionDAGBuilder(*CurDAG, TLI, *FuncInfo, OL)),
  GFI(),
  OptLevel(OL),
  DAGSize(0)
{}

SelectionDAGISel::~SelectionDAGISel() {
  delete SDB;
  delete CurDAG;
  delete FuncInfo;
}

unsigned SelectionDAGISel::MakeReg(EVT VT) {
  return RegInfo->createVirtualRegister(TLI.getRegClassFor(VT));
}

void SelectionDAGISel::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addRequired<GCModuleInfo>();
  AU.addPreserved<GCModuleInfo>();
  AU.addRequired<DwarfWriter>();
  AU.addPreserved<DwarfWriter>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool SelectionDAGISel::runOnMachineFunction(MachineFunction &mf) {
  Function &Fn = *mf.getFunction();

  // Do some sanity-checking on the command-line options.
  assert((!EnableFastISelVerbose || EnableFastISel) &&
         "-fast-isel-verbose requires -fast-isel");
  assert((!EnableFastISelAbort || EnableFastISel) &&
         "-fast-isel-abort requires -fast-isel");

  // Get alias analysis for load/store combining.
  AA = &getAnalysis<AliasAnalysis>();

  MF = &mf;
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const TargetRegisterInfo &TRI = *TM.getRegisterInfo();

  if (Fn.hasGC())
    GFI = &getAnalysis<GCModuleInfo>().getFunctionInfo(Fn);
  else
    GFI = 0;
  RegInfo = &MF->getRegInfo();
  DEBUG(dbgs() << "\n\n\n=== " << Fn.getName() << "\n");

  MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  DwarfWriter *DW = getAnalysisIfAvailable<DwarfWriter>();
  CurDAG->init(*MF, MMI, DW);
  FuncInfo->set(Fn, *MF, EnableFastISel);
  SDB->init(GFI, *AA);

  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    if (InvokeInst *Invoke = dyn_cast<InvokeInst>(I->getTerminator()))
      // Mark landing pad.
      FuncInfo->MBBMap[Invoke->getSuccessor(1)]->setIsLandingPad();

  SelectAllBasicBlocks(Fn, *MF, MMI, DW, TII);

  // If the first basic block in the function has live ins that need to be
  // copied into vregs, emit the copies into the top of the block before
  // emitting the code for the block.
  EmitLiveInCopies(MF->begin(), *RegInfo, TRI, TII);

  // Add function live-ins to entry block live-in set.
  for (MachineRegisterInfo::livein_iterator I = RegInfo->livein_begin(),
         E = RegInfo->livein_end(); I != E; ++I)
    MF->begin()->addLiveIn(I->first);

#ifndef NDEBUG
  assert(FuncInfo->CatchInfoFound.size() == FuncInfo->CatchInfoLost.size() &&
         "Not all catch info was assigned to a landing pad!");
#endif

  FuncInfo->clear();

  return true;
}

/// SetDebugLoc - Update MF's and SDB's DebugLocs if debug information is
/// attached with this instruction.
static void SetDebugLoc(unsigned MDDbgKind, Instruction *I,
                        SelectionDAGBuilder *SDB,
                        FastISel *FastIS, MachineFunction *MF) {
  if (isa<DbgInfoIntrinsic>(I)) return;
  
  if (MDNode *Dbg = I->getMetadata(MDDbgKind)) {
    DILocation DILoc(Dbg);
    DebugLoc Loc = ExtractDebugLocation(DILoc, MF->getDebugLocInfo());

    SDB->setCurDebugLoc(Loc);

    if (FastIS)
      FastIS->setCurDebugLoc(Loc);

    // If the function doesn't have a default debug location yet, set
    // it. This is kind of a hack.
    if (MF->getDefaultDebugLoc().isUnknown())
      MF->setDefaultDebugLoc(Loc);
  }
}

/// ResetDebugLoc - Set MF's and SDB's DebugLocs to Unknown.
static void ResetDebugLoc(SelectionDAGBuilder *SDB, FastISel *FastIS) {
  SDB->setCurDebugLoc(DebugLoc::getUnknownLoc());
  if (FastIS)
    FastIS->setCurDebugLoc(DebugLoc::getUnknownLoc());
}

void SelectionDAGISel::SelectBasicBlock(BasicBlock *LLVMBB,
                                        BasicBlock::iterator Begin,
                                        BasicBlock::iterator End,
                                        bool &HadTailCall) {
  SDB->setCurrentBasicBlock(BB);
  unsigned MDDbgKind = LLVMBB->getContext().getMDKindID("dbg");

  // Lower all of the non-terminator instructions. If a call is emitted
  // as a tail call, cease emitting nodes for this block.
  for (BasicBlock::iterator I = Begin; I != End && !SDB->HasTailCall; ++I) {
    SetDebugLoc(MDDbgKind, I, SDB, 0, MF);

    if (!isa<TerminatorInst>(I)) {
      SDB->visit(*I);

      // Set the current debug location back to "unknown" so that it doesn't
      // spuriously apply to subsequent instructions.
      ResetDebugLoc(SDB, 0);
    }
  }

  if (!SDB->HasTailCall) {
    // Ensure that all instructions which are used outside of their defining
    // blocks are available as virtual registers.  Invoke is handled elsewhere.
    for (BasicBlock::iterator I = Begin; I != End; ++I)
      if (!isa<PHINode>(I) && !isa<InvokeInst>(I))
        SDB->CopyToExportRegsIfNeeded(I);

    // Handle PHI nodes in successor blocks.
    if (End == LLVMBB->end()) {
      HandlePHINodesInSuccessorBlocks(LLVMBB);

      // Lower the terminator after the copies are emitted.
      SetDebugLoc(MDDbgKind, LLVMBB->getTerminator(), SDB, 0, MF);
      SDB->visit(*LLVMBB->getTerminator());
      ResetDebugLoc(SDB, 0);
    }
  }

  // Make sure the root of the DAG is up-to-date.
  CurDAG->setRoot(SDB->getControlRoot());

  // Final step, emit the lowered DAG as machine code.
  CodeGenAndEmitDAG();
  HadTailCall = SDB->HasTailCall;
  SDB->clear();
}

void SelectionDAGISel::ComputeLiveOutVRegInfo() {
  SmallPtrSet<SDNode*, 128> VisitedNodes;
  SmallVector<SDNode*, 128> Worklist;

  Worklist.push_back(CurDAG->getRoot().getNode());

  APInt Mask;
  APInt KnownZero;
  APInt KnownOne;

  while (!Worklist.empty()) {
    SDNode *N = Worklist.back();
    Worklist.pop_back();

    // If we've already seen this node, ignore it.
    if (!VisitedNodes.insert(N))
      continue;

    // Otherwise, add all chain operands to the worklist.
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
      if (N->getOperand(i).getValueType() == MVT::Other)
        Worklist.push_back(N->getOperand(i).getNode());

    // If this is a CopyToReg with a vreg dest, process it.
    if (N->getOpcode() != ISD::CopyToReg)
      continue;

    unsigned DestReg = cast<RegisterSDNode>(N->getOperand(1))->getReg();
    if (!TargetRegisterInfo::isVirtualRegister(DestReg))
      continue;

    // Ignore non-scalar or non-integer values.
    SDValue Src = N->getOperand(2);
    EVT SrcVT = Src.getValueType();
    if (!SrcVT.isInteger() || SrcVT.isVector())
      continue;

    unsigned NumSignBits = CurDAG->ComputeNumSignBits(Src);
    Mask = APInt::getAllOnesValue(SrcVT.getSizeInBits());
    CurDAG->ComputeMaskedBits(Src, Mask, KnownZero, KnownOne);

    // Only install this information if it tells us something.
    if (NumSignBits != 1 || KnownZero != 0 || KnownOne != 0) {
      DestReg -= TargetRegisterInfo::FirstVirtualRegister;
      if (DestReg >= FuncInfo->LiveOutRegInfo.size())
        FuncInfo->LiveOutRegInfo.resize(DestReg+1);
      FunctionLoweringInfo::LiveOutInfo &LOI =
        FuncInfo->LiveOutRegInfo[DestReg];
      LOI.NumSignBits = NumSignBits;
      LOI.KnownOne = KnownOne;
      LOI.KnownZero = KnownZero;
    }
  }
}

void SelectionDAGISel::CodeGenAndEmitDAG() {
  std::string GroupName;
  if (TimePassesIsEnabled)
    GroupName = "Instruction Selection and Scheduling";
  std::string BlockName;
  if (ViewDAGCombine1 || ViewLegalizeTypesDAGs || ViewLegalizeDAGs ||
      ViewDAGCombine2 || ViewDAGCombineLT || ViewISelDAGs || ViewSchedDAGs ||
      ViewSUnitDAGs)
    BlockName = MF->getFunction()->getNameStr() + ":" +
                BB->getBasicBlock()->getNameStr();

  DEBUG(dbgs() << "Initial selection DAG:\n");
  DEBUG(CurDAG->dump());

  if (ViewDAGCombine1) CurDAG->viewGraph("dag-combine1 input for " + BlockName);

  // Run the DAG combiner in pre-legalize mode.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("DAG Combining 1", GroupName);
    CurDAG->Combine(Unrestricted, *AA, OptLevel);
  } else {
    CurDAG->Combine(Unrestricted, *AA, OptLevel);
  }

  DEBUG(dbgs() << "Optimized lowered selection DAG:\n");
  DEBUG(CurDAG->dump());

  // Second step, hack on the DAG until it only uses operations and types that
  // the target supports.
  if (ViewLegalizeTypesDAGs) CurDAG->viewGraph("legalize-types input for " +
                                               BlockName);

  bool Changed;
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Type Legalization", GroupName);
    Changed = CurDAG->LegalizeTypes();
  } else {
    Changed = CurDAG->LegalizeTypes();
  }

  DEBUG(dbgs() << "Type-legalized selection DAG:\n");
  DEBUG(CurDAG->dump());

  if (Changed) {
    if (ViewDAGCombineLT)
      CurDAG->viewGraph("dag-combine-lt input for " + BlockName);

    // Run the DAG combiner in post-type-legalize mode.
    if (TimePassesIsEnabled) {
      NamedRegionTimer T("DAG Combining after legalize types", GroupName);
      CurDAG->Combine(NoIllegalTypes, *AA, OptLevel);
    } else {
      CurDAG->Combine(NoIllegalTypes, *AA, OptLevel);
    }

    DEBUG(dbgs() << "Optimized type-legalized selection DAG:\n");
    DEBUG(CurDAG->dump());
  }

  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Vector Legalization", GroupName);
    Changed = CurDAG->LegalizeVectors();
  } else {
    Changed = CurDAG->LegalizeVectors();
  }

  if (Changed) {
    if (TimePassesIsEnabled) {
      NamedRegionTimer T("Type Legalization 2", GroupName);
      CurDAG->LegalizeTypes();
    } else {
      CurDAG->LegalizeTypes();
    }

    if (ViewDAGCombineLT)
      CurDAG->viewGraph("dag-combine-lv input for " + BlockName);

    // Run the DAG combiner in post-type-legalize mode.
    if (TimePassesIsEnabled) {
      NamedRegionTimer T("DAG Combining after legalize vectors", GroupName);
      CurDAG->Combine(NoIllegalOperations, *AA, OptLevel);
    } else {
      CurDAG->Combine(NoIllegalOperations, *AA, OptLevel);
    }

    DEBUG(dbgs() << "Optimized vector-legalized selection DAG:\n");
    DEBUG(CurDAG->dump());
  }

  if (ViewLegalizeDAGs) CurDAG->viewGraph("legalize input for " + BlockName);

  if (TimePassesIsEnabled) {
    NamedRegionTimer T("DAG Legalization", GroupName);
    CurDAG->Legalize(OptLevel);
  } else {
    CurDAG->Legalize(OptLevel);
  }

  DEBUG(dbgs() << "Legalized selection DAG:\n");
  DEBUG(CurDAG->dump());

  if (ViewDAGCombine2) CurDAG->viewGraph("dag-combine2 input for " + BlockName);

  // Run the DAG combiner in post-legalize mode.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("DAG Combining 2", GroupName);
    CurDAG->Combine(NoIllegalOperations, *AA, OptLevel);
  } else {
    CurDAG->Combine(NoIllegalOperations, *AA, OptLevel);
  }

  DEBUG(dbgs() << "Optimized legalized selection DAG:\n");
  DEBUG(CurDAG->dump());

  if (ViewISelDAGs) CurDAG->viewGraph("isel input for " + BlockName);

  if (OptLevel != CodeGenOpt::None)
    ComputeLiveOutVRegInfo();

  // Third, instruction select all of the operations to machine code, adding the
  // code to the MachineBasicBlock.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Instruction Selection", GroupName);
    InstructionSelect();
  } else {
    InstructionSelect();
  }

  DEBUG(dbgs() << "Selected selection DAG:\n");
  DEBUG(CurDAG->dump());

  if (ViewSchedDAGs) CurDAG->viewGraph("scheduler input for " + BlockName);

  // Schedule machine code.
  ScheduleDAGSDNodes *Scheduler = CreateScheduler();
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Instruction Scheduling", GroupName);
    Scheduler->Run(CurDAG, BB, BB->end());
  } else {
    Scheduler->Run(CurDAG, BB, BB->end());
  }

  if (ViewSUnitDAGs) Scheduler->viewGraph();

  // Emit machine code to BB.  This can change 'BB' to the last block being
  // inserted into.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Instruction Creation", GroupName);
    BB = Scheduler->EmitSchedule(&SDB->EdgeMapping);
  } else {
    BB = Scheduler->EmitSchedule(&SDB->EdgeMapping);
  }

  // Free the scheduler state.
  if (TimePassesIsEnabled) {
    NamedRegionTimer T("Instruction Scheduling Cleanup", GroupName);
    delete Scheduler;
  } else {
    delete Scheduler;
  }

  DEBUG(dbgs() << "Selected machine code:\n");
  DEBUG(BB->dump());
}

void SelectionDAGISel::SelectAllBasicBlocks(Function &Fn,
                                            MachineFunction &MF,
                                            MachineModuleInfo *MMI,
                                            DwarfWriter *DW,
                                            const TargetInstrInfo &TII) {
  // Initialize the Fast-ISel state, if needed.
  FastISel *FastIS = 0;
  if (EnableFastISel)
    FastIS = TLI.createFastISel(MF, MMI, DW,
                                FuncInfo->ValueMap,
                                FuncInfo->MBBMap,
                                FuncInfo->StaticAllocaMap
#ifndef NDEBUG
                                , FuncInfo->CatchInfoLost
#endif
                                );

  unsigned MDDbgKind = Fn.getContext().getMDKindID("dbg");

  // Iterate over all basic blocks in the function.
  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    BasicBlock *LLVMBB = &*I;
    BB = FuncInfo->MBBMap[LLVMBB];

    BasicBlock::iterator const Begin = LLVMBB->begin();
    BasicBlock::iterator const End = LLVMBB->end();
    BasicBlock::iterator BI = Begin;

    // Lower any arguments needed in this block if this is the entry block.
    bool SuppressFastISel = false;
    if (LLVMBB == &Fn.getEntryBlock()) {
      LowerArguments(LLVMBB);

      // If any of the arguments has the byval attribute, forgo
      // fast-isel in the entry block.
      if (FastIS) {
        unsigned j = 1;
        for (Function::arg_iterator I = Fn.arg_begin(), E = Fn.arg_end();
             I != E; ++I, ++j)
          if (Fn.paramHasAttr(j, Attribute::ByVal)) {
            if (EnableFastISelVerbose || EnableFastISelAbort)
              dbgs() << "FastISel skips entry block due to byval argument\n";
            SuppressFastISel = true;
            break;
          }
      }
    }

    if (MMI && BB->isLandingPad()) {
      // Add a label to mark the beginning of the landing pad.  Deletion of the
      // landing pad can thus be detected via the MachineModuleInfo.
      unsigned LabelID = MMI->addLandingPad(BB);

      const TargetInstrDesc &II = TII.get(TargetInstrInfo::EH_LABEL);
      BuildMI(BB, SDB->getCurDebugLoc(), II).addImm(LabelID);

      // Mark exception register as live in.
      unsigned Reg = TLI.getExceptionAddressRegister();
      if (Reg) BB->addLiveIn(Reg);

      // Mark exception selector register as live in.
      Reg = TLI.getExceptionSelectorRegister();
      if (Reg) BB->addLiveIn(Reg);

      // FIXME: Hack around an exception handling flaw (PR1508): the personality
      // function and list of typeids logically belong to the invoke (or, if you
      // like, the basic block containing the invoke), and need to be associated
      // with it in the dwarf exception handling tables.  Currently however the
      // information is provided by an intrinsic (eh.selector) that can be moved
      // to unexpected places by the optimizers: if the unwind edge is critical,
      // then breaking it can result in the intrinsics being in the successor of
      // the landing pad, not the landing pad itself.  This results in exceptions
      // not being caught because no typeids are associated with the invoke.
      // This may not be the only way things can go wrong, but it is the only way
      // we try to work around for the moment.
      BranchInst *Br = dyn_cast<BranchInst>(LLVMBB->getTerminator());

      if (Br && Br->isUnconditional()) { // Critical edge?
        BasicBlock::iterator I, E;
        for (I = LLVMBB->begin(), E = --LLVMBB->end(); I != E; ++I)
          if (isa<EHSelectorInst>(I))
            break;

        if (I == E)
          // No catch info found - try to extract some from the successor.
          CopyCatchInfo(Br->getSuccessor(0), LLVMBB, MMI, *FuncInfo);
      }
    }

    // Before doing SelectionDAG ISel, see if FastISel has been requested.
    if (FastIS && !SuppressFastISel) {
      // Emit code for any incoming arguments. This must happen before
      // beginning FastISel on the entry block.
      if (LLVMBB == &Fn.getEntryBlock()) {
        CurDAG->setRoot(SDB->getControlRoot());
        CodeGenAndEmitDAG();
        SDB->clear();
      }
      FastIS->startNewBlock(BB);
      // Do FastISel on as many instructions as possible.
      for (; BI != End; ++BI) {
        // Just before the terminator instruction, insert instructions to
        // feed PHI nodes in successor blocks.
        if (isa<TerminatorInst>(BI))
          if (!HandlePHINodesInSuccessorBlocksFast(LLVMBB, FastIS)) {
            ResetDebugLoc(SDB, FastIS);
            if (EnableFastISelVerbose || EnableFastISelAbort) {
              dbgs() << "FastISel miss: ";
              BI->dump();
            }
            assert(!EnableFastISelAbort &&
                   "FastISel didn't handle a PHI in a successor");
            break;
          }

        SetDebugLoc(MDDbgKind, BI, SDB, FastIS, &MF);

        // First try normal tablegen-generated "fast" selection.
        if (FastIS->SelectInstruction(BI)) {
          ResetDebugLoc(SDB, FastIS);
          continue;
        }

        // Clear out the debug location so that it doesn't carry over to
        // unrelated instructions.
        ResetDebugLoc(SDB, FastIS);

        // Then handle certain instructions as single-LLVM-Instruction blocks.
        if (isa<CallInst>(BI)) {
          if (EnableFastISelVerbose || EnableFastISelAbort) {
            dbgs() << "FastISel missed call: ";
            BI->dump();
          }

          if (!BI->getType()->isVoidTy()) {
            unsigned &R = FuncInfo->ValueMap[BI];
            if (!R)
              R = FuncInfo->CreateRegForValue(BI);
          }

          bool HadTailCall = false;
          SelectBasicBlock(LLVMBB, BI, llvm::next(BI), HadTailCall);

          // If the call was emitted as a tail call, we're done with the block.
          if (HadTailCall) {
            BI = End;
            break;
          }

          // If the instruction was codegen'd with multiple blocks,
          // inform the FastISel object where to resume inserting.
          FastIS->setCurrentBlock(BB);
          continue;
        }

        // Otherwise, give up on FastISel for the rest of the block.
        // For now, be a little lenient about non-branch terminators.
        if (!isa<TerminatorInst>(BI) || isa<BranchInst>(BI)) {
          if (EnableFastISelVerbose || EnableFastISelAbort) {
            dbgs() << "FastISel miss: ";
            BI->dump();
          }
          if (EnableFastISelAbort)
            // The "fast" selector couldn't handle something and bailed.
            // For the purpose of debugging, just abort.
            llvm_unreachable("FastISel didn't select the entire block");
        }
        break;
      }
    }

    // Run SelectionDAG instruction selection on the remainder of the block
    // not handled by FastISel. If FastISel is not run, this is the entire
    // block.
    if (BI != End) {
      bool HadTailCall;
      SelectBasicBlock(LLVMBB, BI, End, HadTailCall);
    }

    FinishBasicBlock();
  }

  delete FastIS;
}

void
SelectionDAGISel::FinishBasicBlock() {

  DEBUG(dbgs() << "Target-post-processed machine code:\n");
  DEBUG(BB->dump());

  DEBUG(dbgs() << "Total amount of phi nodes to update: "
               << SDB->PHINodesToUpdate.size() << "\n");
  DEBUG(for (unsigned i = 0, e = SDB->PHINodesToUpdate.size(); i != e; ++i)
          dbgs() << "Node " << i << " : ("
                 << SDB->PHINodesToUpdate[i].first
                 << ", " << SDB->PHINodesToUpdate[i].second << ")\n");

  // Next, now that we know what the last MBB the LLVM BB expanded is, update
  // PHI nodes in successors.
  if (SDB->SwitchCases.empty() &&
      SDB->JTCases.empty() &&
      SDB->BitTestCases.empty()) {
    for (unsigned i = 0, e = SDB->PHINodesToUpdate.size(); i != e; ++i) {
      MachineInstr *PHI = SDB->PHINodesToUpdate[i].first;
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      PHI->addOperand(MachineOperand::CreateReg(SDB->PHINodesToUpdate[i].second,
                                                false));
      PHI->addOperand(MachineOperand::CreateMBB(BB));
    }
    SDB->PHINodesToUpdate.clear();
    return;
  }

  for (unsigned i = 0, e = SDB->BitTestCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!SDB->BitTestCases[i].Emitted) {
      // Set the current basic block to the mbb we wish to insert the code into
      BB = SDB->BitTestCases[i].Parent;
      SDB->setCurrentBasicBlock(BB);
      // Emit the code
      SDB->visitBitTestHeader(SDB->BitTestCases[i]);
      CurDAG->setRoot(SDB->getRoot());
      CodeGenAndEmitDAG();
      SDB->clear();
    }

    for (unsigned j = 0, ej = SDB->BitTestCases[i].Cases.size(); j != ej; ++j) {
      // Set the current basic block to the mbb we wish to insert the code into
      BB = SDB->BitTestCases[i].Cases[j].ThisBB;
      SDB->setCurrentBasicBlock(BB);
      // Emit the code
      if (j+1 != ej)
        SDB->visitBitTestCase(SDB->BitTestCases[i].Cases[j+1].ThisBB,
                              SDB->BitTestCases[i].Reg,
                              SDB->BitTestCases[i].Cases[j]);
      else
        SDB->visitBitTestCase(SDB->BitTestCases[i].Default,
                              SDB->BitTestCases[i].Reg,
                              SDB->BitTestCases[i].Cases[j]);


      CurDAG->setRoot(SDB->getRoot());
      CodeGenAndEmitDAG();
      SDB->clear();
    }

    // Update PHI Nodes
    for (unsigned pi = 0, pe = SDB->PHINodesToUpdate.size(); pi != pe; ++pi) {
      MachineInstr *PHI = SDB->PHINodesToUpdate[pi].first;
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      // This is "default" BB. We have two jumps to it. From "header" BB and
      // from last "case" BB.
      if (PHIBB == SDB->BitTestCases[i].Default) {
        PHI->addOperand(MachineOperand::CreateReg(SDB->PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(SDB->BitTestCases[i].Parent));
        PHI->addOperand(MachineOperand::CreateReg(SDB->PHINodesToUpdate[pi].second,
                                                  false));
        PHI->addOperand(MachineOperand::CreateMBB(SDB->BitTestCases[i].Cases.
                                                  back().ThisBB));
      }
      // One of "cases" BB.
      for (unsigned j = 0, ej = SDB->BitTestCases[i].Cases.size();
           j != ej; ++j) {
        MachineBasicBlock* cBB = SDB->BitTestCases[i].Cases[j].ThisBB;
        if (cBB->succ_end() !=
            std::find(cBB->succ_begin(),cBB->succ_end(), PHIBB)) {
          PHI->addOperand(MachineOperand::CreateReg(SDB->PHINodesToUpdate[pi].second,
                                                    false));
          PHI->addOperand(MachineOperand::CreateMBB(cBB));
        }
      }
    }
  }
  SDB->BitTestCases.clear();

  // If the JumpTable record is filled in, then we need to emit a jump table.
  // Updating the PHI nodes is tricky in this case, since we need to determine
  // whether the PHI is a successor of the range check MBB or the jump table MBB
  for (unsigned i = 0, e = SDB->JTCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!SDB->JTCases[i].first.Emitted) {
      // Set the current basic block to the mbb we wish to insert the code into
      BB = SDB->JTCases[i].first.HeaderBB;
      SDB->setCurrentBasicBlock(BB);
      // Emit the code
      SDB->visitJumpTableHeader(SDB->JTCases[i].second, SDB->JTCases[i].first);
      CurDAG->setRoot(SDB->getRoot());
      CodeGenAndEmitDAG();
      SDB->clear();
    }

    // Set the current basic block to the mbb we wish to insert the code into
    BB = SDB->JTCases[i].second.MBB;
    SDB->setCurrentBasicBlock(BB);
    // Emit the code
    SDB->visitJumpTable(SDB->JTCases[i].second);
    CurDAG->setRoot(SDB->getRoot());
    CodeGenAndEmitDAG();
    SDB->clear();

    // Update PHI Nodes
    for (unsigned pi = 0, pe = SDB->PHINodesToUpdate.size(); pi != pe; ++pi) {
      MachineInstr *PHI = SDB->PHINodesToUpdate[pi].first;
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      // "default" BB. We can go there only from header BB.
      if (PHIBB == SDB->JTCases[i].second.Default) {
        PHI->addOperand
          (MachineOperand::CreateReg(SDB->PHINodesToUpdate[pi].second, false));
        PHI->addOperand
          (MachineOperand::CreateMBB(SDB->JTCases[i].first.HeaderBB));
      }
      // JT BB. Just iterate over successors here
      if (BB->succ_end() != std::find(BB->succ_begin(),BB->succ_end(), PHIBB)) {
        PHI->addOperand
          (MachineOperand::CreateReg(SDB->PHINodesToUpdate[pi].second, false));
        PHI->addOperand(MachineOperand::CreateMBB(BB));
      }
    }
  }
  SDB->JTCases.clear();

  // If the switch block involved a branch to one of the actual successors, we
  // need to update PHI nodes in that block.
  for (unsigned i = 0, e = SDB->PHINodesToUpdate.size(); i != e; ++i) {
    MachineInstr *PHI = SDB->PHINodesToUpdate[i].first;
    assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
           "This is not a machine PHI node that we are updating!");
    if (BB->isSuccessor(PHI->getParent())) {
      PHI->addOperand(MachineOperand::CreateReg(SDB->PHINodesToUpdate[i].second,
                                                false));
      PHI->addOperand(MachineOperand::CreateMBB(BB));
    }
  }

  // If we generated any switch lowering information, build and codegen any
  // additional DAGs necessary.
  for (unsigned i = 0, e = SDB->SwitchCases.size(); i != e; ++i) {
    // Set the current basic block to the mbb we wish to insert the code into
    MachineBasicBlock *ThisBB = BB = SDB->SwitchCases[i].ThisBB;
    SDB->setCurrentBasicBlock(BB);

    // Emit the code
    SDB->visitSwitchCase(SDB->SwitchCases[i]);
    CurDAG->setRoot(SDB->getRoot());
    CodeGenAndEmitDAG();

    // Handle any PHI nodes in successors of this chunk, as if we were coming
    // from the original BB before switch expansion.  Note that PHI nodes can
    // occur multiple times in PHINodesToUpdate.  We have to be very careful to
    // handle them the right number of times.
    while ((BB = SDB->SwitchCases[i].TrueBB)) {  // Handle LHS and RHS.
      // If new BB's are created during scheduling, the edges may have been
      // updated. That is, the edge from ThisBB to BB may have been split and
      // BB's predecessor is now another block.
      DenseMap<MachineBasicBlock*, MachineBasicBlock*>::iterator EI =
        SDB->EdgeMapping.find(BB);
      if (EI != SDB->EdgeMapping.end())
        ThisBB = EI->second;
      for (MachineBasicBlock::iterator Phi = BB->begin();
           Phi != BB->end() && Phi->getOpcode() == TargetInstrInfo::PHI; ++Phi){
        // This value for this PHI node is recorded in PHINodesToUpdate, get it.
        for (unsigned pn = 0; ; ++pn) {
          assert(pn != SDB->PHINodesToUpdate.size() &&
                 "Didn't find PHI entry!");
          if (SDB->PHINodesToUpdate[pn].first == Phi) {
            Phi->addOperand(MachineOperand::CreateReg(SDB->PHINodesToUpdate[pn].
                                                      second, false));
            Phi->addOperand(MachineOperand::CreateMBB(ThisBB));
            break;
          }
        }
      }

      // Don't process RHS if same block as LHS.
      if (BB == SDB->SwitchCases[i].FalseBB)
        SDB->SwitchCases[i].FalseBB = 0;

      // If we haven't handled the RHS, do so now.  Otherwise, we're done.
      SDB->SwitchCases[i].TrueBB = SDB->SwitchCases[i].FalseBB;
      SDB->SwitchCases[i].FalseBB = 0;
    }
    assert(SDB->SwitchCases[i].TrueBB == 0 && SDB->SwitchCases[i].FalseBB == 0);
    SDB->clear();
  }
  SDB->SwitchCases.clear();

  SDB->PHINodesToUpdate.clear();
}


/// Create the scheduler. If a specific scheduler was specified
/// via the SchedulerRegistry, use it, otherwise select the
/// one preferred by the target.
///
ScheduleDAGSDNodes *SelectionDAGISel::CreateScheduler() {
  RegisterScheduler::FunctionPassCtor Ctor = RegisterScheduler::getDefault();

  if (!Ctor) {
    Ctor = ISHeuristic;
    RegisterScheduler::setDefault(Ctor);
  }

  return Ctor(this, OptLevel);
}

ScheduleHazardRecognizer *SelectionDAGISel::CreateTargetHazardRecognizer() {
  return new ScheduleHazardRecognizer();
}

//===----------------------------------------------------------------------===//
// Helper functions used by the generated instruction selector.
//===----------------------------------------------------------------------===//
// Calls to these methods are generated by tblgen.

/// CheckAndMask - The isel is trying to match something like (and X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an AND, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckAndMask(SDValue LHS, ConstantSDNode *RHS,
                                    int64_t DesiredMaskS) const {
  const APInt &ActualMask = RHS->getAPIntValue();
  const APInt &DesiredMask = APInt(LHS.getValueSizeInBits(), DesiredMaskS);

  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;

  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask.intersects(~DesiredMask))
    return false;

  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  APInt NeededMask = DesiredMask & ~ActualMask;
  if (CurDAG->MaskedValueIsZero(LHS, NeededMask))
    return true;

  // TODO: check to see if missing bits are just not demanded.

  // Otherwise, this pattern doesn't match.
  return false;
}

/// CheckOrMask - The isel is trying to match something like (or X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an OR, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckOrMask(SDValue LHS, ConstantSDNode *RHS,
                                   int64_t DesiredMaskS) const {
  const APInt &ActualMask = RHS->getAPIntValue();
  const APInt &DesiredMask = APInt(LHS.getValueSizeInBits(), DesiredMaskS);

  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;

  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask.intersects(~DesiredMask))
    return false;

  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  APInt NeededMask = DesiredMask & ~ActualMask;

  APInt KnownZero, KnownOne;
  CurDAG->ComputeMaskedBits(LHS, NeededMask, KnownZero, KnownOne);

  // If all the missing bits in the or are already known to be set, match!
  if ((NeededMask & KnownOne) == NeededMask)
    return true;

  // TODO: check to see if missing bits are just not demanded.

  // Otherwise, this pattern doesn't match.
  return false;
}


/// SelectInlineAsmMemoryOperands - Calls to this are automatically generated
/// by tblgen.  Others should not call it.
void SelectionDAGISel::
SelectInlineAsmMemoryOperands(std::vector<SDValue> &Ops) {
  std::vector<SDValue> InOps;
  std::swap(InOps, Ops);

  Ops.push_back(InOps[0]);  // input chain.
  Ops.push_back(InOps[1]);  // input asm string.

  unsigned i = 2, e = InOps.size();
  if (InOps[e-1].getValueType() == MVT::Flag)
    --e;  // Don't process a flag operand if it is here.

  while (i != e) {
    unsigned Flags = cast<ConstantSDNode>(InOps[i])->getZExtValue();
    if ((Flags & 7) != 4 /*MEM*/) {
      // Just skip over this operand, copying the operands verbatim.
      Ops.insert(Ops.end(), InOps.begin()+i,
                 InOps.begin()+i+InlineAsm::getNumOperandRegisters(Flags) + 1);
      i += InlineAsm::getNumOperandRegisters(Flags) + 1;
    } else {
      assert(InlineAsm::getNumOperandRegisters(Flags) == 1 &&
             "Memory operand with multiple values?");
      // Otherwise, this is a memory operand.  Ask the target to select it.
      std::vector<SDValue> SelOps;
      if (SelectInlineAsmMemoryOperand(InOps[i+1], 'm', SelOps)) {
        llvm_report_error("Could not match memory address.  Inline asm"
                          " failure!");
      }

      // Add this to the output node.
      Ops.push_back(CurDAG->getTargetConstant(4/*MEM*/ | (SelOps.size()<< 3),
                                              MVT::i32));
      Ops.insert(Ops.end(), SelOps.begin(), SelOps.end());
      i += 2;
    }
  }

  // Add the flag input back if present.
  if (e != InOps.size())
    Ops.push_back(InOps.back());
}

/// findFlagUse - Return use of EVT::Flag value produced by the specified
/// SDNode.
///
static SDNode *findFlagUse(SDNode *N) {
  unsigned FlagResNo = N->getNumValues()-1;
  for (SDNode::use_iterator I = N->use_begin(), E = N->use_end(); I != E; ++I) {
    SDUse &Use = I.getUse();
    if (Use.getResNo() == FlagResNo)
      return Use.getUser();
  }
  return NULL;
}

/// findNonImmUse - Return true if "Use" is a non-immediate use of "Def".
/// This function recursively traverses up the operand chain, ignoring
/// certain nodes.
static bool findNonImmUse(SDNode *Use, SDNode* Def, SDNode *ImmedUse,
                          SDNode *Root,
                          SmallPtrSet<SDNode*, 16> &Visited) {
  if (Use->getNodeId() < Def->getNodeId() ||
      !Visited.insert(Use))
    return false;

  for (unsigned i = 0, e = Use->getNumOperands(); i != e; ++i) {
    SDNode *N = Use->getOperand(i).getNode();
    if (N == Def) {
      if (Use == ImmedUse || Use == Root)
        continue;  // We are not looking for immediate use.
      assert(N != Root);
      return true;
    }

    // Traverse up the operand chain.
    if (findNonImmUse(N, Def, ImmedUse, Root, Visited))
      return true;
  }
  return false;
}

/// isNonImmUse - Start searching from Root up the DAG to check is Def can
/// be reached. Return true if that's the case. However, ignore direct uses
/// by ImmedUse (which would be U in the example illustrated in
/// IsLegalAndProfitableToFold) and by Root (which can happen in the store
/// case).
/// FIXME: to be really generic, we should allow direct use by any node
/// that is being folded. But realisticly since we only fold loads which
/// have one non-chain use, we only need to watch out for load/op/store
/// and load/op/cmp case where the root (store / cmp) may reach the load via
/// its chain operand.
static inline bool isNonImmUse(SDNode *Root, SDNode *Def, SDNode *ImmedUse) {
  SmallPtrSet<SDNode*, 16> Visited;
  return findNonImmUse(Root, Def, ImmedUse, Root, Visited);
}

/// IsLegalAndProfitableToFold - Returns true if the specific operand node N of
/// U can be folded during instruction selection that starts at Root and
/// folding N is profitable.
bool SelectionDAGISel::IsLegalAndProfitableToFold(SDNode *N, SDNode *U,
                                                  SDNode *Root) const {
  if (OptLevel == CodeGenOpt::None) return false;

  // If Root use can somehow reach N through a path that that doesn't contain
  // U then folding N would create a cycle. e.g. In the following
  // diagram, Root can reach N through X. If N is folded into into Root, then
  // X is both a predecessor and a successor of U.
  //
  //          [N*]           //
  //         ^   ^           //
  //        /     \          //
  //      [U*]    [X]?       //
  //        ^     ^          //
  //         \   /           //
  //          \ /            //
  //         [Root*]         //
  //
  // * indicates nodes to be folded together.
  //
  // If Root produces a flag, then it gets (even more) interesting. Since it
  // will be "glued" together with its flag use in the scheduler, we need to
  // check if it might reach N.
  //
  //          [N*]           //
  //         ^   ^           //
  //        /     \          //
  //      [U*]    [X]?       //
  //        ^       ^        //
  //         \       \       //
  //          \      |       //
  //         [Root*] |       //
  //          ^      |       //
  //          f      |       //
  //          |      /       //
  //         [Y]    /        //
  //           ^   /         //
  //           f  /          //
  //           | /           //
  //          [FU]           //
  //
  // If FU (flag use) indirectly reaches N (the load), and Root folds N
  // (call it Fold), then X is a predecessor of FU and a successor of
  // Fold. But since Fold and FU are flagged together, this will create
  // a cycle in the scheduling graph.

  EVT VT = Root->getValueType(Root->getNumValues()-1);
  while (VT == MVT::Flag) {
    SDNode *FU = findFlagUse(Root);
    if (FU == NULL)
      break;
    Root = FU;
    VT = Root->getValueType(Root->getNumValues()-1);
  }

  return !isNonImmUse(Root, N, U);
}

SDNode *SelectionDAGISel::Select_INLINEASM(SDNode *N) {
  std::vector<SDValue> Ops(N->op_begin(), N->op_end());
  SelectInlineAsmMemoryOperands(Ops);
    
  std::vector<EVT> VTs;
  VTs.push_back(MVT::Other);
  VTs.push_back(MVT::Flag);
  SDValue New = CurDAG->getNode(ISD::INLINEASM, N->getDebugLoc(),
                                VTs, &Ops[0], Ops.size());
  return New.getNode();
}

SDNode *SelectionDAGISel::Select_UNDEF(SDNode *N) {
  return CurDAG->SelectNodeTo(N, TargetInstrInfo::IMPLICIT_DEF,
                              N->getValueType(0));
}

SDNode *SelectionDAGISel::Select_EH_LABEL(SDNode *N) {
  SDValue Chain = N->getOperand(0);
  unsigned C = cast<LabelSDNode>(N)->getLabelID();
  SDValue Tmp = CurDAG->getTargetConstant(C, MVT::i32);
  return CurDAG->SelectNodeTo(N, TargetInstrInfo::EH_LABEL,
                              MVT::Other, Tmp, Chain);
}

void SelectionDAGISel::CannotYetSelect(SDNode *N) {
  std::string msg;
  raw_string_ostream Msg(msg);
  Msg << "Cannot yet select: ";
  N->print(Msg, CurDAG);
  llvm_report_error(Msg.str());
}

void SelectionDAGISel::CannotYetSelectIntrinsic(SDNode *N) {
  dbgs() << "Cannot yet select: ";
  unsigned iid =
    cast<ConstantSDNode>(N->getOperand(N->getOperand(0).getValueType() == MVT::Other))->getZExtValue();
  if (iid < Intrinsic::num_intrinsics)
    llvm_report_error("Cannot yet select: intrinsic %" + Intrinsic::getName((Intrinsic::ID)iid));
  else if (const TargetIntrinsicInfo *tii = TM.getIntrinsicInfo())
    llvm_report_error(Twine("Cannot yet select: target intrinsic %") +
                      tii->getName(iid));
}

char SelectionDAGISel::ID = 0;

//===-- HexagonTargetMachine.cpp - Define TargetMachine for Hexagon -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Hexagon target spec.
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetMachine.h"
#include "Hexagon.h"
#include "HexagonISelLowering.h"
#include "HexagonMachineScheduler.h"
#include "HexagonTargetObjectFile.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

static cl:: opt<bool> DisableHardwareLoops("disable-hexagon-hwloops",
      cl::Hidden, cl::desc("Disable Hardware Loops for Hexagon target"));

static cl::opt<bool> DisableHexagonMISched("disable-hexagon-misched",
      cl::Hidden, cl::ZeroOrMore, cl::init(false),
      cl::desc("Disable Hexagon MI Scheduling"));

static cl::opt<bool> DisableHexagonCFGOpt("disable-hexagon-cfgopt",
      cl::Hidden, cl::ZeroOrMore, cl::init(false),
      cl::desc("Disable Hexagon CFG Optimization"));


/// HexagonTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int HexagonTargetMachineModule;
int HexagonTargetMachineModule = 0;

extern "C" void LLVMInitializeHexagonTarget() {
  // Register the target.
  RegisterTargetMachine<HexagonTargetMachine> X(TheHexagonTarget);
}

static ScheduleDAGInstrs *createVLIWMachineSched(MachineSchedContext *C) {
  return new VLIWMachineScheduler(C, new ConvergingVLIWScheduler());
}

static MachineSchedRegistry
SchedCustomRegistry("hexagon", "Run Hexagon's custom scheduler",
                    createVLIWMachineSched);

/// HexagonTargetMachine ctor - Create an ILP32 architecture model.
///

/// Hexagon_TODO: Do I need an aggregate alignment?
///
HexagonTargetMachine::HexagonTargetMachine(const Target &T, StringRef TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    DL("e-m:e-p:32:32-i1:32-i64:64-a:0-n32") ,
    Subtarget(TT, CPU, FS), InstrInfo(Subtarget), TLInfo(*this),
    TSInfo(*this),
    FrameLowering(Subtarget),
    InstrItins(&Subtarget.getInstrItineraryData()) {
    setMCUseCFI(false);
    initAsmInfo();
}

// addPassesForOptimizations - Allow the backend (target) to add Target
// Independent Optimization passes to the Pass Manager.
bool HexagonTargetMachine::addPassesForOptimizations(PassManagerBase &PM) {
  if (getOptLevel() != CodeGenOpt::None) {
    PM.add(createConstantPropagationPass());
    PM.add(createLoopSimplifyPass());
    PM.add(createDeadCodeEliminationPass());
    PM.add(createConstantPropagationPass());
    PM.add(createLoopUnrollPass());
    PM.add(createLoopStrengthReducePass());
  }
  return true;
}

namespace {
/// Hexagon Code Generator Pass Configuration Options.
class HexagonPassConfig : public TargetPassConfig {
public:
  HexagonPassConfig(HexagonTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {
    // FIXME: Rather than calling enablePass(&MachineSchedulerID) below, define
    // HexagonSubtarget::enableMachineScheduler() { return true; }.
    // That will bypass the SelectionDAG VLIW scheduler, which is probably just
    // hurting compile time and will be removed eventually anyway.
    if (DisableHexagonMISched)
      disablePass(&MachineSchedulerID);
    else
      enablePass(&MachineSchedulerID);
  }

  HexagonTargetMachine &getHexagonTargetMachine() const {
    return getTM<HexagonTargetMachine>();
  }

  virtual ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const {
    return createVLIWMachineSched(C);
  }

  virtual bool addInstSelector();
  virtual bool addPreRegAlloc();
  virtual bool addPostRegAlloc();
  virtual bool addPreSched2();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *HexagonTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new HexagonPassConfig(this, PM);
}

bool HexagonPassConfig::addInstSelector() {
  HexagonTargetMachine &TM = getHexagonTargetMachine();
  bool NoOpt = (getOptLevel() == CodeGenOpt::None);

  if (!NoOpt)
    addPass(createHexagonRemoveExtendArgs(TM));

  addPass(createHexagonISelDag(TM, getOptLevel()));

  if (!NoOpt) {
    addPass(createHexagonPeephole());
    printAndVerify("After hexagon peephole pass");
  }

  return false;
}

bool HexagonPassConfig::addPreRegAlloc() {
  if (getOptLevel() != CodeGenOpt::None)
    if (!DisableHardwareLoops)
      addPass(createHexagonHardwareLoops());
  return false;
}

bool HexagonPassConfig::addPostRegAlloc() {
  const HexagonTargetMachine &TM = getHexagonTargetMachine();
  if (getOptLevel() != CodeGenOpt::None)
    if (!DisableHexagonCFGOpt)
      addPass(createHexagonCFGOptimizer(TM));
  return false;
}

bool HexagonPassConfig::addPreSched2() {
  const HexagonTargetMachine &TM = getHexagonTargetMachine();
  const HexagonTargetObjectFile &TLOF =
    (const HexagonTargetObjectFile &)getTargetLowering()->getObjFileLowering();

  addPass(createHexagonCopyToCombine());
  if (getOptLevel() != CodeGenOpt::None)
    addPass(&IfConverterID);
  if (!TLOF.IsSmallDataEnabled()) {
    addPass(createHexagonSplitConst32AndConst64(TM));
    printAndVerify("After hexagon split const32/64 pass");
  }
  return true;
}

bool HexagonPassConfig::addPreEmitPass() {
  const HexagonTargetMachine &TM = getHexagonTargetMachine();
  bool NoOpt = (getOptLevel() == CodeGenOpt::None);

  if (!NoOpt)
    addPass(createHexagonNewValueJump());

  // Expand Spill code for predicate registers.
  addPass(createHexagonExpandPredSpillCode(TM));

  // Split up TFRcondsets into conditional transfers.
  addPass(createHexagonSplitTFRCondSets(TM));

  // Create Packets.
  if (!NoOpt) {
    if (!DisableHardwareLoops)
      addPass(createHexagonFixupHwLoops());
    addPass(createHexagonPacketizer());
  }

  return false;
}

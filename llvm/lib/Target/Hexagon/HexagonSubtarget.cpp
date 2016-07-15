//===-- HexagonSubtarget.cpp - Hexagon Subtarget Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Hexagon specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "HexagonSubtarget.h"
#include "Hexagon.h"
#include "HexagonRegisterInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <map>

using namespace llvm;

#define DEBUG_TYPE "hexagon-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "HexagonGenSubtargetInfo.inc"

static cl::opt<bool> EnableMemOps("enable-hexagon-memops",
  cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed, cl::init(true),
  cl::desc("Generate V4 MEMOP in code generation for Hexagon target"));

static cl::opt<bool> DisableMemOps("disable-hexagon-memops",
  cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed, cl::init(false),
  cl::desc("Do not generate V4 MEMOP in code generation for Hexagon target"));

static cl::opt<bool> EnableIEEERndNear("enable-hexagon-ieee-rnd-near",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Generate non-chopped conversion from fp to int."));

static cl::opt<bool> EnableBSBSched("enable-bsb-sched",
  cl::Hidden, cl::ZeroOrMore, cl::init(true));

static cl::opt<bool> EnableHexagonHVXDouble("enable-hexagon-hvx-double",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Enable Hexagon Double Vector eXtensions"));

static cl::opt<bool> EnableHexagonHVX("enable-hexagon-hvx",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Enable Hexagon Vector eXtensions"));

static cl::opt<bool> DisableHexagonMISched("disable-hexagon-misched",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Disable Hexagon MI Scheduling"));

static cl::opt<bool> EnableSubregLiveness("hexagon-subreg-liveness",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Enable subregister liveness tracking for Hexagon"));

void HexagonSubtarget::initializeEnvironment() {
  UseMemOps = false;
  ModeIEEERndNear = false;
  UseBSBScheduling = false;
}

HexagonSubtarget &
HexagonSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  CPUString = HEXAGON_MC::selectHexagonCPU(getTargetTriple(), CPU);

  static std::map<StringRef, HexagonArchEnum> CpuTable {
    { "hexagonv4", V4 },
    { "hexagonv5", V5 },
    { "hexagonv55", V55 },
    { "hexagonv60", V60 },
  };

  auto foundIt = CpuTable.find(CPUString);
  if (foundIt != CpuTable.end())
    HexagonArchVersion = foundIt->second;
  else
    llvm_unreachable("Unrecognized Hexagon processor version");

  UseHVXOps = false;
  UseHVXDblOps = false;
  ParseSubtargetFeatures(CPUString, FS);

  if (EnableHexagonHVX.getPosition())
    UseHVXOps = EnableHexagonHVX;
  if (EnableHexagonHVXDouble.getPosition())
    UseHVXDblOps = EnableHexagonHVXDouble;

  return *this;
}

HexagonSubtarget::HexagonSubtarget(const Triple &TT, StringRef CPU,
                                   StringRef FS, const TargetMachine &TM)
    : HexagonGenSubtargetInfo(TT, CPU, FS), CPUString(CPU),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      FrameLowering() {

  initializeEnvironment();

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUString);

  // UseMemOps on by default unless disabled explicitly
  if (DisableMemOps)
    UseMemOps = false;
  else if (EnableMemOps)
    UseMemOps = true;
  else
    UseMemOps = false;

  if (EnableIEEERndNear)
    ModeIEEERndNear = true;
  else
    ModeIEEERndNear = false;

  UseBSBScheduling = hasV60TOps() && EnableBSBSched;
}


void HexagonSubtarget::HexagonDAGMutation::apply(ScheduleDAGInstrs *DAG) {
  for (auto &SU : DAG->SUnits) {
    if (!SU.isInstr())
      continue;
    SmallVector<SDep, 4> Erase;
    for (auto &D : SU.Preds)
      if (D.getKind() == SDep::Output && D.getReg() == Hexagon::USR_OVF)
        Erase.push_back(D);
    for (auto &E : Erase)
      SU.removePred(E);
  }

  for (auto &SU : DAG->SUnits) {
    // Update the latency of chain edges between v60 vector load or store
    // instructions to be 1. These instructions cannot be scheduled in the
    // same packet.
    MachineInstr *MI1 = SU.getInstr();
    auto *QII = static_cast<const HexagonInstrInfo*>(DAG->TII);
    bool IsStoreMI1 = MI1->mayStore();
    bool IsLoadMI1 = MI1->mayLoad();
    if (!QII->isV60VectorInstruction(MI1) || !(IsStoreMI1 || IsLoadMI1))
      continue;
    for (auto &SI : SU.Succs) {
      if (SI.getKind() != SDep::Order || SI.getLatency() != 0)
        continue;
      MachineInstr *MI2 = SI.getSUnit()->getInstr();
      if (!QII->isV60VectorInstruction(MI2))
        continue;
      if ((IsStoreMI1 && MI2->mayStore()) || (IsLoadMI1 && MI2->mayLoad())) {
        SI.setLatency(1);
        SU.setHeightDirty();
        // Change the dependence in the opposite direction too.
        for (auto &PI : SI.getSUnit()->Preds) {
          if (PI.getSUnit() != &SU || PI.getKind() != SDep::Order)
            continue;
          PI.setLatency(1);
          SI.getSUnit()->setDepthDirty();
        }
      }
    }
  }
}


void HexagonSubtarget::getPostRAMutations(
      std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(make_unique<HexagonSubtarget::HexagonDAGMutation>());
}


// Pin the vtable to this file.
void HexagonSubtarget::anchor() {}

bool HexagonSubtarget::enableMachineScheduler() const {
  if (DisableHexagonMISched.getNumOccurrences())
    return !DisableHexagonMISched;
  return true;
}

bool HexagonSubtarget::enableSubRegLiveness() const {
  return EnableSubregLiveness;
}


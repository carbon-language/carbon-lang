//===----- R600Packetizer.cpp - VLIW packetizer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass implements instructions packetization for R600. It unsets isLast
/// bit of instructions inside a bundle and substitutes src register with
/// PreviousVector when applicable.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "packets"
#include "llvm/Support/Debug.h"
#include "AMDGPU.h"
#include "R600InstrInfo.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class R600Packetizer : public MachineFunctionPass {

public:
  static char ID;
  R600Packetizer(const TargetMachine &TM) : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    AU.addPreserved<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  const char *getPassName() const {
    return "R600 Packetizer";
  }

  bool runOnMachineFunction(MachineFunction &Fn);
};
char R600Packetizer::ID = 0;

class R600PacketizerList : public VLIWPacketizerList {

private:
  const R600InstrInfo *TII;
  const R600RegisterInfo &TRI;

  unsigned getSlot(const MachineInstr *MI) const {
    return TRI.getHWRegChan(MI->getOperand(0).getReg());
  }

  /// \returns register to PV chan mapping for bundle/single instructions that
  /// immediatly precedes I.
  DenseMap<unsigned, unsigned> getPreviousVector(MachineBasicBlock::iterator I)
      const {
    DenseMap<unsigned, unsigned> Result;
    I--;
    if (!TII->isALUInstr(I->getOpcode()) && !I->isBundle())
      return Result;
    MachineBasicBlock::instr_iterator BI = I.getInstrIterator();
    if (I->isBundle())
      BI++;
    do {
      if (TII->isPredicated(BI))
        continue;
      if (TII->isTransOnly(BI))
        continue;
      int OperandIdx = TII->getOperandIdx(BI->getOpcode(), AMDGPU::OpName::write);
      if (OperandIdx > -1 && BI->getOperand(OperandIdx).getImm() == 0)
        continue;
      int DstIdx = TII->getOperandIdx(BI->getOpcode(), AMDGPU::OpName::dst);
      if (DstIdx == -1) {
        continue;
      }
      unsigned Dst = BI->getOperand(DstIdx).getReg();
      if (BI->getOpcode() == AMDGPU::DOT4_r600 ||
          BI->getOpcode() == AMDGPU::DOT4_eg) {
        Result[Dst] = AMDGPU::PV_X;
        continue;
      }
      if (Dst == AMDGPU::OQAP) {
        continue;
      }
      unsigned PVReg = 0;
      switch (TRI.getHWRegChan(Dst)) {
      case 0:
        PVReg = AMDGPU::PV_X;
        break;
      case 1:
        PVReg = AMDGPU::PV_Y;
        break;
      case 2:
        PVReg = AMDGPU::PV_Z;
        break;
      case 3:
        PVReg = AMDGPU::PV_W;
        break;
      default:
        llvm_unreachable("Invalid Chan");
      }
      Result[Dst] = PVReg;
    } while ((++BI)->isBundledWithPred());
    return Result;
  }

  void substitutePV(MachineInstr *MI, const DenseMap<unsigned, unsigned> &PVs)
      const {
    unsigned Ops[] = {
      AMDGPU::OpName::src0,
      AMDGPU::OpName::src1,
      AMDGPU::OpName::src2
    };
    for (unsigned i = 0; i < 3; i++) {
      int OperandIdx = TII->getOperandIdx(MI->getOpcode(), Ops[i]);
      if (OperandIdx < 0)
        continue;
      unsigned Src = MI->getOperand(OperandIdx).getReg();
      const DenseMap<unsigned, unsigned>::const_iterator It = PVs.find(Src);
      if (It != PVs.end())
        MI->getOperand(OperandIdx).setReg(It->second);
    }
  }
public:
  // Ctor.
  R600PacketizerList(MachineFunction &MF, MachineLoopInfo &MLI,
                        MachineDominatorTree &MDT)
  : VLIWPacketizerList(MF, MLI, MDT, true),
    TII (static_cast<const R600InstrInfo *>(MF.getTarget().getInstrInfo())),
    TRI(TII->getRegisterInfo()) { }

  // initPacketizerState - initialize some internal flags.
  void initPacketizerState() { }

  // ignorePseudoInstruction - Ignore bundling of pseudo instructions.
  bool ignorePseudoInstruction(MachineInstr *MI, MachineBasicBlock *MBB) {
    return false;
  }

  // isSoloInstruction - return true if instruction MI can not be packetized
  // with any other instruction, which means that MI itself is a packet.
  bool isSoloInstruction(MachineInstr *MI) {
    if (TII->isVector(*MI))
      return true;
    if (!TII->isALUInstr(MI->getOpcode()))
      return true;
    if (TII->get(MI->getOpcode()).TSFlags & R600_InstFlag::TRANS_ONLY)
      return true;
    if (TII->isTransOnly(MI))
      return true;
    if (MI->getOpcode() == AMDGPU::GROUP_BARRIER)
      return true;
    return false;
  }

  // isLegalToPacketizeTogether - Is it legal to packetize SUI and SUJ
  // together.
  bool isLegalToPacketizeTogether(SUnit *SUI, SUnit *SUJ) {
    MachineInstr *MII = SUI->getInstr(), *MIJ = SUJ->getInstr();
    if (getSlot(MII) <= getSlot(MIJ))
      return false;
    // Does MII and MIJ share the same pred_sel ?
    int OpI = TII->getOperandIdx(MII->getOpcode(), AMDGPU::OpName::pred_sel),
        OpJ = TII->getOperandIdx(MIJ->getOpcode(), AMDGPU::OpName::pred_sel);
    unsigned PredI = (OpI > -1)?MII->getOperand(OpI).getReg():0,
        PredJ = (OpJ > -1)?MIJ->getOperand(OpJ).getReg():0;
    if (PredI != PredJ)
      return false;
    if (SUJ->isSucc(SUI)) {
      for (unsigned i = 0, e = SUJ->Succs.size(); i < e; ++i) {
        const SDep &Dep = SUJ->Succs[i];
        if (Dep.getSUnit() != SUI)
          continue;
        if (Dep.getKind() == SDep::Anti)
          continue;
        if (Dep.getKind() == SDep::Output)
          if (MII->getOperand(0).getReg() != MIJ->getOperand(0).getReg())
            continue;
        return false;
      }
    }
    return true;
  }

  // isLegalToPruneDependencies - Is it legal to prune dependece between SUI
  // and SUJ.
  bool isLegalToPruneDependencies(SUnit *SUI, SUnit *SUJ) {return false;}

  void setIsLastBit(MachineInstr *MI, unsigned Bit) const {
    unsigned LastOp = TII->getOperandIdx(MI->getOpcode(), AMDGPU::OpName::last);
    MI->getOperand(LastOp).setImm(Bit);
  }

  MachineBasicBlock::iterator addToPacket(MachineInstr *MI) {
    CurrentPacketMIs.push_back(MI);
    bool FitsConstLimits = TII->canBundle(CurrentPacketMIs);
    DEBUG(
      if (!FitsConstLimits) {
        dbgs() << "Couldn't pack :\n";
        MI->dump();
        dbgs() << "with the following packets :\n";
        for (unsigned i = 0, e = CurrentPacketMIs.size() - 1; i < e; i++) {
          CurrentPacketMIs[i]->dump();
          dbgs() << "\n";
        }
        dbgs() << "because of Consts read limitations\n";
      });
    const DenseMap<unsigned, unsigned> &PV =
        getPreviousVector(CurrentPacketMIs.front());
    std::vector<R600InstrInfo::BankSwizzle> BS;
    bool FitsReadPortLimits =
        TII->fitsReadPortLimitations(CurrentPacketMIs, PV, BS);
    DEBUG(
      if (!FitsReadPortLimits) {
        dbgs() << "Couldn't pack :\n";
        MI->dump();
        dbgs() << "with the following packets :\n";
        for (unsigned i = 0, e = CurrentPacketMIs.size() - 1; i < e; i++) {
          CurrentPacketMIs[i]->dump();
          dbgs() << "\n";
        }
        dbgs() << "because of Read port limitations\n";
      });
    bool isBundlable = FitsConstLimits && FitsReadPortLimits;
    if (isBundlable) {
      for (unsigned i = 0, e = CurrentPacketMIs.size(); i < e; i++) {
        MachineInstr *MI = CurrentPacketMIs[i];
            unsigned Op = TII->getOperandIdx(MI->getOpcode(),
                AMDGPU::OpName::bank_swizzle);
            MI->getOperand(Op).setImm(BS[i]);
      }
    }
    CurrentPacketMIs.pop_back();
    if (!isBundlable) {
      endPacket(MI->getParent(), MI);
      substitutePV(MI, getPreviousVector(MI));
      return VLIWPacketizerList::addToPacket(MI);
    }
    if (!CurrentPacketMIs.empty())
      setIsLastBit(CurrentPacketMIs.back(), 0);
    substitutePV(MI, PV);
    return VLIWPacketizerList::addToPacket(MI);
  }
};

bool R600Packetizer::runOnMachineFunction(MachineFunction &Fn) {
  const TargetInstrInfo *TII = Fn.getTarget().getInstrInfo();
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfo>();
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();

  // Instantiate the packetizer.
  R600PacketizerList Packetizer(Fn, MLI, MDT);

  // DFA state table should not be empty.
  assert(Packetizer.getResourceTracker() && "Empty DFA table!");

  //
  // Loop over all basic blocks and remove KILL pseudo-instructions
  // These instructions confuse the dependence analysis. Consider:
  // D0 = ...   (Insn 0)
  // R0 = KILL R0, D0 (Insn 1)
  // R0 = ... (Insn 2)
  // Here, Insn 1 will result in the dependence graph not emitting an output
  // dependence between Insn 0 and Insn 2. This can lead to incorrect
  // packetization
  //
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    MachineBasicBlock::iterator End = MBB->end();
    MachineBasicBlock::iterator MI = MBB->begin();
    while (MI != End) {
      if (MI->isKill()) {
        MachineBasicBlock::iterator DeleteMI = MI;
        ++MI;
        MBB->erase(DeleteMI);
        End = MBB->end();
        continue;
      }
      ++MI;
    }
  }

  // Loop over all of the basic blocks.
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    // Find scheduling regions and schedule / packetize each region.
    unsigned RemainingCount = MBB->size();
    for(MachineBasicBlock::iterator RegionEnd = MBB->end();
        RegionEnd != MBB->begin();) {
      // The next region starts above the previous region. Look backward in the
      // instruction stream until we find the nearest boundary.
      MachineBasicBlock::iterator I = RegionEnd;
      for(;I != MBB->begin(); --I, --RemainingCount) {
        if (TII->isSchedulingBoundary(llvm::prior(I), MBB, Fn))
          break;
      }
      I = MBB->begin();

      // Skip empty scheduling regions.
      if (I == RegionEnd) {
        RegionEnd = llvm::prior(RegionEnd);
        --RemainingCount;
        continue;
      }
      // Skip regions with one instruction.
      if (I == llvm::prior(RegionEnd)) {
        RegionEnd = llvm::prior(RegionEnd);
        continue;
      }

      Packetizer.PacketizeMIs(MBB, I, RegionEnd);
      RegionEnd = I;
    }
  }

  return true;

}

} // end anonymous namespace

llvm::FunctionPass *llvm::createR600Packetizer(TargetMachine &tm) {
  return new R600Packetizer(tm);
}

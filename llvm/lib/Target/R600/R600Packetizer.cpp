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

#ifndef R600PACKETIZER_CPP
#define R600PACKETIZER_CPP

#define DEBUG_TYPE "packets"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "AMDGPU.h"
#include "R600InstrInfo.h"

namespace llvm {

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

  enum BankSwizzle {
    ALU_VEC_012 = 0,
    ALU_VEC_021,
    ALU_VEC_120,
    ALU_VEC_102,
    ALU_VEC_201,
    ALU_VEC_210
  };

  unsigned getSlot(const MachineInstr *MI) const {
    return TRI.getHWRegChan(MI->getOperand(0).getReg());
  }

  std::vector<unsigned> getPreviousVector(MachineBasicBlock::iterator I) const {
    std::vector<unsigned> Result;
    I--;
    if (!TII->isALUInstr(I->getOpcode()) && !I->isBundle())
      return Result;
    MachineBasicBlock::instr_iterator BI = I.getInstrIterator();
    if (I->isBundle())
      BI++;
    while (BI->isBundledWithPred() && !TII->isPredicated(BI)) {
      int OperandIdx = TII->getOperandIdx(BI->getOpcode(), R600Operands::WRITE);
      if (OperandIdx > -1 && BI->getOperand(OperandIdx).getImm())
        Result.push_back(BI->getOperand(0).getReg());
      BI++;
    }
    return Result;
  }

  void substitutePV(MachineInstr *MI, const std::vector<unsigned> &PV) const {
    R600Operands::Ops Ops[] = {
      R600Operands::SRC0,
      R600Operands::SRC1,
      R600Operands::SRC2
    };
    for (unsigned i = 0; i < 3; i++) {
      int OperandIdx = TII->getOperandIdx(MI->getOpcode(), Ops[i]);
      if (OperandIdx < 0)
        continue;
      unsigned Src = MI->getOperand(OperandIdx).getReg();
      for (unsigned j = 0, e = PV.size(); j < e; j++) {
        if (Src == PV[j]) {
          unsigned Chan = TRI.getHWRegChan(Src);
          unsigned PVReg;
          switch (Chan) {
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
          MI->getOperand(OperandIdx).setReg(PVReg);
          break;
        }
      }
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
    return false;
  }

  // isLegalToPacketizeTogether - Is it legal to packetize SUI and SUJ
  // together.
  bool isLegalToPacketizeTogether(SUnit *SUI, SUnit *SUJ) {
    MachineInstr *MII = SUI->getInstr(), *MIJ = SUJ->getInstr();
    if (getSlot(MII) <= getSlot(MIJ))
      return false;
    // Does MII and MIJ share the same pred_sel ?
    int OpI = TII->getOperandIdx(MII->getOpcode(), R600Operands::PRED_SEL),
        OpJ = TII->getOperandIdx(MIJ->getOpcode(), R600Operands::PRED_SEL);
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
    unsigned LastOp = TII->getOperandIdx(MI->getOpcode(), R600Operands::LAST);
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
    const std::vector<unsigned> &PV = getPreviousVector(MI);
    bool FitsReadPortLimits = fitsReadPortLimitation(CurrentPacketMIs, PV);
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
private:
  std::vector<std::pair<int, unsigned> >
  ExtractSrcs(const MachineInstr *MI, const std::vector<unsigned> &PV) const {
    R600Operands::Ops Ops[] = {
      R600Operands::SRC0,
      R600Operands::SRC1,
      R600Operands::SRC2
    };
    std::vector<std::pair<int, unsigned> > Result;
    for (unsigned i = 0; i < 3; i++) {
      int OperandIdx = TII->getOperandIdx(MI->getOpcode(), Ops[i]);
      if (OperandIdx < 0){
        Result.push_back(std::pair<int, unsigned>(-1,0));
        continue;
      }
      unsigned Src = MI->getOperand(OperandIdx).getReg();
      if (std::find(PV.begin(), PV.end(), Src) != PV.end()) {
        Result.push_back(std::pair<int, unsigned>(-1,0));
        continue;
      }
      unsigned Reg = TRI.getEncodingValue(Src) & 0xff;
      if (Reg > 127) {
        Result.push_back(std::pair<int, unsigned>(-1,0));
        continue;
      }
      unsigned Chan = TRI.getHWRegChan(Src);
      Result.push_back(std::pair<int, unsigned>(Reg, Chan));
    }
    return Result;
  }

  std::vector<std::pair<int, unsigned> >
  Swizzle(std::vector<std::pair<int, unsigned> > Src,
  BankSwizzle Swz) const {
    switch (Swz) {
    case ALU_VEC_012:
      break;
    case ALU_VEC_021:
      std::swap(Src[1], Src[2]);
      break;
    case ALU_VEC_102:
      std::swap(Src[0], Src[1]);
      break;
    case ALU_VEC_120:
      std::swap(Src[0], Src[1]);
      std::swap(Src[0], Src[2]);
      break;
    case ALU_VEC_201:
      std::swap(Src[0], Src[2]);
      std::swap(Src[0], Src[1]);
      break;
    case ALU_VEC_210:
      std::swap(Src[0], Src[2]);
      break;
    }
    return Src;
  }

  bool isLegal(const std::vector<MachineInstr *> &IG,
      const std::vector<BankSwizzle> &Swz,
      const std::vector<unsigned> &PV) const {
    assert (Swz.size() == IG.size());
    int Vector[4][3];
    memset(Vector, -1, sizeof(Vector));
    for (unsigned i = 0, e = IG.size(); i < e; i++) {
      const std::vector<std::pair<int, unsigned> > &Srcs =
          Swizzle(ExtractSrcs(IG[i], PV), Swz[i]);
      for (unsigned j = 0; j < 3; j++) {
        const std::pair<int, unsigned> &Src = Srcs[j];
        if (Src.first < 0)
          continue;
        if (Vector[Src.second][j] < 0)
          Vector[Src.second][j] = Src.first;
        if (Vector[Src.second][j] != Src.first)
          return false;
      }
    }
    return true;
  }

  bool recursiveFitsFPLimitation(
  std::vector<MachineInstr *> IG,
  const std::vector<unsigned> &PV,
  std::vector<BankSwizzle> &SwzCandidate,
  std::vector<MachineInstr *> CurrentlyChecked)
      const {
    if (!isLegal(CurrentlyChecked, SwzCandidate, PV))
      return false;
    if (IG.size() == CurrentlyChecked.size()) {
      return true;
    }
    BankSwizzle AvailableSwizzle[] = {
      ALU_VEC_012,
      ALU_VEC_021,
      ALU_VEC_120,
      ALU_VEC_102,
      ALU_VEC_201,
      ALU_VEC_210
    };
    CurrentlyChecked.push_back(IG[CurrentlyChecked.size()]);
    for (unsigned i = 0; i < 6; i++) {
      SwzCandidate.push_back(AvailableSwizzle[i]);
      if (recursiveFitsFPLimitation(IG, PV, SwzCandidate, CurrentlyChecked))
        return true;
      SwzCandidate.pop_back();
    }
    return false;
  }

  bool fitsReadPortLimitation(
  std::vector<MachineInstr *> IG,
  const std::vector<unsigned> &PV)
      const {
    //Todo : support shared src0 - src1 operand
    std::vector<BankSwizzle> SwzCandidate;
    bool Result = recursiveFitsFPLimitation(IG, PV, SwzCandidate,
        std::vector<MachineInstr *>());
    if (!Result)
      return false;
    for (unsigned i = 0, e = IG.size(); i < e; i++) {
      MachineInstr *MI = IG[i];
      unsigned Op = TII->getOperandIdx(MI->getOpcode(),
          R600Operands::BANK_SWIZZLE);
      MI->getOperand(Op).setImm(SwzCandidate[i]);
    }
    return true;
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

}

llvm::FunctionPass *llvm::createR600Packetizer(TargetMachine &tm) {
  return new R600Packetizer(tm);
}

#endif // R600PACKETIZER_CPP

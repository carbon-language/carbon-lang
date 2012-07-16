//===-- R600InstrInfo.cpp - R600 Instruction Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// R600 Implementation of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#include "R600InstrInfo.h"
#include "AMDGPUTargetMachine.h"
#include "AMDILSubtarget.h"
#include "R600RegisterInfo.h"

#define GET_INSTRINFO_CTOR
#include "AMDGPUGenDFAPacketizer.inc"

using namespace llvm;

R600InstrInfo::R600InstrInfo(AMDGPUTargetMachine &tm)
  : AMDGPUInstrInfo(tm),
    RI(tm, *this)
  { }

const R600RegisterInfo &R600InstrInfo::getRegisterInfo() const
{
  return RI;
}

bool R600InstrInfo::isTrig(const MachineInstr &MI) const
{
  return get(MI.getOpcode()).TSFlags & R600_InstFlag::TRIG;
}

bool R600InstrInfo::isVector(const MachineInstr &MI) const
{
  return get(MI.getOpcode()).TSFlags & R600_InstFlag::VECTOR;
}

void
R600InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const
{

  unsigned subRegMap[4] = {AMDGPU::sel_x, AMDGPU::sel_y,
                           AMDGPU::sel_z, AMDGPU::sel_w};

  if (AMDGPU::R600_Reg128RegClass.contains(DestReg)
      && AMDGPU::R600_Reg128RegClass.contains(SrcReg)) {
    for (unsigned i = 0; i < 4; i++) {
      BuildMI(MBB, MI, DL, get(AMDGPU::MOV))
              .addReg(RI.getSubReg(DestReg, subRegMap[i]), RegState::Define)
              .addReg(RI.getSubReg(SrcReg, subRegMap[i]))
              .addReg(DestReg, RegState::Define | RegState::Implicit);
    }
  } else {

    /* We can't copy vec4 registers */
    assert(!AMDGPU::R600_Reg128RegClass.contains(DestReg)
           && !AMDGPU::R600_Reg128RegClass.contains(SrcReg));

    BuildMI(MBB, MI, DL, get(AMDGPU::MOV), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
  }
}

MachineInstr * R600InstrInfo::getMovImmInstr(MachineFunction *MF,
                                             unsigned DstReg, int64_t Imm) const
{
  MachineInstr * MI = MF->CreateMachineInstr(get(AMDGPU::MOV), DebugLoc());
  MachineInstrBuilder(MI).addReg(DstReg, RegState::Define);
  MachineInstrBuilder(MI).addReg(AMDGPU::ALU_LITERAL_X);
  MachineInstrBuilder(MI).addImm(Imm);

  return MI;
}

unsigned R600InstrInfo::getIEQOpcode() const
{
  return AMDGPU::SETE_INT;
}

bool R600InstrInfo::isMov(unsigned Opcode) const
{
  switch(Opcode) {
  default: return false;
  case AMDGPU::MOV:
  case AMDGPU::MOV_IMM_F32:
  case AMDGPU::MOV_IMM_I32:
    return true;
  }
}

DFAPacketizer *R600InstrInfo::CreateTargetScheduleState(const TargetMachine *TM,
    const ScheduleDAG *DAG) const
{
  const InstrItineraryData *II = TM->getInstrItineraryData();
  return TM->getSubtarget<AMDILSubtarget>().createDFAPacketizer(II);
}

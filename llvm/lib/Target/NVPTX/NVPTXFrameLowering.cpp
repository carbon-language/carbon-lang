//=======- NVPTXFrameLowering.cpp - NVPTX Frame Information ---*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "NVPTXFrameLowering.h"
#include "NVPTX.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "NVPTXTargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

NVPTXFrameLowering::NVPTXFrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, 8, 0) {}

bool NVPTXFrameLowering::hasFP(const MachineFunction &MF) const { return true; }

void NVPTXFrameLowering::emitPrologue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  if (MF.getFrameInfo()->hasStackObjects()) {
    assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
    // Insert "mov.u32 %SP, %Depot"
    MachineBasicBlock::iterator MBBI = MBB.begin();
    // This instruction really occurs before first instruction
    // in the BB, so giving it no debug location.
    DebugLoc dl = DebugLoc();

    MachineRegisterInfo &MRI = MF.getRegInfo();

    // mov %SPL, %depot;
    // cvta.local %SP, %SPL;
    if (static_cast<const NVPTXTargetMachine &>(MF.getTarget()).is64Bit()) {
      unsigned LocalReg = MRI.createVirtualRegister(&NVPTX::Int64RegsRegClass);
      MachineInstr *MI =
          BuildMI(MBB, MBBI, dl, MF.getSubtarget().getInstrInfo()->get(
                                     NVPTX::cvta_local_yes_64),
                  NVPTX::VRFrame).addReg(LocalReg);
      BuildMI(MBB, MI, dl,
              MF.getSubtarget().getInstrInfo()->get(NVPTX::MOV_DEPOT_ADDR_64),
              LocalReg).addImm(MF.getFunctionNumber());
    } else {
      unsigned LocalReg = MRI.createVirtualRegister(&NVPTX::Int32RegsRegClass);
      MachineInstr *MI =
          BuildMI(MBB, MBBI, dl,
                  MF.getSubtarget().getInstrInfo()->get(NVPTX::cvta_local_yes),
                  NVPTX::VRFrame).addReg(LocalReg);
      BuildMI(MBB, MI, dl,
              MF.getSubtarget().getInstrInfo()->get(NVPTX::MOV_DEPOT_ADDR),
              LocalReg).addImm(MF.getFunctionNumber());
    }
  }
}

void NVPTXFrameLowering::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {}

// This function eliminates ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
void NVPTXFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN,
  // ADJCALLSTACKUP instructions.
  MBB.erase(I);
}

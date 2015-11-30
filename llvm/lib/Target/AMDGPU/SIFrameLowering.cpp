//===----------------------- SIFrameLowering.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//

#include "SIFrameLowering.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"

using namespace llvm;


static bool hasOnlySGPRSpills(const SIMachineFunctionInfo *FuncInfo,
                              const MachineFrameInfo *FrameInfo) {
  if (!FuncInfo->hasSpilledSGPRs())
    return false;

  if (FuncInfo->hasSpilledVGPRs())
    return false;

  for (int I = FrameInfo->getObjectIndexBegin(),
         E = FrameInfo->getObjectIndexEnd(); I != E; ++I) {
    if (!FrameInfo->isSpillSlotObjectIndex(I))
      return false;
  }

  return true;
}

void SIFrameLowering::emitPrologue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  if (!MF.getFrameInfo()->hasStackObjects())
    return;

  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  // If we only have SGPR spills, we won't actually be using scratch memory
  // since these spill to VGPRs.
  //
  // FIXME: We should be cleaning up these unused SGPR spill frame indices
  // somewhere.
  if (hasOnlySGPRSpills(MFI, MF.getFrameInfo()))
    return;

  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();

  // We need to insert initialization of the scratch resource descriptor.
  unsigned ScratchRsrcReg = MFI->getScratchRSrcReg();
  assert(ScratchRsrcReg != AMDGPU::NoRegister);

  uint64_t Rsrc23 = TII->getScratchRsrcWords23();
  MachineBasicBlock::iterator I = MBB.begin();
  DebugLoc DL;

  unsigned Rsrc0 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0);
  unsigned Rsrc1 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub1);
  unsigned Rsrc2 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub2);
  unsigned Rsrc3 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub3);

  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), Rsrc0)
    .addExternalSymbol("SCRATCH_RSRC_DWORD0");

  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), Rsrc1)
    .addExternalSymbol("SCRATCH_RSRC_DWORD1");

  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), Rsrc2)
    .addImm(Rsrc23 & 0xffffffff);

  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), Rsrc3)
    .addImm(Rsrc23 >> 32);
}

void SIFrameLowering::processFunctionBeforeFrameFinalized(
  MachineFunction &MF,
  RegScavenger *RS) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();

  if (!MFI->hasStackObjects())
    return;

  bool MayNeedScavengingEmergencySlot = MFI->hasStackObjects();

  assert((RS || !MayNeedScavengingEmergencySlot) &&
         "RegScavenger required if spilling");

  if (MayNeedScavengingEmergencySlot) {
    int ScavengeFI = MFI->CreateSpillStackObject(
      AMDGPU::SGPR_32RegClass.getSize(),
      AMDGPU::SGPR_32RegClass.getAlignment());
    RS->addScavengingFrameIndex(ScavengeFI);
  }
}

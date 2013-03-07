//===-- AMDGPUIndirectAddressing.cpp - Indirect Adressing Support ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// Instructions can use indirect addressing to index the register file as if it
/// were memory.  This pass lowers RegisterLoad and RegisterStore instructions
/// to either a COPY or a MOV that uses indirect addressing.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "R600InstrInfo.h"
#include "R600MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace {

class AMDGPUIndirectAddressingPass : public MachineFunctionPass {

private:
  static char ID;
  const AMDGPUInstrInfo *TII;

  bool regHasExplicitDef(MachineRegisterInfo &MRI, unsigned Reg) const;

public:
  AMDGPUIndirectAddressingPass(TargetMachine &tm) :
    MachineFunctionPass(ID),
    TII(static_cast<const AMDGPUInstrInfo*>(tm.getInstrInfo()))
    { }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  const char *getPassName() const { return "R600 Handle indirect addressing"; }

};

} // End anonymous namespace

char AMDGPUIndirectAddressingPass::ID = 0;

FunctionPass *llvm::createAMDGPUIndirectAddressingPass(TargetMachine &tm) {
  return new AMDGPUIndirectAddressingPass(tm);
}

bool AMDGPUIndirectAddressingPass::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();

  int IndirectBegin = TII->getIndirectIndexBegin(MF);
  int IndirectEnd = TII->getIndirectIndexEnd(MF);

  if (IndirectBegin == -1) {
    // No indirect addressing, we can skip this pass
    assert(IndirectEnd == -1);
    return false;
  }

  // The map keeps track of the indirect address that is represented by
  // each virtual register. The key is the register and the value is the
  // indirect address it uses.
  std::map<unsigned, unsigned> RegisterAddressMap;

  // First pass - Lower all of the RegisterStore instructions and track which
  // registers are live.
  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                      BB != BB_E; ++BB) {
    // This map keeps track of the current live indirect registers.
    // The key is the address and the value is the register
    std::map<unsigned, unsigned> LiveAddressRegisterMap;
    MachineBasicBlock &MBB = *BB;

    for (MachineBasicBlock::iterator I = MBB.begin(), Next = llvm::next(I);
                               I != MBB.end(); I = Next) {
      Next = llvm::next(I);
      MachineInstr &MI = *I;

      if (!TII->isRegisterStore(MI)) {
        continue;
      }

      // Lower RegisterStore

      unsigned RegIndex = MI.getOperand(2).getImm();
      unsigned Channel = MI.getOperand(3).getImm();
      unsigned Address = TII->calculateIndirectAddress(RegIndex, Channel);
      const TargetRegisterClass *IndirectStoreRegClass =
                   TII->getIndirectAddrStoreRegClass(MI.getOperand(0).getReg());

      if (MI.getOperand(1).getReg() == AMDGPU::INDIRECT_BASE_ADDR) {
        // Direct register access.
        unsigned DstReg = MRI.createVirtualRegister(IndirectStoreRegClass);

        BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::COPY), DstReg)
                .addOperand(MI.getOperand(0));

        RegisterAddressMap[DstReg] = Address;
        LiveAddressRegisterMap[Address] = DstReg;
      } else {
        // Indirect register access.
        MachineInstrBuilder MOV = TII->buildIndirectWrite(BB, I,
                                           MI.getOperand(0).getReg(), // Value
                                           Address,
                                           MI.getOperand(1).getReg()); // Offset
        for (int i = IndirectBegin; i <= IndirectEnd; ++i) {
          unsigned Addr = TII->calculateIndirectAddress(i, Channel);
          unsigned DstReg = MRI.createVirtualRegister(IndirectStoreRegClass);
          MOV.addReg(DstReg, RegState::Define | RegState::Implicit);
          RegisterAddressMap[DstReg] = Addr;
          LiveAddressRegisterMap[Addr] = DstReg;
        }
      }
      MI.eraseFromParent();
    }

    // Update the live-ins of the succesor blocks
    for (MachineBasicBlock::succ_iterator Succ = MBB.succ_begin(),
                                          SuccEnd = MBB.succ_end();
                                          SuccEnd != Succ; ++Succ) {
      std::map<unsigned, unsigned>::const_iterator Key, KeyEnd;
      for (Key = LiveAddressRegisterMap.begin(),
           KeyEnd = LiveAddressRegisterMap.end(); KeyEnd != Key; ++Key) {
        (*Succ)->addLiveIn(Key->second);
      }
    }
  }

  // Second pass - Lower the RegisterLoad instructions
  for (MachineFunction::iterator BB = MF.begin(), BB_E = MF.end();
                                                      BB != BB_E; ++BB) {
    // Key is the address and the value is the register
    std::map<unsigned, unsigned> LiveAddressRegisterMap;
    MachineBasicBlock &MBB = *BB;

    MachineBasicBlock::livein_iterator LI = MBB.livein_begin();
    while (LI != MBB.livein_end()) {
      std::vector<unsigned> PhiRegisters;

      // Make sure this live in is used for indirect addressing
      if (RegisterAddressMap.find(*LI) == RegisterAddressMap.end()) {
        ++LI;
        continue;
      }

      unsigned Address = RegisterAddressMap[*LI];
      LiveAddressRegisterMap[Address] = *LI;
      PhiRegisters.push_back(*LI);

      // Check if there are other live in registers which map to the same
      // indirect address.
      for (MachineBasicBlock::livein_iterator LJ = llvm::next(LI),
                                              LE = MBB.livein_end();
                                              LJ != LE; ++LJ) {
        unsigned Reg = *LJ;
        if (RegisterAddressMap.find(Reg) == RegisterAddressMap.end()) {
          continue;
        }

        if (RegisterAddressMap[Reg] == Address) {
          PhiRegisters.push_back(Reg);
        }
      }

      if (PhiRegisters.size() == 1) {
        // We don't need to insert a Phi instruction, so we can just add the
        // registers to the live list for the block.
        LiveAddressRegisterMap[Address] = *LI;
        MBB.removeLiveIn(*LI);
      } else {
        // We need to insert a PHI, because we have the same address being
        // written in multiple predecessor blocks.
        const TargetRegisterClass *PhiDstClass =
                   TII->getIndirectAddrStoreRegClass(*(PhiRegisters.begin()));
        unsigned PhiDstReg = MRI.createVirtualRegister(PhiDstClass);
        MachineInstrBuilder Phi = BuildMI(MBB, MBB.begin(),
                                          MBB.findDebugLoc(MBB.begin()),
                                          TII->get(AMDGPU::PHI), PhiDstReg);

        for (std::vector<unsigned>::const_iterator RI = PhiRegisters.begin(),
                                                   RE = PhiRegisters.end();
                                                   RI != RE; ++RI) {
          unsigned Reg = *RI;
          MachineInstr *DefInst = MRI.getVRegDef(Reg);
          assert(DefInst);
          MachineBasicBlock *RegBlock = DefInst->getParent();
          Phi.addReg(Reg);
          Phi.addMBB(RegBlock);
          MBB.removeLiveIn(Reg);
        }
        RegisterAddressMap[PhiDstReg] = Address;
        LiveAddressRegisterMap[Address] = PhiDstReg;
      }
      LI = MBB.livein_begin();
    }

    for (MachineBasicBlock::iterator I = MBB.begin(), Next = llvm::next(I);
                               I != MBB.end(); I = Next) {
      Next = llvm::next(I);
      MachineInstr &MI = *I;

      if (!TII->isRegisterLoad(MI)) {
        if (MI.getOpcode() == AMDGPU::PHI) {
          continue;
        }
        // Check for indirect register defs
        for (unsigned OpIdx = 0, NumOperands = MI.getNumOperands();
                                 OpIdx < NumOperands; ++OpIdx) {
          MachineOperand &MO = MI.getOperand(OpIdx);
          if (MO.isReg() && MO.isDef() &&
              RegisterAddressMap.find(MO.getReg()) != RegisterAddressMap.end()) {
            unsigned Reg = MO.getReg();
            unsigned LiveAddress = RegisterAddressMap[Reg];
            // Chain the live-ins
            if (LiveAddressRegisterMap.find(LiveAddress) !=
                                                     RegisterAddressMap.end()) {
              MI.addOperand(MachineOperand::CreateReg(
                                  LiveAddressRegisterMap[LiveAddress],
                                  false, // isDef
                                  true,  // isImp
                                  true));  // isKill
            }
            LiveAddressRegisterMap[LiveAddress] = Reg;
          }
        }
        continue;
      }

      const TargetRegisterClass *SuperIndirectRegClass =
                                                TII->getSuperIndirectRegClass();
      const TargetRegisterClass *IndirectLoadRegClass =
                                             TII->getIndirectAddrLoadRegClass();
      unsigned IndirectReg = MRI.createVirtualRegister(SuperIndirectRegClass);

      unsigned RegIndex = MI.getOperand(2).getImm();
      unsigned Channel = MI.getOperand(3).getImm();
      unsigned Address = TII->calculateIndirectAddress(RegIndex, Channel);

      if (MI.getOperand(1).getReg() == AMDGPU::INDIRECT_BASE_ADDR) {
        // Direct register access
        unsigned Reg = LiveAddressRegisterMap[Address];
        unsigned AddrReg = IndirectLoadRegClass->getRegister(Address);

        if (regHasExplicitDef(MRI, Reg)) {
          // If the register we are reading from has an explicit def, then that
          // means it was written via a direct register access (i.e. COPY
          // or other instruction that doesn't use indirect addressing).  In
          // this case we know where the value has been stored, so we can just
          // issue a copy.
          BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::COPY),
                  MI.getOperand(0).getReg())
                  .addReg(Reg);
        } else {
          // If the register we are reading has an implicit def, then that
          // means it was written by an indirect register access (i.e. An
          // instruction that uses indirect addressing. 
          BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(AMDGPU::COPY),
                   MI.getOperand(0).getReg())
                   .addReg(AddrReg)
                   .addReg(Reg, RegState::Implicit);
        }
      } else {
        // Indirect register access

        // Note on REQ_SEQUENCE instructons: You can't actually use the register
        // it defines unless  you have an instruction that takes the defined
        // register class as an operand.

        MachineInstrBuilder Sequence = BuildMI(MBB, I, MBB.findDebugLoc(I),
                                               TII->get(AMDGPU::REG_SEQUENCE),
                                               IndirectReg);
        for (int i = IndirectBegin; i <= IndirectEnd; ++i) {
          unsigned Addr = TII->calculateIndirectAddress(i, Channel);
          if (LiveAddressRegisterMap.find(Addr) == LiveAddressRegisterMap.end()) {
            continue;
          }
          unsigned Reg = LiveAddressRegisterMap[Addr];

          // We only need to use REG_SEQUENCE for explicit defs, since the
          // register coalescer won't do anything with the implicit defs.
          if (!regHasExplicitDef(MRI, Reg)) {
            continue;
          }

          // Insert a REQ_SEQUENCE instruction to force the register allocator
          // to allocate the virtual register to the correct physical register.
          Sequence.addReg(LiveAddressRegisterMap[Addr]);
          Sequence.addImm(TII->getRegisterInfo().getIndirectSubReg(Addr));
        }
        MachineInstrBuilder Mov = TII->buildIndirectRead(BB, I,
                                           MI.getOperand(0).getReg(), // Value
                                           Address,
                                           MI.getOperand(1).getReg()); // Offset



        Mov.addReg(IndirectReg, RegState::Implicit | RegState::Kill);
        Mov.addReg(LiveAddressRegisterMap[Address], RegState::Implicit);

      }
      MI.eraseFromParent();
    }
  }
  return false;
}

bool AMDGPUIndirectAddressingPass::regHasExplicitDef(MachineRegisterInfo &MRI,
                                                  unsigned Reg) const {
  MachineInstr *DefInstr = MRI.getVRegDef(Reg);

  if (!DefInstr) {
    return false;
  }

  if (DefInstr->getOpcode() == AMDGPU::PHI) {
    bool Explicit = false;
    for (MachineInstr::const_mop_iterator I = DefInstr->operands_begin(),
                                          E = DefInstr->operands_end();
                                          I != E; ++I) {
      const MachineOperand &MO = *I;
      if (!MO.isReg() || MO.isDef()) {
        continue;
      }

      Explicit = Explicit || regHasExplicitDef(MRI, MO.getReg());
    }
    return Explicit;
  }

  return DefInstr->getOperand(0).isReg() &&
         DefInstr->getOperand(0).getReg() == Reg;
}

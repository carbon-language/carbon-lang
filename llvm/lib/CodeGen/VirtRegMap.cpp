//===-- llvm/CodeGen/VirtRegMap.cpp - Virtual Register Map ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the VirtRegMap class.
//
// It also contains implementations of the the Spiller interface, which, given a
// virtual register map and a machine function, eliminates all virtual
// references by replacing them with physical register references - adding spill
// code as necessary.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "spiller"
#include "VirtRegMap.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

namespace {
  Statistic<> NumSpills("spiller", "Number of register spills");
  Statistic<> NumStores("spiller", "Number of stores added");
  Statistic<> NumLoads ("spiller", "Number of loads added");

  enum SpillerName { simple, local };

  cl::opt<SpillerName>
  SpillerOpt("spiller",
             cl::desc("Spiller to use: (default: simple)"),
             cl::Prefix,
             cl::values(clEnumVal(simple, "  simple spiller"),
                        clEnumVal(local,  "  local spiller"),
                        clEnumValEnd),
             cl::init(simple));
}

//===----------------------------------------------------------------------===//
//  VirtRegMap implementation
//===----------------------------------------------------------------------===//

void VirtRegMap::grow() {
  Virt2PhysMap.grow(MF.getSSARegMap()->getLastVirtReg());
  Virt2StackSlotMap.grow(MF.getSSARegMap()->getLastVirtReg());
}

int VirtRegMap::assignVirt2StackSlot(unsigned virtReg) {
  assert(MRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2StackSlotMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign stack slot to already spilled register");
  const TargetRegisterClass* RC = MF.getSSARegMap()->getRegClass(virtReg);
  int frameIndex = MF.getFrameInfo()->CreateStackObject(RC->getSize(),
                                                        RC->getAlignment());
  Virt2StackSlotMap[virtReg] = frameIndex;
  ++NumSpills;
  return frameIndex;
}

void VirtRegMap::assignVirt2StackSlot(unsigned virtReg, int frameIndex) {
  assert(MRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2StackSlotMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign stack slot to already spilled register");
  Virt2StackSlotMap[virtReg] = frameIndex;
}

void VirtRegMap::virtFolded(unsigned virtReg,
                            MachineInstr* oldMI,
                            MachineInstr* newMI) {
  // move previous memory references folded to new instruction
  MI2VirtMapTy::iterator i, e;
  std::vector<MI2VirtMapTy::mapped_type> regs;
  for (tie(i, e) = MI2VirtMap.equal_range(oldMI); i != e; ) {
    regs.push_back(i->second);
    MI2VirtMap.erase(i++);
  }
  for (unsigned i = 0, e = regs.size(); i != e; ++i)
    MI2VirtMap.insert(std::make_pair(newMI, i));

  // add new memory reference
  MI2VirtMap.insert(std::make_pair(newMI, virtReg));
}

void VirtRegMap::print(std::ostream &OS) const {
  const MRegisterInfo* MRI = MF.getTarget().getRegisterInfo();

  OS << "********** REGISTER MAP **********\n";
  for (unsigned i = MRegisterInfo::FirstVirtualRegister,
         e = MF.getSSARegMap()->getLastVirtReg(); i <= e; ++i) {
    if (Virt2PhysMap[i] != (unsigned)VirtRegMap::NO_PHYS_REG)
      OS << "[reg" << i << " -> " << MRI->getName(Virt2PhysMap[i]) << "]\n";
         
  }

  for (unsigned i = MRegisterInfo::FirstVirtualRegister,
         e = MF.getSSARegMap()->getLastVirtReg(); i <= e; ++i)
    if (Virt2StackSlotMap[i] != VirtRegMap::NO_STACK_SLOT)
      OS << "[reg" << i << " -> fi#" << Virt2StackSlotMap[i] << "]\n";
  OS << '\n';
}

void VirtRegMap::dump() const { print(std::cerr); }


//===----------------------------------------------------------------------===//
// Simple Spiller Implementation
//===----------------------------------------------------------------------===//

Spiller::~Spiller() {}

namespace {
  struct SimpleSpiller : public Spiller {
    bool runOnMachineFunction(MachineFunction& mf, const VirtRegMap &VRM);
  };
}

bool SimpleSpiller::runOnMachineFunction(MachineFunction& MF,
                                         const VirtRegMap& VRM) {
  DEBUG(std::cerr << "********** REWRITE MACHINE CODE **********\n");
  DEBUG(std::cerr << "********** Function: "
                  << MF.getFunction()->getName() << '\n');
  const TargetMachine& TM = MF.getTarget();
  const MRegisterInfo& MRI = *TM.getRegisterInfo();

  // LoadedRegs - Keep track of which vregs are loaded, so that we only load
  // each vreg once (in the case where a spilled vreg is used by multiple
  // operands).  This is always smaller than the number of operands to the
  // current machine instr, so it should be small.
  std::vector<unsigned> LoadedRegs;

  for (MachineFunction::iterator MBBI = MF.begin(), E = MF.end();
       MBBI != E; ++MBBI) {
    DEBUG(std::cerr << MBBI->getBasicBlock()->getName() << ":\n");
    MachineBasicBlock &MBB = *MBBI;
    for (MachineBasicBlock::iterator MII = MBB.begin(),
           E = MBB.end(); MII != E; ++MII) {
      MachineInstr &MI = *MII;
      for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
        MachineOperand &MOP = MI.getOperand(i);
        if (MOP.isRegister() && MOP.getReg() &&
            MRegisterInfo::isVirtualRegister(MOP.getReg())){
          unsigned VirtReg = MOP.getReg();
          unsigned PhysReg = VRM.getPhys(VirtReg);
          if (VRM.hasStackSlot(VirtReg)) {
            int StackSlot = VRM.getStackSlot(VirtReg);

            if (MOP.isUse() &&
                std::find(LoadedRegs.begin(), LoadedRegs.end(), VirtReg)
                           == LoadedRegs.end()) {
              MRI.loadRegFromStackSlot(MBB, &MI, PhysReg, StackSlot);
              LoadedRegs.push_back(VirtReg);
              ++NumLoads;
              DEBUG(std::cerr << '\t' << *prior(MII));
            }

            if (MOP.isDef()) {
              MRI.storeRegToStackSlot(MBB, next(MII), PhysReg,
                                      VRM.getStackSlot(VirtReg));
              ++NumStores;
            }
          }
          MI.SetMachineOperandReg(i, PhysReg);
        }
      }
      DEBUG(std::cerr << '\t' << MI);
      LoadedRegs.clear();
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
//  Local Spiller Implementation
//===----------------------------------------------------------------------===//

namespace {
  class LocalSpiller : public Spiller {
    typedef std::vector<unsigned> Phys2VirtMap;
    typedef std::vector<bool> PhysFlag;
    typedef DenseMap<MachineInstr*, VirtReg2IndexFunctor> Virt2MI;

    MachineFunction *MF;
    const TargetMachine *TM;
    const TargetInstrInfo *TII;
    const MRegisterInfo *MRI;
    const VirtRegMap *VRM;
    Phys2VirtMap p2vMap_;
    PhysFlag dirty_;
    Virt2MI lastDef_;

  public:
    bool runOnMachineFunction(MachineFunction &MF, const VirtRegMap &VRM);

  private:
    void vacateJustPhysReg(MachineBasicBlock& MBB, 
                           MachineBasicBlock::iterator MII,
                           unsigned PhysReg);

    void vacatePhysReg(MachineBasicBlock& MBB,
                       MachineBasicBlock::iterator MII,
                       unsigned PhysReg) {
      vacateJustPhysReg(MBB, MII, PhysReg);
      for (const unsigned* as = MRI->getAliasSet(PhysReg); *as; ++as)
        vacateJustPhysReg(MBB, MII, *as);
    }

    void handleUse(MachineBasicBlock& MBB,
                   MachineBasicBlock::iterator MII,
                   unsigned VirtReg,
                   unsigned PhysReg) {
      // check if we are replacing a previous mapping
      if (p2vMap_[PhysReg] != VirtReg) {
        vacatePhysReg(MBB, MII, PhysReg);
        p2vMap_[PhysReg] = VirtReg;
        // load if necessary
        if (VRM->hasStackSlot(VirtReg)) {
          MRI->loadRegFromStackSlot(MBB, MII, PhysReg,
                                     VRM->getStackSlot(VirtReg));
          ++NumLoads;
          DEBUG(std::cerr << "added: " << *prior(MII));
          lastDef_[VirtReg] = MII;
        }
      }
    }

    void handleDef(MachineBasicBlock& MBB,
                   MachineBasicBlock::iterator MII,
                   unsigned VirtReg,
                   unsigned PhysReg) {
      // check if we are replacing a previous mapping
      if (p2vMap_[PhysReg] != VirtReg)
        vacatePhysReg(MBB, MII, PhysReg);

      p2vMap_[PhysReg] = VirtReg;
      dirty_[PhysReg] = true;
      lastDef_[VirtReg] = MII;
    }

    void eliminateVirtRegsInMBB(MachineBasicBlock& MBB);
  };
}

bool LocalSpiller::runOnMachineFunction(MachineFunction &mf,
                                        const VirtRegMap &vrm) {
  MF = &mf;
  TM = &MF->getTarget();
  TII = TM->getInstrInfo();
  MRI = TM->getRegisterInfo();
  VRM = &vrm;
  p2vMap_.assign(MRI->getNumRegs(), 0);
  dirty_.assign(MRI->getNumRegs(), false);

  DEBUG(std::cerr << "********** REWRITE MACHINE CODE **********\n");
  DEBUG(std::cerr << "********** Function: "
        << MF->getFunction()->getName() << '\n');

  for (MachineFunction::iterator MBB = MF->begin(), E = MF->end();
       MBB != E; ++MBB) {
    lastDef_.grow(MF->getSSARegMap()->getLastVirtReg());
    DEBUG(std::cerr << MBB->getBasicBlock()->getName() << ":\n");
    eliminateVirtRegsInMBB(*MBB);
    // clear map, dirty flag and last ref
    p2vMap_.assign(p2vMap_.size(), 0);
    dirty_.assign(dirty_.size(), false);
    lastDef_.clear();
  }
  return true;
}

void LocalSpiller::vacateJustPhysReg(MachineBasicBlock& MBB,
                                     MachineBasicBlock::iterator MII,
                                     unsigned PhysReg) {
  unsigned VirtReg = p2vMap_[PhysReg];
  if (dirty_[PhysReg] && VRM->hasStackSlot(VirtReg)) {
    assert(lastDef_[VirtReg] && "virtual register is mapped "
           "to a register and but was not defined!");
    MachineBasicBlock::iterator lastDef = lastDef_[VirtReg];
    MachineBasicBlock::iterator nextLastRef = next(lastDef);
    MRI->storeRegToStackSlot(*lastDef->getParent(),
                              nextLastRef,
                              PhysReg,
                              VRM->getStackSlot(VirtReg));
    ++NumStores;
    DEBUG(std::cerr << "added: " << *prior(nextLastRef);
          std::cerr << "after: " << *lastDef);
    lastDef_[VirtReg] = 0;
  }
  p2vMap_[PhysReg] = 0;
  dirty_[PhysReg] = false;
}

void LocalSpiller::eliminateVirtRegsInMBB(MachineBasicBlock &MBB) {
  for (MachineBasicBlock::iterator MI = MBB.begin(), E = MBB.end();
       MI != E; ++MI) {

    // if we have references to memory operands make sure
    // we clear all physical registers that may contain
    // the value of the spilled virtual register
    VirtRegMap::MI2VirtMapTy::const_iterator i, e;
    for (tie(i, e) = VRM->getFoldedVirts(MI); i != e; ++i) {
      if (VRM->hasPhys(i->second))
        vacateJustPhysReg(MBB, MI, VRM->getPhys(i->second));
    }

    // rewrite all used operands
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand& op = MI->getOperand(i);
      if (op.isRegister() && op.getReg() && op.isUse() &&
          MRegisterInfo::isVirtualRegister(op.getReg())) {
        unsigned VirtReg = op.getReg();
        unsigned PhysReg = VRM->getPhys(VirtReg);
        handleUse(MBB, MI, VirtReg, PhysReg);
        MI->SetMachineOperandReg(i, PhysReg);
        // mark as dirty if this is def&use
        if (op.isDef()) {
          dirty_[PhysReg] = true;
          lastDef_[VirtReg] = MI;
        }
      }
    }

    // spill implicit physical register defs
    const TargetInstrDescriptor& tid = TII->get(MI->getOpcode());
    for (const unsigned* id = tid.ImplicitDefs; *id; ++id)
      vacatePhysReg(MBB, MI, *id);

    // spill explicit physical register defs
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand& op = MI->getOperand(i);
      if (op.isRegister() && op.getReg() && !op.isUse() &&
          MRegisterInfo::isPhysicalRegister(op.getReg()))
        vacatePhysReg(MBB, MI, op.getReg());
    }

    // rewrite def operands (def&use was handled with the
    // uses so don't check for those here)
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand& op = MI->getOperand(i);
      if (op.isRegister() && op.getReg() && !op.isUse())
        if (MRegisterInfo::isPhysicalRegister(op.getReg()))
          vacatePhysReg(MBB, MI, op.getReg());
        else {
          unsigned PhysReg = VRM->getPhys(op.getReg());
          handleDef(MBB, MI, op.getReg(), PhysReg);
          MI->SetMachineOperandReg(i, PhysReg);
        }
    }

    DEBUG(std::cerr << '\t' << *MI);
  }

  for (unsigned i = 1, e = p2vMap_.size(); i != e; ++i)
    vacateJustPhysReg(MBB, MBB.getFirstTerminator(), i);
}


llvm::Spiller* llvm::createSpiller() {
  switch (SpillerOpt) {
  default: assert(0 && "Unreachable!");
  case local:
    return new LocalSpiller();
  case simple:
    return new SimpleSpiller();
  }
}

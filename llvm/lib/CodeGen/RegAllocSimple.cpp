//===-- RegAllocSimple.cpp - A simple generic register allocator --- ------===//
//
// This file implements a simple register allocator. *Very* simple.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/Statistic.h"
#include <iostream>

/// PhysRegClassMap - Construct a mapping of physical register numbers to their
/// register classes.
///
/// NOTE: This class will eventually be pulled out to somewhere shared.
///
class PhysRegClassMap {
  std::map<unsigned, const TargetRegisterClass*> PhysReg2RegClassMap;
public:
  PhysRegClassMap(const MRegisterInfo *RI) {
    for (MRegisterInfo::const_iterator I = RI->regclass_begin(),
           E = RI->regclass_end(); I != E; ++I)
      for (unsigned i=0; i < (*I)->getNumRegs(); ++i)
        PhysReg2RegClassMap[(*I)->getRegister(i)] = *I;
  }

  const TargetRegisterClass *operator[](unsigned Reg) {
    assert(PhysReg2RegClassMap[Reg] && "Register is not a known physreg!");
    return PhysReg2RegClassMap[Reg];
  }

  const TargetRegisterClass *get(unsigned Reg) { return operator[](Reg); }
};


namespace {
  struct RegAllocSimple : public FunctionPass {
    TargetMachine &TM;
    MachineFunction *MF;
    const MRegisterInfo *RegInfo;
    unsigned NumBytesAllocated;
    
    // Maps SSA Regs => offsets on the stack where these values are stored
    std::map<unsigned, unsigned> VirtReg2OffsetMap;

    // Maps SSA Regs => physical regs
    std::map<unsigned, unsigned> SSA2PhysRegMap;

    // Maps physical register to their register classes
    PhysRegClassMap PhysRegClasses;

    // Made to combat the incorrect allocation of r2 = add r1, r1
    std::map<unsigned, unsigned> VirtReg2PhysRegMap;
    
    // Maps RegClass => which index we can take a register from. Since this is a
    // simple register allocator, when we need a register of a certain class, we
    // just take the next available one.
    std::map<unsigned, unsigned> RegsUsed;
    std::map<const TargetRegisterClass*, unsigned> RegClassIdx;

    RegAllocSimple(TargetMachine &tm)
      : TM(tm), RegInfo(tm.getRegisterInfo()), PhysRegClasses(RegInfo) {
      RegsUsed[RegInfo->getFramePointer()] = 1;
      RegsUsed[RegInfo->getStackPointer()] = 1;

      cleanupAfterFunction();
    }

    bool isAvailableReg(unsigned Reg) {
      // assert(Reg < MRegisterInfo::FirstVirtualReg && "...");
      return RegsUsed.find(Reg) == RegsUsed.end();
    }

    /// allocateStackSpaceFor - This allocates space for the specified virtual
    /// register to be held on the stack.
    unsigned allocateStackSpaceFor(unsigned VirtReg, 
                                   const TargetRegisterClass *regClass);

    /// Given size (in bytes), returns a register that is currently unused
    /// Side effect: marks that register as being used until manually cleared
    unsigned getFreeReg(unsigned virtualReg);

    /// Returns all `borrowed' registers back to the free pool
    void clearAllRegs() {
      RegClassIdx.clear();
    }

    /// Invalidates any references, real or implicit, to physical registers
    ///
    void invalidatePhysRegs(const MachineInstr *MI) {
      unsigned Opcode = MI->getOpcode();
      const MachineInstrInfo &MII = TM.getInstrInfo();
      const MachineInstrDescriptor &Desc = MII.get(Opcode);
      const unsigned *regs = Desc.ImplicitUses;
      while (*regs)
        RegsUsed[*regs++] = 1;

      regs = Desc.ImplicitDefs;
      while (*regs)
        RegsUsed[*regs++] = 1;
    }

    void cleanupAfterFunction() {
      VirtReg2OffsetMap.clear();
      SSA2PhysRegMap.clear();
      NumBytesAllocated = 4;   // FIXME: This is X86 specific
    }

    /// Moves value from memory into that register
    MachineBasicBlock::iterator
    moveUseToReg (MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I, unsigned VirtReg,
                  unsigned &PhysReg);

    /// Saves reg value on the stack (maps virtual register to stack value)
    MachineBasicBlock::iterator
    saveVirtRegToStack (MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator I, unsigned VirtReg,
                        unsigned PhysReg);

    MachineBasicBlock::iterator
    savePhysRegToStack (MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator I, unsigned PhysReg);

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnMachineFunction(MachineFunction &Fn);

    /// AllocateBasicBlock - Register allocate the specified basic block.
    void AllocateBasicBlock(MachineBasicBlock &MBB);

    /// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions
    /// in predecessor basic blocks.
    void EliminatePHINodes(MachineBasicBlock &MBB);

    bool runOnFunction(Function &Fn) {
      return runOnMachineFunction(MachineFunction::get(&Fn));
    }
  };

}

/// allocateStackSpaceFor - This allocates space for the specified virtual
/// register to be held on the stack.
unsigned RegAllocSimple::allocateStackSpaceFor(unsigned VirtReg,
                                            const TargetRegisterClass *regClass)
{
  if (VirtReg2OffsetMap.find(VirtReg) == VirtReg2OffsetMap.end()) {
    unsigned RegSize = regClass->getDataSize();

    // Align NumBytesAllocated.  We should be using TargetData alignment stuff
    // to determine this, but we don't know the LLVM type associated with the
    // virtual register.  Instead, just align to a multiple of the size for now.
    NumBytesAllocated += RegSize-1;
    NumBytesAllocated = NumBytesAllocated/RegSize*RegSize;

    // Assign the slot...
    VirtReg2OffsetMap[VirtReg] = NumBytesAllocated;

    // Reserve the space!
    NumBytesAllocated += RegSize;
  }
  return VirtReg2OffsetMap[VirtReg];
}

unsigned RegAllocSimple::getFreeReg(unsigned virtualReg) {
  const TargetRegisterClass* regClass = MF->getRegClass(virtualReg);
  unsigned physReg;
  assert(regClass);
  if (RegClassIdx.find(regClass) != RegClassIdx.end()) {
    unsigned regIdx = RegClassIdx[regClass]++;
    assert(regIdx < regClass->getNumRegs() && "Not enough registers!");
    physReg = regClass->getRegister(regIdx);
  } else {
    physReg = regClass->getRegister(0);
    // assert(physReg < regClass->getNumRegs() && "No registers in class!");
    RegClassIdx[regClass] = 1;
  }

  if (isAvailableReg(physReg))
    return physReg;
  else {
    return getFreeReg(virtualReg);
  }
}

MachineBasicBlock::iterator
RegAllocSimple::moveUseToReg (MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I,
                              unsigned VirtReg, unsigned &PhysReg)
{
  const TargetRegisterClass* regClass = MF->getRegClass(VirtReg);
  assert(regClass);

  unsigned stackOffset = allocateStackSpaceFor(VirtReg, regClass);
  PhysReg = getFreeReg(VirtReg);

  // Add move instruction(s)
  return RegInfo->loadRegOffset2Reg(MBB, I, PhysReg,
                                    RegInfo->getFramePointer(),
                                    -stackOffset, regClass->getDataSize());
}

MachineBasicBlock::iterator
RegAllocSimple::saveVirtRegToStack (MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    unsigned VirtReg, unsigned PhysReg)
{
  const TargetRegisterClass* regClass = MF->getRegClass(VirtReg);
  assert(regClass);

  unsigned stackOffset = allocateStackSpaceFor(VirtReg, regClass);

  // Add move instruction(s)
  return RegInfo->storeReg2RegOffset(MBB, I, PhysReg,
                                     RegInfo->getFramePointer(),
                                     -stackOffset, regClass->getDataSize());
}

MachineBasicBlock::iterator
RegAllocSimple::savePhysRegToStack (MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    unsigned PhysReg)
{
  const TargetRegisterClass* regClass = MF->getRegClass(PhysReg);
  assert(regClass);

  unsigned offset = allocateStackSpaceFor(PhysReg, regClass);

  // Add move instruction(s)
  return RegInfo->storeReg2RegOffset(MBB, I, PhysReg,
                                     RegInfo->getFramePointer(),
                                     offset, regClass->getDataSize());
}

/// EliminatePHINodes - Eliminate phi nodes by inserting copy instructions in
/// predecessor basic blocks.
void RegAllocSimple::EliminatePHINodes(MachineBasicBlock &MBB) {
  while (MBB.front()->getOpcode() == 0) {
    MachineInstr *MI = MBB.front();
    // get rid of the phi
    MBB.erase(MBB.begin());
    
    // a preliminary pass that will invalidate any registers that
    // are used by the instruction (including implicit uses)
    invalidatePhysRegs(MI);
    
    DEBUG(std::cerr << "num invalid regs: " << RegsUsed.size() << "\n");
    
    DEBUG(std::cerr << "num ops: " << MI->getNumOperands() << "\n");
    MachineOperand &targetReg = MI->getOperand(0);
    
    // If it's a virtual register, allocate a physical one
    // otherwise, just use whatever register is there now
    // note: it MUST be a register -- we're assigning to it
    unsigned virtualReg = (unsigned) targetReg.getAllocatedRegNum();
    unsigned physReg;
    if (targetReg.isVirtualRegister()) {
      physReg = getFreeReg(virtualReg);
    } else {
      physReg = virtualReg;
    }
    
    // Find the register class of the target register: should be the
    // same as the values we're trying to store there
    const TargetRegisterClass* regClass = PhysRegClasses[physReg];
    assert(regClass && "Target register class not found!");
    unsigned dataSize = regClass->getDataSize();
    
    for (int i = MI->getNumOperands() - 1; i >= 2; i-=2) {
      MachineOperand &opVal = MI->getOperand(i-1);
      
      // Get the MachineBasicBlock equivalent of the BasicBlock that is the
      // source path the phi
      MachineBasicBlock &opBlock = *MI->getOperand(i).getMachineBasicBlock();
      MachineBasicBlock::iterator opI = opBlock.end();
      MachineInstr *opMI = *--opI;
      const MachineInstrInfo &MII = TM.getInstrInfo();

      // must backtrack over ALL the branches in the previous block, until no
      // more
      while (MII.isBranch(opMI->getOpcode()) && opI != opBlock.begin())
        opMI = *--opI;

      // move back to the first branch instruction so new instructions
      // are inserted right in front of it and not in front of a non-branch
      if (!MII.isBranch(opMI->getOpcode()))
        ++opI;
      
      // Retrieve the constant value from this op, move it to target
      // register of the phi
      if (opVal.isImmediate()) {
        opI = RegInfo->moveImm2Reg(opBlock, opI, physReg,
                                   (unsigned) opVal.getImmedValue(),
                                   dataSize);
        saveVirtRegToStack(opBlock, opI, virtualReg, physReg);
      } else {
        // Allocate a physical register and add a move in the BB
        unsigned opVirtualReg = (unsigned) opVal.getAllocatedRegNum();
        unsigned opPhysReg; // = getFreeReg(opVirtualReg);
        opI = moveUseToReg(opBlock, opI, opVirtualReg, physReg);
        //opI = RegInfo->moveReg2Reg(opBlock, opI, physReg, opPhysReg,
        //                           dataSize);
        // Save that register value to the stack of the TARGET REG
        saveVirtRegToStack(opBlock, opI, virtualReg, physReg);
      }
      
      // make regs available to other instructions
      clearAllRegs();
    }
    
    // really delete the instruction
    delete MI;
  }
}


void RegAllocSimple::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // Handle PHI instructions specially: add moves to each pred block
  EliminatePHINodes(MBB);
  
  //loop over each basic block
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
    MachineInstr *MI = *I;
    
    // a preliminary pass that will invalidate any registers that
    // are used by the instruction (including implicit uses)
    invalidatePhysRegs(MI);
    
    // Loop over uses, move from memory into registers
    for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
      MachineOperand &op = MI->getOperand(i);
      
      if (op.isImmediate()) {
        DEBUG(std::cerr << "const\n");
      } else if (op.isVirtualRegister()) {
        unsigned virtualReg = (unsigned) op.getAllocatedRegNum();
        DEBUG(std::cerr << "op: " << op << "\n");
        DEBUG(std::cerr << "\t inst[" << i << "]: ";
              MI->print(std::cerr, TM));
        
        // make sure the same virtual register maps to the same physical
        // register in any given instruction
        unsigned physReg;
        if (VirtReg2PhysRegMap.find(virtualReg) != VirtReg2PhysRegMap.end()) {
          physReg = VirtReg2PhysRegMap[virtualReg];
        } else {
          if (op.opIsDef()) {
            if (TM.getInstrInfo().isTwoAddrInstr(MI->getOpcode()) && i == 0) {
              // must be same register number as the first operand
              // This maps a = b + c into b += c, and saves b into a's spot
              physReg = (unsigned) MI->getOperand(1).getAllocatedRegNum();
            } else {
              physReg = getFreeReg(virtualReg);
            }
            MachineBasicBlock::iterator J = I;
            J = saveVirtRegToStack(MBB, ++J, virtualReg, physReg);
            I = --J;
          } else {
            I = moveUseToReg(MBB, I, virtualReg, physReg);
          }
          VirtReg2PhysRegMap[virtualReg] = physReg;
        }
        MI->SetMachineOperandReg(i, physReg);
        DEBUG(std::cerr << "virt: " << virtualReg << 
              ", phys: " << op.getAllocatedRegNum() << "\n");
      }
    }
    
    clearAllRegs();
    VirtReg2PhysRegMap.clear();
  }
}

bool RegAllocSimple::runOnMachineFunction(MachineFunction &Fn) {
  DEBUG(std::cerr << "Machine Function " << "\n");
  MF = &Fn;

  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  // add prologue we should preserve callee-save registers...
  RegInfo->emitPrologue(Fn, NumBytesAllocated);

  const MachineInstrInfo &MII = TM.getInstrInfo();

  // add epilogue to restore the callee-save registers
  // loop over the basic block
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    // check if last instruction is a RET
    MachineBasicBlock::iterator I = MBB->end();
    MachineInstr *MI = *--I;
    if (MII.isReturn(MI->getOpcode())) {
      // this block has a return instruction, add epilogue
      RegInfo->emitEpilogue(*MBB, NumBytesAllocated);
    }
  }

  cleanupAfterFunction();
  return false;  // We never modify the LLVM itself.
}

Pass *createSimpleX86RegisterAllocator(TargetMachine &TM) {
  return new RegAllocSimple(TM);
}

//===-- RegAllocSimple.cpp - A simple generic register allocator --- ------===//
//
// This file implements a simple register allocator. *Very* simple.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/MachineRegInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/Statistic.h"
#include <map>

namespace {
  struct RegAllocSimple : public FunctionPass {
    TargetMachine &TM;
    MachineBasicBlock *CurrMBB;
    MachineFunction *MF;
    unsigned maxOffset;
    const MRegisterInfo *RegInfo;
    unsigned NumBytesAllocated, ByteAlignment;
    
    // Maps SSA Regs => offsets on the stack where these values are stored
    // FIXME: change name to VirtReg2OffsetMap
    std::map<unsigned, unsigned> RegMap;

    // Maps SSA Regs => physical regs
    std::map<unsigned, unsigned> SSA2PhysRegMap;

    // Maps physical register to their register classes
    std::map<unsigned, const TargetRegisterClass*> PhysReg2RegClassMap;

    // Made to combat the incorrect allocation of r2 = add r1, r1
    std::map<unsigned, unsigned> VirtReg2PhysRegMap;
    
    // Maps RegClass => which index we can take a register from. Since this is a
    // simple register allocator, when we need a register of a certain class, we
    // just take the next available one.
    std::map<unsigned, unsigned> RegsUsed;
    std::map<const TargetRegisterClass*, unsigned> RegClassIdx;

    RegAllocSimple(TargetMachine &tm) : TM(tm), CurrMBB(0), maxOffset(0), 
                                        RegInfo(tm.getRegisterInfo()),
                                        ByteAlignment(4)
    {
      // build reverse mapping for physReg -> register class
      RegInfo->buildReg2RegClassMap(PhysReg2RegClassMap);

      RegsUsed[RegInfo->getFramePointer()] = 1;
      RegsUsed[RegInfo->getStackPointer()] = 1;

      cleanupAfterFunction();
    }

    bool isAvailableReg(unsigned Reg) {
      // assert(Reg < MRegisterInfo::FirstVirtualReg && "...");
      return RegsUsed.find(Reg) == RegsUsed.end();
    }

    ///
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
      RegMap.clear();
      SSA2PhysRegMap.clear();
      NumBytesAllocated = ByteAlignment;
    }

    /// Moves value from memory into that register
    MachineBasicBlock::iterator
    moveUseToReg (MachineBasicBlock *MBB,
                  MachineBasicBlock::iterator I, unsigned VirtReg,
                  unsigned &PhysReg);

    /// Saves reg value on the stack (maps virtual register to stack value)
    MachineBasicBlock::iterator
    saveVirtRegToStack (MachineBasicBlock *MBB,
                        MachineBasicBlock::iterator I, unsigned VirtReg,
                        unsigned PhysReg);

    MachineBasicBlock::iterator
    savePhysRegToStack (MachineBasicBlock *MBB,
                        MachineBasicBlock::iterator I, unsigned PhysReg);

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnMachineFunction(MachineFunction &Fn);

    bool runOnFunction(Function &Fn) {
      return runOnMachineFunction(MachineFunction::get(&Fn));
    }
  };

}

unsigned RegAllocSimple::allocateStackSpaceFor(unsigned VirtReg,
                                            const TargetRegisterClass *regClass)
{
  if (RegMap.find(VirtReg) == RegMap.end()) {
#if 0
    unsigned size = regClass->getDataSize();
    unsigned over = NumBytesAllocated - (NumBytesAllocated % ByteAlignment);
    if (size >= ByteAlignment - over) {
      // need to pad by (ByteAlignment - over)
      NumBytesAllocated += ByteAlignment - over;
    }
    RegMap[VirtReg] = NumBytesAllocated;
    NumBytesAllocated += size;
#endif
    // FIXME: forcing each arg to take 4 bytes on the stack
    RegMap[VirtReg] = NumBytesAllocated;
    NumBytesAllocated += ByteAlignment;
  }
  return RegMap[VirtReg];
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
RegAllocSimple::moveUseToReg (MachineBasicBlock *MBB,
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
RegAllocSimple::saveVirtRegToStack (MachineBasicBlock *MBB,
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
RegAllocSimple::savePhysRegToStack (MachineBasicBlock *MBB,
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

bool RegAllocSimple::runOnMachineFunction(MachineFunction &Fn) {
  cleanupAfterFunction();

  unsigned virtualReg, physReg;
  DEBUG(std::cerr << "Machine Function " << "\n");
  MF = &Fn;

  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
  {
    CurrMBB = &(*MBB);

    // Handle PHI instructions specially: add moves to each pred block
    while (MBB->front()->getOpcode() == 0) {
      MachineInstr *MI = MBB->front();
      // get rid of the phi
      MBB->erase(MBB->begin());
    
      // a preliminary pass that will invalidate any registers that
      // are used by the instruction (including implicit uses)
      invalidatePhysRegs(MI);

      DEBUG(std::cerr << "num invalid regs: " << RegsUsed.size() << "\n");

      DEBUG(std::cerr << "num ops: " << MI->getNumOperands() << "\n");
      MachineOperand &targetReg = MI->getOperand(0);

      // If it's a virtual register, allocate a physical one
      // otherwise, just use whatever register is there now
      // note: it MUST be a register -- we're assigning to it
      virtualReg = (unsigned) targetReg.getAllocatedRegNum();
      if (targetReg.isVirtualRegister()) {
        physReg = getFreeReg(virtualReg);
      } else {
        physReg = targetReg.getAllocatedRegNum();
      }

      // Find the register class of the target register: should be the
      // same as the values we're trying to store there
      const TargetRegisterClass* regClass = PhysReg2RegClassMap[physReg];
      assert(regClass && "Target register class not found!");
      unsigned dataSize = regClass->getDataSize();

      for (int i = MI->getNumOperands() - 1; i >= 2; i-=2) {
        MachineOperand &opVal = MI->getOperand(i-1);

        // Get the MachineBasicBlock equivalent of the BasicBlock that is the
        // source path the phi
        BasicBlock *opBB =
          cast<BasicBlock>(MI->getOperand(i).getVRegValue());
        MachineBasicBlock *opBlock = NULL;
        for (MachineFunction::iterator opFi = Fn.begin(), opFe = Fn.end();
             opFi != opFe; ++opFi)
        {
          if (opFi->getBasicBlock() == opBB) {
            opBlock = opFi; break;
          }
        }
        assert(opBlock && "MachineBasicBlock object not found for specified block!");

        MachineBasicBlock::iterator opI = opBlock->end();
        MachineInstr *opMI = *(--opI);
        const MachineInstrInfo &MII = TM.getInstrInfo();
        // must backtrack over ALL the branches in the previous block, until no more
        while ((MII.isBranch(opMI->getOpcode()) || MII.isReturn(opMI->getOpcode()))
               && opI != opBlock->begin())
        {
          opMI = *(--opI);
        }
        // move back to the first branch instruction so new instructions
        // are inserted right in front of it and not in front of a non-branch
        ++opI; 


        // insert the move just before the return/branch
        if (MII.isReturn(opMI->getOpcode()) || MII.isBranch(opMI->getOpcode()))
        {
          // Retrieve the constant value from this op, move it to target
          // register of the phi
          if (opVal.getType() == MachineOperand::MO_SignExtendedImmed ||
              opVal.getType() == MachineOperand::MO_UnextendedImmed)
          {
            opI = RegInfo->moveImm2Reg(opBlock, opI, physReg,
                                       (unsigned) opVal.getImmedValue(),
                                       dataSize);
            saveVirtRegToStack(opBlock, opI, virtualReg, physReg);
          } else {
            // Allocate a physical register and add a move in the BB
            unsigned opVirtualReg = (unsigned) opVal.getAllocatedRegNum();
            unsigned opPhysReg; // = getFreeReg(opVirtualReg);
            opI = moveUseToReg(opBlock, opI, opVirtualReg, opPhysReg);
            opI = RegInfo->moveReg2Reg(opBlock, opI, physReg, opPhysReg,
                                       dataSize);
            // Save that register value to the stack of the TARGET REG
            saveVirtRegToStack(opBlock, opI, virtualReg, opPhysReg);
          }
        } 

        // make regs available to other instructions
        clearAllRegs();
      }
      
      // really delete the instruction
      delete MI;
    }

    //loop over each basic block
    for (MachineBasicBlock::iterator I = MBB->begin(); I != MBB->end(); ++I)
    {
      MachineInstr *MI = *I;

      // a preliminary pass that will invalidate any registers that
      // are used by the instruction (including implicit uses)
      invalidatePhysRegs(MI);

      // Loop over uses, move from memory into registers
      for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
        MachineOperand &op = MI->getOperand(i);

        if (op.getType() == MachineOperand::MO_SignExtendedImmed ||
            op.getType() == MachineOperand::MO_UnextendedImmed)
        {
          DEBUG(std::cerr << "const\n");
        } else if (op.isVirtualRegister()) {
          virtualReg = (unsigned) op.getAllocatedRegNum();
          DEBUG(std::cerr << "op: " << op << "\n");
          DEBUG(std::cerr << "\t inst[" << i << "]: ";
                MI->print(std::cerr, TM));

          // make sure the same virtual register maps to the same physical
          // register in any given instruction
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
              J = saveVirtRegToStack(CurrMBB, ++J, virtualReg, physReg);
              I = --J;
            } else {
              I = moveUseToReg(CurrMBB, I, virtualReg, physReg);
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

  // add prologue we should preserve callee-save registers...
  MachineFunction::iterator Fi = Fn.begin();
  MachineBasicBlock *MBB = Fi;
  MachineBasicBlock::iterator MBBi = MBB->begin();
  RegInfo->emitPrologue(MBB, MBBi, NumBytesAllocated);

  // add epilogue to restore the callee-save registers
  // loop over the basic block
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
  {
    // check if last instruction is a RET
    MachineBasicBlock::iterator I = (*MBB).end();
    MachineInstr *MI = *(--I);
    const MachineInstrInfo &MII = TM.getInstrInfo();
    if (MII.isReturn(MI->getOpcode())) {
      // this block has a return instruction, add epilogue
      RegInfo->emitEpilogue(MBB, I, NumBytesAllocated);
    }
  }

  return false;  // We never modify the LLVM itself.
}

Pass *createSimpleX86RegisterAllocator(TargetMachine &TM) {
  return new RegAllocSimple(TM);
}

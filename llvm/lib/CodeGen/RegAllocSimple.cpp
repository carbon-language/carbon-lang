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
    std::map<unsigned, unsigned> RegMap; // FIXME: change name to OffsetMap

    // Maps SSA Regs => physical regs
    std::map<unsigned, unsigned> SSA2PhysRegMap;
    
    // Maps RegClass => which index we can take a register from. Since this is a
    // simple register allocator, when we need a register of a certain class, we
    // just take the next available one.
    std::map<unsigned, unsigned> RegsUsed;
    std::map<const TargetRegisterClass*, unsigned> RegClassIdx;

    RegAllocSimple(TargetMachine &tm) : TM(tm), CurrMBB(0), maxOffset(0), 
                                        RegInfo(tm.getRegisterInfo()),
                                        NumBytesAllocated(0), ByteAlignment(4)
    {
      RegsUsed[RegInfo->getFramePointer()] = 1;
      RegsUsed[RegInfo->getStackPointer()] = 1;
    }

    bool isAvailableReg(unsigned Reg) {
      // assert(Reg < MRegisterInfo::FirstVirtualReg && "...");
      return RegsUsed.find(Reg) == RegsUsed.end();
    }

    /// Given size (in bytes), returns a register that is currently unused
    /// Side effect: marks that register as being used until manually cleared
    unsigned getFreeReg(unsigned virtualReg);

    /// Returns all `borrowed' registers back to the free pool
    void clearAllRegs() {
        RegClassIdx.clear();
    }

    /// Moves value from memory into that register
    MachineBasicBlock::iterator
    moveUseToReg (MachineBasicBlock::iterator I, unsigned VirtReg,
                  unsigned &PhysReg);

    /// Saves reg value on the stack (maps virtual register to stack value)
    MachineBasicBlock::iterator
    saveRegToStack (MachineBasicBlock::iterator I, unsigned VirtReg,
                    unsigned PhysReg);

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnMachineFunction(MachineFunction &Fn);

    bool runOnFunction(Function &Fn) {
      return runOnMachineFunction(MachineFunction::get(&Fn));
    }
  };

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
RegAllocSimple::moveUseToReg (MachineBasicBlock::iterator I,
                              unsigned VirtReg, unsigned &PhysReg)
{
  const TargetRegisterClass* regClass = MF->getRegClass(VirtReg);
  assert(regClass);

  unsigned stackOffset;
  if (RegMap.find(VirtReg) == RegMap.end()) {
    unsigned size = regClass->getDataSize();
    unsigned over = NumBytesAllocated - (NumBytesAllocated % ByteAlignment);
    if (size >= ByteAlignment - over) {
      // need to pad by (ByteAlignment - over)
      NumBytesAllocated += ByteAlignment - over;
    }
    RegMap[VirtReg] = NumBytesAllocated;
    NumBytesAllocated += size;
  }
  stackOffset = RegMap[VirtReg];
  PhysReg = getFreeReg(VirtReg);

  // Add move instruction(s)
  MachineBasicBlock::iterator newI =
    RegInfo->loadRegOffset2Reg(CurrMBB, I, PhysReg,
                               RegInfo->getFramePointer(),
                               stackOffset, regClass->getDataSize());

  // FIXME: increment the frame pointer

  return newI;
}

MachineBasicBlock::iterator
RegAllocSimple::saveRegToStack (MachineBasicBlock::iterator I,
                                unsigned VirtReg, unsigned PhysReg)
{
  const TargetRegisterClass* regClass = MF->getRegClass(VirtReg);
  assert(regClass);
  assert(RegMap.find(VirtReg) != RegMap.end() &&
         "Virtual reg has no stack offset mapping!");

  unsigned offset = RegMap[VirtReg];
  // Add move instruction(s)
  return RegInfo->storeReg2RegOffset(CurrMBB, I, PhysReg,
                                     RegInfo->getFramePointer(),
                                     offset, regClass->getDataSize());
}

bool RegAllocSimple::runOnMachineFunction(MachineFunction &Fn) {
  RegMap.clear();
  unsigned virtualReg, physReg;
  DEBUG(std::cerr << "Machine Function " << "\n");
  MF = &Fn;
  // FIXME: add prolog. we should preserve callee-save registers...

  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
  {
    CurrMBB = &(*MBB);

    // FIXME: if return, special case => into return register
    //loop over each basic block
    for (MachineBasicBlock::iterator I = MBB->begin(); I != MBB->end(); ++I)
    {
      MachineInstr *MI = *I;

      DEBUG(std::cerr << "instr: ";
            MI->print(std::cerr, TM));

      // Loop over each instruction:
      // uses, move from memory into registers
      for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
        MachineOperand &op = MI->getOperand(i);

        if (op.getType() == MachineOperand::MO_SignExtendedImmed ||
            op.getType() == MachineOperand::MO_UnextendedImmed)
        {
          DEBUG(std::cerr << "const\n");
        } else if (op.isVirtualRegister()) {
          virtualReg = (unsigned) op.getAllocatedRegNum();
#if 0
          // FIXME: save register to stack
          if (op.opIsDef()) {
            MachineBasicBlock::iterator J = I;
            saveRegToStack(++J, virtualReg, physReg);
          }
#endif
          DEBUG(std::cerr << "op: " << op << "\n");
          DEBUG(std::cerr << "\t inst[" << i << "]: ";
                MI->print(std::cerr, TM));
          I = moveUseToReg(I, virtualReg, physReg);
          //MI = *I;
          bool def = op.opIsDef() || op.opIsDefAndUse();
          MI->SetMachineOperandReg(i, physReg, def);
          DEBUG(std::cerr << "virt: " << virtualReg << 
                ", phys: " << op.getAllocatedRegNum() << "\n");
        }
      }

      clearAllRegs();
    }

  }

  // FIXME: add epilog. we should preserve callee-save registers...

  return false;  // We never modify the LLVM itself.
}

Pass *createSimpleX86RegisterAllocator(TargetMachine &TM) {
  return new RegAllocSimple(TM);
}

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
    std::map<unsigned, unsigned> RegMap; // FIXME: change name to VirtReg2OffsetMap

    // Maps SSA Regs => physical regs
    std::map<unsigned, unsigned> SSA2PhysRegMap;

    // Maps physical register to their register classes
    std::map<unsigned, const TargetRegisterClass*> PhysReg2RegClassMap;
    
    // Maps RegClass => which index we can take a register from. Since this is a
    // simple register allocator, when we need a register of a certain class, we
    // just take the next available one.
    std::map<unsigned, unsigned> RegsUsed;
    std::map<const TargetRegisterClass*, unsigned> RegClassIdx;

    RegAllocSimple(TargetMachine &tm) : TM(tm), CurrMBB(0), maxOffset(0), 
                                        RegInfo(tm.getRegisterInfo()),
                                        NumBytesAllocated(0), ByteAlignment(4)
    {
      // build reverse mapping for physReg -> register class
      RegInfo->buildReg2RegClassMap(PhysReg2RegClassMap);

      RegsUsed[RegInfo->getFramePointer()] = 1;
      RegsUsed[RegInfo->getStackPointer()] = 1;
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

    /// Moves value from memory into that register
    MachineBasicBlock::iterator
    moveUseToReg (MachineBasicBlock::iterator I, unsigned VirtReg,
                  unsigned &PhysReg);

    /// Saves reg value on the stack (maps virtual register to stack value)
    MachineBasicBlock::iterator
    saveVirtRegToStack (MachineBasicBlock::iterator I, unsigned VirtReg,
                        unsigned PhysReg);

    MachineBasicBlock::iterator
    savePhysRegToStack (MachineBasicBlock::iterator I, unsigned PhysReg);

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
    unsigned size = regClass->getDataSize();
    unsigned over = NumBytesAllocated - (NumBytesAllocated % ByteAlignment);
    if (size >= ByteAlignment - over) {
      // need to pad by (ByteAlignment - over)
      NumBytesAllocated += ByteAlignment - over;
    }
    RegMap[VirtReg] = NumBytesAllocated;
    NumBytesAllocated += size;
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
RegAllocSimple::moveUseToReg (MachineBasicBlock::iterator I,
                              unsigned VirtReg, unsigned &PhysReg)
{
  const TargetRegisterClass* regClass = MF->getRegClass(VirtReg);
  assert(regClass);

  unsigned stackOffset = allocateStackSpaceFor(VirtReg, regClass);
  PhysReg = getFreeReg(VirtReg);

  // FIXME: increment the frame pointer

  // Add move instruction(s)
  return RegInfo->loadRegOffset2Reg(CurrMBB, I, PhysReg,
                                    RegInfo->getFramePointer(),
                                    stackOffset, regClass->getDataSize());
}

MachineBasicBlock::iterator
RegAllocSimple::saveVirtRegToStack (MachineBasicBlock::iterator I,
                                    unsigned VirtReg, unsigned PhysReg)
{
  const TargetRegisterClass* regClass = MF->getRegClass(VirtReg);
  assert(regClass);

  unsigned offset = allocateStackSpaceFor(VirtReg, regClass);

  // Add move instruction(s)
  return RegInfo->storeReg2RegOffset(CurrMBB, I, PhysReg,
                                     RegInfo->getFramePointer(),
                                     offset, regClass->getDataSize());
}

MachineBasicBlock::iterator
RegAllocSimple::savePhysRegToStack (MachineBasicBlock::iterator I,
                                    unsigned PhysReg)
{
  const TargetRegisterClass* regClass = MF->getRegClass(PhysReg);
  assert(regClass);

  unsigned offset = allocateStackSpaceFor(PhysReg, regClass);

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

#if 0
  // FIXME: add prolog. we should preserve callee-save registers...
  MachineFunction::iterator Fi = Fn.begin();
  MachineBasicBlock &MBB = *Fi;
  MachineBasicBlock::iterator MBBi = MBB.begin()
  const unsigned* calleeSaveRegs = tm.getCalleeSaveRegs();
  while (*calleeSaveRegs) {
    //MBBi = saveRegToStack(MBBi, *calleeSaveRegs, 
    ++calleeSaveRegs;
  }
#endif

  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
  {
    CurrMBB = &(*MBB);

    //loop over each basic block
    for (MachineBasicBlock::iterator I = MBB->begin(); I != MBB->end(); ++I)
    {
      MachineInstr *MI = *I;

      DEBUG(std::cerr << "instr: ";
            MI->print(std::cerr, TM));

      // FIXME: add a preliminary pass that will invalidate any registers that
      // are used by the instruction (including implicit uses)


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
          // save register to stack if it's a def
          DEBUG(std::cerr << "op: " << op << "\n");
          DEBUG(std::cerr << "\t inst[" << i << "]: ";
                MI->print(std::cerr, TM));
          if (op.opIsDef()) {
            physReg = getFreeReg(virtualReg);
            MachineBasicBlock::iterator J = I;
            I = saveVirtRegToStack(J, virtualReg, physReg);
          } else {
            I = moveUseToReg(I, virtualReg, physReg);
          }
          MI->SetMachineOperandReg(i, physReg);
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

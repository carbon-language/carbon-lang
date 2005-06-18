//===-- SparcV9CodeEmitter.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SPARC-specific backend for emitting machine code to memory.
//
// This module also contains the code for lazily resolving the targets of call
// instructions, including the callback used to redirect calls to functions for
// which the code has not yet been generated into the JIT compiler.
//
// This file #includes SparcV9GenCodeEmitter.inc, which contains the code for
// getBinaryCodeForInstr(), a method that converts a MachineInstr into the
// corresponding binary machine code word.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/Debug.h"
#include "SparcV9Internals.h"
#include "SparcV9TargetMachine.h"
#include "SparcV9RegInfo.h"
#include "SparcV9CodeEmitter.h"
#include "SparcV9Relocations.h"
#include "MachineFunctionInfo.h"
using namespace llvm;

bool SparcV9TargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                      MachineCodeEmitter &MCE) {
  PM.add(new SparcV9CodeEmitter(*this, MCE));
  PM.add(createSparcV9MachineCodeDestructionPass());
  return false;
}

SparcV9CodeEmitter::SparcV9CodeEmitter(TargetMachine &tm,
                                       MachineCodeEmitter &M): TM(tm), MCE(M) {}

void SparcV9CodeEmitter::emitWord(unsigned Val) {
  MCE.emitWord(Val);
}

unsigned
SparcV9CodeEmitter::getRealRegNum(unsigned fakeReg,
                                  MachineInstr &MI) {
  const SparcV9RegInfo &RI = *TM.getRegInfo();
  unsigned regClass = 0, regType = RI.getRegType(fakeReg);
  // At least map fakeReg into its class
  fakeReg = RI.getClassRegNum(fakeReg, regClass);

  switch (regClass) {
  case SparcV9RegInfo::IntRegClassID: {
    // SparcV9 manual, p31
    static const unsigned IntRegMap[] = {
      // "o0", "o1", "o2", "o3", "o4", "o5",       "o7",
      8, 9, 10, 11, 12, 13, 15,
      // "l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7",
      16, 17, 18, 19, 20, 21, 22, 23,
      // "i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7",
      24, 25, 26, 27, 28, 29, 30, 31,
      // "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7",
      0, 1, 2, 3, 4, 5, 6, 7,
      // "o6"
      14
    };

    return IntRegMap[fakeReg];
    break;
  }
  case SparcV9RegInfo::FloatRegClassID: {
    DEBUG(std::cerr << "FP reg: " << fakeReg << "\n");
    if (regType == SparcV9RegInfo::FPSingleRegType) {
      // only numbered 0-31, hence can already fit into 5 bits (and 6)
      DEBUG(std::cerr << "FP single reg, returning: " << fakeReg << "\n");
    } else if (regType == SparcV9RegInfo::FPDoubleRegType) {
      // FIXME: This assumes that we only have 5-bit register fields!
      // From SparcV9 Manual, page 40.
      // The bit layout becomes: b[4], b[3], b[2], b[1], b[5]
      fakeReg |= (fakeReg >> 5) & 1;
      fakeReg &= 0x1f;
      DEBUG(std::cerr << "FP double reg, returning: " << fakeReg << "\n");
    }
    return fakeReg;
  }
  case SparcV9RegInfo::IntCCRegClassID: {
    /*                                   xcc, icc, ccr */
    static const unsigned IntCCReg[] = {  6,   4,   2 };

    assert(fakeReg < sizeof(IntCCReg)/sizeof(IntCCReg[0])
             && "CC register out of bounds for IntCCReg map");
    DEBUG(std::cerr << "IntCC reg: " << IntCCReg[fakeReg] << "\n");
    return IntCCReg[fakeReg];
  }
  case SparcV9RegInfo::FloatCCRegClassID: {
    /* These are laid out %fcc0 - %fcc3 => 0 - 3, so are correct */
    DEBUG(std::cerr << "FP CC reg: " << fakeReg << "\n");
    return fakeReg;
  }
  case SparcV9RegInfo::SpecialRegClassID: {
    // Currently only "special" reg is %fsr, which is encoded as 1 in
    // instructions and 0 in SparcV9SpecialRegClass.
    static const unsigned SpecialReg[] = {  1 };
    assert(fakeReg < sizeof(SpecialReg)/sizeof(SpecialReg[0])
             && "Special register out of bounds for SpecialReg map");
    DEBUG(std::cerr << "Special reg: " << SpecialReg[fakeReg] << "\n");
    return SpecialReg[fakeReg];
  }
  default:
    assert(0 && "Invalid unified register number in getRealRegNum");
    return fakeReg;
  }
}



int64_t SparcV9CodeEmitter::getMachineOpValue(MachineInstr &MI,
                                              MachineOperand &MO) {
  int64_t rv = 0; // Return value; defaults to 0 for unhandled cases
                  // or things that get fixed up later by the JIT.
  if (MO.isPCRelativeDisp() || MO.isGlobalAddress()) {
    DEBUG(std::cerr << "PCRelativeDisp: ");
    Value *V = MO.getVRegValue();
    if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
      DEBUG(std::cerr << "Saving reference to BB (VReg)\n");
      unsigned* CurrPC = (unsigned*)(intptr_t)MCE.getCurrentPCValue();
      BBRefs.push_back(std::make_pair(BB, std::make_pair(CurrPC, &MI)));
    } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      // The real target of the branch is CI = PC + (rv * 4)
      // So undo that: give the instruction (CI - PC) / 4
      rv = (CI->getRawValue() - MCE.getCurrentPCValue()) / 4;
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
      unsigned Reloc = 0;
      if (MI.getOpcode() == V9::CALL) {
        Reloc = V9::reloc_pcrel_call;
      } else if (MI.getOpcode() == V9::SETHI) {
        if (MO.isHiBits64())
          Reloc = V9::reloc_sethi_hh;
        else if (MO.isHiBits32())
          Reloc = V9::reloc_sethi_lm;
        else
          assert(0 && "Unknown relocation!");
      } else if (MI.getOpcode() == V9::ORi) {
        if (MO.isLoBits32())
          Reloc = V9::reloc_or_lo;
        else if (MO.isLoBits64())
          Reloc = V9::reloc_or_hm;
        else
          assert(0 && "Unknown relocation!");
      } else {
        assert(0 && "Unknown relocation!");
      }

      MCE.addRelocation(MachineRelocation(MCE.getCurrentPCOffset(), Reloc, GV));
      rv = 0;
    } else {
      std::cerr << "ERROR: PC relative disp unhandled:" << MO << "\n";
      abort();
    }
  } else if (MO.isRegister() || MO.getType() == MachineOperand::MO_CCRegister)
  {
    // This is necessary because the SparcV9 backend doesn't actually lay out
    // registers in the real fashion -- it skips those that it chooses not to
    // allocate, i.e. those that are the FP, SP, etc.
    unsigned fakeReg = MO.getReg();
    unsigned realRegByClass = getRealRegNum(fakeReg, MI);
    DEBUG(std::cerr << MO << ": Reg[" << std::dec << fakeReg << "] => "
                    << realRegByClass << " (LLC: "
                    << TM.getRegInfo()->getUnifiedRegName(fakeReg) << ")\n");
    rv = realRegByClass;
  } else if (MO.isImmediate()) {
    rv = MO.getImmedValue();
    DEBUG(std::cerr << "immed: " << rv << "\n");
  } else if (MO.isMachineBasicBlock()) {
    // Duplicate code of the above case for VirtualRegister, BasicBlock...
    // It should really hit this case, but SparcV9 backend uses VRegs instead
    DEBUG(std::cerr << "Saving reference to MBB\n");
    const BasicBlock *BB = MO.getMachineBasicBlock()->getBasicBlock();
    unsigned* CurrPC = (unsigned*)(intptr_t)MCE.getCurrentPCValue();
    BBRefs.push_back(std::make_pair(BB, std::make_pair(CurrPC, &MI)));
  } else if (MO.isExternalSymbol()) {
    // SparcV9 backend doesn't generate this (yet...)
    std::cerr << "ERROR: External symbol unhandled: " << MO << "\n";
    abort();
  } else if (MO.isFrameIndex()) {
    // SparcV9 backend doesn't generate this (yet...)
    int FrameIndex = MO.getFrameIndex();
    std::cerr << "ERROR: Frame index unhandled.\n";
    abort();
  } else if (MO.isConstantPoolIndex()) {
    unsigned Index = MO.getConstantPoolIndex();
    rv = MCE.getConstantPoolEntryAddress(Index);
  } else {
    std::cerr << "ERROR: Unknown type of MachineOperand: " << MO << "\n";
    abort();
  }

  // Finally, deal with the various bitfield-extracting functions that
  // are used in SPARC assembly. (Some of these make no sense in combination
  // with some of the above; we'll trust that the instruction selector
  // will not produce nonsense, and not check for valid combinations here.)
  if (MO.isLoBits32()) {          // %lo(val) == %lo() in SparcV9 ABI doc
    return rv & 0x03ff;
  } else if (MO.isHiBits32()) {   // %lm(val) == %hi() in SparcV9 ABI doc
    return (rv >> 10) & 0x03fffff;
  } else if (MO.isLoBits64()) {   // %hm(val) == %ulo() in SparcV9 ABI doc
    return (rv >> 32) & 0x03ff;
  } else if (MO.isHiBits64()) {   // %hh(val) == %uhi() in SparcV9 ABI doc
    return rv >> 42;
  } else {                        // (unadorned) val
    return rv;
  }
}

unsigned SparcV9CodeEmitter::getValueBit(int64_t Val, unsigned bit) {
  Val >>= bit;
  return (Val & 1);
}

bool SparcV9CodeEmitter::runOnMachineFunction(MachineFunction &MF) {
  MCE.startFunction(MF);
  DEBUG(std::cerr << "Starting function " << MF.getFunction()->getName()
            << ", address: " << "0x" << std::hex
            << (long)MCE.getCurrentPCValue() << "\n");

  MCE.emitConstantPool(MF.getConstantPool());
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE.finishFunction(MF);

  DEBUG(std::cerr << "Finishing fn " << MF.getFunction()->getName() << "\n");

  // Resolve branches to BasicBlocks for the entire function
  for (unsigned i = 0, e = BBRefs.size(); i != e; ++i) {
    long Location = BBLocations[BBRefs[i].first];
    unsigned *Ref = BBRefs[i].second.first;
    MachineInstr *MI = BBRefs[i].second.second;
    DEBUG(std::cerr << "Fixup @ " << std::hex << Ref << " to 0x" << Location
                    << " in instr: " << std::dec << *MI);
    for (unsigned ii = 0, ee = MI->getNumOperands(); ii != ee; ++ii) {
      MachineOperand &op = MI->getOperand(ii);
      if (op.isPCRelativeDisp()) {
        // the instruction's branch target is made such that it branches to
        // PC + (branchTarget * 4), so undo that arithmetic here:
        // Location is the target of the branch
        // Ref is the location of the instruction, and hence the PC
        int64_t branchTarget = (Location - (long)Ref) >> 2;
        // Save the flags.
        bool loBits32=false, hiBits32=false, loBits64=false, hiBits64=false;
        if (op.isLoBits32()) { loBits32=true; }
        if (op.isHiBits32()) { hiBits32=true; }
        if (op.isLoBits64()) { loBits64=true; }
        if (op.isHiBits64()) { hiBits64=true; }
        MI->SetMachineOperandConst(ii, MachineOperand::MO_SignExtendedImmed,
                                   branchTarget);
        if (loBits32) { MI->getOperand(ii).markLo32(); }
        else if (hiBits32) { MI->getOperand(ii).markHi32(); }
        else if (loBits64) { MI->getOperand(ii).markLo64(); }
        else if (hiBits64) { MI->getOperand(ii).markHi64(); }
        DEBUG(std::cerr << "Rewrote BB ref: ");
        unsigned fixedInstr = SparcV9CodeEmitter::getBinaryCodeForInstr(*MI);
        MCE.emitWordAt (fixedInstr, Ref);
        break;
      }
    }
  }
  BBRefs.clear();
  BBLocations.clear();

  return false;
}

void SparcV9CodeEmitter::emitBasicBlock(MachineBasicBlock &MBB) {
  currBB = MBB.getBasicBlock();
  BBLocations[currBB] = MCE.getCurrentPCValue();
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I)
    if (I->getOpcode() != V9::RDCCR) {
      emitWord(getBinaryCodeForInstr(*I));
    } else {
      // FIXME: The tblgen produced code emitter cannot deal with the fact that
      // machine operand #0 of the RDCCR instruction should be ignored.  This is
      // really a bug in the representation of the RDCCR instruction (which has
      // no need to explicitly represent the CCR dest), but we hack around it
      // here.
      unsigned RegNo = getMachineOpValue(*I, I->getOperand(1));
      RegNo &= (1<<5)-1;
      emitWord((RegNo << 25) | 2168487936U);
    }
}

#include "SparcV9GenCodeEmitter.inc"


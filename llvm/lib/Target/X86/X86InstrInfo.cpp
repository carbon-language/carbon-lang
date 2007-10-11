//===- X86InstrInfo.cpp - X86 Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86InstrInfo.h"
#include "X86.h"
#include "X86GenInstrInfo.inc"
#include "X86InstrBuilder.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

X86InstrInfo::X86InstrInfo(X86TargetMachine &tm)
  : TargetInstrInfo(X86Insts, array_lengthof(X86Insts)),
    TM(tm), RI(tm, *this) {
}

bool X86InstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned& sourceReg,
                               unsigned& destReg) const {
  MachineOpCode oc = MI.getOpcode();
  if (oc == X86::MOV8rr || oc == X86::MOV16rr ||
      oc == X86::MOV32rr || oc == X86::MOV64rr ||
      oc == X86::MOV16to16_ || oc == X86::MOV32to32_ ||
      oc == X86::MOV_Fp3232  || oc == X86::MOVSSrr || oc == X86::MOVSDrr ||
      oc == X86::MOV_Fp3264 || oc == X86::MOV_Fp6432 || oc == X86::MOV_Fp6464 ||
      oc == X86::FsMOVAPSrr || oc == X86::FsMOVAPDrr ||
      oc == X86::MOVAPSrr || oc == X86::MOVAPDrr ||
      oc == X86::MOVSS2PSrr || oc == X86::MOVSD2PDrr ||
      oc == X86::MOVPS2SSrr || oc == X86::MOVPD2SDrr ||
      oc == X86::MMX_MOVD64rr || oc == X86::MMX_MOVQ64rr) {
      assert(MI.getNumOperands() >= 2 &&
             MI.getOperand(0).isRegister() &&
             MI.getOperand(1).isRegister() &&
             "invalid register-register move instruction");
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
  }
  return false;
}

unsigned X86InstrInfo::isLoadFromStackSlot(MachineInstr *MI, 
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV16_rm:
  case X86::MOV32rm:
  case X86::MOV32_rm:
  case X86::MOV64rm:
  case X86::LD_Fp64m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MOVAPSrm:
  case X86::MOVAPDrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
    if (MI->getOperand(1).isFrameIndex() && MI->getOperand(2).isImmediate() &&
        MI->getOperand(3).isRegister() && MI->getOperand(4).isImmediate() &&
        MI->getOperand(2).getImmedValue() == 1 &&
        MI->getOperand(3).getReg() == 0 &&
        MI->getOperand(4).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned X86InstrInfo::isStoreToStackSlot(MachineInstr *MI,
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::MOV8mr:
  case X86::MOV16mr:
  case X86::MOV16_mr:
  case X86::MOV32mr:
  case X86::MOV32_mr:
  case X86::MOV64mr:
  case X86::ST_FpP64m:
  case X86::MOVSSmr:
  case X86::MOVSDmr:
  case X86::MOVAPSmr:
  case X86::MOVAPDmr:
  case X86::MMX_MOVD64mr:
  case X86::MMX_MOVQ64mr:
  case X86::MMX_MOVNTQmr:
    if (MI->getOperand(0).isFrameIndex() && MI->getOperand(1).isImmediate() &&
        MI->getOperand(2).isRegister() && MI->getOperand(3).isImmediate() &&
        MI->getOperand(1).getImmedValue() == 1 &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(0).getFrameIndex();
      return MI->getOperand(4).getReg();
    }
    break;
  }
  return 0;
}


bool X86InstrInfo::isReallyTriviallyReMaterializable(MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV16_rm:
  case X86::MOV32rm:
  case X86::MOV32_rm:
  case X86::MOV64rm:
  case X86::LD_Fp64m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MOVAPSrm:
  case X86::MOVAPDrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
    // Loads from constant pools are trivially rematerializable.
    return MI->getOperand(1).isRegister() && MI->getOperand(2).isImmediate() &&
           MI->getOperand(3).isRegister() && MI->getOperand(4).isConstantPoolIndex() &&
           MI->getOperand(1).getReg() == 0 &&
           MI->getOperand(2).getImmedValue() == 1 &&
           MI->getOperand(3).getReg() == 0;
  }
  // All other instructions marked M_REMATERIALIZABLE are always trivially
  // rematerializable.
  return true;
}

/// hasLiveCondCodeDef - True if MI has a condition code def, e.g. EFLAGS, that
/// is not marked dead.
static bool hasLiveCondCodeDef(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegister() && MO.isDef() &&
        MO.getReg() == X86::EFLAGS && !MO.isDead()) {
      return true;
    }
  }
  return false;
}

/// convertToThreeAddress - This method must be implemented by targets that
/// set the M_CONVERTIBLE_TO_3_ADDR flag.  When this flag is set, the target
/// may be able to convert a two-address instruction into a true
/// three-address instruction on demand.  This allows the X86 target (for
/// example) to convert ADD and SHL instructions into LEA instructions if they
/// would require register copies due to two-addressness.
///
/// This method returns a null pointer if the transformation cannot be
/// performed, otherwise it returns the new instruction.
///
MachineInstr *
X86InstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                    MachineBasicBlock::iterator &MBBI,
                                    LiveVariables &LV) const {
  MachineInstr *MI = MBBI;
  // All instructions input are two-addr instructions.  Get the known operands.
  unsigned Dest = MI->getOperand(0).getReg();
  unsigned Src = MI->getOperand(1).getReg();

  MachineInstr *NewMI = NULL;
  // FIXME: 16-bit LEA's are really slow on Athlons, but not bad on P4's.  When
  // we have better subtarget support, enable the 16-bit LEA generation here.
  bool DisableLEA16 = true;

  unsigned MIOpc = MI->getOpcode();
  switch (MIOpc) {
  case X86::SHUFPSrri: {
    assert(MI->getNumOperands() == 4 && "Unknown shufps instruction!");
    if (!TM.getSubtarget<X86Subtarget>().hasSSE2()) return 0;
    
    unsigned A = MI->getOperand(0).getReg();
    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    unsigned M = MI->getOperand(3).getImm();
    if (B != C) return 0;
    NewMI = BuildMI(get(X86::PSHUFDri), A).addReg(B).addImm(M);
    break;
  }
  case X86::SHL64ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned Dest = MI->getOperand(0).getReg();
    unsigned Src = MI->getOperand(1).getReg();
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;
    
    NewMI = BuildMI(get(X86::LEA64r), Dest)
      .addReg(0).addImm(1 << ShAmt).addReg(Src).addImm(0);
    break;
  }
  case X86::SHL32ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned Dest = MI->getOperand(0).getReg();
    unsigned Src = MI->getOperand(1).getReg();
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;
    
    unsigned Opc = TM.getSubtarget<X86Subtarget>().is64Bit() ?
      X86::LEA64_32r : X86::LEA32r;
    NewMI = BuildMI(get(Opc), Dest)
      .addReg(0).addImm(1 << ShAmt).addReg(Src).addImm(0);
    break;
  }
  case X86::SHL16ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned Dest = MI->getOperand(0).getReg();
    unsigned Src = MI->getOperand(1).getReg();
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;
    
    if (DisableLEA16) {
      // If 16-bit LEA is disabled, use 32-bit LEA via subregisters.
      SSARegMap *RegMap = MFI->getParent()->getSSARegMap();
      unsigned Opc = TM.getSubtarget<X86Subtarget>().is64Bit()
        ? X86::LEA64_32r : X86::LEA32r;
      unsigned leaInReg = RegMap->createVirtualRegister(&X86::GR32RegClass);
      unsigned leaOutReg = RegMap->createVirtualRegister(&X86::GR32RegClass);
            
      MachineInstr *Ins =
        BuildMI(get(X86::INSERT_SUBREG), leaInReg).addReg(Src).addImm(2);
      Ins->copyKillDeadInfo(MI);
      
      NewMI = BuildMI(get(Opc), leaOutReg)
        .addReg(0).addImm(1 << ShAmt).addReg(leaInReg).addImm(0);
      
      MachineInstr *Ext =
        BuildMI(get(X86::EXTRACT_SUBREG), Dest).addReg(leaOutReg).addImm(2);
      Ext->copyKillDeadInfo(MI);
      
      MFI->insert(MBBI, Ins);            // Insert the insert_subreg
      LV.instructionChanged(MI, NewMI);  // Update live variables
      LV.addVirtualRegisterKilled(leaInReg, NewMI);
      MFI->insert(MBBI, NewMI);          // Insert the new inst
      LV.addVirtualRegisterKilled(leaOutReg, Ext);
      MFI->insert(MBBI, Ext);            // Insert the extract_subreg      
      return Ext;
    } else {
      NewMI = BuildMI(get(X86::LEA16r), Dest)
        .addReg(0).addImm(1 << ShAmt).addReg(Src).addImm(0);
    }
    break;
  }
  default: {
    // The following opcodes also sets the condition code register(s). Only
    // convert them to equivalent lea if the condition code register def's
    // are dead!
    if (hasLiveCondCodeDef(MI))
      return 0;

    bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();
    switch (MIOpc) {
    default: return 0;
    case X86::INC64r:
    case X86::INC32r: {
      assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
      unsigned Opc = MIOpc == X86::INC64r ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
      NewMI = addRegOffset(BuildMI(get(Opc), Dest), Src, 1);
      break;
    }
    case X86::INC16r:
    case X86::INC64_16r:
      if (DisableLEA16) return 0;
      assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
      NewMI = addRegOffset(BuildMI(get(X86::LEA16r), Dest), Src, 1);
      break;
    case X86::DEC64r:
    case X86::DEC32r: {
      assert(MI->getNumOperands() >= 2 && "Unknown dec instruction!");
      unsigned Opc = MIOpc == X86::DEC64r ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
      NewMI = addRegOffset(BuildMI(get(Opc), Dest), Src, -1);
      break;
    }
    case X86::DEC16r:
    case X86::DEC64_16r:
      if (DisableLEA16) return 0;
      assert(MI->getNumOperands() >= 2 && "Unknown dec instruction!");
      NewMI = addRegOffset(BuildMI(get(X86::LEA16r), Dest), Src, -1);
      break;
    case X86::ADD64rr:
    case X86::ADD32rr: {
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      unsigned Opc = MIOpc == X86::ADD64rr ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
      NewMI = addRegReg(BuildMI(get(Opc), Dest), Src,
                        MI->getOperand(2).getReg());
      break;
    }
    case X86::ADD16rr:
      if (DisableLEA16) return 0;
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      NewMI = addRegReg(BuildMI(get(X86::LEA16r), Dest), Src,
                        MI->getOperand(2).getReg());
      break;
    case X86::ADD64ri32:
    case X86::ADD64ri8:
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      if (MI->getOperand(2).isImmediate())
        NewMI = addRegOffset(BuildMI(get(X86::LEA64r), Dest), Src,
                             MI->getOperand(2).getImmedValue());
      break;
    case X86::ADD32ri:
    case X86::ADD32ri8:
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      if (MI->getOperand(2).isImmediate()) {
        unsigned Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;
        NewMI = addRegOffset(BuildMI(get(Opc), Dest), Src,
                             MI->getOperand(2).getImmedValue());
      }
      break;
    case X86::ADD16ri:
    case X86::ADD16ri8:
      if (DisableLEA16) return 0;
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      if (MI->getOperand(2).isImmediate())
        NewMI = addRegOffset(BuildMI(get(X86::LEA16r), Dest), Src,
                             MI->getOperand(2).getImmedValue());
      break;
    case X86::SHL16ri:
      if (DisableLEA16) return 0;
    case X86::SHL32ri:
    case X86::SHL64ri: {
      assert(MI->getNumOperands() >= 3 && MI->getOperand(2).isImmediate() &&
             "Unknown shl instruction!");
      unsigned ShAmt = MI->getOperand(2).getImmedValue();
      if (ShAmt == 1 || ShAmt == 2 || ShAmt == 3) {
        X86AddressMode AM;
        AM.Scale = 1 << ShAmt;
        AM.IndexReg = Src;
        unsigned Opc = MIOpc == X86::SHL64ri ? X86::LEA64r
          : (MIOpc == X86::SHL32ri
             ? (is64Bit ? X86::LEA64_32r : X86::LEA32r) : X86::LEA16r);
        NewMI = addFullAddress(BuildMI(get(Opc), Dest), AM);
      }
      break;
    }
    }
  }
  }

  NewMI->copyKillDeadInfo(MI);
  LV.instructionChanged(MI, NewMI);  // Update live variables
  MFI->insert(MBBI, NewMI);          // Insert the new inst    
  return NewMI;
}

/// commuteInstruction - We have a few instructions that must be hacked on to
/// commute them.
///
MachineInstr *X86InstrInfo::commuteInstruction(MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  case X86::SHRD16rri8: // A = SHRD16rri8 B, C, I -> A = SHLD16rri8 C, B, (16-I)
  case X86::SHLD16rri8: // A = SHLD16rri8 B, C, I -> A = SHRD16rri8 C, B, (16-I)
  case X86::SHRD32rri8: // A = SHRD32rri8 B, C, I -> A = SHLD32rri8 C, B, (32-I)
  case X86::SHLD32rri8: // A = SHLD32rri8 B, C, I -> A = SHRD32rri8 C, B, (32-I)
  case X86::SHRD64rri8: // A = SHRD64rri8 B, C, I -> A = SHLD64rri8 C, B, (64-I)
  case X86::SHLD64rri8:{// A = SHLD64rri8 B, C, I -> A = SHRD64rri8 C, B, (64-I)
    unsigned Opc;
    unsigned Size;
    switch (MI->getOpcode()) {
    default: assert(0 && "Unreachable!");
    case X86::SHRD16rri8: Size = 16; Opc = X86::SHLD16rri8; break;
    case X86::SHLD16rri8: Size = 16; Opc = X86::SHRD16rri8; break;
    case X86::SHRD32rri8: Size = 32; Opc = X86::SHLD32rri8; break;
    case X86::SHLD32rri8: Size = 32; Opc = X86::SHRD32rri8; break;
    case X86::SHRD64rri8: Size = 64; Opc = X86::SHLD64rri8; break;
    case X86::SHLD64rri8: Size = 64; Opc = X86::SHRD64rri8; break;
    }
    unsigned Amt = MI->getOperand(3).getImmedValue();
    unsigned A = MI->getOperand(0).getReg();
    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    bool BisKill = MI->getOperand(1).isKill();
    bool CisKill = MI->getOperand(2).isKill();
    return BuildMI(get(Opc), A).addReg(C, false, false, CisKill)
      .addReg(B, false, false, BisKill).addImm(Size-Amt);
  }
  case X86::CMOVB16rr:
  case X86::CMOVB32rr:
  case X86::CMOVB64rr:
  case X86::CMOVAE16rr:
  case X86::CMOVAE32rr:
  case X86::CMOVAE64rr:
  case X86::CMOVE16rr:
  case X86::CMOVE32rr:
  case X86::CMOVE64rr:
  case X86::CMOVNE16rr:
  case X86::CMOVNE32rr:
  case X86::CMOVNE64rr:
  case X86::CMOVBE16rr:
  case X86::CMOVBE32rr:
  case X86::CMOVBE64rr:
  case X86::CMOVA16rr:
  case X86::CMOVA32rr:
  case X86::CMOVA64rr:
  case X86::CMOVL16rr:
  case X86::CMOVL32rr:
  case X86::CMOVL64rr:
  case X86::CMOVGE16rr:
  case X86::CMOVGE32rr:
  case X86::CMOVGE64rr:
  case X86::CMOVLE16rr:
  case X86::CMOVLE32rr:
  case X86::CMOVLE64rr:
  case X86::CMOVG16rr:
  case X86::CMOVG32rr:
  case X86::CMOVG64rr:
  case X86::CMOVS16rr:
  case X86::CMOVS32rr:
  case X86::CMOVS64rr:
  case X86::CMOVNS16rr:
  case X86::CMOVNS32rr:
  case X86::CMOVNS64rr:
  case X86::CMOVP16rr:
  case X86::CMOVP32rr:
  case X86::CMOVP64rr:
  case X86::CMOVNP16rr:
  case X86::CMOVNP32rr:
  case X86::CMOVNP64rr: {
    unsigned Opc = 0;
    switch (MI->getOpcode()) {
    default: break;
    case X86::CMOVB16rr:  Opc = X86::CMOVAE16rr; break;
    case X86::CMOVB32rr:  Opc = X86::CMOVAE32rr; break;
    case X86::CMOVB64rr:  Opc = X86::CMOVAE64rr; break;
    case X86::CMOVAE16rr: Opc = X86::CMOVB16rr; break;
    case X86::CMOVAE32rr: Opc = X86::CMOVB32rr; break;
    case X86::CMOVAE64rr: Opc = X86::CMOVB64rr; break;
    case X86::CMOVE16rr:  Opc = X86::CMOVNE16rr; break;
    case X86::CMOVE32rr:  Opc = X86::CMOVNE32rr; break;
    case X86::CMOVE64rr:  Opc = X86::CMOVNE64rr; break;
    case X86::CMOVNE16rr: Opc = X86::CMOVE16rr; break;
    case X86::CMOVNE32rr: Opc = X86::CMOVE32rr; break;
    case X86::CMOVNE64rr: Opc = X86::CMOVE64rr; break;
    case X86::CMOVBE16rr: Opc = X86::CMOVA16rr; break;
    case X86::CMOVBE32rr: Opc = X86::CMOVA32rr; break;
    case X86::CMOVBE64rr: Opc = X86::CMOVA64rr; break;
    case X86::CMOVA16rr:  Opc = X86::CMOVBE16rr; break;
    case X86::CMOVA32rr:  Opc = X86::CMOVBE32rr; break;
    case X86::CMOVA64rr:  Opc = X86::CMOVBE64rr; break;
    case X86::CMOVL16rr:  Opc = X86::CMOVGE16rr; break;
    case X86::CMOVL32rr:  Opc = X86::CMOVGE32rr; break;
    case X86::CMOVL64rr:  Opc = X86::CMOVGE64rr; break;
    case X86::CMOVGE16rr: Opc = X86::CMOVL16rr; break;
    case X86::CMOVGE32rr: Opc = X86::CMOVL32rr; break;
    case X86::CMOVGE64rr: Opc = X86::CMOVL64rr; break;
    case X86::CMOVLE16rr: Opc = X86::CMOVG16rr; break;
    case X86::CMOVLE32rr: Opc = X86::CMOVG32rr; break;
    case X86::CMOVLE64rr: Opc = X86::CMOVG64rr; break;
    case X86::CMOVG16rr:  Opc = X86::CMOVLE16rr; break;
    case X86::CMOVG32rr:  Opc = X86::CMOVLE32rr; break;
    case X86::CMOVG64rr:  Opc = X86::CMOVLE64rr; break;
    case X86::CMOVS16rr:  Opc = X86::CMOVNS16rr; break;
    case X86::CMOVS32rr:  Opc = X86::CMOVNS32rr; break;
    case X86::CMOVS64rr:  Opc = X86::CMOVNS32rr; break;
    case X86::CMOVNS16rr: Opc = X86::CMOVS16rr; break;
    case X86::CMOVNS32rr: Opc = X86::CMOVS32rr; break;
    case X86::CMOVNS64rr: Opc = X86::CMOVS64rr; break;
    case X86::CMOVP16rr:  Opc = X86::CMOVNP16rr; break;
    case X86::CMOVP32rr:  Opc = X86::CMOVNP32rr; break;
    case X86::CMOVP64rr:  Opc = X86::CMOVNP32rr; break;
    case X86::CMOVNP16rr: Opc = X86::CMOVP16rr; break;
    case X86::CMOVNP32rr: Opc = X86::CMOVP32rr; break;
    case X86::CMOVNP64rr: Opc = X86::CMOVP64rr; break;
    }

    MI->setInstrDescriptor(get(Opc));
    // Fallthrough intended.
  }
  default:
    return TargetInstrInfo::commuteInstruction(MI);
  }
}

static X86::CondCode GetCondFromBranchOpc(unsigned BrOpc) {
  switch (BrOpc) {
  default: return X86::COND_INVALID;
  case X86::JE:  return X86::COND_E;
  case X86::JNE: return X86::COND_NE;
  case X86::JL:  return X86::COND_L;
  case X86::JLE: return X86::COND_LE;
  case X86::JG:  return X86::COND_G;
  case X86::JGE: return X86::COND_GE;
  case X86::JB:  return X86::COND_B;
  case X86::JBE: return X86::COND_BE;
  case X86::JA:  return X86::COND_A;
  case X86::JAE: return X86::COND_AE;
  case X86::JS:  return X86::COND_S;
  case X86::JNS: return X86::COND_NS;
  case X86::JP:  return X86::COND_P;
  case X86::JNP: return X86::COND_NP;
  case X86::JO:  return X86::COND_O;
  case X86::JNO: return X86::COND_NO;
  }
}

unsigned X86::GetCondBranchFromCond(X86::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case X86::COND_E:  return X86::JE;
  case X86::COND_NE: return X86::JNE;
  case X86::COND_L:  return X86::JL;
  case X86::COND_LE: return X86::JLE;
  case X86::COND_G:  return X86::JG;
  case X86::COND_GE: return X86::JGE;
  case X86::COND_B:  return X86::JB;
  case X86::COND_BE: return X86::JBE;
  case X86::COND_A:  return X86::JA;
  case X86::COND_AE: return X86::JAE;
  case X86::COND_S:  return X86::JS;
  case X86::COND_NS: return X86::JNS;
  case X86::COND_P:  return X86::JP;
  case X86::COND_NP: return X86::JNP;
  case X86::COND_O:  return X86::JO;
  case X86::COND_NO: return X86::JNO;
  }
}

/// GetOppositeBranchCondition - Return the inverse of the specified condition,
/// e.g. turning COND_E to COND_NE.
X86::CondCode X86::GetOppositeBranchCondition(X86::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case X86::COND_E:  return X86::COND_NE;
  case X86::COND_NE: return X86::COND_E;
  case X86::COND_L:  return X86::COND_GE;
  case X86::COND_LE: return X86::COND_G;
  case X86::COND_G:  return X86::COND_LE;
  case X86::COND_GE: return X86::COND_L;
  case X86::COND_B:  return X86::COND_AE;
  case X86::COND_BE: return X86::COND_A;
  case X86::COND_A:  return X86::COND_BE;
  case X86::COND_AE: return X86::COND_B;
  case X86::COND_S:  return X86::COND_NS;
  case X86::COND_NS: return X86::COND_S;
  case X86::COND_P:  return X86::COND_NP;
  case X86::COND_NP: return X86::COND_P;
  case X86::COND_O:  return X86::COND_NO;
  case X86::COND_NO: return X86::COND_O;
  }
}

bool X86InstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  const TargetInstrDescriptor *TID = MI->getInstrDescriptor();
  if (TID->Flags & M_TERMINATOR_FLAG) {
    // Conditional branch is a special case.
    if ((TID->Flags & M_BRANCH_FLAG) != 0 && (TID->Flags & M_BARRIER_FLAG) == 0)
      return true;
    if ((TID->Flags & M_PREDICABLE) == 0)
      return true;
    return !isPredicated(MI);
  }
  return false;
}

// For purposes of branch analysis do not count FP_REG_KILL as a terminator.
static bool isBrAnalysisUnpredicatedTerminator(const MachineInstr *MI,
                                               const X86InstrInfo &TII) {
  if (MI->getOpcode() == X86::FP_REG_KILL)
    return false;
  return TII.isUnpredicatedTerminator(MI);
}

bool X86InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB, 
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 std::vector<MachineOperand> &Cond) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isBrAnalysisUnpredicatedTerminator(--I, *this))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isBrAnalysisUnpredicatedTerminator(--I, *this)) {
    if (!isBranch(LastInst->getOpcode()))
      return true;
    
    // If the block ends with a branch there are 3 possibilities:
    // it's an unconditional, conditional, or indirect branch.
    
    if (LastInst->getOpcode() == X86::JMP) {
      TBB = LastInst->getOperand(0).getMachineBasicBlock();
      return false;
    }
    X86::CondCode BranchCode = GetCondFromBranchOpc(LastInst->getOpcode());
    if (BranchCode == X86::COND_INVALID)
      return true;  // Can't handle indirect branch.

    // Otherwise, block ends with fall-through condbranch.
    TBB = LastInst->getOperand(0).getMachineBasicBlock();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));
    return false;
  }
  
  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;
  
  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isBrAnalysisUnpredicatedTerminator(--I, *this))
    return true;

  // If the block ends with X86::JMP and a conditional branch, handle it.
  X86::CondCode BranchCode = GetCondFromBranchOpc(SecondLastInst->getOpcode());
  if (BranchCode != X86::COND_INVALID && LastInst->getOpcode() == X86::JMP) {
    TBB = SecondLastInst->getOperand(0).getMachineBasicBlock();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));
    FBB = LastInst->getOperand(0).getMachineBasicBlock();
    return false;
  }

  // If the block ends with two X86::JMPs, handle it.  The second one is not
  // executed, so remove it.
  if (SecondLastInst->getOpcode() == X86::JMP && 
      LastInst->getOpcode() == X86::JMP) {
    TBB = SecondLastInst->getOperand(0).getMachineBasicBlock();
    I = LastInst;
    I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned X86InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != X86::JMP && 
      GetCondFromBranchOpc(I->getOpcode()) == X86::COND_INVALID)
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();
  
  if (I == MBB.begin()) return 1;
  --I;
  if (GetCondFromBranchOpc(I->getOpcode()) == X86::COND_INVALID)
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned
X86InstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                           MachineBasicBlock *FBB,
                           const std::vector<MachineOperand> &Cond) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "X86 branch conditions have one component!");

  if (FBB == 0) { // One way branch.
    if (Cond.empty()) {
      // Unconditional branch?
      BuildMI(&MBB, get(X86::JMP)).addMBB(TBB);
    } else {
      // Conditional branch.
      unsigned Opc = GetCondBranchFromCond((X86::CondCode)Cond[0].getImm());
      BuildMI(&MBB, get(Opc)).addMBB(TBB);
    }
    return 1;
  }
  
  // Two-way Conditional branch.
  unsigned Opc = GetCondBranchFromCond((X86::CondCode)Cond[0].getImm());
  BuildMI(&MBB, get(Opc)).addMBB(TBB);
  BuildMI(&MBB, get(X86::JMP)).addMBB(FBB);
  return 2;
}

bool X86InstrInfo::BlockHasNoFallThrough(MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case X86::TCRETURNri:
  case X86::TCRETURNdi:
  case X86::RET:     // Return.
  case X86::RETI:
  case X86::TAILJMPd:
  case X86::TAILJMPr:
  case X86::TAILJMPm:
  case X86::JMP:     // Uncond branch.
  case X86::JMP32r:  // Indirect branch.
  case X86::JMP64r:  // Indirect branch (64-bit).
  case X86::JMP32m:  // Indirect branch through mem.
  case X86::JMP64m:  // Indirect branch through mem (64-bit).
    return true;
  default: return false;
  }
}

bool X86InstrInfo::
ReverseBranchCondition(std::vector<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid X86 branch condition!");
  Cond[0].setImm(GetOppositeBranchCondition((X86::CondCode)Cond[0].getImm()));
  return false;
}

const TargetRegisterClass *X86InstrInfo::getPointerRegClass() const {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  if (Subtarget->is64Bit())
    return &X86::GR64RegClass;
  else
    return &X86::GR32RegClass;
}

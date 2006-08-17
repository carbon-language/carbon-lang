//===- X86RegisterInfo.cpp - X86 Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the MRegisterInfo class.  This
// file is responsible for the frame pointer elimination optimization on X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86TargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>

using namespace llvm;

namespace {
  cl::opt<bool>
  NoFusing("disable-spill-fusing",
           cl::desc("Disable fusing of spill code into instructions"));
  cl::opt<bool>
  PrintFailedFusing("print-failed-fuse-candidates",
                    cl::desc("Print instructions that the allocator wants to"
                             " fuse, but the X86 backend currently can't"),
                    cl::Hidden);
}

X86RegisterInfo::X86RegisterInfo()
  : X86GenRegisterInfo(X86::ADJCALLSTACKDOWN, X86::ADJCALLSTACKUP) {}

void X86RegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                          unsigned SrcReg, int FrameIdx,
                                          const TargetRegisterClass *RC) const {
  unsigned Opc;
  if (RC == &X86::GR32RegClass) {
    Opc = X86::MOV32mr;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16mr;
  } else if (RC == &X86::GR8RegClass) {
    Opc = X86::MOV8mr;
  } else if (RC == &X86::GR32_RegClass) {
    Opc = X86::MOV32_mr;
  } else if (RC == &X86::GR16_RegClass) {
    Opc = X86::MOV16_mr;
  } else if (RC == &X86::RFPRegClass || RC == &X86::RSTRegClass) {
    Opc = X86::FpST64m;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::MOVSSmr;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::MOVSDmr;
  } else if (RC == &X86::VR128RegClass) {
    Opc = X86::MOVAPSmr;
  } else {
    assert(0 && "Unknown regclass");
    abort();
  }
  addFrameReference(BuildMI(MBB, MI, Opc, 5), FrameIdx).addReg(SrcReg);
}

void X86RegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                           unsigned DestReg, int FrameIdx,
                                           const TargetRegisterClass *RC) const{
  unsigned Opc;
  if (RC == &X86::GR32RegClass) {
    Opc = X86::MOV32rm;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16rm;
  } else if (RC == &X86::GR8RegClass) {
    Opc = X86::MOV8rm;
  } else if (RC == &X86::GR32_RegClass) {
    Opc = X86::MOV32_rm;
  } else if (RC == &X86::GR16_RegClass) {
    Opc = X86::MOV16_rm;
  } else if (RC == &X86::RFPRegClass || RC == &X86::RSTRegClass) {
    Opc = X86::FpLD64m;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::MOVSSrm;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::MOVSDrm;
  } else if (RC == &X86::VR128RegClass) {
    Opc = X86::MOVAPSrm;
  } else {
    assert(0 && "Unknown regclass");
    abort();
  }
  addFrameReference(BuildMI(MBB, MI, Opc, 4, DestReg), FrameIdx);
}

void X86RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *RC) const {
  unsigned Opc;
  if (RC == &X86::GR32RegClass) {
    Opc = X86::MOV32rr;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16rr;
  } else if (RC == &X86::GR8RegClass) {
    Opc = X86::MOV8rr;
  } else if (RC == &X86::GR32_RegClass) {
    Opc = X86::MOV32_rr;
  } else if (RC == &X86::GR16_RegClass) {
    Opc = X86::MOV16_rr;
  } else if (RC == &X86::RFPRegClass || RC == &X86::RSTRegClass) {
    Opc = X86::FpMOV;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::FsMOVAPSrr;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::FsMOVAPDrr;
  } else if (RC == &X86::VR128RegClass) {
    Opc = X86::MOVAPSrr;
  } else {
    assert(0 && "Unknown regclass");
    abort();
  }
  BuildMI(MBB, MI, Opc, 1, DestReg).addReg(SrcReg);
}


static MachineInstr *MakeMInst(unsigned Opcode, unsigned FrameIndex,
                               MachineInstr *MI) {
  return addFrameReference(BuildMI(Opcode, 4), FrameIndex);
}

static MachineInstr *MakeMRInst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
                 .addReg(MI->getOperand(1).getReg());
}

static MachineInstr *MakeMRIInst(unsigned Opcode, unsigned FrameIndex,
                                 MachineInstr *MI) {
  return addFrameReference(BuildMI(Opcode, 6), FrameIndex)
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImmedValue());
}

static MachineInstr *MakeMIInst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  if (MI->getOperand(1).isImmediate())
    return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
      .addImm(MI->getOperand(1).getImmedValue());
  else if (MI->getOperand(1).isGlobalAddress())
    return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
      .addGlobalAddress(MI->getOperand(1).getGlobal(),
                        MI->getOperand(1).getOffset());
  else if (MI->getOperand(1).isJumpTableIndex())
    return addFrameReference(BuildMI(Opcode, 5), FrameIndex)
      .addJumpTableIndex(MI->getOperand(1).getJumpTableIndex());
  assert(0 && "Unknown operand for MakeMI!");
  return 0;
}

static MachineInstr *MakeM0Inst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  return addFrameReference(BuildMI(Opcode, 5), FrameIndex).addImm(0);
}

static MachineInstr *MakeRMInst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  const MachineOperand& op = MI->getOperand(0);
  return addFrameReference(BuildMI(Opcode, 5, op.getReg(), op.getUseType()),
                           FrameIndex);
}

static MachineInstr *MakeRMIInst(unsigned Opcode, unsigned FrameIndex,
                                 MachineInstr *MI) {
  const MachineOperand& op = MI->getOperand(0);
  return addFrameReference(BuildMI(Opcode, 6, op.getReg(), op.getUseType()),
                        FrameIndex).addImm(MI->getOperand(2).getImmedValue());
}


//===----------------------------------------------------------------------===//
// Efficient Lookup Table Support
//===----------------------------------------------------------------------===//

namespace {
  /// TableEntry - Maps the 'from' opcode to a fused form of the 'to' opcode.
  ///
  struct TableEntry {
    unsigned from;                      // Original opcode.
    unsigned to;                        // New opcode.
    unsigned make;                      // Form of make required to produce the
                                        // new instruction.
                                        
    // less operators used by STL search.                                    
    bool operator<(const TableEntry &TE) const { return from < TE.from; }
    friend bool operator<(const TableEntry &TE, unsigned V) {
      return TE.from < V;
    }
    friend bool operator<(unsigned V, const TableEntry &TE) {
      return V < TE.from;
    }
  };
}

/// TableIsSorted - Return true if the table is in 'from' opcode order.
///
static bool TableIsSorted(const TableEntry *Table, unsigned NumEntries) {
  for (unsigned i = 1; i != NumEntries; ++i)
    if (!(Table[i-1] < Table[i])) {
      std::cerr << "Entries out of order " << Table[i-1].from
                << " " << Table[i].from << "\n";
      return false;
    }
  return true;
}

/// TableLookup - Return the table entry matching the specified opcode.
/// Otherwise return NULL.
static const TableEntry *TableLookup(const TableEntry *Table, unsigned N,
                                unsigned Opcode) {
  const TableEntry *I = std::lower_bound(Table, Table+N, Opcode);
  if (I != Table+N && I->from == Opcode)
    return I;
  return NULL;
}

#define ARRAY_SIZE(TABLE)  \
   (sizeof(TABLE)/sizeof(TABLE[0]))

#ifdef NDEBUG
#define ASSERT_SORTED(TABLE)
#else
#define ASSERT_SORTED(TABLE)                                              \
  { static bool TABLE##Checked = false;                                   \
    if (!TABLE##Checked) {                                                \
       assert(TableIsSorted(TABLE, ARRAY_SIZE(TABLE)) &&                  \
              "All lookup tables must be sorted for efficient access!");  \
       TABLE##Checked = true;                                             \
    }                                                                     \
  }
#endif


MachineInstr* X86RegisterInfo::foldMemoryOperand(MachineInstr* MI,
                                                 unsigned i,
                                                 int FrameIndex) const {
  // Check switch flag 
  if (NoFusing) return NULL;

  // Selection of instruction makes
  enum {
    makeM0Inst,
    makeMIInst,
    makeMInst,
    makeMRIInst,
    makeMRInst,
    makeRMIInst,
    makeRMInst
  };
  
  // Table (and size) to search
  const TableEntry *OpcodeTablePtr = NULL;
  unsigned OpcodeTableSize = 0;

  if (i == 0) { // If operand 0
    static const TableEntry OpcodeTable[] = {
      { X86::ADC32ri,     X86::ADC32mi,     makeMIInst },
      { X86::ADC32ri8,    X86::ADC32mi8,    makeMIInst },
      { X86::ADC32rr,     X86::ADC32mr,     makeMRInst },
      { X86::ADD16ri,     X86::ADD16mi,     makeMIInst },
      { X86::ADD16ri8,    X86::ADD16mi8,    makeMIInst },
      { X86::ADD16rr,     X86::ADD16mr,     makeMRInst },
      { X86::ADD32ri,     X86::ADD32mi,     makeMIInst },
      { X86::ADD32ri8,    X86::ADD32mi8,    makeMIInst },
      { X86::ADD32rr,     X86::ADD32mr,     makeMRInst },
      { X86::ADD8ri,      X86::ADD8mi,      makeMIInst },
      { X86::ADD8rr,      X86::ADD8mr,      makeMRInst },
      { X86::AND16ri,     X86::AND16mi,     makeMIInst },
      { X86::AND16ri8,    X86::AND16mi8,    makeMIInst },
      { X86::AND16rr,     X86::AND16mr,     makeMRInst },
      { X86::AND32ri,     X86::AND32mi,     makeMIInst },
      { X86::AND32ri8,    X86::AND32mi8,    makeMIInst },
      { X86::AND32rr,     X86::AND32mr,     makeMRInst },
      { X86::AND8ri,      X86::AND8mi,      makeMIInst },
      { X86::AND8rr,      X86::AND8mr,      makeMRInst },
      { X86::DEC16r,      X86::DEC16m,      makeMInst },
      { X86::DEC32r,      X86::DEC32m,      makeMInst },
      { X86::DEC8r,       X86::DEC8m,       makeMInst },
      { X86::DIV16r,      X86::DIV16m,      makeMInst },
      { X86::DIV32r,      X86::DIV32m,      makeMInst },
      { X86::DIV8r,       X86::DIV8m,       makeMInst },
      { X86::FsMOVAPDrr,  X86::MOVSDmr,     makeMRInst },
      { X86::FsMOVAPSrr,  X86::MOVSSmr,     makeMRInst },
      { X86::IDIV16r,     X86::IDIV16m,     makeMInst },
      { X86::IDIV32r,     X86::IDIV32m,     makeMInst },
      { X86::IDIV8r,      X86::IDIV8m,      makeMInst },
      { X86::IMUL16r,     X86::IMUL16m,     makeMInst },
      { X86::IMUL32r,     X86::IMUL32m,     makeMInst },
      { X86::IMUL8r,      X86::IMUL8m,      makeMInst },
      { X86::INC16r,      X86::INC16m,      makeMInst },
      { X86::INC32r,      X86::INC32m,      makeMInst },
      { X86::INC8r,       X86::INC8m,       makeMInst },
      { X86::MOV16r0,     X86::MOV16mi,     makeM0Inst },
      { X86::MOV16ri,     X86::MOV16mi,     makeMIInst },
      { X86::MOV16rr,     X86::MOV16mr,     makeMRInst },
      { X86::MOV32r0,     X86::MOV32mi,     makeM0Inst },
      { X86::MOV32ri,     X86::MOV32mi,     makeMIInst },
      { X86::MOV32rr,     X86::MOV32mr,     makeMRInst },
      { X86::MOV8r0,      X86::MOV8mi,      makeM0Inst },
      { X86::MOV8ri,      X86::MOV8mi,      makeMIInst },
      { X86::MOV8rr,      X86::MOV8mr,      makeMRInst },
      { X86::MOVAPDrr,    X86::MOVAPDmr,    makeMRInst },
      { X86::MOVAPSrr,    X86::MOVAPSmr,    makeMRInst },
      { X86::MOVPDI2DIrr, X86::MOVPDI2DImr, makeMRInst },
      { X86::MOVPS2SSrr,  X86::MOVPS2SSmr,  makeMRInst },
      { X86::MOVSDrr,     X86::MOVSDmr,     makeMRInst },
      { X86::MOVSSrr,     X86::MOVSSmr,     makeMRInst },
      { X86::MOVUPDrr,    X86::MOVUPDmr,    makeMRInst },
      { X86::MOVUPSrr,    X86::MOVUPSmr,    makeMRInst },
      { X86::MUL16r,      X86::MUL16m,      makeMInst },
      { X86::MUL32r,      X86::MUL32m,      makeMInst },
      { X86::MUL8r,       X86::MUL8m,       makeMInst },
      { X86::NEG16r,      X86::NEG16m,      makeMInst },
      { X86::NEG32r,      X86::NEG32m,      makeMInst },
      { X86::NEG8r,       X86::NEG8m,       makeMInst },
      { X86::NOT16r,      X86::NOT16m,      makeMInst },
      { X86::NOT32r,      X86::NOT32m,      makeMInst },
      { X86::NOT8r,       X86::NOT8m,       makeMInst },
      { X86::OR16ri,      X86::OR16mi,      makeMIInst },
      { X86::OR16ri8,     X86::OR16mi8,     makeMIInst },
      { X86::OR16rr,      X86::OR16mr,      makeMRInst },
      { X86::OR32ri,      X86::OR32mi,      makeMIInst },
      { X86::OR32ri8,     X86::OR32mi8,     makeMIInst },
      { X86::OR32rr,      X86::OR32mr,      makeMRInst },
      { X86::OR8ri,       X86::OR8mi,       makeMIInst },
      { X86::OR8rr,       X86::OR8mr,       makeMRInst },
      { X86::ROL16r1,     X86::ROL16m1,     makeMInst },
      { X86::ROL16rCL,    X86::ROL16mCL,    makeMInst },
      { X86::ROL16ri,     X86::ROL16mi,     makeMIInst },
      { X86::ROL32r1,     X86::ROL32m1,     makeMInst },
      { X86::ROL32rCL,    X86::ROL32mCL,    makeMInst },
      { X86::ROL32ri,     X86::ROL32mi,     makeMIInst },
      { X86::ROL8r1,      X86::ROL8m1,      makeMInst },
      { X86::ROL8rCL,     X86::ROL8mCL,     makeMInst },
      { X86::ROL8ri,      X86::ROL8mi,      makeMIInst },
      { X86::ROR16r1,     X86::ROR16m1,     makeMInst },
      { X86::ROR16rCL,    X86::ROR16mCL,    makeMInst },
      { X86::ROR16ri,     X86::ROR16mi,     makeMIInst },
      { X86::ROR32r1,     X86::ROR32m1,     makeMInst },
      { X86::ROR32rCL,    X86::ROR32mCL,    makeMInst },
      { X86::ROR32ri,     X86::ROR32mi,     makeMIInst },
      { X86::ROR8r1,      X86::ROR8m1,      makeMInst },
      { X86::ROR8rCL,     X86::ROR8mCL,     makeMInst },
      { X86::ROR8ri,      X86::ROR8mi,      makeMIInst },
      { X86::SAR16r1,     X86::SAR16m1,     makeMInst },
      { X86::SAR16rCL,    X86::SAR16mCL,    makeMInst },
      { X86::SAR16ri,     X86::SAR16mi,     makeMIInst },
      { X86::SAR32r1,     X86::SAR32m1,     makeMInst },
      { X86::SAR32rCL,    X86::SAR32mCL,    makeMInst },
      { X86::SAR32ri,     X86::SAR32mi,     makeMIInst },
      { X86::SAR8r1,      X86::SAR8m1,      makeMInst },
      { X86::SAR8rCL,     X86::SAR8mCL,     makeMInst },
      { X86::SAR8ri,      X86::SAR8mi,      makeMIInst },
      { X86::SBB32ri,     X86::SBB32mi,     makeMIInst },
      { X86::SBB32ri8,    X86::SBB32mi8,    makeMIInst },
      { X86::SBB32rr,     X86::SBB32mr,     makeMRInst },
      { X86::SETAEr,      X86::SETAEm,      makeMInst },
      { X86::SETAr,       X86::SETAm,       makeMInst },
      { X86::SETBEr,      X86::SETBEm,      makeMInst },
      { X86::SETBr,       X86::SETBm,       makeMInst },
      { X86::SETEr,       X86::SETEm,       makeMInst },
      { X86::SETGEr,      X86::SETGEm,      makeMInst },
      { X86::SETGr,       X86::SETGm,       makeMInst },
      { X86::SETLEr,      X86::SETLEm,      makeMInst },
      { X86::SETLr,       X86::SETLm,       makeMInst },
      { X86::SETNEr,      X86::SETNEm,      makeMInst },
      { X86::SETNPr,      X86::SETNPm,      makeMInst },
      { X86::SETNSr,      X86::SETNSm,      makeMInst },
      { X86::SETPr,       X86::SETPm,       makeMInst },
      { X86::SETSr,       X86::SETSm,       makeMInst },
      { X86::SHL16r1,     X86::SHL16m1,     makeMInst },
      { X86::SHL16rCL,    X86::SHL16mCL,    makeMInst },
      { X86::SHL16ri,     X86::SHL16mi,     makeMIInst },
      { X86::SHL32r1,     X86::SHL32m1,     makeMInst },
      { X86::SHL32rCL,    X86::SHL32mCL,    makeMInst },
      { X86::SHL32ri,     X86::SHL32mi,     makeMIInst },
      { X86::SHL8r1,      X86::SHL8m1,      makeMInst },
      { X86::SHL8rCL,     X86::SHL8mCL,     makeMInst },
      { X86::SHL8ri,      X86::SHL8mi,      makeMIInst },
      { X86::SHLD16rrCL,  X86::SHLD16mrCL,  makeMRInst },
      { X86::SHLD16rri8,  X86::SHLD16mri8,  makeMRIInst },
      { X86::SHLD32rrCL,  X86::SHLD32mrCL,  makeMRInst },
      { X86::SHLD32rri8,  X86::SHLD32mri8,  makeMRIInst },
      { X86::SHR16r1,     X86::SHR16m1,     makeMInst },
      { X86::SHR16rCL,    X86::SHR16mCL,    makeMInst },
      { X86::SHR16ri,     X86::SHR16mi,     makeMIInst },
      { X86::SHR32r1,     X86::SHR32m1,     makeMInst },
      { X86::SHR32rCL,    X86::SHR32mCL,    makeMInst },
      { X86::SHR32ri,     X86::SHR32mi,     makeMIInst },
      { X86::SHR8r1,      X86::SHR8m1,      makeMInst },
      { X86::SHR8rCL,     X86::SHR8mCL,     makeMInst },
      { X86::SHR8ri,      X86::SHR8mi,      makeMIInst },
      { X86::SHRD16rrCL,  X86::SHRD16mrCL,  makeMRInst },
      { X86::SHRD16rri8,  X86::SHRD16mri8,  makeMRIInst },
      { X86::SHRD32rrCL,  X86::SHRD32mrCL,  makeMRInst },
      { X86::SHRD32rri8,  X86::SHRD32mri8,  makeMRIInst },
      { X86::SUB16ri,     X86::SUB16mi,     makeMIInst },
      { X86::SUB16ri8,    X86::SUB16mi8,    makeMIInst },
      { X86::SUB16rr,     X86::SUB16mr,     makeMRInst },
      { X86::SUB32ri,     X86::SUB32mi,     makeMIInst },
      { X86::SUB32ri8,    X86::SUB32mi8,    makeMIInst },
      { X86::SUB32rr,     X86::SUB32mr,     makeMRInst },
      { X86::SUB8ri,      X86::SUB8mi,      makeMIInst },
      { X86::SUB8rr,      X86::SUB8mr,      makeMRInst },
      { X86::XCHG16rr,    X86::XCHG16mr,    makeMRInst },
      { X86::XCHG32rr,    X86::XCHG32mr,    makeMRInst },
      { X86::XCHG8rr,     X86::XCHG8mr,     makeMRInst },
      { X86::XOR16ri,     X86::XOR16mi,     makeMIInst },
      { X86::XOR16ri8,    X86::XOR16mi8,    makeMIInst },
      { X86::XOR16rr,     X86::XOR16mr,     makeMRInst },
      { X86::XOR32ri,     X86::XOR32mi,     makeMIInst },
      { X86::XOR32ri8,    X86::XOR32mi8,    makeMIInst },
      { X86::XOR32rr,     X86::XOR32mr,     makeMRInst },
      { X86::XOR8ri,      X86::XOR8mi,      makeMIInst },
      { X86::XOR8rr,      X86::XOR8mr,      makeMRInst }
    };
    ASSERT_SORTED(OpcodeTable);
    OpcodeTablePtr = OpcodeTable;
    OpcodeTableSize = ARRAY_SIZE(OpcodeTable);
  } else if (i == 1) {
    static const TableEntry OpcodeTable[] = {
      { X86::ADC32rr,         X86::ADC32rm,         makeRMInst },
      { X86::ADD16rr,         X86::ADD16rm,         makeRMInst },
      { X86::ADD32rr,         X86::ADD32rm,         makeRMInst },
      { X86::ADD8rr,          X86::ADD8rm,          makeRMInst },
      { X86::ADDPDrr,         X86::ADDPDrm,         makeRMInst },
      { X86::ADDPSrr,         X86::ADDPSrm,         makeRMInst },
      { X86::ADDSDrr,         X86::ADDSDrm,         makeRMInst },
      { X86::ADDSSrr,         X86::ADDSSrm,         makeRMInst },
      { X86::ADDSUBPDrr,      X86::ADDSUBPDrm,      makeRMInst },
      { X86::ADDSUBPSrr,      X86::ADDSUBPSrm,      makeRMInst },
      { X86::AND16rr,         X86::AND16rm,         makeRMInst },
      { X86::AND32rr,         X86::AND32rm,         makeRMInst },
      { X86::AND8rr,          X86::AND8rm,          makeRMInst },
      { X86::ANDNPDrr,        X86::ANDNPDrm,        makeRMInst },
      { X86::ANDNPSrr,        X86::ANDNPSrm,        makeRMInst },
      { X86::ANDPDrr,         X86::ANDPDrm,         makeRMInst },
      { X86::ANDPSrr,         X86::ANDPSrm,         makeRMInst },
      { X86::CMOVA16rr,       X86::CMOVA16rm,       makeRMInst },
      { X86::CMOVA32rr,       X86::CMOVA32rm,       makeRMInst },
      { X86::CMOVAE16rr,      X86::CMOVAE16rm,      makeRMInst },
      { X86::CMOVAE32rr,      X86::CMOVAE32rm,      makeRMInst },
      { X86::CMOVB16rr,       X86::CMOVB16rm,       makeRMInst },
      { X86::CMOVB32rr,       X86::CMOVB32rm,       makeRMInst },
      { X86::CMOVBE16rr,      X86::CMOVBE16rm,      makeRMInst },
      { X86::CMOVBE32rr,      X86::CMOVBE32rm,      makeRMInst },
      { X86::CMOVE16rr,       X86::CMOVE16rm,       makeRMInst },
      { X86::CMOVE32rr,       X86::CMOVE32rm,       makeRMInst },
      { X86::CMOVG16rr,       X86::CMOVG16rm,       makeRMInst },
      { X86::CMOVG32rr,       X86::CMOVG32rm,       makeRMInst },
      { X86::CMOVGE16rr,      X86::CMOVGE16rm,      makeRMInst },
      { X86::CMOVGE32rr,      X86::CMOVGE32rm,      makeRMInst },
      { X86::CMOVL16rr,       X86::CMOVL16rm,       makeRMInst },
      { X86::CMOVL32rr,       X86::CMOVL32rm,       makeRMInst },
      { X86::CMOVLE16rr,      X86::CMOVLE16rm,      makeRMInst },
      { X86::CMOVLE32rr,      X86::CMOVLE32rm,      makeRMInst },
      { X86::CMOVNE16rr,      X86::CMOVNE16rm,      makeRMInst },
      { X86::CMOVNE32rr,      X86::CMOVNE32rm,      makeRMInst },
      { X86::CMOVNP16rr,      X86::CMOVNP16rm,      makeRMInst },
      { X86::CMOVNP32rr,      X86::CMOVNP32rm,      makeRMInst },
      { X86::CMOVNS16rr,      X86::CMOVNS16rm,      makeRMInst },
      { X86::CMOVNS32rr,      X86::CMOVNS32rm,      makeRMInst },
      { X86::CMOVP16rr,       X86::CMOVP16rm,       makeRMInst },
      { X86::CMOVP32rr,       X86::CMOVP32rm,       makeRMInst },
      { X86::CMOVS16rr,       X86::CMOVS16rm,       makeRMInst },
      { X86::CMOVS32rr,       X86::CMOVS32rm,       makeRMInst },
      { X86::CMP16ri,         X86::CMP16mi,         makeMIInst },
      { X86::CMP16ri8,        X86::CMP16mi8,        makeMIInst },
      { X86::CMP16rr,         X86::CMP16rm,         makeRMInst },
      { X86::CMP32ri,         X86::CMP32mi,         makeMIInst },
      { X86::CMP32ri8,        X86::CMP32mi8,        makeRMInst },
      { X86::CMP32rr,         X86::CMP32rm,         makeRMInst },
      { X86::CMP8ri,          X86::CMP8mi,          makeRMInst },
      { X86::CMP8rr,          X86::CMP8rm,          makeRMInst },
      { X86::CMPPDrri,        X86::CMPPDrmi,        makeRMIInst },
      { X86::CMPPSrri,        X86::CMPPSrmi,        makeRMIInst },
      { X86::CMPSDrr,         X86::CMPSDrm,         makeRMInst },
      { X86::CMPSSrr,         X86::CMPSSrm,         makeRMInst },
      { X86::CVTSD2SSrr,      X86::CVTSD2SSrm,      makeRMInst },
      { X86::CVTSI2SDrr,      X86::CVTSI2SDrm,      makeRMInst },
      { X86::CVTSI2SSrr,      X86::CVTSI2SSrm,      makeRMInst },
      { X86::CVTSS2SDrr,      X86::CVTSS2SDrm,      makeRMInst },
      { X86::CVTTSD2SIrr,     X86::CVTTSD2SIrm,     makeRMInst },
      { X86::CVTTSS2SIrr,     X86::CVTTSS2SIrm,     makeRMInst },
      { X86::DIVPDrr,         X86::DIVPDrm,         makeRMInst },
      { X86::DIVPSrr,         X86::DIVPSrm,         makeRMInst },
      { X86::DIVSDrr,         X86::DIVSDrm,         makeRMInst },
      { X86::DIVSSrr,         X86::DIVSSrm,         makeRMInst },
      { X86::FsMOVAPDrr,      X86::MOVSDrm,         makeRMInst },
      { X86::FsMOVAPSrr,      X86::MOVSSrm,         makeRMInst },
      { X86::HADDPDrr,        X86::HADDPDrm,        makeRMInst },
      { X86::HADDPSrr,        X86::HADDPSrm,        makeRMInst },
      { X86::HSUBPDrr,        X86::HSUBPDrm,        makeRMInst },
      { X86::HSUBPSrr,        X86::HSUBPSrm,        makeRMInst },
      { X86::IMUL16rr,        X86::IMUL16rm,        makeRMInst },
      { X86::IMUL16rri,       X86::IMUL16rmi,       makeRMIInst },
      { X86::IMUL16rri8,      X86::IMUL16rmi8,      makeRMIInst },
      { X86::IMUL32rr,        X86::IMUL32rm,        makeRMInst },
      { X86::IMUL32rri,       X86::IMUL32rmi,       makeRMIInst },
      { X86::IMUL32rri8,      X86::IMUL32rmi8,      makeRMIInst },
      { X86::Int_CMPSDrr,     X86::Int_CMPSDrm,     makeRMInst },
      { X86::Int_CMPSSrr,     X86::Int_CMPSSrm,     makeRMInst },
      { X86::Int_COMISDrr,    X86::Int_COMISDrm,    makeRMInst },
      { X86::Int_COMISSrr,    X86::Int_COMISSrm,    makeRMInst },
      { X86::Int_CVTDQ2PDrr,  X86::Int_CVTDQ2PDrm,  makeRMInst },
      { X86::Int_CVTDQ2PSrr,  X86::Int_CVTDQ2PSrm,  makeRMInst },
      { X86::Int_CVTPD2DQrr,  X86::Int_CVTPD2DQrm,  makeRMInst },
      { X86::Int_CVTPD2PSrr,  X86::Int_CVTPD2PSrm,  makeRMInst },
      { X86::Int_CVTPS2DQrr,  X86::Int_CVTPS2DQrm,  makeRMInst },
      { X86::Int_CVTPS2PDrr,  X86::Int_CVTPS2PDrm,  makeRMInst },
      { X86::Int_CVTSD2SIrr,  X86::Int_CVTSD2SIrm,  makeRMInst },
      { X86::Int_CVTSD2SSrr,  X86::Int_CVTSD2SSrm,  makeRMInst },
      { X86::Int_CVTSI2SDrr,  X86::Int_CVTSI2SDrm,  makeRMInst },
      { X86::Int_CVTSI2SSrr,  X86::Int_CVTSI2SSrm,  makeRMInst },
      { X86::Int_CVTSS2SDrr,  X86::Int_CVTSS2SDrm,  makeRMInst },
      { X86::Int_CVTSS2SIrr,  X86::Int_CVTSS2SIrm,  makeRMInst },
      { X86::Int_CVTTPD2DQrr, X86::Int_CVTTPD2DQrm, makeRMInst },
      { X86::Int_CVTTPS2DQrr, X86::Int_CVTTPS2DQrm, makeRMInst },
      { X86::Int_CVTTSD2SIrr, X86::Int_CVTTSD2SIrm, makeRMInst },
      { X86::Int_CVTTSS2SIrr, X86::Int_CVTTSS2SIrm, makeRMInst },
      { X86::Int_UCOMISDrr,   X86::Int_UCOMISDrm,   makeRMInst },
      { X86::Int_UCOMISSrr,   X86::Int_UCOMISSrm,   makeRMInst },
      { X86::MAXPDrr,         X86::MAXPDrm,         makeRMInst },
      { X86::MAXPSrr,         X86::MAXPSrm,         makeRMInst },
      { X86::MINPDrr,         X86::MINPDrm,         makeRMInst },
      { X86::MINPSrr,         X86::MINPSrm,         makeRMInst },
      { X86::MOV16rr,         X86::MOV16rm,         makeRMInst },
      { X86::MOV32rr,         X86::MOV32rm,         makeRMInst },
      { X86::MOV8rr,          X86::MOV8rm,          makeRMInst },
      { X86::MOVAPDrr,        X86::MOVAPDrm,        makeRMInst },
      { X86::MOVAPSrr,        X86::MOVAPSrm,        makeRMInst },
      { X86::MOVDDUPrr,       X86::MOVDDUPrm,       makeRMInst },
      { X86::MOVDI2PDIrr,     X86::MOVDI2PDIrm,     makeRMInst },
      { X86::MOVQI2PQIrr,     X86::MOVQI2PQIrm,     makeRMInst },
      { X86::MOVSD2PDrr,      X86::MOVSD2PDrm,      makeRMInst },
      { X86::MOVSDrr,         X86::MOVSDrm,         makeRMInst },
      { X86::MOVSHDUPrr,      X86::MOVSHDUPrm,      makeRMInst },
      { X86::MOVSLDUPrr,      X86::MOVSLDUPrm,      makeRMInst },
      { X86::MOVSS2PSrr,      X86::MOVSS2PSrm,      makeRMInst },
      { X86::MOVSSrr,         X86::MOVSSrm,         makeRMInst },
      { X86::MOVSX16rr8,      X86::MOVSX16rm8,      makeRMInst },
      { X86::MOVSX32rr16,     X86::MOVSX32rm16,     makeRMInst },
      { X86::MOVSX32rr8,      X86::MOVSX32rm8,      makeRMInst },
      { X86::MOVUPDrr,        X86::MOVUPDrm,        makeRMInst },
      { X86::MOVUPSrr,        X86::MOVUPSrm,        makeRMInst },
      { X86::MOVZX16rr8,      X86::MOVZX16rm8,      makeRMInst },
      { X86::MOVZX32rr16,     X86::MOVZX32rm16,     makeRMInst },
      { X86::MOVZX32rr8,      X86::MOVZX32rm8,      makeRMInst },
      { X86::MULPDrr,         X86::MULPDrm,         makeRMInst },
      { X86::MULPSrr,         X86::MULPSrm,         makeRMInst },
      { X86::MULSDrr,         X86::MULSDrm,         makeRMInst },
      { X86::MULSSrr,         X86::MULSSrm,         makeRMInst },
      { X86::OR16rr,          X86::OR16rm,          makeRMInst },
      { X86::OR32rr,          X86::OR32rm,          makeRMInst },
      { X86::OR8rr,           X86::OR8rm,           makeRMInst },
      { X86::ORPDrr,          X86::ORPDrm,          makeRMInst },
      { X86::ORPSrr,          X86::ORPSrm,          makeRMInst },
      { X86::PACKSSDWrr,      X86::PACKSSDWrm,      makeRMInst },
      { X86::PACKSSWBrr,      X86::PACKSSWBrm,      makeRMInst },
      { X86::PACKUSWBrr,      X86::PACKUSWBrm,      makeRMInst },
      { X86::PADDBrr,         X86::PADDBrm,         makeRMInst },
      { X86::PADDDrr,         X86::PADDDrm,         makeRMInst },
      { X86::PADDSBrr,        X86::PADDSBrm,        makeRMInst },
      { X86::PADDSWrr,        X86::PADDSWrm,        makeRMInst },
      { X86::PADDWrr,         X86::PADDWrm,         makeRMInst },
      { X86::PANDNrr,         X86::PANDNrm,         makeRMInst },
      { X86::PANDrr,          X86::PANDrm,          makeRMInst },
      { X86::PAVGBrr,         X86::PAVGBrm,         makeRMInst },
      { X86::PAVGWrr,         X86::PAVGWrm,         makeRMInst },
      { X86::PCMPEQBrr,       X86::PCMPEQBrm,       makeRMInst },
      { X86::PCMPEQDrr,       X86::PCMPEQDrm,       makeRMInst },
      { X86::PCMPEQWrr,       X86::PCMPEQWrm,       makeRMInst },
      { X86::PCMPGTBrr,       X86::PCMPGTBrm,       makeRMInst },
      { X86::PCMPGTDrr,       X86::PCMPGTDrm,       makeRMInst },
      { X86::PCMPGTWrr,       X86::PCMPGTWrm,       makeRMInst },
      { X86::PINSRWrri,       X86::PINSRWrmi,       makeRMIInst },
      { X86::PMADDWDrr,       X86::PMADDWDrm,       makeRMInst },
      { X86::PMAXSWrr,        X86::PMAXSWrm,        makeRMInst },
      { X86::PMAXUBrr,        X86::PMAXUBrm,        makeRMInst },
      { X86::PMINSWrr,        X86::PMINSWrm,        makeRMInst },
      { X86::PMINUBrr,        X86::PMINUBrm,        makeRMInst },
      { X86::PMULHUWrr,       X86::PMULHUWrm,       makeRMInst },
      { X86::PMULHWrr,        X86::PMULHWrm,        makeRMInst },
      { X86::PMULLWrr,        X86::PMULLWrm,        makeRMInst },
      { X86::PMULUDQrr,       X86::PMULUDQrm,       makeRMInst },
      { X86::PORrr,           X86::PORrm,           makeRMInst },
      { X86::PSADBWrr,        X86::PSADBWrm,        makeRMInst },
      { X86::PSHUFDri,        X86::PSHUFDmi,        makeRMIInst },
      { X86::PSHUFHWri,       X86::PSHUFHWmi,       makeRMIInst },
      { X86::PSHUFLWri,       X86::PSHUFLWmi,       makeRMIInst },
      { X86::PSLLDrr,         X86::PSLLDrm,         makeRMInst },
      { X86::PSLLQrr,         X86::PSLLQrm,         makeRMInst },
      { X86::PSLLWrr,         X86::PSLLWrm,         makeRMInst },
      { X86::PSRADrr,         X86::PSRADrm,         makeRMInst },
      { X86::PSRAWrr,         X86::PSRAWrm,         makeRMInst },
      { X86::PSRLDrr,         X86::PSRLDrm,         makeRMInst },
      { X86::PSRLQrr,         X86::PSRLQrm,         makeRMInst },
      { X86::PSRLWrr,         X86::PSRLWrm,         makeRMInst },
      { X86::PSUBBrr,         X86::PSUBBrm,         makeRMInst },
      { X86::PSUBDrr,         X86::PSUBDrm,         makeRMInst },
      { X86::PSUBSBrr,        X86::PSUBSBrm,        makeRMInst },
      { X86::PSUBSWrr,        X86::PSUBSWrm,        makeRMInst },
      { X86::PSUBWrr,         X86::PSUBWrm,         makeRMInst },
      { X86::PUNPCKHBWrr,     X86::PUNPCKHBWrm,     makeRMInst },
      { X86::PUNPCKHDQrr,     X86::PUNPCKHDQrm,     makeRMInst },
      { X86::PUNPCKHQDQrr,    X86::PUNPCKHQDQrm,    makeRMInst },
      { X86::PUNPCKHWDrr,     X86::PUNPCKHWDrm,     makeRMInst },
      { X86::PUNPCKLBWrr,     X86::PUNPCKLBWrm,     makeRMInst },
      { X86::PUNPCKLDQrr,     X86::PUNPCKLDQrm,     makeRMInst },
      { X86::PUNPCKLQDQrr,    X86::PUNPCKLQDQrm,    makeRMInst },
      { X86::PUNPCKLWDrr,     X86::PUNPCKLWDrm,     makeRMInst },
      { X86::PXORrr,          X86::PXORrm,          makeRMInst },
      { X86::RCPPSr,          X86::RCPPSm,          makeRMInst },
      { X86::RSQRTPSr,        X86::RSQRTPSm,        makeRMInst },
      { X86::SBB32rr,         X86::SBB32rm,         makeRMInst },
      { X86::SHUFPDrri,       X86::SHUFPDrmi,       makeRMIInst },
      { X86::SHUFPSrri,       X86::SHUFPSrmi,       makeRMIInst },
      { X86::SQRTPDr,         X86::SQRTPDm,         makeRMInst },
      { X86::SQRTPSr,         X86::SQRTPSm,         makeRMInst },
      { X86::SQRTSDr,         X86::SQRTSDm,         makeRMInst },
      { X86::SQRTSSr,         X86::SQRTSSm,         makeRMInst },
      { X86::SUB16rr,         X86::SUB16rm,         makeRMInst },
      { X86::SUB32rr,         X86::SUB32rm,         makeRMInst },
      { X86::SUB8rr,          X86::SUB8rm,          makeRMInst },
      { X86::SUBPDrr,         X86::SUBPDrm,         makeRMInst },
      { X86::SUBPSrr,         X86::SUBPSrm,         makeRMInst },
      { X86::SUBSDrr,         X86::SUBSDrm,         makeRMInst },
      { X86::SUBSSrr,         X86::SUBSSrm,         makeRMInst },
      { X86::TEST16ri,        X86::TEST16mi,        makeMIInst },
      { X86::TEST16rr,        X86::TEST16rm,        makeRMInst },
      { X86::TEST32ri,        X86::TEST32mi,        makeMIInst },
      { X86::TEST32rr,        X86::TEST32rm,        makeRMInst },
      { X86::TEST8ri,         X86::TEST8mi,         makeMIInst },
      { X86::TEST8rr,         X86::TEST8rm,         makeRMInst },
      { X86::UCOMISDrr,       X86::UCOMISDrm,       makeRMInst },
      { X86::UCOMISSrr,       X86::UCOMISSrm,       makeRMInst },
      { X86::UNPCKHPDrr,      X86::UNPCKHPDrm,      makeRMInst },
      { X86::UNPCKHPSrr,      X86::UNPCKHPSrm,      makeRMInst },
      { X86::UNPCKLPDrr,      X86::UNPCKLPDrm,      makeRMInst },
      { X86::UNPCKLPSrr,      X86::UNPCKLPSrm,      makeRMInst },
      { X86::XCHG16rr,        X86::XCHG16rm,        makeRMInst },
      { X86::XCHG32rr,        X86::XCHG32rm,        makeRMInst },
      { X86::XCHG8rr,         X86::XCHG8rm,         makeRMInst },
      { X86::XOR16rr,         X86::XOR16rm,         makeRMInst },
      { X86::XOR32rr,         X86::XOR32rm,         makeRMInst },
      { X86::XOR8rr,          X86::XOR8rm,          makeRMInst },
      { X86::XORPDrr,         X86::XORPDrm,         makeRMInst },
      { X86::XORPSrr,         X86::XORPSrm,         makeRMInst }
    };
    ASSERT_SORTED(OpcodeTable);
    OpcodeTablePtr = OpcodeTable;
    OpcodeTableSize = ARRAY_SIZE(OpcodeTable);
  }
  
  // If table selected
  if (OpcodeTablePtr) {
    // Opcode to fuse
    unsigned fromOpcode = MI->getOpcode();
    // Lookup fromOpcode in table
    const TableEntry *entry = TableLookup(OpcodeTablePtr, OpcodeTableSize,
                                          fromOpcode);
    
    // If opcode found in table
    if (entry) {
      // Fused opcode
      unsigned toOpcode = entry->to;
      
      // Make new instruction
      switch (entry->make) {
      case makeM0Inst:  return MakeM0Inst(toOpcode, FrameIndex, MI);
      case makeMIInst:  return MakeMIInst(toOpcode, FrameIndex, MI);
      case makeMInst:   return MakeMInst(toOpcode, FrameIndex, MI);
      case makeMRIInst: return MakeMRIInst(toOpcode, FrameIndex, MI);
      case makeMRInst:  return MakeMRInst(toOpcode, FrameIndex, MI);
      case makeRMIInst: return MakeRMIInst(toOpcode, FrameIndex, MI);
      case makeRMInst:  return MakeRMInst(toOpcode, FrameIndex, MI);
      default: assert(0 && "Unknown instruction make");
      }
    }
  }
  
  // No fusion 
  if (PrintFailedFusing)
    std::cerr << "We failed to fuse ("
              << ((i == 1) ? "r" : "s") << "): " << *MI;
  return NULL;
}


const unsigned *X86RegisterInfo::getCalleeSaveRegs() const {
  static const unsigned CalleeSaveRegs[] = {
    X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };
  return CalleeSaveRegs;
}

const TargetRegisterClass* const*
X86RegisterInfo::getCalleeSaveRegClasses() const {
  static const TargetRegisterClass * const CalleeSaveRegClasses[] = {
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,  0
  };
  return CalleeSaveRegClasses;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool hasFP(const MachineFunction &MF) {
  return (NoFramePointerElim || 
          MF.getFrameInfo()->hasVarSizedObjects() ||
          MF.getInfo<X86FunctionInfo>()->getForceFramePointer());
}

void X86RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (hasFP(MF)) {
    // If we have a frame pointer, turn the adjcallstackup instruction into a
    // 'sub ESP, <amt>' and the adjcallstackdown instruction into 'add ESP,
    // <amt>'
    MachineInstr *Old = I;
    unsigned Amount = Old->getOperand(0).getImmedValue();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      MachineInstr *New = 0;
      if (Old->getOpcode() == X86::ADJCALLSTACKDOWN) {
        New=BuildMI(X86::SUB32ri, 1, X86::ESP, MachineOperand::UseAndDef)
              .addImm(Amount);
      } else {
        assert(Old->getOpcode() == X86::ADJCALLSTACKUP);
        // factor out the amount the callee already popped.
        unsigned CalleeAmt = Old->getOperand(1).getImmedValue();
        Amount -= CalleeAmt;
        if (Amount) {
          unsigned Opc = Amount < 128 ? X86::ADD32ri8 : X86::ADD32ri;
          New = BuildMI(Opc, 1, X86::ESP,
                        MachineOperand::UseAndDef).addImm(Amount);
        }
      }

      // Replace the pseudo instruction with a new instruction...
      if (New) MBB.insert(I, New);
    }
  } else if (I->getOpcode() == X86::ADJCALLSTACKUP) {
    // If we are performing frame pointer elimination and if the callee pops
    // something off the stack pointer, add it back.  We do this until we have
    // more advanced stack pointer tracking ability.
    if (unsigned CalleeAmt = I->getOperand(1).getImmedValue()) {
      unsigned Opc = CalleeAmt < 128 ? X86::SUB32ri8 : X86::SUB32ri;
      MachineInstr *New =
        BuildMI(Opc, 1, X86::ESP,
                MachineOperand::UseAndDef).addImm(CalleeAmt);
      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

void X86RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const{
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();

  // This must be part of a four operand memory reference.  Replace the
  // FrameIndex with base register with EBP.  Add add an offset to the offset.
  MI.getOperand(i).ChangeToRegister(hasFP(MF) ? X86::EBP : X86::ESP);

  // Now add the frame object offset to the offset from EBP.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(i+3).getImmedValue()+4;

  if (!hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();
  else
    Offset += 4;  // Skip the saved EBP

  MI.getOperand(i+3).ChangeToImmediate(Offset);
}

void
X86RegisterInfo::processFunctionBeforeFrameFinalized(MachineFunction &MF) const{
  if (hasFP(MF)) {
    // Create a frame entry for the EBP register that must be saved.
    int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, -8);
    assert(FrameIdx == MF.getFrameInfo()->getObjectIndexBegin() &&
           "Slot for EBP register must be last in order to be found!");
  }
}

void X86RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
  const Function* Fn = MF.getFunction();
  const X86Subtarget* Subtarget = &MF.getTarget().getSubtarget<X86Subtarget>();
  MachineInstr *MI;
  
  // Get the number of bytes to allocate from the FrameInfo
  unsigned NumBytes = MFI->getStackSize();
  if (MFI->hasCalls() || MF.getFrameInfo()->hasVarSizedObjects()) {
    // When we have no frame pointer, we reserve argument space for call sites
    // in the function immediately on entry to the current function.  This
    // eliminates the need for add/sub ESP brackets around call sites.
    //
    if (!hasFP(MF))
      NumBytes += MFI->getMaxCallFrameSize();

    // Round the size to a multiple of the alignment (don't forget the 4 byte
    // offset though).
    NumBytes = ((NumBytes+4)+Align-1)/Align*Align - 4;
  }

  // Update frame info to pretend that this is part of the stack...
  MFI->setStackSize(NumBytes);

  if (NumBytes) {   // adjust stack pointer: ESP -= numbytes
    if (NumBytes >= 4096 && Subtarget->TargetType == X86Subtarget::isCygwin) {
      // Function prologue calls _alloca to probe the stack when allocating  
      // more than 4k bytes in one go. Touching the stack at 4K increments is  
      // necessary to ensure that the guard pages used by the OS virtual memory
      // manager are allocated in correct sequence.
      MI = BuildMI(X86::MOV32ri, 2, X86::EAX).addImm(NumBytes);
      MBB.insert(MBBI, MI);
      MI = BuildMI(X86::CALLpcrel32, 1).addExternalSymbol("_alloca");
      MBB.insert(MBBI, MI);
    } else {
      unsigned Opc = NumBytes < 128 ? X86::SUB32ri8 : X86::SUB32ri;
      MI = BuildMI(Opc, 1, X86::ESP,MachineOperand::UseAndDef).addImm(NumBytes);
      MBB.insert(MBBI, MI);
    }
  }

  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexBegin())+4;

    // Save EBP into the appropriate stack slot...
    MI = addRegOffset(BuildMI(X86::MOV32mr, 5),    // mov [ESP-<offset>], EBP
                      X86::ESP, EBPOffset+NumBytes).addReg(X86::EBP);
    MBB.insert(MBBI, MI);

    // Update EBP with the new base value...
    if (NumBytes == 4)    // mov EBP, ESP
      MI = BuildMI(X86::MOV32rr, 2, X86::EBP).addReg(X86::ESP);
    else                  // lea EBP, [ESP+StackSize]
      MI = addRegOffset(BuildMI(X86::LEA32r, 5, X86::EBP), X86::ESP,NumBytes-4);

    MBB.insert(MBBI, MI);
  }

  // If it's main() on Cygwin\Mingw32 we should align stack as well
  if (Fn->hasExternalLinkage() && Fn->getName() == "main" &&
      Subtarget->TargetType == X86Subtarget::isCygwin) {
    MI = BuildMI(X86::AND32ri, 2, X86::ESP).addImm(-Align);
    MBB.insert(MBBI, MI);

    // Probe the stack
    MI = BuildMI(X86::MOV32ri, 2, X86::EAX).addImm(Align);
    MBB.insert(MBBI, MI);
    MI = BuildMI(X86::CALLpcrel32, 1).addExternalSymbol("_alloca");
    MBB.insert(MBBI, MI);
  }
}

void X86RegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());

  switch (MBBI->getOpcode()) {
  case X86::RET:
  case X86::RETI:
  case X86::TAILJMPd:
  case X86::TAILJMPr:
  case X86::TAILJMPm: break;  // These are ok
  default:
    assert(0 && "Can only insert epilog into returning blocks");
  }

  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexEnd()-1)+4;

    // mov ESP, EBP
    BuildMI(MBB, MBBI, X86::MOV32rr, 1,X86::ESP).addReg(X86::EBP);

    // pop EBP
    BuildMI(MBB, MBBI, X86::POP32r, 0, X86::EBP);
  } else {
    // Get the number of bytes allocated from the FrameInfo...
    unsigned NumBytes = MFI->getStackSize();

    if (NumBytes) {    // adjust stack pointer back: ESP += numbytes
      // If there is an ADD32ri or SUB32ri of ESP immediately before this
      // instruction, merge the two instructions.
      if (MBBI != MBB.begin()) {
        MachineBasicBlock::iterator PI = prior(MBBI);
        if ((PI->getOpcode() == X86::ADD32ri || 
             PI->getOpcode() == X86::ADD32ri8) &&
            PI->getOperand(0).getReg() == X86::ESP) {
          NumBytes += PI->getOperand(1).getImmedValue();
          MBB.erase(PI);
        } else if ((PI->getOpcode() == X86::SUB32ri ||
                    PI->getOpcode() == X86::SUB32ri8) &&
                   PI->getOperand(0).getReg() == X86::ESP) {
          NumBytes -= PI->getOperand(1).getImmedValue();
          MBB.erase(PI);
        } else if (PI->getOpcode() == X86::ADJSTACKPTRri) {
          NumBytes += PI->getOperand(1).getImmedValue();
          MBB.erase(PI);
        }
      }

      if (NumBytes > 0) {
        unsigned Opc = NumBytes < 128 ? X86::ADD32ri8 : X86::ADD32ri;
        BuildMI(MBB, MBBI, Opc, 2)
          .addReg(X86::ESP, MachineOperand::UseAndDef).addImm(NumBytes);
      } else if ((int)NumBytes < 0) {
        unsigned Opc = -NumBytes < 128 ? X86::SUB32ri8 : X86::SUB32ri;
        BuildMI(MBB, MBBI, Opc, 2)
          .addReg(X86::ESP, MachineOperand::UseAndDef).addImm(-NumBytes);
      }
    }
  }
}

unsigned X86RegisterInfo::getRARegister() const {
  return X86::ST0;  // use a non-register register
}

unsigned X86RegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? X86::EBP : X86::ESP;
}

namespace llvm {
unsigned getX86SubSuperRegister(unsigned Reg, MVT::ValueType VT, bool High) {
  switch (VT) {
  default: return Reg;
  case MVT::i8:
    if (High) {
      switch (Reg) {
      default: return Reg;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
        return X86::AH;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
        return X86::DH;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
        return X86::CH;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
        return X86::BH;
      }
    } else {
      switch (Reg) {
      default: return Reg;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
        return X86::AL;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
        return X86::DL;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
        return X86::CL;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
        return X86::BL;
      }
    }
  case MVT::i16:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
      return X86::AX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
      return X86::DX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
      return X86::CX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
      return X86::BX;
    case X86::ESI:
      return X86::SI;
    case X86::EDI:
      return X86::DI;
    case X86::EBP:
      return X86::BP;
    case X86::ESP:
      return X86::SP;
    }
  case MVT::i32:
    switch (Reg) {
    default: return true;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX:
      return X86::EAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX:
      return X86::EDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX:
      return X86::ECX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX:
      return X86::EBX;
    case X86::SI:
      return X86::ESI;
    case X86::DI:
      return X86::EDI;
    case X86::BP:
      return X86::EBP;
    case X86::SP:
      return X86::ESP;
    }
  }

  return Reg;
}
}

#include "X86GenRegisterInfo.inc"


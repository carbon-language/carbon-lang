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

X86RegisterInfo::X86RegisterInfo(const TargetInstrInfo &tii)
  : X86GenRegisterInfo(X86::ADJCALLSTACKDOWN, X86::ADJCALLSTACKUP), TII(tii) {}

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

static MachineInstr *FuseTwoAddrInst(unsigned Opcode, unsigned FrameIndex,
                                     MachineInstr *MI) {
  unsigned NumOps = MI->getNumOperands()-2;
  // Create the base instruction with the memory operand as the first part.
  MachineInstrBuilder MIB = addFrameReference(BuildMI(Opcode, 4+NumOps),
                                              FrameIndex);
  
  // Loop over the rest of the ri operands, converting them over.
  for (unsigned i = 0; i != NumOps; ++i) {
    if (MI->getOperand(i+2).isReg())
      MIB = MIB.addReg(MI->getOperand(i+2).getReg());
    else {
      assert(MI->getOperand(i+2).isImm() && "Unknown operand type!");
      MIB = MIB.addImm(MI->getOperand(i+2).getImm());
    }
  }
  return MIB;
}

static MachineInstr *FuseInst(unsigned Opcode, unsigned OpNo,
                              unsigned FrameIndex, MachineInstr *MI) {
  MachineInstrBuilder MIB = BuildMI(Opcode, MI->getNumOperands()+3);
  
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (i == OpNo) {
      assert(MO.isReg() && "Expected to fold into reg operand!");
      MIB = addFrameReference(MIB, FrameIndex);
    } else if (MO.isReg())
      MIB = MIB.addReg(MO.getReg(), MO.getUseType());
    else if (MO.isImm())
      MIB = MIB.addImm(MO.getImm());
    else if (MO.isGlobalAddress())
      MIB = MIB.addGlobalAddress(MO.getGlobal(), MO.getOffset());
    else if (MO.isJumpTableIndex())
      MIB = MIB.addJumpTableIndex(MO.getJumpTableIndex());
    else
      assert(0 && "Unknown operand for FuseInst!");
  }
  return MIB;
}

static MachineInstr *MakeM0Inst(unsigned Opcode, unsigned FrameIndex,
                                MachineInstr *MI) {
  return addFrameReference(BuildMI(Opcode, 5), FrameIndex).addImm(0);
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


MachineInstr* X86RegisterInfo::foldMemoryOperand(MachineInstr *MI,
                                                 unsigned i,
                                                 int FrameIndex) const {
  // Check switch flag 
  if (NoFusing) return NULL;

  // Table (and size) to search
  const TableEntry *OpcodeTablePtr = NULL;
  unsigned OpcodeTableSize = 0;
  bool isTwoAddrFold = false;

  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  if (MI->getNumOperands() >= 2 && MI->getOperand(0).isReg() && 
      MI->getOperand(1).isReg() && i < 2 &&
      MI->getOperand(0).getReg() == MI->getOperand(1).getReg() &&
      TII.isTwoAddrInstr(MI->getOpcode())) {
    static const TableEntry OpcodeTable[] = {
      { X86::ADC32ri,     X86::ADC32mi },
      { X86::ADC32ri8,    X86::ADC32mi8 },
      { X86::ADC32rr,     X86::ADC32mr },
      { X86::ADD16ri,     X86::ADD16mi },
      { X86::ADD16ri8,    X86::ADD16mi8 },
      { X86::ADD16rr,     X86::ADD16mr },
      { X86::ADD32ri,     X86::ADD32mi },
      { X86::ADD32ri8,    X86::ADD32mi8 },
      { X86::ADD32rr,     X86::ADD32mr },
      { X86::ADD8ri,      X86::ADD8mi },
      { X86::ADD8rr,      X86::ADD8mr },
      { X86::AND16ri,     X86::AND16mi },
      { X86::AND16ri8,    X86::AND16mi8 },
      { X86::AND16rr,     X86::AND16mr },
      { X86::AND32ri,     X86::AND32mi },
      { X86::AND32ri8,    X86::AND32mi8 },
      { X86::AND32rr,     X86::AND32mr },
      { X86::AND8ri,      X86::AND8mi },
      { X86::AND8rr,      X86::AND8mr },
      { X86::DEC16r,      X86::DEC16m },
      { X86::DEC32r,      X86::DEC32m },
      { X86::DEC8r,       X86::DEC8m },
      { X86::INC16r,      X86::INC16m },
      { X86::INC32r,      X86::INC32m },
      { X86::INC8r,       X86::INC8m },
      { X86::NEG16r,      X86::NEG16m },
      { X86::NEG32r,      X86::NEG32m },
      { X86::NEG8r,       X86::NEG8m },
      { X86::NOT16r,      X86::NOT16m },
      { X86::NOT32r,      X86::NOT32m },
      { X86::NOT8r,       X86::NOT8m },
      { X86::OR16ri,      X86::OR16mi },
      { X86::OR16ri8,     X86::OR16mi8 },
      { X86::OR16rr,      X86::OR16mr },
      { X86::OR32ri,      X86::OR32mi },
      { X86::OR32ri8,     X86::OR32mi8 },
      { X86::OR32rr,      X86::OR32mr },
      { X86::OR8ri,       X86::OR8mi },
      { X86::OR8rr,       X86::OR8mr },
      { X86::ROL16r1,     X86::ROL16m1 },
      { X86::ROL16rCL,    X86::ROL16mCL },
      { X86::ROL16ri,     X86::ROL16mi },
      { X86::ROL32r1,     X86::ROL32m1 },
      { X86::ROL32rCL,    X86::ROL32mCL },
      { X86::ROL32ri,     X86::ROL32mi },
      { X86::ROL8r1,      X86::ROL8m1 },
      { X86::ROL8rCL,     X86::ROL8mCL },
      { X86::ROL8ri,      X86::ROL8mi },
      { X86::ROR16r1,     X86::ROR16m1 },
      { X86::ROR16rCL,    X86::ROR16mCL },
      { X86::ROR16ri,     X86::ROR16mi },
      { X86::ROR32r1,     X86::ROR32m1 },
      { X86::ROR32rCL,    X86::ROR32mCL },
      { X86::ROR32ri,     X86::ROR32mi },
      { X86::ROR8r1,      X86::ROR8m1 },
      { X86::ROR8rCL,     X86::ROR8mCL },
      { X86::ROR8ri,      X86::ROR8mi },
      { X86::SAR16r1,     X86::SAR16m1 },
      { X86::SAR16rCL,    X86::SAR16mCL },
      { X86::SAR16ri,     X86::SAR16mi },
      { X86::SAR32r1,     X86::SAR32m1 },
      { X86::SAR32rCL,    X86::SAR32mCL },
      { X86::SAR32ri,     X86::SAR32mi },
      { X86::SAR8r1,      X86::SAR8m1 },
      { X86::SAR8rCL,     X86::SAR8mCL },
      { X86::SAR8ri,      X86::SAR8mi },
      { X86::SBB32ri,     X86::SBB32mi },
      { X86::SBB32ri8,    X86::SBB32mi8 },
      { X86::SBB32rr,     X86::SBB32mr },
      { X86::SHL16r1,     X86::SHL16m1 },
      { X86::SHL16rCL,    X86::SHL16mCL },
      { X86::SHL16ri,     X86::SHL16mi },
      { X86::SHL32r1,     X86::SHL32m1 },
      { X86::SHL32rCL,    X86::SHL32mCL },
      { X86::SHL32ri,     X86::SHL32mi },
      { X86::SHL8r1,      X86::SHL8m1 },
      { X86::SHL8rCL,     X86::SHL8mCL },
      { X86::SHL8ri,      X86::SHL8mi },
      { X86::SHLD16rrCL,  X86::SHLD16mrCL },
      { X86::SHLD16rri8,  X86::SHLD16mri8 },
      { X86::SHLD32rrCL,  X86::SHLD32mrCL },
      { X86::SHLD32rri8,  X86::SHLD32mri8 },
      { X86::SHR16r1,     X86::SHR16m1 },
      { X86::SHR16rCL,    X86::SHR16mCL },
      { X86::SHR16ri,     X86::SHR16mi },
      { X86::SHR32r1,     X86::SHR32m1 },
      { X86::SHR32rCL,    X86::SHR32mCL },
      { X86::SHR32ri,     X86::SHR32mi },
      { X86::SHR8r1,      X86::SHR8m1 },
      { X86::SHR8rCL,     X86::SHR8mCL },
      { X86::SHR8ri,      X86::SHR8mi },
      { X86::SHRD16rrCL,  X86::SHRD16mrCL },
      { X86::SHRD16rri8,  X86::SHRD16mri8 },
      { X86::SHRD32rrCL,  X86::SHRD32mrCL },
      { X86::SHRD32rri8,  X86::SHRD32mri8 },
      { X86::SUB16ri,     X86::SUB16mi },
      { X86::SUB16ri8,    X86::SUB16mi8 },
      { X86::SUB16rr,     X86::SUB16mr },
      { X86::SUB32ri,     X86::SUB32mi },
      { X86::SUB32ri8,    X86::SUB32mi8 },
      { X86::SUB32rr,     X86::SUB32mr },
      { X86::SUB8ri,      X86::SUB8mi },
      { X86::SUB8rr,      X86::SUB8mr },
      { X86::XOR16ri,     X86::XOR16mi },
      { X86::XOR16ri8,    X86::XOR16mi8 },
      { X86::XOR16rr,     X86::XOR16mr },
      { X86::XOR32ri,     X86::XOR32mi },
      { X86::XOR32ri8,    X86::XOR32mi8 },
      { X86::XOR32rr,     X86::XOR32mr },
      { X86::XOR8ri,      X86::XOR8mi },
      { X86::XOR8rr,      X86::XOR8mr }
    };
    ASSERT_SORTED(OpcodeTable);
    OpcodeTablePtr = OpcodeTable;
    OpcodeTableSize = ARRAY_SIZE(OpcodeTable);
    isTwoAddrFold = true;
  } else if (i == 0) { // If operand 0
    if (MI->getOpcode() == X86::MOV16r0)
      return MakeM0Inst(X86::MOV16mi, FrameIndex, MI);
    else if (MI->getOpcode() == X86::MOV32r0)
      return MakeM0Inst(X86::MOV32mi, FrameIndex, MI);
    else if (MI->getOpcode() == X86::MOV8r0)
      return MakeM0Inst(X86::MOV8mi, FrameIndex, MI);
    
    static const TableEntry OpcodeTable[] = {
      { X86::CMP16ri,     X86::CMP16mi },
      { X86::CMP16ri8,    X86::CMP16mi8 },
      { X86::CMP32ri,     X86::CMP32mi },
      { X86::CMP32ri8,    X86::CMP32mi8 },
      { X86::CMP8ri,      X86::CMP8mi },
      { X86::DIV16r,      X86::DIV16m },
      { X86::DIV32r,      X86::DIV32m },
      { X86::DIV8r,       X86::DIV8m },
      { X86::FsMOVAPDrr,  X86::MOVSDmr },
      { X86::FsMOVAPSrr,  X86::MOVSSmr },
      { X86::IDIV16r,     X86::IDIV16m },
      { X86::IDIV32r,     X86::IDIV32m },
      { X86::IDIV8r,      X86::IDIV8m },
      { X86::IMUL16r,     X86::IMUL16m },
      { X86::IMUL32r,     X86::IMUL32m },
      { X86::IMUL8r,      X86::IMUL8m },
      { X86::MOV16ri,     X86::MOV16mi },
      { X86::MOV16rr,     X86::MOV16mr },
      { X86::MOV32ri,     X86::MOV32mi },
      { X86::MOV32rr,     X86::MOV32mr },
      { X86::MOV8ri,      X86::MOV8mi },
      { X86::MOV8rr,      X86::MOV8mr },
      { X86::MOVAPDrr,    X86::MOVAPDmr },
      { X86::MOVAPSrr,    X86::MOVAPSmr },
      { X86::MOVPDI2DIrr, X86::MOVPDI2DImr },
      { X86::MOVPS2SSrr,  X86::MOVPS2SSmr },
      { X86::MOVSDrr,     X86::MOVSDmr },
      { X86::MOVSSrr,     X86::MOVSSmr },
      { X86::MOVUPDrr,    X86::MOVUPDmr },
      { X86::MOVUPSrr,    X86::MOVUPSmr },
      { X86::MUL16r,      X86::MUL16m },
      { X86::MUL32r,      X86::MUL32m },
      { X86::MUL8r,       X86::MUL8m },
      { X86::SETAEr,      X86::SETAEm },
      { X86::SETAr,       X86::SETAm },
      { X86::SETBEr,      X86::SETBEm },
      { X86::SETBr,       X86::SETBm },
      { X86::SETEr,       X86::SETEm },
      { X86::SETGEr,      X86::SETGEm },
      { X86::SETGr,       X86::SETGm },
      { X86::SETLEr,      X86::SETLEm },
      { X86::SETLr,       X86::SETLm },
      { X86::SETNEr,      X86::SETNEm },
      { X86::SETNPr,      X86::SETNPm },
      { X86::SETNSr,      X86::SETNSm },
      { X86::SETPr,       X86::SETPm },
      { X86::SETSr,       X86::SETSm },
      { X86::TEST16ri,    X86::TEST16mi },
      { X86::TEST32ri,    X86::TEST32mi },
      { X86::TEST8ri,     X86::TEST8mi },
      { X86::XCHG16rr,    X86::XCHG16mr },
      { X86::XCHG32rr,    X86::XCHG32mr },
      { X86::XCHG8rr,     X86::XCHG8mr }
    };
    ASSERT_SORTED(OpcodeTable);
    OpcodeTablePtr = OpcodeTable;
    OpcodeTableSize = ARRAY_SIZE(OpcodeTable);
  } else if (i == 1) {
    static const TableEntry OpcodeTable[] = {
      { X86::CMP16rr,         X86::CMP16rm },
      { X86::CMP32rr,         X86::CMP32rm },
      { X86::CMP8rr,          X86::CMP8rm },
      { X86::CMPPDrri,        X86::CMPPDrmi },
      { X86::CMPPSrri,        X86::CMPPSrmi },
      { X86::CMPSDrr,         X86::CMPSDrm },
      { X86::CMPSSrr,         X86::CMPSSrm },
      { X86::CVTSD2SSrr,      X86::CVTSD2SSrm },
      { X86::CVTSI2SDrr,      X86::CVTSI2SDrm },
      { X86::CVTSI2SSrr,      X86::CVTSI2SSrm },
      { X86::CVTSS2SDrr,      X86::CVTSS2SDrm },
      { X86::CVTTSD2SIrr,     X86::CVTTSD2SIrm },
      { X86::CVTTSS2SIrr,     X86::CVTTSS2SIrm },
      { X86::FsMOVAPDrr,      X86::MOVSDrm },
      { X86::FsMOVAPSrr,      X86::MOVSSrm },
      { X86::IMUL16rri,       X86::IMUL16rmi },
      { X86::IMUL16rri8,      X86::IMUL16rmi8 },
      { X86::IMUL32rri,       X86::IMUL32rmi },
      { X86::IMUL32rri8,      X86::IMUL32rmi8 },
      { X86::Int_CMPSDrr,     X86::Int_CMPSDrm },
      { X86::Int_CMPSSrr,     X86::Int_CMPSSrm },
      { X86::Int_COMISDrr,    X86::Int_COMISDrm },
      { X86::Int_COMISSrr,    X86::Int_COMISSrm },
      { X86::Int_CVTDQ2PDrr,  X86::Int_CVTDQ2PDrm },
      { X86::Int_CVTDQ2PSrr,  X86::Int_CVTDQ2PSrm },
      { X86::Int_CVTPD2DQrr,  X86::Int_CVTPD2DQrm },
      { X86::Int_CVTPD2PSrr,  X86::Int_CVTPD2PSrm },
      { X86::Int_CVTPS2DQrr,  X86::Int_CVTPS2DQrm },
      { X86::Int_CVTPS2PDrr,  X86::Int_CVTPS2PDrm },
      { X86::Int_CVTSD2SIrr,  X86::Int_CVTSD2SIrm },
      { X86::Int_CVTSD2SSrr,  X86::Int_CVTSD2SSrm },
      { X86::Int_CVTSI2SDrr,  X86::Int_CVTSI2SDrm },
      { X86::Int_CVTSI2SSrr,  X86::Int_CVTSI2SSrm },
      { X86::Int_CVTSS2SDrr,  X86::Int_CVTSS2SDrm },
      { X86::Int_CVTSS2SIrr,  X86::Int_CVTSS2SIrm },
      { X86::Int_CVTTPD2DQrr, X86::Int_CVTTPD2DQrm },
      { X86::Int_CVTTPS2DQrr, X86::Int_CVTTPS2DQrm },
      { X86::Int_CVTTSD2SIrr, X86::Int_CVTTSD2SIrm },
      { X86::Int_CVTTSS2SIrr, X86::Int_CVTTSS2SIrm },
      { X86::Int_UCOMISDrr,   X86::Int_UCOMISDrm },
      { X86::Int_UCOMISSrr,   X86::Int_UCOMISSrm },
      { X86::MOV16rr,         X86::MOV16rm },
      { X86::MOV32rr,         X86::MOV32rm },
      { X86::MOV8rr,          X86::MOV8rm },
      { X86::MOVAPDrr,        X86::MOVAPDrm },
      { X86::MOVAPSrr,        X86::MOVAPSrm },
      { X86::MOVDDUPrr,       X86::MOVDDUPrm },
      { X86::MOVDI2PDIrr,     X86::MOVDI2PDIrm },
      { X86::MOVQI2PQIrr,     X86::MOVQI2PQIrm },
      { X86::MOVSD2PDrr,      X86::MOVSD2PDrm },
      { X86::MOVSDrr,         X86::MOVSDrm },
      { X86::MOVSHDUPrr,      X86::MOVSHDUPrm },
      { X86::MOVSLDUPrr,      X86::MOVSLDUPrm },
      { X86::MOVSS2PSrr,      X86::MOVSS2PSrm },
      { X86::MOVSSrr,         X86::MOVSSrm },
      { X86::MOVSX16rr8,      X86::MOVSX16rm8 },
      { X86::MOVSX32rr16,     X86::MOVSX32rm16 },
      { X86::MOVSX32rr8,      X86::MOVSX32rm8 },
      { X86::MOVUPDrr,        X86::MOVUPDrm },
      { X86::MOVUPSrr,        X86::MOVUPSrm },
      { X86::MOVZX16rr8,      X86::MOVZX16rm8 },
      { X86::MOVZX32rr16,     X86::MOVZX32rm16 },
      { X86::MOVZX32rr8,      X86::MOVZX32rm8 },
      { X86::PSHUFDri,        X86::PSHUFDmi },
      { X86::PSHUFHWri,       X86::PSHUFHWmi },
      { X86::PSHUFLWri,       X86::PSHUFLWmi },
      { X86::TEST16rr,        X86::TEST16rm },
      { X86::TEST32rr,        X86::TEST32rm },
      { X86::TEST8rr,         X86::TEST8rm },
      { X86::UCOMISDrr,       X86::UCOMISDrm },
      { X86::UCOMISSrr,       X86::UCOMISSrm },
      { X86::XCHG16rr,        X86::XCHG16rm },
      { X86::XCHG32rr,        X86::XCHG32rm },
      { X86::XCHG8rr,         X86::XCHG8rm }
    };
    ASSERT_SORTED(OpcodeTable);
    OpcodeTablePtr = OpcodeTable;
    OpcodeTableSize = ARRAY_SIZE(OpcodeTable);
  } else if (i == 2) {
    static const TableEntry OpcodeTable[] = {
      { X86::ADC32rr,         X86::ADC32rm },
      { X86::ADD16rr,         X86::ADD16rm },
      { X86::ADD32rr,         X86::ADD32rm },
      { X86::ADD8rr,          X86::ADD8rm },
      { X86::ADDPDrr,         X86::ADDPDrm },
      { X86::ADDPSrr,         X86::ADDPSrm },
      { X86::ADDSDrr,         X86::ADDSDrm },
      { X86::ADDSSrr,         X86::ADDSSrm },
      { X86::ADDSUBPDrr,      X86::ADDSUBPDrm },
      { X86::ADDSUBPSrr,      X86::ADDSUBPSrm },
      { X86::AND16rr,         X86::AND16rm },
      { X86::AND32rr,         X86::AND32rm },
      { X86::AND8rr,          X86::AND8rm },
      { X86::ANDNPDrr,        X86::ANDNPDrm },
      { X86::ANDNPSrr,        X86::ANDNPSrm },
      { X86::ANDPDrr,         X86::ANDPDrm },
      { X86::ANDPSrr,         X86::ANDPSrm },
      { X86::CMOVA16rr,       X86::CMOVA16rm },
      { X86::CMOVA32rr,       X86::CMOVA32rm },
      { X86::CMOVAE16rr,      X86::CMOVAE16rm },
      { X86::CMOVAE32rr,      X86::CMOVAE32rm },
      { X86::CMOVB16rr,       X86::CMOVB16rm },
      { X86::CMOVB32rr,       X86::CMOVB32rm },
      { X86::CMOVBE16rr,      X86::CMOVBE16rm },
      { X86::CMOVBE32rr,      X86::CMOVBE32rm },
      { X86::CMOVE16rr,       X86::CMOVE16rm },
      { X86::CMOVE32rr,       X86::CMOVE32rm },
      { X86::CMOVG16rr,       X86::CMOVG16rm },
      { X86::CMOVG32rr,       X86::CMOVG32rm },
      { X86::CMOVGE16rr,      X86::CMOVGE16rm },
      { X86::CMOVGE32rr,      X86::CMOVGE32rm },
      { X86::CMOVL16rr,       X86::CMOVL16rm },
      { X86::CMOVL32rr,       X86::CMOVL32rm },
      { X86::CMOVLE16rr,      X86::CMOVLE16rm },
      { X86::CMOVLE32rr,      X86::CMOVLE32rm },
      { X86::CMOVNE16rr,      X86::CMOVNE16rm },
      { X86::CMOVNE32rr,      X86::CMOVNE32rm },
      { X86::CMOVNP16rr,      X86::CMOVNP16rm },
      { X86::CMOVNP32rr,      X86::CMOVNP32rm },
      { X86::CMOVNS16rr,      X86::CMOVNS16rm },
      { X86::CMOVNS32rr,      X86::CMOVNS32rm },
      { X86::CMOVP16rr,       X86::CMOVP16rm },
      { X86::CMOVP32rr,       X86::CMOVP32rm },
      { X86::CMOVS16rr,       X86::CMOVS16rm },
      { X86::CMOVS32rr,       X86::CMOVS32rm },
      { X86::DIVPDrr,         X86::DIVPDrm },
      { X86::DIVPSrr,         X86::DIVPSrm },
      { X86::DIVSDrr,         X86::DIVSDrm },
      { X86::DIVSSrr,         X86::DIVSSrm },
      { X86::HADDPDrr,        X86::HADDPDrm },
      { X86::HADDPSrr,        X86::HADDPSrm },
      { X86::HSUBPDrr,        X86::HSUBPDrm },
      { X86::HSUBPSrr,        X86::HSUBPSrm },
      { X86::IMUL16rr,        X86::IMUL16rm },
      { X86::IMUL32rr,        X86::IMUL32rm },
      { X86::MAXPDrr,         X86::MAXPDrm },
      { X86::MAXPSrr,         X86::MAXPSrm },
      { X86::MINPDrr,         X86::MINPDrm },
      { X86::MINPSrr,         X86::MINPSrm },
      { X86::MULPDrr,         X86::MULPDrm },
      { X86::MULPSrr,         X86::MULPSrm },
      { X86::MULSDrr,         X86::MULSDrm },
      { X86::MULSSrr,         X86::MULSSrm },
      { X86::OR16rr,          X86::OR16rm },
      { X86::OR32rr,          X86::OR32rm },
      { X86::OR8rr,           X86::OR8rm },
      { X86::ORPDrr,          X86::ORPDrm },
      { X86::ORPSrr,          X86::ORPSrm },
      { X86::PACKSSDWrr,      X86::PACKSSDWrm },
      { X86::PACKSSWBrr,      X86::PACKSSWBrm },
      { X86::PACKUSWBrr,      X86::PACKUSWBrm },
      { X86::PADDBrr,         X86::PADDBrm },
      { X86::PADDDrr,         X86::PADDDrm },
      { X86::PADDSBrr,        X86::PADDSBrm },
      { X86::PADDSWrr,        X86::PADDSWrm },
      { X86::PADDWrr,         X86::PADDWrm },
      { X86::PANDNrr,         X86::PANDNrm },
      { X86::PANDrr,          X86::PANDrm },
      { X86::PAVGBrr,         X86::PAVGBrm },
      { X86::PAVGWrr,         X86::PAVGWrm },
      { X86::PCMPEQBrr,       X86::PCMPEQBrm },
      { X86::PCMPEQDrr,       X86::PCMPEQDrm },
      { X86::PCMPEQWrr,       X86::PCMPEQWrm },
      { X86::PCMPGTBrr,       X86::PCMPGTBrm },
      { X86::PCMPGTDrr,       X86::PCMPGTDrm },
      { X86::PCMPGTWrr,       X86::PCMPGTWrm },
      { X86::PINSRWrri,       X86::PINSRWrmi },
      { X86::PMADDWDrr,       X86::PMADDWDrm },
      { X86::PMAXSWrr,        X86::PMAXSWrm },
      { X86::PMAXUBrr,        X86::PMAXUBrm },
      { X86::PMINSWrr,        X86::PMINSWrm },
      { X86::PMINUBrr,        X86::PMINUBrm },
      { X86::PMULHUWrr,       X86::PMULHUWrm },
      { X86::PMULHWrr,        X86::PMULHWrm },
      { X86::PMULLWrr,        X86::PMULLWrm },
      { X86::PMULUDQrr,       X86::PMULUDQrm },
      { X86::PORrr,           X86::PORrm },
      { X86::PSADBWrr,        X86::PSADBWrm },
      { X86::PSLLDrr,         X86::PSLLDrm },
      { X86::PSLLQrr,         X86::PSLLQrm },
      { X86::PSLLWrr,         X86::PSLLWrm },
      { X86::PSRADrr,         X86::PSRADrm },
      { X86::PSRAWrr,         X86::PSRAWrm },
      { X86::PSRLDrr,         X86::PSRLDrm },
      { X86::PSRLQrr,         X86::PSRLQrm },
      { X86::PSRLWrr,         X86::PSRLWrm },
      { X86::PSUBBrr,         X86::PSUBBrm },
      { X86::PSUBDrr,         X86::PSUBDrm },
      { X86::PSUBSBrr,        X86::PSUBSBrm },
      { X86::PSUBSWrr,        X86::PSUBSWrm },
      { X86::PSUBWrr,         X86::PSUBWrm },
      { X86::PUNPCKHBWrr,     X86::PUNPCKHBWrm },
      { X86::PUNPCKHDQrr,     X86::PUNPCKHDQrm },
      { X86::PUNPCKHQDQrr,    X86::PUNPCKHQDQrm },
      { X86::PUNPCKHWDrr,     X86::PUNPCKHWDrm },
      { X86::PUNPCKLBWrr,     X86::PUNPCKLBWrm },
      { X86::PUNPCKLDQrr,     X86::PUNPCKLDQrm },
      { X86::PUNPCKLQDQrr,    X86::PUNPCKLQDQrm },
      { X86::PUNPCKLWDrr,     X86::PUNPCKLWDrm },
      { X86::PXORrr,          X86::PXORrm },
      { X86::RCPPSr,          X86::RCPPSm },
      { X86::RSQRTPSr,        X86::RSQRTPSm },
      { X86::SBB32rr,         X86::SBB32rm },
      { X86::SHUFPDrri,       X86::SHUFPDrmi },
      { X86::SHUFPSrri,       X86::SHUFPSrmi },
      { X86::SQRTPDr,         X86::SQRTPDm },
      { X86::SQRTPSr,         X86::SQRTPSm },
      { X86::SQRTSDr,         X86::SQRTSDm },
      { X86::SQRTSSr,         X86::SQRTSSm },
      { X86::SUB16rr,         X86::SUB16rm },
      { X86::SUB32rr,         X86::SUB32rm },
      { X86::SUB8rr,          X86::SUB8rm },
      { X86::SUBPDrr,         X86::SUBPDrm },
      { X86::SUBPSrr,         X86::SUBPSrm },
      { X86::SUBSDrr,         X86::SUBSDrm },
      { X86::SUBSSrr,         X86::SUBSSrm },
      { X86::UNPCKHPDrr,      X86::UNPCKHPDrm },
      { X86::UNPCKHPSrr,      X86::UNPCKHPSrm },
      { X86::UNPCKLPDrr,      X86::UNPCKLPDrm },
      { X86::UNPCKLPSrr,      X86::UNPCKLPSrm },
      { X86::XOR16rr,         X86::XOR16rm },
      { X86::XOR32rr,         X86::XOR32rm },
      { X86::XOR8rr,          X86::XOR8rm },
      { X86::XORPDrr,         X86::XORPDrm },
      { X86::XORPSrr,         X86::XORPSrm }
    };
    ASSERT_SORTED(OpcodeTable);
    OpcodeTablePtr = OpcodeTable;
    OpcodeTableSize = ARRAY_SIZE(OpcodeTable);
  }
  
  // If table selected...
  if (OpcodeTablePtr) {
    // Find the Opcode to fuse
    unsigned fromOpcode = MI->getOpcode();
    // Lookup fromOpcode in table
    if (const TableEntry *Entry = TableLookup(OpcodeTablePtr, OpcodeTableSize,
                                              fromOpcode)) {
      if (isTwoAddrFold)
        return FuseTwoAddrInst(Entry->to, FrameIndex, MI);
      
      return FuseInst(Entry->to, i, FrameIndex, MI);
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
        New=BuildMI(X86::SUB32ri, 2, X86::ESP).addReg(X86::ESP).addImm(Amount);
      } else {
        assert(Old->getOpcode() == X86::ADJCALLSTACKUP);
        // factor out the amount the callee already popped.
        unsigned CalleeAmt = Old->getOperand(1).getImmedValue();
        Amount -= CalleeAmt;
        if (Amount) {
          unsigned Opc = Amount < 128 ? X86::ADD32ri8 : X86::ADD32ri;
          New = BuildMI(Opc, 2, X86::ESP).addReg(X86::ESP).addImm(Amount);
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
        BuildMI(Opc, 1, X86::ESP).addReg(X86::ESP).addImm(CalleeAmt);
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
      MI = BuildMI(Opc, 2, X86::ESP).addReg(X86::ESP).addImm(NumBytes);
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
    MI = BuildMI(X86::AND32ri, 2, X86::ESP).addReg(X86::ESP).addImm(-Align);
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
    BuildMI(MBB, MBBI, X86::MOV32rr, 1, X86::ESP).addReg(X86::EBP);

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
        BuildMI(MBB, MBBI, Opc, 2, X86::ESP).addReg(X86::ESP).addImm(NumBytes);
      } else if ((int)NumBytes < 0) {
        unsigned Opc = -NumBytes < 128 ? X86::SUB32ri8 : X86::SUB32ri;
        BuildMI(MBB, MBBI, Opc, 2, X86::ESP).addReg(X86::ESP).addImm(-NumBytes);
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


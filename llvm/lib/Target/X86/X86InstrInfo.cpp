//===- X86InstrInfo.cpp - X86 Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86InstrInfo.h"
#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/MC/MCAsmInfo.h"
#include <limits>

#define GET_INSTRINFO_MC_DESC
#include "X86GenInstrInfo.inc"

using namespace llvm;

static cl::opt<bool>
NoFusing("disable-spill-fusing",
         cl::desc("Disable fusing of spill code into instructions"));
static cl::opt<bool>
PrintFailedFusing("print-failed-fuse-candidates",
                  cl::desc("Print instructions that the allocator wants to"
                           " fuse, but the X86 backend currently can't"),
                  cl::Hidden);
static cl::opt<bool>
ReMatPICStubLoad("remat-pic-stub-load",
                 cl::desc("Re-materialize load from stub in PIC mode"),
                 cl::init(false), cl::Hidden);

X86InstrInfo::X86InstrInfo(X86TargetMachine &tm)
  : TargetInstrInfoImpl(X86Insts, array_lengthof(X86Insts)),
    TM(tm), RI(tm, *this) {
  enum {
    TB_NOT_REVERSABLE = 1U << 31,
    TB_FLAGS = TB_NOT_REVERSABLE
  };

  static const unsigned OpTbl2Addr[][2] = {
    { X86::ADC32ri,     X86::ADC32mi },
    { X86::ADC32ri8,    X86::ADC32mi8 },
    { X86::ADC32rr,     X86::ADC32mr },
    { X86::ADC64ri32,   X86::ADC64mi32 },
    { X86::ADC64ri8,    X86::ADC64mi8 },
    { X86::ADC64rr,     X86::ADC64mr },
    { X86::ADD16ri,     X86::ADD16mi },
    { X86::ADD16ri8,    X86::ADD16mi8 },
    { X86::ADD16ri_DB,  X86::ADD16mi  | TB_NOT_REVERSABLE },
    { X86::ADD16ri8_DB, X86::ADD16mi8 | TB_NOT_REVERSABLE },
    { X86::ADD16rr,     X86::ADD16mr },
    { X86::ADD16rr_DB,  X86::ADD16mr | TB_NOT_REVERSABLE },
    { X86::ADD32ri,     X86::ADD32mi },
    { X86::ADD32ri8,    X86::ADD32mi8 },
    { X86::ADD32ri_DB,  X86::ADD32mi | TB_NOT_REVERSABLE },
    { X86::ADD32ri8_DB, X86::ADD32mi8 | TB_NOT_REVERSABLE },
    { X86::ADD32rr,     X86::ADD32mr },
    { X86::ADD32rr_DB,  X86::ADD32mr | TB_NOT_REVERSABLE },
    { X86::ADD64ri32,   X86::ADD64mi32 },
    { X86::ADD64ri8,    X86::ADD64mi8 },
    { X86::ADD64ri32_DB,X86::ADD64mi32 | TB_NOT_REVERSABLE },
    { X86::ADD64ri8_DB, X86::ADD64mi8 | TB_NOT_REVERSABLE },
    { X86::ADD64rr,     X86::ADD64mr },
    { X86::ADD64rr_DB,  X86::ADD64mr | TB_NOT_REVERSABLE },
    { X86::ADD8ri,      X86::ADD8mi },
    { X86::ADD8rr,      X86::ADD8mr },
    { X86::AND16ri,     X86::AND16mi },
    { X86::AND16ri8,    X86::AND16mi8 },
    { X86::AND16rr,     X86::AND16mr },
    { X86::AND32ri,     X86::AND32mi },
    { X86::AND32ri8,    X86::AND32mi8 },
    { X86::AND32rr,     X86::AND32mr },
    { X86::AND64ri32,   X86::AND64mi32 },
    { X86::AND64ri8,    X86::AND64mi8 },
    { X86::AND64rr,     X86::AND64mr },
    { X86::AND8ri,      X86::AND8mi },
    { X86::AND8rr,      X86::AND8mr },
    { X86::DEC16r,      X86::DEC16m },
    { X86::DEC32r,      X86::DEC32m },
    { X86::DEC64_16r,   X86::DEC64_16m },
    { X86::DEC64_32r,   X86::DEC64_32m },
    { X86::DEC64r,      X86::DEC64m },
    { X86::DEC8r,       X86::DEC8m },
    { X86::INC16r,      X86::INC16m },
    { X86::INC32r,      X86::INC32m },
    { X86::INC64_16r,   X86::INC64_16m },
    { X86::INC64_32r,   X86::INC64_32m },
    { X86::INC64r,      X86::INC64m },
    { X86::INC8r,       X86::INC8m },
    { X86::NEG16r,      X86::NEG16m },
    { X86::NEG32r,      X86::NEG32m },
    { X86::NEG64r,      X86::NEG64m },
    { X86::NEG8r,       X86::NEG8m },
    { X86::NOT16r,      X86::NOT16m },
    { X86::NOT32r,      X86::NOT32m },
    { X86::NOT64r,      X86::NOT64m },
    { X86::NOT8r,       X86::NOT8m },
    { X86::OR16ri,      X86::OR16mi },
    { X86::OR16ri8,     X86::OR16mi8 },
    { X86::OR16rr,      X86::OR16mr },
    { X86::OR32ri,      X86::OR32mi },
    { X86::OR32ri8,     X86::OR32mi8 },
    { X86::OR32rr,      X86::OR32mr },
    { X86::OR64ri32,    X86::OR64mi32 },
    { X86::OR64ri8,     X86::OR64mi8 },
    { X86::OR64rr,      X86::OR64mr },
    { X86::OR8ri,       X86::OR8mi },
    { X86::OR8rr,       X86::OR8mr },
    { X86::ROL16r1,     X86::ROL16m1 },
    { X86::ROL16rCL,    X86::ROL16mCL },
    { X86::ROL16ri,     X86::ROL16mi },
    { X86::ROL32r1,     X86::ROL32m1 },
    { X86::ROL32rCL,    X86::ROL32mCL },
    { X86::ROL32ri,     X86::ROL32mi },
    { X86::ROL64r1,     X86::ROL64m1 },
    { X86::ROL64rCL,    X86::ROL64mCL },
    { X86::ROL64ri,     X86::ROL64mi },
    { X86::ROL8r1,      X86::ROL8m1 },
    { X86::ROL8rCL,     X86::ROL8mCL },
    { X86::ROL8ri,      X86::ROL8mi },
    { X86::ROR16r1,     X86::ROR16m1 },
    { X86::ROR16rCL,    X86::ROR16mCL },
    { X86::ROR16ri,     X86::ROR16mi },
    { X86::ROR32r1,     X86::ROR32m1 },
    { X86::ROR32rCL,    X86::ROR32mCL },
    { X86::ROR32ri,     X86::ROR32mi },
    { X86::ROR64r1,     X86::ROR64m1 },
    { X86::ROR64rCL,    X86::ROR64mCL },
    { X86::ROR64ri,     X86::ROR64mi },
    { X86::ROR8r1,      X86::ROR8m1 },
    { X86::ROR8rCL,     X86::ROR8mCL },
    { X86::ROR8ri,      X86::ROR8mi },
    { X86::SAR16r1,     X86::SAR16m1 },
    { X86::SAR16rCL,    X86::SAR16mCL },
    { X86::SAR16ri,     X86::SAR16mi },
    { X86::SAR32r1,     X86::SAR32m1 },
    { X86::SAR32rCL,    X86::SAR32mCL },
    { X86::SAR32ri,     X86::SAR32mi },
    { X86::SAR64r1,     X86::SAR64m1 },
    { X86::SAR64rCL,    X86::SAR64mCL },
    { X86::SAR64ri,     X86::SAR64mi },
    { X86::SAR8r1,      X86::SAR8m1 },
    { X86::SAR8rCL,     X86::SAR8mCL },
    { X86::SAR8ri,      X86::SAR8mi },
    { X86::SBB32ri,     X86::SBB32mi },
    { X86::SBB32ri8,    X86::SBB32mi8 },
    { X86::SBB32rr,     X86::SBB32mr },
    { X86::SBB64ri32,   X86::SBB64mi32 },
    { X86::SBB64ri8,    X86::SBB64mi8 },
    { X86::SBB64rr,     X86::SBB64mr },
    { X86::SHL16rCL,    X86::SHL16mCL },
    { X86::SHL16ri,     X86::SHL16mi },
    { X86::SHL32rCL,    X86::SHL32mCL },
    { X86::SHL32ri,     X86::SHL32mi },
    { X86::SHL64rCL,    X86::SHL64mCL },
    { X86::SHL64ri,     X86::SHL64mi },
    { X86::SHL8rCL,     X86::SHL8mCL },
    { X86::SHL8ri,      X86::SHL8mi },
    { X86::SHLD16rrCL,  X86::SHLD16mrCL },
    { X86::SHLD16rri8,  X86::SHLD16mri8 },
    { X86::SHLD32rrCL,  X86::SHLD32mrCL },
    { X86::SHLD32rri8,  X86::SHLD32mri8 },
    { X86::SHLD64rrCL,  X86::SHLD64mrCL },
    { X86::SHLD64rri8,  X86::SHLD64mri8 },
    { X86::SHR16r1,     X86::SHR16m1 },
    { X86::SHR16rCL,    X86::SHR16mCL },
    { X86::SHR16ri,     X86::SHR16mi },
    { X86::SHR32r1,     X86::SHR32m1 },
    { X86::SHR32rCL,    X86::SHR32mCL },
    { X86::SHR32ri,     X86::SHR32mi },
    { X86::SHR64r1,     X86::SHR64m1 },
    { X86::SHR64rCL,    X86::SHR64mCL },
    { X86::SHR64ri,     X86::SHR64mi },
    { X86::SHR8r1,      X86::SHR8m1 },
    { X86::SHR8rCL,     X86::SHR8mCL },
    { X86::SHR8ri,      X86::SHR8mi },
    { X86::SHRD16rrCL,  X86::SHRD16mrCL },
    { X86::SHRD16rri8,  X86::SHRD16mri8 },
    { X86::SHRD32rrCL,  X86::SHRD32mrCL },
    { X86::SHRD32rri8,  X86::SHRD32mri8 },
    { X86::SHRD64rrCL,  X86::SHRD64mrCL },
    { X86::SHRD64rri8,  X86::SHRD64mri8 },
    { X86::SUB16ri,     X86::SUB16mi },
    { X86::SUB16ri8,    X86::SUB16mi8 },
    { X86::SUB16rr,     X86::SUB16mr },
    { X86::SUB32ri,     X86::SUB32mi },
    { X86::SUB32ri8,    X86::SUB32mi8 },
    { X86::SUB32rr,     X86::SUB32mr },
    { X86::SUB64ri32,   X86::SUB64mi32 },
    { X86::SUB64ri8,    X86::SUB64mi8 },
    { X86::SUB64rr,     X86::SUB64mr },
    { X86::SUB8ri,      X86::SUB8mi },
    { X86::SUB8rr,      X86::SUB8mr },
    { X86::XOR16ri,     X86::XOR16mi },
    { X86::XOR16ri8,    X86::XOR16mi8 },
    { X86::XOR16rr,     X86::XOR16mr },
    { X86::XOR32ri,     X86::XOR32mi },
    { X86::XOR32ri8,    X86::XOR32mi8 },
    { X86::XOR32rr,     X86::XOR32mr },
    { X86::XOR64ri32,   X86::XOR64mi32 },
    { X86::XOR64ri8,    X86::XOR64mi8 },
    { X86::XOR64rr,     X86::XOR64mr },
    { X86::XOR8ri,      X86::XOR8mi },
    { X86::XOR8rr,      X86::XOR8mr }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl2Addr); i != e; ++i) {
    unsigned RegOp = OpTbl2Addr[i][0];
    unsigned MemOp = OpTbl2Addr[i][1] & ~TB_FLAGS;
    assert(!RegOp2MemOpTable2Addr.count(RegOp) && "Duplicated entries?");
    RegOp2MemOpTable2Addr[RegOp] = std::make_pair(MemOp, 0U);

    // If this is not a reversible operation (because there is a many->one)
    // mapping, don't insert the reverse of the operation into MemOp2RegOpTable.
    if (OpTbl2Addr[i][1] & TB_NOT_REVERSABLE)
      continue;

    // Index 0, folded load and store, no alignment requirement.
    unsigned AuxInfo = 0 | (1 << 4) | (1 << 5);

    assert(!MemOp2RegOpTable.count(MemOp) &&
            "Duplicated entries in unfolding maps?");
    MemOp2RegOpTable[MemOp] = std::make_pair(RegOp, AuxInfo);
  }

  // If the third value is 1, then it's folding either a load or a store.
  static const unsigned OpTbl0[][4] = {
    { X86::BT16ri8,     X86::BT16mi8, 1, 0 },
    { X86::BT32ri8,     X86::BT32mi8, 1, 0 },
    { X86::BT64ri8,     X86::BT64mi8, 1, 0 },
    { X86::CALL32r,     X86::CALL32m, 1, 0 },
    { X86::CALL64r,     X86::CALL64m, 1, 0 },
    { X86::WINCALL64r,  X86::WINCALL64m, 1, 0 },
    { X86::CMP16ri,     X86::CMP16mi, 1, 0 },
    { X86::CMP16ri8,    X86::CMP16mi8, 1, 0 },
    { X86::CMP16rr,     X86::CMP16mr, 1, 0 },
    { X86::CMP32ri,     X86::CMP32mi, 1, 0 },
    { X86::CMP32ri8,    X86::CMP32mi8, 1, 0 },
    { X86::CMP32rr,     X86::CMP32mr, 1, 0 },
    { X86::CMP64ri32,   X86::CMP64mi32, 1, 0 },
    { X86::CMP64ri8,    X86::CMP64mi8, 1, 0 },
    { X86::CMP64rr,     X86::CMP64mr, 1, 0 },
    { X86::CMP8ri,      X86::CMP8mi, 1, 0 },
    { X86::CMP8rr,      X86::CMP8mr, 1, 0 },
    { X86::DIV16r,      X86::DIV16m, 1, 0 },
    { X86::DIV32r,      X86::DIV32m, 1, 0 },
    { X86::DIV64r,      X86::DIV64m, 1, 0 },
    { X86::DIV8r,       X86::DIV8m, 1, 0 },
    { X86::EXTRACTPSrr, X86::EXTRACTPSmr, 0, 16 },
    { X86::FsMOVAPDrr,  X86::MOVSDmr | TB_NOT_REVERSABLE , 0, 0 },
    { X86::FsMOVAPSrr,  X86::MOVSSmr | TB_NOT_REVERSABLE , 0, 0 },
    { X86::IDIV16r,     X86::IDIV16m, 1, 0 },
    { X86::IDIV32r,     X86::IDIV32m, 1, 0 },
    { X86::IDIV64r,     X86::IDIV64m, 1, 0 },
    { X86::IDIV8r,      X86::IDIV8m, 1, 0 },
    { X86::IMUL16r,     X86::IMUL16m, 1, 0 },
    { X86::IMUL32r,     X86::IMUL32m, 1, 0 },
    { X86::IMUL64r,     X86::IMUL64m, 1, 0 },
    { X86::IMUL8r,      X86::IMUL8m, 1, 0 },
    { X86::JMP32r,      X86::JMP32m, 1, 0 },
    { X86::JMP64r,      X86::JMP64m, 1, 0 },
    { X86::MOV16ri,     X86::MOV16mi, 0, 0 },
    { X86::MOV16rr,     X86::MOV16mr, 0, 0 },
    { X86::MOV32ri,     X86::MOV32mi, 0, 0 },
    { X86::MOV32rr,     X86::MOV32mr, 0, 0 },
    { X86::MOV64ri32,   X86::MOV64mi32, 0, 0 },
    { X86::MOV64rr,     X86::MOV64mr, 0, 0 },
    { X86::MOV8ri,      X86::MOV8mi, 0, 0 },
    { X86::MOV8rr,      X86::MOV8mr, 0, 0 },
    { X86::MOV8rr_NOREX, X86::MOV8mr_NOREX, 0, 0 },
    { X86::MOVAPDrr,    X86::MOVAPDmr, 0, 16 },
    { X86::MOVAPSrr,    X86::MOVAPSmr, 0, 16 },
    { X86::MOVDQArr,    X86::MOVDQAmr, 0, 16 },
    { X86::MOVPDI2DIrr, X86::MOVPDI2DImr, 0, 0 },
    { X86::MOVPQIto64rr,X86::MOVPQI2QImr, 0, 0 },
    { X86::MOVSDto64rr, X86::MOVSDto64mr, 0, 0 },
    { X86::MOVSS2DIrr,  X86::MOVSS2DImr, 0, 0 },
    { X86::MOVUPDrr,    X86::MOVUPDmr, 0, 0 },
    { X86::MOVUPSrr,    X86::MOVUPSmr, 0, 0 },
    { X86::MUL16r,      X86::MUL16m, 1, 0 },
    { X86::MUL32r,      X86::MUL32m, 1, 0 },
    { X86::MUL64r,      X86::MUL64m, 1, 0 },
    { X86::MUL8r,       X86::MUL8m, 1, 0 },
    { X86::SETAEr,      X86::SETAEm, 0, 0 },
    { X86::SETAr,       X86::SETAm, 0, 0 },
    { X86::SETBEr,      X86::SETBEm, 0, 0 },
    { X86::SETBr,       X86::SETBm, 0, 0 },
    { X86::SETEr,       X86::SETEm, 0, 0 },
    { X86::SETGEr,      X86::SETGEm, 0, 0 },
    { X86::SETGr,       X86::SETGm, 0, 0 },
    { X86::SETLEr,      X86::SETLEm, 0, 0 },
    { X86::SETLr,       X86::SETLm, 0, 0 },
    { X86::SETNEr,      X86::SETNEm, 0, 0 },
    { X86::SETNOr,      X86::SETNOm, 0, 0 },
    { X86::SETNPr,      X86::SETNPm, 0, 0 },
    { X86::SETNSr,      X86::SETNSm, 0, 0 },
    { X86::SETOr,       X86::SETOm, 0, 0 },
    { X86::SETPr,       X86::SETPm, 0, 0 },
    { X86::SETSr,       X86::SETSm, 0, 0 },
    { X86::TAILJMPr,    X86::TAILJMPm, 1, 0 },
    { X86::TAILJMPr64,  X86::TAILJMPm64, 1, 0 },
    { X86::TEST16ri,    X86::TEST16mi, 1, 0 },
    { X86::TEST32ri,    X86::TEST32mi, 1, 0 },
    { X86::TEST64ri32,  X86::TEST64mi32, 1, 0 },
    { X86::TEST8ri,     X86::TEST8mi, 1, 0 }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl0); i != e; ++i) {
    unsigned RegOp      = OpTbl0[i][0];
    unsigned MemOp      = OpTbl0[i][1] & ~TB_FLAGS;
    unsigned FoldedLoad = OpTbl0[i][2];
    unsigned Align      = OpTbl0[i][3];
    assert(!RegOp2MemOpTable0.count(RegOp) && "Duplicated entries?");
    RegOp2MemOpTable0[RegOp] = std::make_pair(MemOp, Align);

    // If this is not a reversible operation (because there is a many->one)
    // mapping, don't insert the reverse of the operation into MemOp2RegOpTable.
    if (OpTbl0[i][1] & TB_NOT_REVERSABLE)
      continue;

    // Index 0, folded load or store.
    unsigned AuxInfo = 0 | (FoldedLoad << 4) | ((FoldedLoad^1) << 5);
    assert(!MemOp2RegOpTable.count(MemOp) && "Duplicated entries?");
    MemOp2RegOpTable[MemOp] = std::make_pair(RegOp, AuxInfo);
  }

  static const unsigned OpTbl1[][3] = {
    { X86::CMP16rr,         X86::CMP16rm, 0 },
    { X86::CMP32rr,         X86::CMP32rm, 0 },
    { X86::CMP64rr,         X86::CMP64rm, 0 },
    { X86::CMP8rr,          X86::CMP8rm, 0 },
    { X86::CVTSD2SSrr,      X86::CVTSD2SSrm, 0 },
    { X86::CVTSI2SD64rr,    X86::CVTSI2SD64rm, 0 },
    { X86::CVTSI2SDrr,      X86::CVTSI2SDrm, 0 },
    { X86::CVTSI2SS64rr,    X86::CVTSI2SS64rm, 0 },
    { X86::CVTSI2SSrr,      X86::CVTSI2SSrm, 0 },
    { X86::CVTSS2SDrr,      X86::CVTSS2SDrm, 0 },
    { X86::CVTTSD2SI64rr,   X86::CVTTSD2SI64rm, 0 },
    { X86::CVTTSD2SIrr,     X86::CVTTSD2SIrm, 0 },
    { X86::CVTTSS2SI64rr,   X86::CVTTSS2SI64rm, 0 },
    { X86::CVTTSS2SIrr,     X86::CVTTSS2SIrm, 0 },
    { X86::FsMOVAPDrr,      X86::MOVSDrm | TB_NOT_REVERSABLE , 0 },
    { X86::FsMOVAPSrr,      X86::MOVSSrm | TB_NOT_REVERSABLE , 0 },
    { X86::IMUL16rri,       X86::IMUL16rmi, 0 },
    { X86::IMUL16rri8,      X86::IMUL16rmi8, 0 },
    { X86::IMUL32rri,       X86::IMUL32rmi, 0 },
    { X86::IMUL32rri8,      X86::IMUL32rmi8, 0 },
    { X86::IMUL64rri32,     X86::IMUL64rmi32, 0 },
    { X86::IMUL64rri8,      X86::IMUL64rmi8, 0 },
    { X86::Int_COMISDrr,    X86::Int_COMISDrm, 0 },
    { X86::Int_COMISSrr,    X86::Int_COMISSrm, 0 },
    { X86::Int_CVTDQ2PDrr,  X86::Int_CVTDQ2PDrm, 16 },
    { X86::Int_CVTDQ2PSrr,  X86::Int_CVTDQ2PSrm, 16 },
    { X86::Int_CVTPD2DQrr,  X86::Int_CVTPD2DQrm, 16 },
    { X86::Int_CVTPD2PSrr,  X86::Int_CVTPD2PSrm, 16 },
    { X86::Int_CVTPS2DQrr,  X86::Int_CVTPS2DQrm, 16 },
    { X86::Int_CVTPS2PDrr,  X86::Int_CVTPS2PDrm, 0 },
    { X86::CVTSD2SI64rr,    X86::CVTSD2SI64rm, 0 },
    { X86::CVTSD2SIrr,      X86::CVTSD2SIrm, 0 },
    { X86::Int_CVTSD2SSrr,  X86::Int_CVTSD2SSrm, 0 },
    { X86::Int_CVTSI2SD64rr,X86::Int_CVTSI2SD64rm, 0 },
    { X86::Int_CVTSI2SDrr,  X86::Int_CVTSI2SDrm, 0 },
    { X86::Int_CVTSI2SS64rr,X86::Int_CVTSI2SS64rm, 0 },
    { X86::Int_CVTSI2SSrr,  X86::Int_CVTSI2SSrm, 0 },
    { X86::Int_CVTSS2SDrr,  X86::Int_CVTSS2SDrm, 0 },
    { X86::Int_CVTSS2SI64rr,X86::Int_CVTSS2SI64rm, 0 },
    { X86::Int_CVTSS2SIrr,  X86::Int_CVTSS2SIrm, 0 },
    { X86::CVTTPD2DQrr,     X86::CVTTPD2DQrm, 16 },
    { X86::CVTTPS2DQrr,     X86::CVTTPS2DQrm, 16 },
    { X86::Int_CVTTSD2SI64rr,X86::Int_CVTTSD2SI64rm, 0 },
    { X86::Int_CVTTSD2SIrr, X86::Int_CVTTSD2SIrm, 0 },
    { X86::Int_CVTTSS2SI64rr,X86::Int_CVTTSS2SI64rm, 0 },
    { X86::Int_CVTTSS2SIrr, X86::Int_CVTTSS2SIrm, 0 },
    { X86::Int_UCOMISDrr,   X86::Int_UCOMISDrm, 0 },
    { X86::Int_UCOMISSrr,   X86::Int_UCOMISSrm, 0 },
    { X86::MOV16rr,         X86::MOV16rm, 0 },
    { X86::MOV32rr,         X86::MOV32rm, 0 },
    { X86::MOV64rr,         X86::MOV64rm, 0 },
    { X86::MOV64toPQIrr,    X86::MOVQI2PQIrm, 0 },
    { X86::MOV64toSDrr,     X86::MOV64toSDrm, 0 },
    { X86::MOV8rr,          X86::MOV8rm, 0 },
    { X86::MOVAPDrr,        X86::MOVAPDrm, 16 },
    { X86::MOVAPSrr,        X86::MOVAPSrm, 16 },
    { X86::MOVDDUPrr,       X86::MOVDDUPrm, 0 },
    { X86::MOVDI2PDIrr,     X86::MOVDI2PDIrm, 0 },
    { X86::MOVDI2SSrr,      X86::MOVDI2SSrm, 0 },
    { X86::MOVDQArr,        X86::MOVDQArm, 16 },
    { X86::MOVSHDUPrr,      X86::MOVSHDUPrm, 16 },
    { X86::MOVSLDUPrr,      X86::MOVSLDUPrm, 16 },
    { X86::MOVSX16rr8,      X86::MOVSX16rm8, 0 },
    { X86::MOVSX32rr16,     X86::MOVSX32rm16, 0 },
    { X86::MOVSX32rr8,      X86::MOVSX32rm8, 0 },
    { X86::MOVSX64rr16,     X86::MOVSX64rm16, 0 },
    { X86::MOVSX64rr32,     X86::MOVSX64rm32, 0 },
    { X86::MOVSX64rr8,      X86::MOVSX64rm8, 0 },
    { X86::MOVUPDrr,        X86::MOVUPDrm, 16 },
    { X86::MOVUPSrr,        X86::MOVUPSrm, 0 },
    { X86::MOVZDI2PDIrr,    X86::MOVZDI2PDIrm, 0 },
    { X86::MOVZQI2PQIrr,    X86::MOVZQI2PQIrm, 0 },
    { X86::MOVZPQILo2PQIrr, X86::MOVZPQILo2PQIrm, 16 },
    { X86::MOVZX16rr8,      X86::MOVZX16rm8, 0 },
    { X86::MOVZX32rr16,     X86::MOVZX32rm16, 0 },
    { X86::MOVZX32_NOREXrr8, X86::MOVZX32_NOREXrm8, 0 },
    { X86::MOVZX32rr8,      X86::MOVZX32rm8, 0 },
    { X86::MOVZX64rr16,     X86::MOVZX64rm16, 0 },
    { X86::MOVZX64rr32,     X86::MOVZX64rm32, 0 },
    { X86::MOVZX64rr8,      X86::MOVZX64rm8, 0 },
    { X86::PSHUFDri,        X86::PSHUFDmi, 16 },
    { X86::PSHUFHWri,       X86::PSHUFHWmi, 16 },
    { X86::PSHUFLWri,       X86::PSHUFLWmi, 16 },
    { X86::RCPPSr,          X86::RCPPSm, 16 },
    { X86::RCPPSr_Int,      X86::RCPPSm_Int, 16 },
    { X86::RSQRTPSr,        X86::RSQRTPSm, 16 },
    { X86::RSQRTPSr_Int,    X86::RSQRTPSm_Int, 16 },
    { X86::RSQRTSSr,        X86::RSQRTSSm, 0 },
    { X86::RSQRTSSr_Int,    X86::RSQRTSSm_Int, 0 },
    { X86::SQRTPDr,         X86::SQRTPDm, 16 },
    { X86::SQRTPDr_Int,     X86::SQRTPDm_Int, 16 },
    { X86::SQRTPSr,         X86::SQRTPSm, 16 },
    { X86::SQRTPSr_Int,     X86::SQRTPSm_Int, 16 },
    { X86::SQRTSDr,         X86::SQRTSDm, 0 },
    { X86::SQRTSDr_Int,     X86::SQRTSDm_Int, 0 },
    { X86::SQRTSSr,         X86::SQRTSSm, 0 },
    { X86::SQRTSSr_Int,     X86::SQRTSSm_Int, 0 },
    { X86::TEST16rr,        X86::TEST16rm, 0 },
    { X86::TEST32rr,        X86::TEST32rm, 0 },
    { X86::TEST64rr,        X86::TEST64rm, 0 },
    { X86::TEST8rr,         X86::TEST8rm, 0 },
    // FIXME: TEST*rr EAX,EAX ---> CMP [mem], 0
    { X86::UCOMISDrr,       X86::UCOMISDrm, 0 },
    { X86::UCOMISSrr,       X86::UCOMISSrm, 0 }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl1); i != e; ++i) {
    unsigned RegOp = OpTbl1[i][0];
    unsigned MemOp = OpTbl1[i][1] & ~TB_FLAGS;
    unsigned Align = OpTbl1[i][2];
    assert(!RegOp2MemOpTable1.count(RegOp) && "Duplicate entries");
    RegOp2MemOpTable1[RegOp] = std::make_pair(MemOp, Align);

    // If this is not a reversible operation (because there is a many->one)
    // mapping, don't insert the reverse of the operation into MemOp2RegOpTable.
    if (OpTbl1[i][1] & TB_NOT_REVERSABLE)
      continue;

    // Index 1, folded load
    unsigned AuxInfo = 1 | (1 << 4);
    assert(!MemOp2RegOpTable.count(MemOp) && "Duplicate entries");
    MemOp2RegOpTable[MemOp] = std::make_pair(RegOp, AuxInfo);
  }

  static const unsigned OpTbl2[][3] = {
    { X86::ADC32rr,         X86::ADC32rm, 0 },
    { X86::ADC64rr,         X86::ADC64rm, 0 },
    { X86::ADD16rr,         X86::ADD16rm, 0 },
    { X86::ADD16rr_DB,      X86::ADD16rm | TB_NOT_REVERSABLE, 0 },
    { X86::ADD32rr,         X86::ADD32rm, 0 },
    { X86::ADD32rr_DB,      X86::ADD32rm | TB_NOT_REVERSABLE, 0 },
    { X86::ADD64rr,         X86::ADD64rm, 0 },
    { X86::ADD64rr_DB,      X86::ADD64rm | TB_NOT_REVERSABLE, 0 },
    { X86::ADD8rr,          X86::ADD8rm, 0 },
    { X86::ADDPDrr,         X86::ADDPDrm, 16 },
    { X86::ADDPSrr,         X86::ADDPSrm, 16 },
    { X86::ADDSDrr,         X86::ADDSDrm, 0 },
    { X86::ADDSSrr,         X86::ADDSSrm, 0 },
    { X86::ADDSUBPDrr,      X86::ADDSUBPDrm, 16 },
    { X86::ADDSUBPSrr,      X86::ADDSUBPSrm, 16 },
    { X86::AND16rr,         X86::AND16rm, 0 },
    { X86::AND32rr,         X86::AND32rm, 0 },
    { X86::AND64rr,         X86::AND64rm, 0 },
    { X86::AND8rr,          X86::AND8rm, 0 },
    { X86::ANDNPDrr,        X86::ANDNPDrm, 16 },
    { X86::ANDNPSrr,        X86::ANDNPSrm, 16 },
    { X86::ANDPDrr,         X86::ANDPDrm, 16 },
    { X86::ANDPSrr,         X86::ANDPSrm, 16 },
    { X86::CMOVA16rr,       X86::CMOVA16rm, 0 },
    { X86::CMOVA32rr,       X86::CMOVA32rm, 0 },
    { X86::CMOVA64rr,       X86::CMOVA64rm, 0 },
    { X86::CMOVAE16rr,      X86::CMOVAE16rm, 0 },
    { X86::CMOVAE32rr,      X86::CMOVAE32rm, 0 },
    { X86::CMOVAE64rr,      X86::CMOVAE64rm, 0 },
    { X86::CMOVB16rr,       X86::CMOVB16rm, 0 },
    { X86::CMOVB32rr,       X86::CMOVB32rm, 0 },
    { X86::CMOVB64rr,       X86::CMOVB64rm, 0 },
    { X86::CMOVBE16rr,      X86::CMOVBE16rm, 0 },
    { X86::CMOVBE32rr,      X86::CMOVBE32rm, 0 },
    { X86::CMOVBE64rr,      X86::CMOVBE64rm, 0 },
    { X86::CMOVE16rr,       X86::CMOVE16rm, 0 },
    { X86::CMOVE32rr,       X86::CMOVE32rm, 0 },
    { X86::CMOVE64rr,       X86::CMOVE64rm, 0 },
    { X86::CMOVG16rr,       X86::CMOVG16rm, 0 },
    { X86::CMOVG32rr,       X86::CMOVG32rm, 0 },
    { X86::CMOVG64rr,       X86::CMOVG64rm, 0 },
    { X86::CMOVGE16rr,      X86::CMOVGE16rm, 0 },
    { X86::CMOVGE32rr,      X86::CMOVGE32rm, 0 },
    { X86::CMOVGE64rr,      X86::CMOVGE64rm, 0 },
    { X86::CMOVL16rr,       X86::CMOVL16rm, 0 },
    { X86::CMOVL32rr,       X86::CMOVL32rm, 0 },
    { X86::CMOVL64rr,       X86::CMOVL64rm, 0 },
    { X86::CMOVLE16rr,      X86::CMOVLE16rm, 0 },
    { X86::CMOVLE32rr,      X86::CMOVLE32rm, 0 },
    { X86::CMOVLE64rr,      X86::CMOVLE64rm, 0 },
    { X86::CMOVNE16rr,      X86::CMOVNE16rm, 0 },
    { X86::CMOVNE32rr,      X86::CMOVNE32rm, 0 },
    { X86::CMOVNE64rr,      X86::CMOVNE64rm, 0 },
    { X86::CMOVNO16rr,      X86::CMOVNO16rm, 0 },
    { X86::CMOVNO32rr,      X86::CMOVNO32rm, 0 },
    { X86::CMOVNO64rr,      X86::CMOVNO64rm, 0 },
    { X86::CMOVNP16rr,      X86::CMOVNP16rm, 0 },
    { X86::CMOVNP32rr,      X86::CMOVNP32rm, 0 },
    { X86::CMOVNP64rr,      X86::CMOVNP64rm, 0 },
    { X86::CMOVNS16rr,      X86::CMOVNS16rm, 0 },
    { X86::CMOVNS32rr,      X86::CMOVNS32rm, 0 },
    { X86::CMOVNS64rr,      X86::CMOVNS64rm, 0 },
    { X86::CMOVO16rr,       X86::CMOVO16rm, 0 },
    { X86::CMOVO32rr,       X86::CMOVO32rm, 0 },
    { X86::CMOVO64rr,       X86::CMOVO64rm, 0 },
    { X86::CMOVP16rr,       X86::CMOVP16rm, 0 },
    { X86::CMOVP32rr,       X86::CMOVP32rm, 0 },
    { X86::CMOVP64rr,       X86::CMOVP64rm, 0 },
    { X86::CMOVS16rr,       X86::CMOVS16rm, 0 },
    { X86::CMOVS32rr,       X86::CMOVS32rm, 0 },
    { X86::CMOVS64rr,       X86::CMOVS64rm, 0 },
    { X86::CMPPDrri,        X86::CMPPDrmi, 16 },
    { X86::CMPPSrri,        X86::CMPPSrmi, 16 },
    { X86::CMPSDrr,         X86::CMPSDrm, 0 },
    { X86::CMPSSrr,         X86::CMPSSrm, 0 },
    { X86::DIVPDrr,         X86::DIVPDrm, 16 },
    { X86::DIVPSrr,         X86::DIVPSrm, 16 },
    { X86::DIVSDrr,         X86::DIVSDrm, 0 },
    { X86::DIVSSrr,         X86::DIVSSrm, 0 },
    { X86::FsANDNPDrr,      X86::FsANDNPDrm, 16 },
    { X86::FsANDNPSrr,      X86::FsANDNPSrm, 16 },
    { X86::FsANDPDrr,       X86::FsANDPDrm, 16 },
    { X86::FsANDPSrr,       X86::FsANDPSrm, 16 },
    { X86::FsORPDrr,        X86::FsORPDrm, 16 },
    { X86::FsORPSrr,        X86::FsORPSrm, 16 },
    { X86::FsXORPDrr,       X86::FsXORPDrm, 16 },
    { X86::FsXORPSrr,       X86::FsXORPSrm, 16 },
    { X86::HADDPDrr,        X86::HADDPDrm, 16 },
    { X86::HADDPSrr,        X86::HADDPSrm, 16 },
    { X86::HSUBPDrr,        X86::HSUBPDrm, 16 },
    { X86::HSUBPSrr,        X86::HSUBPSrm, 16 },
    { X86::IMUL16rr,        X86::IMUL16rm, 0 },
    { X86::IMUL32rr,        X86::IMUL32rm, 0 },
    { X86::IMUL64rr,        X86::IMUL64rm, 0 },
    { X86::Int_CMPSDrr,     X86::Int_CMPSDrm, 0 },
    { X86::Int_CMPSSrr,     X86::Int_CMPSSrm, 0 },
    { X86::MAXPDrr,         X86::MAXPDrm, 16 },
    { X86::MAXPDrr_Int,     X86::MAXPDrm_Int, 16 },
    { X86::MAXPSrr,         X86::MAXPSrm, 16 },
    { X86::MAXPSrr_Int,     X86::MAXPSrm_Int, 16 },
    { X86::MAXSDrr,         X86::MAXSDrm, 0 },
    { X86::MAXSDrr_Int,     X86::MAXSDrm_Int, 0 },
    { X86::MAXSSrr,         X86::MAXSSrm, 0 },
    { X86::MAXSSrr_Int,     X86::MAXSSrm_Int, 0 },
    { X86::MINPDrr,         X86::MINPDrm, 16 },
    { X86::MINPDrr_Int,     X86::MINPDrm_Int, 16 },
    { X86::MINPSrr,         X86::MINPSrm, 16 },
    { X86::MINPSrr_Int,     X86::MINPSrm_Int, 16 },
    { X86::MINSDrr,         X86::MINSDrm, 0 },
    { X86::MINSDrr_Int,     X86::MINSDrm_Int, 0 },
    { X86::MINSSrr,         X86::MINSSrm, 0 },
    { X86::MINSSrr_Int,     X86::MINSSrm_Int, 0 },
    { X86::MULPDrr,         X86::MULPDrm, 16 },
    { X86::MULPSrr,         X86::MULPSrm, 16 },
    { X86::MULSDrr,         X86::MULSDrm, 0 },
    { X86::MULSSrr,         X86::MULSSrm, 0 },
    { X86::OR16rr,          X86::OR16rm, 0 },
    { X86::OR32rr,          X86::OR32rm, 0 },
    { X86::OR64rr,          X86::OR64rm, 0 },
    { X86::OR8rr,           X86::OR8rm, 0 },
    { X86::ORPDrr,          X86::ORPDrm, 16 },
    { X86::ORPSrr,          X86::ORPSrm, 16 },
    { X86::PACKSSDWrr,      X86::PACKSSDWrm, 16 },
    { X86::PACKSSWBrr,      X86::PACKSSWBrm, 16 },
    { X86::PACKUSWBrr,      X86::PACKUSWBrm, 16 },
    { X86::PADDBrr,         X86::PADDBrm, 16 },
    { X86::PADDDrr,         X86::PADDDrm, 16 },
    { X86::PADDQrr,         X86::PADDQrm, 16 },
    { X86::PADDSBrr,        X86::PADDSBrm, 16 },
    { X86::PADDSWrr,        X86::PADDSWrm, 16 },
    { X86::PADDWrr,         X86::PADDWrm, 16 },
    { X86::PANDNrr,         X86::PANDNrm, 16 },
    { X86::PANDrr,          X86::PANDrm, 16 },
    { X86::PAVGBrr,         X86::PAVGBrm, 16 },
    { X86::PAVGWrr,         X86::PAVGWrm, 16 },
    { X86::PCMPEQBrr,       X86::PCMPEQBrm, 16 },
    { X86::PCMPEQDrr,       X86::PCMPEQDrm, 16 },
    { X86::PCMPEQWrr,       X86::PCMPEQWrm, 16 },
    { X86::PCMPGTBrr,       X86::PCMPGTBrm, 16 },
    { X86::PCMPGTDrr,       X86::PCMPGTDrm, 16 },
    { X86::PCMPGTWrr,       X86::PCMPGTWrm, 16 },
    { X86::PINSRWrri,       X86::PINSRWrmi, 16 },
    { X86::PMADDWDrr,       X86::PMADDWDrm, 16 },
    { X86::PMAXSWrr,        X86::PMAXSWrm, 16 },
    { X86::PMAXUBrr,        X86::PMAXUBrm, 16 },
    { X86::PMINSWrr,        X86::PMINSWrm, 16 },
    { X86::PMINUBrr,        X86::PMINUBrm, 16 },
    { X86::PMULDQrr,        X86::PMULDQrm, 16 },
    { X86::PMULHUWrr,       X86::PMULHUWrm, 16 },
    { X86::PMULHWrr,        X86::PMULHWrm, 16 },
    { X86::PMULLDrr,        X86::PMULLDrm, 16 },
    { X86::PMULLWrr,        X86::PMULLWrm, 16 },
    { X86::PMULUDQrr,       X86::PMULUDQrm, 16 },
    { X86::PORrr,           X86::PORrm, 16 },
    { X86::PSADBWrr,        X86::PSADBWrm, 16 },
    { X86::PSLLDrr,         X86::PSLLDrm, 16 },
    { X86::PSLLQrr,         X86::PSLLQrm, 16 },
    { X86::PSLLWrr,         X86::PSLLWrm, 16 },
    { X86::PSRADrr,         X86::PSRADrm, 16 },
    { X86::PSRAWrr,         X86::PSRAWrm, 16 },
    { X86::PSRLDrr,         X86::PSRLDrm, 16 },
    { X86::PSRLQrr,         X86::PSRLQrm, 16 },
    { X86::PSRLWrr,         X86::PSRLWrm, 16 },
    { X86::PSUBBrr,         X86::PSUBBrm, 16 },
    { X86::PSUBDrr,         X86::PSUBDrm, 16 },
    { X86::PSUBSBrr,        X86::PSUBSBrm, 16 },
    { X86::PSUBSWrr,        X86::PSUBSWrm, 16 },
    { X86::PSUBWrr,         X86::PSUBWrm, 16 },
    { X86::PUNPCKHBWrr,     X86::PUNPCKHBWrm, 16 },
    { X86::PUNPCKHDQrr,     X86::PUNPCKHDQrm, 16 },
    { X86::PUNPCKHQDQrr,    X86::PUNPCKHQDQrm, 16 },
    { X86::PUNPCKHWDrr,     X86::PUNPCKHWDrm, 16 },
    { X86::PUNPCKLBWrr,     X86::PUNPCKLBWrm, 16 },
    { X86::PUNPCKLDQrr,     X86::PUNPCKLDQrm, 16 },
    { X86::PUNPCKLQDQrr,    X86::PUNPCKLQDQrm, 16 },
    { X86::PUNPCKLWDrr,     X86::PUNPCKLWDrm, 16 },
    { X86::PXORrr,          X86::PXORrm, 16 },
    { X86::SBB32rr,         X86::SBB32rm, 0 },
    { X86::SBB64rr,         X86::SBB64rm, 0 },
    { X86::SHUFPDrri,       X86::SHUFPDrmi, 16 },
    { X86::SHUFPSrri,       X86::SHUFPSrmi, 16 },
    { X86::SUB16rr,         X86::SUB16rm, 0 },
    { X86::SUB32rr,         X86::SUB32rm, 0 },
    { X86::SUB64rr,         X86::SUB64rm, 0 },
    { X86::SUB8rr,          X86::SUB8rm, 0 },
    { X86::SUBPDrr,         X86::SUBPDrm, 16 },
    { X86::SUBPSrr,         X86::SUBPSrm, 16 },
    { X86::SUBSDrr,         X86::SUBSDrm, 0 },
    { X86::SUBSSrr,         X86::SUBSSrm, 0 },
    // FIXME: TEST*rr -> swapped operand of TEST*mr.
    { X86::UNPCKHPDrr,      X86::UNPCKHPDrm, 16 },
    { X86::UNPCKHPSrr,      X86::UNPCKHPSrm, 16 },
    { X86::UNPCKLPDrr,      X86::UNPCKLPDrm, 16 },
    { X86::UNPCKLPSrr,      X86::UNPCKLPSrm, 16 },
    { X86::XOR16rr,         X86::XOR16rm, 0 },
    { X86::XOR32rr,         X86::XOR32rm, 0 },
    { X86::XOR64rr,         X86::XOR64rm, 0 },
    { X86::XOR8rr,          X86::XOR8rm, 0 },
    { X86::XORPDrr,         X86::XORPDrm, 16 },
    { X86::XORPSrr,         X86::XORPSrm, 16 }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl2); i != e; ++i) {
    unsigned RegOp = OpTbl2[i][0];
    unsigned MemOp = OpTbl2[i][1] & ~TB_FLAGS;
    unsigned Align = OpTbl2[i][2];

    assert(!RegOp2MemOpTable2.count(RegOp) && "Duplicate entry!");
    RegOp2MemOpTable2[RegOp] = std::make_pair(MemOp, Align);

    // If this is not a reversible operation (because there is a many->one)
    // mapping, don't insert the reverse of the operation into MemOp2RegOpTable.
    if (OpTbl2[i][1] & TB_NOT_REVERSABLE)
      continue;

    // Index 2, folded load
    unsigned AuxInfo = 2 | (1 << 4);
    assert(!MemOp2RegOpTable.count(MemOp) &&
           "Duplicated entries in unfolding maps?");
    MemOp2RegOpTable[MemOp] = std::make_pair(RegOp, AuxInfo);
  }
}

bool
X86InstrInfo::isCoalescableExtInstr(const MachineInstr &MI,
                                    unsigned &SrcReg, unsigned &DstReg,
                                    unsigned &SubIdx) const {
  switch (MI.getOpcode()) {
  default: break;
  case X86::MOVSX16rr8:
  case X86::MOVZX16rr8:
  case X86::MOVSX32rr8:
  case X86::MOVZX32rr8:
  case X86::MOVSX64rr8:
  case X86::MOVZX64rr8:
    if (!TM.getSubtarget<X86Subtarget>().is64Bit())
      // It's not always legal to reference the low 8-bit of the larger
      // register in 32-bit mode.
      return false;
  case X86::MOVSX32rr16:
  case X86::MOVZX32rr16:
  case X86::MOVSX64rr16:
  case X86::MOVZX64rr16:
  case X86::MOVSX64rr32:
  case X86::MOVZX64rr32: {
    if (MI.getOperand(0).getSubReg() || MI.getOperand(1).getSubReg())
      // Be conservative.
      return false;
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable(0);
      break;
    case X86::MOVSX16rr8:
    case X86::MOVZX16rr8:
    case X86::MOVSX32rr8:
    case X86::MOVZX32rr8:
    case X86::MOVSX64rr8:
    case X86::MOVZX64rr8:
      SubIdx = X86::sub_8bit;
      break;
    case X86::MOVSX32rr16:
    case X86::MOVZX32rr16:
    case X86::MOVSX64rr16:
    case X86::MOVZX64rr16:
      SubIdx = X86::sub_16bit;
      break;
    case X86::MOVSX64rr32:
    case X86::MOVZX64rr32:
      SubIdx = X86::sub_32bit;
      break;
    }
    return true;
  }
  }
  return false;
}

/// isFrameOperand - Return true and the FrameIndex if the specified
/// operand and follow operands form a reference to the stack frame.
bool X86InstrInfo::isFrameOperand(const MachineInstr *MI, unsigned int Op,
                                  int &FrameIndex) const {
  if (MI->getOperand(Op).isFI() && MI->getOperand(Op+1).isImm() &&
      MI->getOperand(Op+2).isReg() && MI->getOperand(Op+3).isImm() &&
      MI->getOperand(Op+1).getImm() == 1 &&
      MI->getOperand(Op+2).getReg() == 0 &&
      MI->getOperand(Op+3).getImm() == 0) {
    FrameIndex = MI->getOperand(Op).getIndex();
    return true;
  }
  return false;
}

static bool isFrameLoadOpcode(int Opcode) {
  switch (Opcode) {
  default: break;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV32rm:
  case X86::MOV64rm:
  case X86::LD_Fp64m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MOVAPSrm:
  case X86::MOVAPDrm:
  case X86::MOVDQArm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
    return true;
    break;
  }
  return false;
}

static bool isFrameStoreOpcode(int Opcode) {
  switch (Opcode) {
  default: break;
  case X86::MOV8mr:
  case X86::MOV16mr:
  case X86::MOV32mr:
  case X86::MOV64mr:
  case X86::ST_FpP64m:
  case X86::MOVSSmr:
  case X86::MOVSDmr:
  case X86::MOVAPSmr:
  case X86::MOVAPDmr:
  case X86::MOVDQAmr:
  case X86::MMX_MOVD64mr:
  case X86::MMX_MOVQ64mr:
  case X86::MMX_MOVNTQmr:
    return true;
  }
  return false;
}

unsigned X86InstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                           int &FrameIndex) const {
  if (isFrameLoadOpcode(MI->getOpcode()))
    if (MI->getOperand(0).getSubReg() == 0 && isFrameOperand(MI, 1, FrameIndex))
      return MI->getOperand(0).getReg();
  return 0;
}

unsigned X86InstrInfo::isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                                 int &FrameIndex) const {
  if (isFrameLoadOpcode(MI->getOpcode())) {
    unsigned Reg;
    if ((Reg = isLoadFromStackSlot(MI, FrameIndex)))
      return Reg;
    // Check for post-frame index elimination operations
    const MachineMemOperand *Dummy;
    return hasLoadFromStackSlot(MI, Dummy, FrameIndex);
  }
  return 0;
}

bool X86InstrInfo::hasLoadFromStackSlot(const MachineInstr *MI,
                                        const MachineMemOperand *&MMO,
                                        int &FrameIndex) const {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end();
       o != oe;
       ++o) {
    if ((*o)->isLoad() && (*o)->getValue())
      if (const FixedStackPseudoSourceValue *Value =
          dyn_cast<const FixedStackPseudoSourceValue>((*o)->getValue())) {
        FrameIndex = Value->getFrameIndex();
        MMO = *o;
        return true;
      }
  }
  return false;
}

unsigned X86InstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                          int &FrameIndex) const {
  if (isFrameStoreOpcode(MI->getOpcode()))
    if (MI->getOperand(X86::AddrNumOperands).getSubReg() == 0 &&
        isFrameOperand(MI, 0, FrameIndex))
      return MI->getOperand(X86::AddrNumOperands).getReg();
  return 0;
}

unsigned X86InstrInfo::isStoreToStackSlotPostFE(const MachineInstr *MI,
                                                int &FrameIndex) const {
  if (isFrameStoreOpcode(MI->getOpcode())) {
    unsigned Reg;
    if ((Reg = isStoreToStackSlot(MI, FrameIndex)))
      return Reg;
    // Check for post-frame index elimination operations
    const MachineMemOperand *Dummy;
    return hasStoreToStackSlot(MI, Dummy, FrameIndex);
  }
  return 0;
}

bool X86InstrInfo::hasStoreToStackSlot(const MachineInstr *MI,
                                       const MachineMemOperand *&MMO,
                                       int &FrameIndex) const {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end();
       o != oe;
       ++o) {
    if ((*o)->isStore() && (*o)->getValue())
      if (const FixedStackPseudoSourceValue *Value =
          dyn_cast<const FixedStackPseudoSourceValue>((*o)->getValue())) {
        FrameIndex = Value->getFrameIndex();
        MMO = *o;
        return true;
      }
  }
  return false;
}

/// regIsPICBase - Return true if register is PIC base (i.e.g defined by
/// X86::MOVPC32r.
static bool regIsPICBase(unsigned BaseReg, const MachineRegisterInfo &MRI) {
  bool isPICBase = false;
  for (MachineRegisterInfo::def_iterator I = MRI.def_begin(BaseReg),
         E = MRI.def_end(); I != E; ++I) {
    MachineInstr *DefMI = I.getOperand().getParent();
    if (DefMI->getOpcode() != X86::MOVPC32r)
      return false;
    assert(!isPICBase && "More than one PIC base?");
    isPICBase = true;
  }
  return isPICBase;
}

bool
X86InstrInfo::isReallyTriviallyReMaterializable(const MachineInstr *MI,
                                                AliasAnalysis *AA) const {
  switch (MI->getOpcode()) {
  default: break;
    case X86::MOV8rm:
    case X86::MOV16rm:
    case X86::MOV32rm:
    case X86::MOV64rm:
    case X86::LD_Fp64m:
    case X86::MOVSSrm:
    case X86::MOVSDrm:
    case X86::MOVAPSrm:
    case X86::MOVUPSrm:
    case X86::MOVAPDrm:
    case X86::MOVDQArm:
    case X86::MMX_MOVD64rm:
    case X86::MMX_MOVQ64rm:
    case X86::FsMOVAPSrm:
    case X86::FsMOVAPDrm: {
      // Loads from constant pools are trivially rematerializable.
      if (MI->getOperand(1).isReg() &&
          MI->getOperand(2).isImm() &&
          MI->getOperand(3).isReg() && MI->getOperand(3).getReg() == 0 &&
          MI->isInvariantLoad(AA)) {
        unsigned BaseReg = MI->getOperand(1).getReg();
        if (BaseReg == 0 || BaseReg == X86::RIP)
          return true;
        // Allow re-materialization of PIC load.
        if (!ReMatPICStubLoad && MI->getOperand(4).isGlobal())
          return false;
        const MachineFunction &MF = *MI->getParent()->getParent();
        const MachineRegisterInfo &MRI = MF.getRegInfo();
        bool isPICBase = false;
        for (MachineRegisterInfo::def_iterator I = MRI.def_begin(BaseReg),
               E = MRI.def_end(); I != E; ++I) {
          MachineInstr *DefMI = I.getOperand().getParent();
          if (DefMI->getOpcode() != X86::MOVPC32r)
            return false;
          assert(!isPICBase && "More than one PIC base?");
          isPICBase = true;
        }
        return isPICBase;
      }
      return false;
    }

     case X86::LEA32r:
     case X86::LEA64r: {
       if (MI->getOperand(2).isImm() &&
           MI->getOperand(3).isReg() && MI->getOperand(3).getReg() == 0 &&
           !MI->getOperand(4).isReg()) {
         // lea fi#, lea GV, etc. are all rematerializable.
         if (!MI->getOperand(1).isReg())
           return true;
         unsigned BaseReg = MI->getOperand(1).getReg();
         if (BaseReg == 0)
           return true;
         // Allow re-materialization of lea PICBase + x.
         const MachineFunction &MF = *MI->getParent()->getParent();
         const MachineRegisterInfo &MRI = MF.getRegInfo();
         return regIsPICBase(BaseReg, MRI);
       }
       return false;
     }
  }

  // All other instructions marked M_REMATERIALIZABLE are always trivially
  // rematerializable.
  return true;
}

/// isSafeToClobberEFLAGS - Return true if it's safe insert an instruction that
/// would clobber the EFLAGS condition register. Note the result may be
/// conservative. If it cannot definitely determine the safety after visiting
/// a few instructions in each direction it assumes it's not safe.
static bool isSafeToClobberEFLAGS(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I) {
  MachineBasicBlock::iterator E = MBB.end();

  // It's always safe to clobber EFLAGS at the end of a block.
  if (I == E)
    return true;

  // For compile time consideration, if we are not able to determine the
  // safety after visiting 4 instructions in each direction, we will assume
  // it's not safe.
  MachineBasicBlock::iterator Iter = I;
  for (unsigned i = 0; i < 4; ++i) {
    bool SeenDef = false;
    for (unsigned j = 0, e = Iter->getNumOperands(); j != e; ++j) {
      MachineOperand &MO = Iter->getOperand(j);
      if (!MO.isReg())
        continue;
      if (MO.getReg() == X86::EFLAGS) {
        if (MO.isUse())
          return false;
        SeenDef = true;
      }
    }

    if (SeenDef)
      // This instruction defines EFLAGS, no need to look any further.
      return true;
    ++Iter;
    // Skip over DBG_VALUE.
    while (Iter != E && Iter->isDebugValue())
      ++Iter;

    // If we make it to the end of the block, it's safe to clobber EFLAGS.
    if (Iter == E)
      return true;
  }

  MachineBasicBlock::iterator B = MBB.begin();
  Iter = I;
  for (unsigned i = 0; i < 4; ++i) {
    // If we make it to the beginning of the block, it's safe to clobber
    // EFLAGS iff EFLAGS is not live-in.
    if (Iter == B)
      return !MBB.isLiveIn(X86::EFLAGS);

    --Iter;
    // Skip over DBG_VALUE.
    while (Iter != B && Iter->isDebugValue())
      --Iter;

    bool SawKill = false;
    for (unsigned j = 0, e = Iter->getNumOperands(); j != e; ++j) {
      MachineOperand &MO = Iter->getOperand(j);
      if (MO.isReg() && MO.getReg() == X86::EFLAGS) {
        if (MO.isDef()) return MO.isDead();
        if (MO.isKill()) SawKill = true;
      }
    }

    if (SawKill)
      // This instruction kills EFLAGS and doesn't redefine it, so
      // there's no need to look further.
      return true;
  }

  // Conservative answer.
  return false;
}

void X86InstrInfo::reMaterialize(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I,
                                 unsigned DestReg, unsigned SubIdx,
                                 const MachineInstr *Orig,
                                 const TargetRegisterInfo &TRI) const {
  DebugLoc DL = Orig->getDebugLoc();

  // MOV32r0 etc. are implemented with xor which clobbers condition code.
  // Re-materialize them as movri instructions to avoid side effects.
  bool Clone = true;
  unsigned Opc = Orig->getOpcode();
  switch (Opc) {
  default: break;
  case X86::MOV8r0:
  case X86::MOV16r0:
  case X86::MOV32r0:
  case X86::MOV64r0: {
    if (!isSafeToClobberEFLAGS(MBB, I)) {
      switch (Opc) {
      default: break;
      case X86::MOV8r0:  Opc = X86::MOV8ri;  break;
      case X86::MOV16r0: Opc = X86::MOV16ri; break;
      case X86::MOV32r0: Opc = X86::MOV32ri; break;
      case X86::MOV64r0: Opc = X86::MOV64ri64i32; break;
      }
      Clone = false;
    }
    break;
  }
  }

  if (Clone) {
    MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
    MBB.insert(I, MI);
  } else {
    BuildMI(MBB, I, DL, get(Opc)).addOperand(Orig->getOperand(0)).addImm(0);
  }

  MachineInstr *NewMI = prior(I);
  NewMI->substituteRegister(Orig->getOperand(0).getReg(), DestReg, SubIdx, TRI);
}

/// hasLiveCondCodeDef - True if MI has a condition code def, e.g. EFLAGS, that
/// is not marked dead.
static bool hasLiveCondCodeDef(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        MO.getReg() == X86::EFLAGS && !MO.isDead()) {
      return true;
    }
  }
  return false;
}

/// convertToThreeAddressWithLEA - Helper for convertToThreeAddress when
/// 16-bit LEA is disabled, use 32-bit LEA to form 3-address code by promoting
/// to a 32-bit superregister and then truncating back down to a 16-bit
/// subregister.
MachineInstr *
X86InstrInfo::convertToThreeAddressWithLEA(unsigned MIOpc,
                                           MachineFunction::iterator &MFI,
                                           MachineBasicBlock::iterator &MBBI,
                                           LiveVariables *LV) const {
  MachineInstr *MI = MBBI;
  unsigned Dest = MI->getOperand(0).getReg();
  unsigned Src = MI->getOperand(1).getReg();
  bool isDead = MI->getOperand(0).isDead();
  bool isKill = MI->getOperand(1).isKill();

  unsigned Opc = TM.getSubtarget<X86Subtarget>().is64Bit()
    ? X86::LEA64_32r : X86::LEA32r;
  MachineRegisterInfo &RegInfo = MFI->getParent()->getRegInfo();
  unsigned leaInReg = RegInfo.createVirtualRegister(&X86::GR32_NOSPRegClass);
  unsigned leaOutReg = RegInfo.createVirtualRegister(&X86::GR32RegClass);

  // Build and insert into an implicit UNDEF value. This is OK because
  // well be shifting and then extracting the lower 16-bits.
  // This has the potential to cause partial register stall. e.g.
  //   movw    (%rbp,%rcx,2), %dx
  //   leal    -65(%rdx), %esi
  // But testing has shown this *does* help performance in 64-bit mode (at
  // least on modern x86 machines).
  BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(X86::IMPLICIT_DEF), leaInReg);
  MachineInstr *InsMI =
    BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(TargetOpcode::COPY))
    .addReg(leaInReg, RegState::Define, X86::sub_16bit)
    .addReg(Src, getKillRegState(isKill));

  MachineInstrBuilder MIB = BuildMI(*MFI, MBBI, MI->getDebugLoc(),
                                    get(Opc), leaOutReg);
  switch (MIOpc) {
  default:
    llvm_unreachable(0);
    break;
  case X86::SHL16ri: {
    unsigned ShAmt = MI->getOperand(2).getImm();
    MIB.addReg(0).addImm(1 << ShAmt)
       .addReg(leaInReg, RegState::Kill).addImm(0).addReg(0);
    break;
  }
  case X86::INC16r:
  case X86::INC64_16r:
    addRegOffset(MIB, leaInReg, true, 1);
    break;
  case X86::DEC16r:
  case X86::DEC64_16r:
    addRegOffset(MIB, leaInReg, true, -1);
    break;
  case X86::ADD16ri:
  case X86::ADD16ri8:
  case X86::ADD16ri_DB:
  case X86::ADD16ri8_DB:
    addRegOffset(MIB, leaInReg, true, MI->getOperand(2).getImm());
    break;
  case X86::ADD16rr:
  case X86::ADD16rr_DB: {
    unsigned Src2 = MI->getOperand(2).getReg();
    bool isKill2 = MI->getOperand(2).isKill();
    unsigned leaInReg2 = 0;
    MachineInstr *InsMI2 = 0;
    if (Src == Src2) {
      // ADD16rr %reg1028<kill>, %reg1028
      // just a single insert_subreg.
      addRegReg(MIB, leaInReg, true, leaInReg, false);
    } else {
      leaInReg2 = RegInfo.createVirtualRegister(&X86::GR32_NOSPRegClass);
      // Build and insert into an implicit UNDEF value. This is OK because
      // well be shifting and then extracting the lower 16-bits.
      BuildMI(*MFI, MIB, MI->getDebugLoc(), get(X86::IMPLICIT_DEF), leaInReg2);
      InsMI2 =
        BuildMI(*MFI, MIB, MI->getDebugLoc(), get(TargetOpcode::COPY))
        .addReg(leaInReg2, RegState::Define, X86::sub_16bit)
        .addReg(Src2, getKillRegState(isKill2));
      addRegReg(MIB, leaInReg, true, leaInReg2, true);
    }
    if (LV && isKill2 && InsMI2)
      LV->replaceKillInstruction(Src2, MI, InsMI2);
    break;
  }
  }

  MachineInstr *NewMI = MIB;
  MachineInstr *ExtMI =
    BuildMI(*MFI, MBBI, MI->getDebugLoc(), get(TargetOpcode::COPY))
    .addReg(Dest, RegState::Define | getDeadRegState(isDead))
    .addReg(leaOutReg, RegState::Kill, X86::sub_16bit);

  if (LV) {
    // Update live variables
    LV->getVarInfo(leaInReg).Kills.push_back(NewMI);
    LV->getVarInfo(leaOutReg).Kills.push_back(ExtMI);
    if (isKill)
      LV->replaceKillInstruction(Src, MI, InsMI);
    if (isDead)
      LV->replaceKillInstruction(Dest, MI, ExtMI);
  }

  return ExtMI;
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
                                    LiveVariables *LV) const {
  MachineInstr *MI = MBBI;
  MachineFunction &MF = *MI->getParent()->getParent();
  // All instructions input are two-addr instructions.  Get the known operands.
  unsigned Dest = MI->getOperand(0).getReg();
  unsigned Src = MI->getOperand(1).getReg();
  bool isDead = MI->getOperand(0).isDead();
  bool isKill = MI->getOperand(1).isKill();

  MachineInstr *NewMI = NULL;
  // FIXME: 16-bit LEA's are really slow on Athlons, but not bad on P4's.  When
  // we have better subtarget support, enable the 16-bit LEA generation here.
  // 16-bit LEA is also slow on Core2.
  bool DisableLEA16 = true;
  bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();

  unsigned MIOpc = MI->getOpcode();
  switch (MIOpc) {
  case X86::SHUFPSrri: {
    assert(MI->getNumOperands() == 4 && "Unknown shufps instruction!");
    if (!TM.getSubtarget<X86Subtarget>().hasSSE2()) return 0;

    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    if (B != C) return 0;
    unsigned A = MI->getOperand(0).getReg();
    unsigned M = MI->getOperand(3).getImm();
    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::PSHUFDri))
      .addReg(A, RegState::Define | getDeadRegState(isDead))
      .addReg(B, getKillRegState(isKill)).addImm(M);
    break;
  }
  case X86::SHL64ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;

    // LEA can't handle RSP.
    if (TargetRegisterInfo::isVirtualRegister(Src) &&
        !MF.getRegInfo().constrainRegClass(Src, &X86::GR64_NOSPRegClass))
      return 0;

    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::LEA64r))
      .addReg(Dest, RegState::Define | getDeadRegState(isDead))
      .addReg(0).addImm(1 << ShAmt)
      .addReg(Src, getKillRegState(isKill))
      .addImm(0).addReg(0);
    break;
  }
  case X86::SHL32ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;

    // LEA can't handle ESP.
    if (TargetRegisterInfo::isVirtualRegister(Src) &&
        !MF.getRegInfo().constrainRegClass(Src, &X86::GR32_NOSPRegClass))
      return 0;

    unsigned Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;
    NewMI = BuildMI(MF, MI->getDebugLoc(), get(Opc))
      .addReg(Dest, RegState::Define | getDeadRegState(isDead))
      .addReg(0).addImm(1 << ShAmt)
      .addReg(Src, getKillRegState(isKill)).addImm(0).addReg(0);
    break;
  }
  case X86::SHL16ri: {
    assert(MI->getNumOperands() >= 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;

    if (DisableLEA16)
      return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
    NewMI = BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
      .addReg(Dest, RegState::Define | getDeadRegState(isDead))
      .addReg(0).addImm(1 << ShAmt)
      .addReg(Src, getKillRegState(isKill))
      .addImm(0).addReg(0);
    break;
  }
  default: {
    // The following opcodes also sets the condition code register(s). Only
    // convert them to equivalent lea if the condition code register def's
    // are dead!
    if (hasLiveCondCodeDef(MI))
      return 0;

    switch (MIOpc) {
    default: return 0;
    case X86::INC64r:
    case X86::INC32r:
    case X86::INC64_32r: {
      assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
      unsigned Opc = MIOpc == X86::INC64r ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);

      // LEA can't handle RSP.
      if (TargetRegisterInfo::isVirtualRegister(Src) &&
          !MF.getRegInfo().constrainRegClass(Src,
                            MIOpc == X86::INC64r ? X86::GR64_NOSPRegisterClass :
                                                   X86::GR32_NOSPRegisterClass))
        return 0;

      NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(Opc))
                              .addReg(Dest, RegState::Define |
                                      getDeadRegState(isDead)),
                              Src, isKill, 1);
      break;
    }
    case X86::INC16r:
    case X86::INC64_16r:
      if (DisableLEA16)
        return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
      assert(MI->getNumOperands() >= 2 && "Unknown inc instruction!");
      NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                           .addReg(Dest, RegState::Define |
                                   getDeadRegState(isDead)),
                           Src, isKill, 1);
      break;
    case X86::DEC64r:
    case X86::DEC32r:
    case X86::DEC64_32r: {
      assert(MI->getNumOperands() >= 2 && "Unknown dec instruction!");
      unsigned Opc = MIOpc == X86::DEC64r ? X86::LEA64r
        : (is64Bit ? X86::LEA64_32r : X86::LEA32r);
      // LEA can't handle RSP.
      if (TargetRegisterInfo::isVirtualRegister(Src) &&
          !MF.getRegInfo().constrainRegClass(Src,
                            MIOpc == X86::DEC64r ? X86::GR64_NOSPRegisterClass :
                                                   X86::GR32_NOSPRegisterClass))
        return 0;

      NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(Opc))
                              .addReg(Dest, RegState::Define |
                                      getDeadRegState(isDead)),
                              Src, isKill, -1);
      break;
    }
    case X86::DEC16r:
    case X86::DEC64_16r:
      if (DisableLEA16)
        return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
      assert(MI->getNumOperands() >= 2 && "Unknown dec instruction!");
      NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                           .addReg(Dest, RegState::Define |
                                   getDeadRegState(isDead)),
                           Src, isKill, -1);
      break;
    case X86::ADD64rr:
    case X86::ADD64rr_DB:
    case X86::ADD32rr:
    case X86::ADD32rr_DB: {
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      unsigned Opc;
      TargetRegisterClass *RC;
      if (MIOpc == X86::ADD64rr || MIOpc == X86::ADD64rr_DB) {
        Opc = X86::LEA64r;
        RC = X86::GR64_NOSPRegisterClass;
      } else {
        Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;
        RC = X86::GR32_NOSPRegisterClass;
      }


      unsigned Src2 = MI->getOperand(2).getReg();
      bool isKill2 = MI->getOperand(2).isKill();

      // LEA can't handle RSP.
      if (TargetRegisterInfo::isVirtualRegister(Src2) &&
          !MF.getRegInfo().constrainRegClass(Src2, RC))
        return 0;

      NewMI = addRegReg(BuildMI(MF, MI->getDebugLoc(), get(Opc))
                        .addReg(Dest, RegState::Define |
                                getDeadRegState(isDead)),
                        Src, isKill, Src2, isKill2);
      if (LV && isKill2)
        LV->replaceKillInstruction(Src2, MI, NewMI);
      break;
    }
    case X86::ADD16rr:
    case X86::ADD16rr_DB: {
      if (DisableLEA16)
        return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      unsigned Src2 = MI->getOperand(2).getReg();
      bool isKill2 = MI->getOperand(2).isKill();
      NewMI = addRegReg(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                        .addReg(Dest, RegState::Define |
                                getDeadRegState(isDead)),
                        Src, isKill, Src2, isKill2);
      if (LV && isKill2)
        LV->replaceKillInstruction(Src2, MI, NewMI);
      break;
    }
    case X86::ADD64ri32:
    case X86::ADD64ri8:
    case X86::ADD64ri32_DB:
    case X86::ADD64ri8_DB:
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA64r))
                              .addReg(Dest, RegState::Define |
                                      getDeadRegState(isDead)),
                              Src, isKill, MI->getOperand(2).getImm());
      break;
    case X86::ADD32ri:
    case X86::ADD32ri8:
    case X86::ADD32ri_DB:
    case X86::ADD32ri8_DB: {
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      unsigned Opc = is64Bit ? X86::LEA64_32r : X86::LEA32r;
      NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(Opc))
                              .addReg(Dest, RegState::Define |
                                      getDeadRegState(isDead)),
                                Src, isKill, MI->getOperand(2).getImm());
      break;
    }
    case X86::ADD16ri:
    case X86::ADD16ri8:
    case X86::ADD16ri_DB:
    case X86::ADD16ri8_DB:
      if (DisableLEA16)
        return is64Bit ? convertToThreeAddressWithLEA(MIOpc, MFI, MBBI, LV) : 0;
      assert(MI->getNumOperands() >= 3 && "Unknown add instruction!");
      NewMI = addRegOffset(BuildMI(MF, MI->getDebugLoc(), get(X86::LEA16r))
                              .addReg(Dest, RegState::Define |
                                      getDeadRegState(isDead)),
                              Src, isKill, MI->getOperand(2).getImm());
      break;
    }
  }
  }

  if (!NewMI) return 0;

  if (LV) {  // Update live variables
    if (isKill)
      LV->replaceKillInstruction(Src, MI, NewMI);
    if (isDead)
      LV->replaceKillInstruction(Dest, MI, NewMI);
  }

  MFI->insert(MBBI, NewMI);          // Insert the new inst
  return NewMI;
}

/// commuteInstruction - We have a few instructions that must be hacked on to
/// commute them.
///
MachineInstr *
X86InstrInfo::commuteInstruction(MachineInstr *MI, bool NewMI) const {
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
    default: llvm_unreachable("Unreachable!");
    case X86::SHRD16rri8: Size = 16; Opc = X86::SHLD16rri8; break;
    case X86::SHLD16rri8: Size = 16; Opc = X86::SHRD16rri8; break;
    case X86::SHRD32rri8: Size = 32; Opc = X86::SHLD32rri8; break;
    case X86::SHLD32rri8: Size = 32; Opc = X86::SHRD32rri8; break;
    case X86::SHRD64rri8: Size = 64; Opc = X86::SHLD64rri8; break;
    case X86::SHLD64rri8: Size = 64; Opc = X86::SHRD64rri8; break;
    }
    unsigned Amt = MI->getOperand(3).getImm();
    if (NewMI) {
      MachineFunction &MF = *MI->getParent()->getParent();
      MI = MF.CloneMachineInstr(MI);
      NewMI = false;
    }
    MI->setDesc(get(Opc));
    MI->getOperand(3).setImm(Size-Amt);
    return TargetInstrInfoImpl::commuteInstruction(MI, NewMI);
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
  case X86::CMOVNP64rr:
  case X86::CMOVO16rr:
  case X86::CMOVO32rr:
  case X86::CMOVO64rr:
  case X86::CMOVNO16rr:
  case X86::CMOVNO32rr:
  case X86::CMOVNO64rr: {
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
    case X86::CMOVS64rr:  Opc = X86::CMOVNS64rr; break;
    case X86::CMOVNS16rr: Opc = X86::CMOVS16rr; break;
    case X86::CMOVNS32rr: Opc = X86::CMOVS32rr; break;
    case X86::CMOVNS64rr: Opc = X86::CMOVS64rr; break;
    case X86::CMOVP16rr:  Opc = X86::CMOVNP16rr; break;
    case X86::CMOVP32rr:  Opc = X86::CMOVNP32rr; break;
    case X86::CMOVP64rr:  Opc = X86::CMOVNP64rr; break;
    case X86::CMOVNP16rr: Opc = X86::CMOVP16rr; break;
    case X86::CMOVNP32rr: Opc = X86::CMOVP32rr; break;
    case X86::CMOVNP64rr: Opc = X86::CMOVP64rr; break;
    case X86::CMOVO16rr:  Opc = X86::CMOVNO16rr; break;
    case X86::CMOVO32rr:  Opc = X86::CMOVNO32rr; break;
    case X86::CMOVO64rr:  Opc = X86::CMOVNO64rr; break;
    case X86::CMOVNO16rr: Opc = X86::CMOVO16rr; break;
    case X86::CMOVNO32rr: Opc = X86::CMOVO32rr; break;
    case X86::CMOVNO64rr: Opc = X86::CMOVO64rr; break;
    }
    if (NewMI) {
      MachineFunction &MF = *MI->getParent()->getParent();
      MI = MF.CloneMachineInstr(MI);
      NewMI = false;
    }
    MI->setDesc(get(Opc));
    // Fallthrough intended.
  }
  default:
    return TargetInstrInfoImpl::commuteInstruction(MI, NewMI);
  }
}

static X86::CondCode GetCondFromBranchOpc(unsigned BrOpc) {
  switch (BrOpc) {
  default: return X86::COND_INVALID;
  case X86::JE_4:  return X86::COND_E;
  case X86::JNE_4: return X86::COND_NE;
  case X86::JL_4:  return X86::COND_L;
  case X86::JLE_4: return X86::COND_LE;
  case X86::JG_4:  return X86::COND_G;
  case X86::JGE_4: return X86::COND_GE;
  case X86::JB_4:  return X86::COND_B;
  case X86::JBE_4: return X86::COND_BE;
  case X86::JA_4:  return X86::COND_A;
  case X86::JAE_4: return X86::COND_AE;
  case X86::JS_4:  return X86::COND_S;
  case X86::JNS_4: return X86::COND_NS;
  case X86::JP_4:  return X86::COND_P;
  case X86::JNP_4: return X86::COND_NP;
  case X86::JO_4:  return X86::COND_O;
  case X86::JNO_4: return X86::COND_NO;
  }
}

unsigned X86::GetCondBranchFromCond(X86::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Illegal condition code!");
  case X86::COND_E:  return X86::JE_4;
  case X86::COND_NE: return X86::JNE_4;
  case X86::COND_L:  return X86::JL_4;
  case X86::COND_LE: return X86::JLE_4;
  case X86::COND_G:  return X86::JG_4;
  case X86::COND_GE: return X86::JGE_4;
  case X86::COND_B:  return X86::JB_4;
  case X86::COND_BE: return X86::JBE_4;
  case X86::COND_A:  return X86::JA_4;
  case X86::COND_AE: return X86::JAE_4;
  case X86::COND_S:  return X86::JS_4;
  case X86::COND_NS: return X86::JNS_4;
  case X86::COND_P:  return X86::JP_4;
  case X86::COND_NP: return X86::JNP_4;
  case X86::COND_O:  return X86::JO_4;
  case X86::COND_NO: return X86::JNO_4;
  }
}

/// GetOppositeBranchCondition - Return the inverse of the specified condition,
/// e.g. turning COND_E to COND_NE.
X86::CondCode X86::GetOppositeBranchCondition(X86::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Illegal condition code!");
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
  const MCInstrDesc &MCID = MI->getDesc();
  if (!MCID.isTerminator()) return false;

  // Conditional branch is a special case.
  if (MCID.isBranch() && !MCID.isBarrier())
    return true;
  if (!MCID.isPredicable())
    return true;
  return !isPredicated(MI);
}

bool X86InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::iterator I = MBB.end();
  MachineBasicBlock::iterator UnCondBrIter = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;

    // Working from the bottom, when we see a non-terminator instruction, we're
    // done.
    if (!isUnpredicatedTerminator(I))
      break;

    // A terminator that isn't a branch can't easily be handled by this
    // analysis.
    if (!I->getDesc().isBranch())
      return true;

    // Handle unconditional branches.
    if (I->getOpcode() == X86::JMP_4) {
      UnCondBrIter = I;

      if (!AllowModify) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a JMP, delete them.
      while (llvm::next(I) != MBB.end())
        llvm::next(I)->eraseFromParent();

      Cond.clear();
      FBB = 0;

      // Delete the JMP if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = 0;
        I->eraseFromParent();
        I = MBB.end();
        UnCondBrIter = MBB.end();
        continue;
      }

      // TBB is used to indicate the unconditional destination.
      TBB = I->getOperand(0).getMBB();
      continue;
    }

    // Handle conditional branches.
    X86::CondCode BranchCode = GetCondFromBranchOpc(I->getOpcode());
    if (BranchCode == X86::COND_INVALID)
      return true;  // Can't handle indirect branch.

    // Working from the bottom, handle the first conditional branch.
    if (Cond.empty()) {
      MachineBasicBlock *TargetBB = I->getOperand(0).getMBB();
      if (AllowModify && UnCondBrIter != MBB.end() &&
          MBB.isLayoutSuccessor(TargetBB)) {
        // If we can modify the code and it ends in something like:
        //
        //     jCC L1
        //     jmp L2
        //   L1:
        //     ...
        //   L2:
        //
        // Then we can change this to:
        //
        //     jnCC L2
        //   L1:
        //     ...
        //   L2:
        //
        // Which is a bit more efficient.
        // We conditionally jump to the fall-through block.
        BranchCode = GetOppositeBranchCondition(BranchCode);
        unsigned JNCC = GetCondBranchFromCond(BranchCode);
        MachineBasicBlock::iterator OldInst = I;

        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), get(JNCC))
          .addMBB(UnCondBrIter->getOperand(0).getMBB());
        BuildMI(MBB, UnCondBrIter, MBB.findDebugLoc(I), get(X86::JMP_4))
          .addMBB(TargetBB);

        OldInst->eraseFromParent();
        UnCondBrIter->eraseFromParent();

        // Restart the analysis.
        UnCondBrIter = MBB.end();
        I = MBB.end();
        continue;
      }

      FBB = TBB;
      TBB = I->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(BranchCode));
      continue;
    }

    // Handle subsequent conditional branches. Only handle the case where all
    // conditional branches branch to the same destination and their condition
    // opcodes fit one of the special multi-branch idioms.
    assert(Cond.size() == 1);
    assert(TBB);

    // Only handle the case where all conditional branches branch to the same
    // destination.
    if (TBB != I->getOperand(0).getMBB())
      return true;

    // If the conditions are the same, we can leave them alone.
    X86::CondCode OldBranchCode = (X86::CondCode)Cond[0].getImm();
    if (OldBranchCode == BranchCode)
      continue;

    // If they differ, see if they fit one of the known patterns. Theoretically,
    // we could handle more patterns here, but we shouldn't expect to see them
    // if instruction selection has done a reasonable job.
    if ((OldBranchCode == X86::COND_NP &&
         BranchCode == X86::COND_E) ||
        (OldBranchCode == X86::COND_E &&
         BranchCode == X86::COND_NP))
      BranchCode = X86::COND_NP_OR_E;
    else if ((OldBranchCode == X86::COND_P &&
              BranchCode == X86::COND_NE) ||
             (OldBranchCode == X86::COND_NE &&
              BranchCode == X86::COND_P))
      BranchCode = X86::COND_NE_OR_P;
    else
      return true;

    // Update the MachineOperand.
    Cond[0].setImm(BranchCode);
  }

  return false;
}

unsigned X86InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;
    if (I->getOpcode() != X86::JMP_4 &&
        GetCondFromBranchOpc(I->getOpcode()) == X86::COND_INVALID)
      break;
    // Remove the branch.
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  return Count;
}

unsigned
X86InstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                           MachineBasicBlock *FBB,
                           const SmallVectorImpl<MachineOperand> &Cond,
                           DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "X86 branch conditions have one component!");

  if (Cond.empty()) {
    // Unconditional branch?
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(X86::JMP_4)).addMBB(TBB);
    return 1;
  }

  // Conditional branch.
  unsigned Count = 0;
  X86::CondCode CC = (X86::CondCode)Cond[0].getImm();
  switch (CC) {
  case X86::COND_NP_OR_E:
    // Synthesize NP_OR_E with two branches.
    BuildMI(&MBB, DL, get(X86::JNP_4)).addMBB(TBB);
    ++Count;
    BuildMI(&MBB, DL, get(X86::JE_4)).addMBB(TBB);
    ++Count;
    break;
  case X86::COND_NE_OR_P:
    // Synthesize NE_OR_P with two branches.
    BuildMI(&MBB, DL, get(X86::JNE_4)).addMBB(TBB);
    ++Count;
    BuildMI(&MBB, DL, get(X86::JP_4)).addMBB(TBB);
    ++Count;
    break;
  default: {
    unsigned Opc = GetCondBranchFromCond(CC);
    BuildMI(&MBB, DL, get(Opc)).addMBB(TBB);
    ++Count;
  }
  }
  if (FBB) {
    // Two-way Conditional branch. Insert the second branch.
    BuildMI(&MBB, DL, get(X86::JMP_4)).addMBB(FBB);
    ++Count;
  }
  return Count;
}

/// isHReg - Test if the given register is a physical h register.
static bool isHReg(unsigned Reg) {
  return X86::GR8_ABCD_HRegClass.contains(Reg);
}

// Try and copy between VR128/VR64 and GR64 registers.
static unsigned CopyToFromAsymmetricReg(unsigned DestReg, unsigned SrcReg) {
  // SrcReg(VR128) -> DestReg(GR64)
  // SrcReg(VR64)  -> DestReg(GR64)
  // SrcReg(GR64)  -> DestReg(VR128)
  // SrcReg(GR64)  -> DestReg(VR64)

  if (X86::GR64RegClass.contains(DestReg)) {
    if (X86::VR128RegClass.contains(SrcReg)) {
      // Copy from a VR128 register to a GR64 register.
      return X86::MOVPQIto64rr;
    } else if (X86::VR64RegClass.contains(SrcReg)) {
      // Copy from a VR64 register to a GR64 register.
      return X86::MOVSDto64rr;
    }
  } else if (X86::GR64RegClass.contains(SrcReg)) {
    // Copy from a GR64 register to a VR128 register.
    if (X86::VR128RegClass.contains(DestReg))
      return X86::MOV64toPQIrr;
    // Copy from a GR64 register to a VR64 register.
    else if (X86::VR64RegClass.contains(DestReg))
      return X86::MOV64toSDrr;
  }

  return 0;
}

void X86InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI, DebugLoc DL,
                               unsigned DestReg, unsigned SrcReg,
                               bool KillSrc) const {
  // First deal with the normal symmetric copies.
  unsigned Opc = 0;
  if (X86::GR64RegClass.contains(DestReg, SrcReg))
    Opc = X86::MOV64rr;
  else if (X86::GR32RegClass.contains(DestReg, SrcReg))
    Opc = X86::MOV32rr;
  else if (X86::GR16RegClass.contains(DestReg, SrcReg))
    Opc = X86::MOV16rr;
  else if (X86::GR8RegClass.contains(DestReg, SrcReg)) {
    // Copying to or from a physical H register on x86-64 requires a NOREX
    // move.  Otherwise use a normal move.
    if ((isHReg(DestReg) || isHReg(SrcReg)) &&
        TM.getSubtarget<X86Subtarget>().is64Bit())
      Opc = X86::MOV8rr_NOREX;
    else
      Opc = X86::MOV8rr;
  } else if (X86::VR128RegClass.contains(DestReg, SrcReg))
    Opc = X86::MOVAPSrr;
  else if (X86::VR64RegClass.contains(DestReg, SrcReg))
    Opc = X86::MMX_MOVQ64rr;
  else
    Opc = CopyToFromAsymmetricReg(DestReg, SrcReg);

  if (Opc) {
    BuildMI(MBB, MI, DL, get(Opc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // Moving EFLAGS to / from another register requires a push and a pop.
  if (SrcReg == X86::EFLAGS) {
    if (X86::GR64RegClass.contains(DestReg)) {
      BuildMI(MBB, MI, DL, get(X86::PUSHF64));
      BuildMI(MBB, MI, DL, get(X86::POP64r), DestReg);
      return;
    } else if (X86::GR32RegClass.contains(DestReg)) {
      BuildMI(MBB, MI, DL, get(X86::PUSHF32));
      BuildMI(MBB, MI, DL, get(X86::POP32r), DestReg);
      return;
    }
  }
  if (DestReg == X86::EFLAGS) {
    if (X86::GR64RegClass.contains(SrcReg)) {
      BuildMI(MBB, MI, DL, get(X86::PUSH64r))
        .addReg(SrcReg, getKillRegState(KillSrc));
      BuildMI(MBB, MI, DL, get(X86::POPF64));
      return;
    } else if (X86::GR32RegClass.contains(SrcReg)) {
      BuildMI(MBB, MI, DL, get(X86::PUSH32r))
        .addReg(SrcReg, getKillRegState(KillSrc));
      BuildMI(MBB, MI, DL, get(X86::POPF32));
      return;
    }
  }

  DEBUG(dbgs() << "Cannot copy " << RI.getName(SrcReg)
               << " to " << RI.getName(DestReg) << '\n');
  llvm_unreachable("Cannot emit physreg copy instruction");
}

static unsigned getLoadStoreRegOpcode(unsigned Reg,
                                      const TargetRegisterClass *RC,
                                      bool isStackAligned,
                                      const TargetMachine &TM,
                                      bool load) {
  switch (RC->getSize()) {
  default:
    llvm_unreachable("Unknown spill size");
  case 1:
    assert(X86::GR8RegClass.hasSubClassEq(RC) && "Unknown 1-byte regclass");
    if (TM.getSubtarget<X86Subtarget>().is64Bit())
      // Copying to or from a physical H register on x86-64 requires a NOREX
      // move.  Otherwise use a normal move.
      if (isHReg(Reg) || X86::GR8_ABCD_HRegClass.hasSubClassEq(RC))
        return load ? X86::MOV8rm_NOREX : X86::MOV8mr_NOREX;
    return load ? X86::MOV8rm : X86::MOV8mr;
  case 2:
    assert(X86::GR16RegClass.hasSubClassEq(RC) && "Unknown 2-byte regclass");
    return load ? X86::MOV16rm : X86::MOV16mr;
  case 4:
    if (X86::GR32RegClass.hasSubClassEq(RC))
      return load ? X86::MOV32rm : X86::MOV32mr;
    if (X86::FR32RegClass.hasSubClassEq(RC))
      return load ? X86::MOVSSrm : X86::MOVSSmr;
    if (X86::RFP32RegClass.hasSubClassEq(RC))
      return load ? X86::LD_Fp32m : X86::ST_Fp32m;
    llvm_unreachable("Unknown 4-byte regclass");
  case 8:
    if (X86::GR64RegClass.hasSubClassEq(RC))
      return load ? X86::MOV64rm : X86::MOV64mr;
    if (X86::FR64RegClass.hasSubClassEq(RC))
      return load ? X86::MOVSDrm : X86::MOVSDmr;
    if (X86::VR64RegClass.hasSubClassEq(RC))
      return load ? X86::MMX_MOVQ64rm : X86::MMX_MOVQ64mr;
    if (X86::RFP64RegClass.hasSubClassEq(RC))
      return load ? X86::LD_Fp64m : X86::ST_Fp64m;
    llvm_unreachable("Unknown 8-byte regclass");
  case 10:
    assert(X86::RFP80RegClass.hasSubClassEq(RC) && "Unknown 10-byte regclass");
    return load ? X86::LD_Fp80m : X86::ST_FpP80m;
  case 16:
    assert(X86::VR128RegClass.hasSubClassEq(RC) && "Unknown 16-byte regclass");
    // If stack is realigned we can use aligned stores.
    if (isStackAligned)
      return load ? X86::MOVAPSrm : X86::MOVAPSmr;
    else
      return load ? X86::MOVUPSrm : X86::MOVUPSmr;
  }
}

static unsigned getStoreRegOpcode(unsigned SrcReg,
                                  const TargetRegisterClass *RC,
                                  bool isStackAligned,
                                  TargetMachine &TM) {
  return getLoadStoreRegOpcode(SrcReg, RC, isStackAligned, TM, false);
}


static unsigned getLoadRegOpcode(unsigned DestReg,
                                 const TargetRegisterClass *RC,
                                 bool isStackAligned,
                                 const TargetMachine &TM) {
  return getLoadStoreRegOpcode(DestReg, RC, isStackAligned, TM, true);
}

void X86InstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       unsigned SrcReg, bool isKill, int FrameIdx,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  const MachineFunction &MF = *MBB.getParent();
  assert(MF.getFrameInfo()->getObjectSize(FrameIdx) >= RC->getSize() &&
         "Stack slot too small for store");
  bool isAligned = (TM.getFrameLowering()->getStackAlignment() >= 16) ||
    RI.canRealignStack(MF);
  unsigned Opc = getStoreRegOpcode(SrcReg, RC, isAligned, TM);
  DebugLoc DL = MBB.findDebugLoc(MI);
  addFrameReference(BuildMI(MBB, MI, DL, get(Opc)), FrameIdx)
    .addReg(SrcReg, getKillRegState(isKill));
}

void X86InstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                  bool isKill,
                                  SmallVectorImpl<MachineOperand> &Addr,
                                  const TargetRegisterClass *RC,
                                  MachineInstr::mmo_iterator MMOBegin,
                                  MachineInstr::mmo_iterator MMOEnd,
                                  SmallVectorImpl<MachineInstr*> &NewMIs) const {
  bool isAligned = MMOBegin != MMOEnd && (*MMOBegin)->getAlignment() >= 16;
  unsigned Opc = getStoreRegOpcode(SrcReg, RC, isAligned, TM);
  DebugLoc DL;
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc));
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  MIB.addReg(SrcReg, getKillRegState(isKill));
  (*MIB).setMemRefs(MMOBegin, MMOEnd);
  NewMIs.push_back(MIB);
}


void X86InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        unsigned DestReg, int FrameIdx,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI) const {
  const MachineFunction &MF = *MBB.getParent();
  bool isAligned = (TM.getFrameLowering()->getStackAlignment() >= 16) ||
    RI.canRealignStack(MF);
  unsigned Opc = getLoadRegOpcode(DestReg, RC, isAligned, TM);
  DebugLoc DL = MBB.findDebugLoc(MI);
  addFrameReference(BuildMI(MBB, MI, DL, get(Opc), DestReg), FrameIdx);
}

void X86InstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                 SmallVectorImpl<MachineOperand> &Addr,
                                 const TargetRegisterClass *RC,
                                 MachineInstr::mmo_iterator MMOBegin,
                                 MachineInstr::mmo_iterator MMOEnd,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  bool isAligned = MMOBegin != MMOEnd && (*MMOBegin)->getAlignment() >= 16;
  unsigned Opc = getLoadRegOpcode(DestReg, RC, isAligned, TM);
  DebugLoc DL;
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  (*MIB).setMemRefs(MMOBegin, MMOEnd);
  NewMIs.push_back(MIB);
}

MachineInstr*
X86InstrInfo::emitFrameIndexDebugValue(MachineFunction &MF,
                                       int FrameIx, uint64_t Offset,
                                       const MDNode *MDPtr,
                                       DebugLoc DL) const {
  X86AddressMode AM;
  AM.BaseType = X86AddressMode::FrameIndexBase;
  AM.Base.FrameIndex = FrameIx;
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(X86::DBG_VALUE));
  addFullAddress(MIB, AM).addImm(Offset).addMetadata(MDPtr);
  return &*MIB;
}

static MachineInstr *FuseTwoAddrInst(MachineFunction &MF, unsigned Opcode,
                                     const SmallVectorImpl<MachineOperand> &MOs,
                                     MachineInstr *MI,
                                     const TargetInstrInfo &TII) {
  // Create the base instruction with the memory operand as the first part.
  MachineInstr *NewMI = MF.CreateMachineInstr(TII.get(Opcode),
                                              MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(NewMI);
  unsigned NumAddrOps = MOs.size();
  for (unsigned i = 0; i != NumAddrOps; ++i)
    MIB.addOperand(MOs[i]);
  if (NumAddrOps < 4)  // FrameIndex only
    addOffset(MIB, 0);

  // Loop over the rest of the ri operands, converting them over.
  unsigned NumOps = MI->getDesc().getNumOperands()-2;
  for (unsigned i = 0; i != NumOps; ++i) {
    MachineOperand &MO = MI->getOperand(i+2);
    MIB.addOperand(MO);
  }
  for (unsigned i = NumOps+2, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    MIB.addOperand(MO);
  }
  return MIB;
}

static MachineInstr *FuseInst(MachineFunction &MF,
                              unsigned Opcode, unsigned OpNo,
                              const SmallVectorImpl<MachineOperand> &MOs,
                              MachineInstr *MI, const TargetInstrInfo &TII) {
  MachineInstr *NewMI = MF.CreateMachineInstr(TII.get(Opcode),
                                              MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(NewMI);

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (i == OpNo) {
      assert(MO.isReg() && "Expected to fold into reg operand!");
      unsigned NumAddrOps = MOs.size();
      for (unsigned i = 0; i != NumAddrOps; ++i)
        MIB.addOperand(MOs[i]);
      if (NumAddrOps < 4)  // FrameIndex only
        addOffset(MIB, 0);
    } else {
      MIB.addOperand(MO);
    }
  }
  return MIB;
}

static MachineInstr *MakeM0Inst(const TargetInstrInfo &TII, unsigned Opcode,
                                const SmallVectorImpl<MachineOperand> &MOs,
                                MachineInstr *MI) {
  MachineFunction &MF = *MI->getParent()->getParent();
  MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), TII.get(Opcode));

  unsigned NumAddrOps = MOs.size();
  for (unsigned i = 0; i != NumAddrOps; ++i)
    MIB.addOperand(MOs[i]);
  if (NumAddrOps < 4)  // FrameIndex only
    addOffset(MIB, 0);
  return MIB.addImm(0);
}

MachineInstr*
X86InstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                    MachineInstr *MI, unsigned i,
                                    const SmallVectorImpl<MachineOperand> &MOs,
                                    unsigned Size, unsigned Align) const {
  const DenseMap<unsigned, std::pair<unsigned,unsigned> > *OpcodeTablePtr = 0;
  bool isTwoAddrFold = false;
  unsigned NumOps = MI->getDesc().getNumOperands();
  bool isTwoAddr = NumOps > 1 &&
    MI->getDesc().getOperandConstraint(1, MCOI::TIED_TO) != -1;

  // FIXME: AsmPrinter doesn't know how to handle
  // X86II::MO_GOT_ABSOLUTE_ADDRESS after folding.
  if (MI->getOpcode() == X86::ADD32ri &&
      MI->getOperand(2).getTargetFlags() == X86II::MO_GOT_ABSOLUTE_ADDRESS)
    return NULL;

  MachineInstr *NewMI = NULL;
  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  if (isTwoAddr && NumOps >= 2 && i < 2 &&
      MI->getOperand(0).isReg() &&
      MI->getOperand(1).isReg() &&
      MI->getOperand(0).getReg() == MI->getOperand(1).getReg()) {
    OpcodeTablePtr = &RegOp2MemOpTable2Addr;
    isTwoAddrFold = true;
  } else if (i == 0) { // If operand 0
    if (MI->getOpcode() == X86::MOV64r0)
      NewMI = MakeM0Inst(*this, X86::MOV64mi32, MOs, MI);
    else if (MI->getOpcode() == X86::MOV32r0)
      NewMI = MakeM0Inst(*this, X86::MOV32mi, MOs, MI);
    else if (MI->getOpcode() == X86::MOV16r0)
      NewMI = MakeM0Inst(*this, X86::MOV16mi, MOs, MI);
    else if (MI->getOpcode() == X86::MOV8r0)
      NewMI = MakeM0Inst(*this, X86::MOV8mi, MOs, MI);
    if (NewMI)
      return NewMI;

    OpcodeTablePtr = &RegOp2MemOpTable0;
  } else if (i == 1) {
    OpcodeTablePtr = &RegOp2MemOpTable1;
  } else if (i == 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2;
  }

  // If table selected...
  if (OpcodeTablePtr) {
    // Find the Opcode to fuse
    DenseMap<unsigned, std::pair<unsigned,unsigned> >::const_iterator I =
      OpcodeTablePtr->find(MI->getOpcode());
    if (I != OpcodeTablePtr->end()) {
      unsigned Opcode = I->second.first;
      unsigned MinAlign = I->second.second;
      if (Align < MinAlign)
        return NULL;
      bool NarrowToMOV32rm = false;
      if (Size) {
        unsigned RCSize = getRegClass(MI->getDesc(), i, &RI)->getSize();
        if (Size < RCSize) {
          // Check if it's safe to fold the load. If the size of the object is
          // narrower than the load width, then it's not.
          if (Opcode != X86::MOV64rm || RCSize != 8 || Size != 4)
            return NULL;
          // If this is a 64-bit load, but the spill slot is 32, then we can do
          // a 32-bit load which is implicitly zero-extended. This likely is due
          // to liveintervalanalysis remat'ing a load from stack slot.
          if (MI->getOperand(0).getSubReg() || MI->getOperand(1).getSubReg())
            return NULL;
          Opcode = X86::MOV32rm;
          NarrowToMOV32rm = true;
        }
      }

      if (isTwoAddrFold)
        NewMI = FuseTwoAddrInst(MF, Opcode, MOs, MI, *this);
      else
        NewMI = FuseInst(MF, Opcode, i, MOs, MI, *this);

      if (NarrowToMOV32rm) {
        // If this is the special case where we use a MOV32rm to load a 32-bit
        // value and zero-extend the top bits. Change the destination register
        // to a 32-bit one.
        unsigned DstReg = NewMI->getOperand(0).getReg();
        if (TargetRegisterInfo::isPhysicalRegister(DstReg))
          NewMI->getOperand(0).setReg(RI.getSubReg(DstReg,
                                                   X86::sub_32bit));
        else
          NewMI->getOperand(0).setSubReg(X86::sub_32bit);
      }
      return NewMI;
    }
  }

  // No fusion
  if (PrintFailedFusing && !MI->isCopy())
    dbgs() << "We failed to fuse operand " << i << " in " << *MI;
  return NULL;
}


MachineInstr* X86InstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                  MachineInstr *MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                                  int FrameIndex) const {
  // Check switch flag
  if (NoFusing) return NULL;

  if (!MF.getFunction()->hasFnAttr(Attribute::OptimizeForSize))
    switch (MI->getOpcode()) {
    case X86::CVTSD2SSrr:
    case X86::Int_CVTSD2SSrr:
    case X86::CVTSS2SDrr:
    case X86::Int_CVTSS2SDrr:
    case X86::RCPSSr:
    case X86::RCPSSr_Int:
    case X86::ROUNDSDr:
    case X86::ROUNDSSr:
    case X86::RSQRTSSr:
    case X86::RSQRTSSr_Int:
    case X86::SQRTSSr:
    case X86::SQRTSSr_Int:
      return 0;
    }

  const MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned Size = MFI->getObjectSize(FrameIndex);
  unsigned Alignment = MFI->getObjectAlignment(FrameIndex);
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    unsigned RCSize = 0;
    switch (MI->getOpcode()) {
    default: return NULL;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; RCSize = 1; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri8; RCSize = 2; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri8; RCSize = 4; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri8; RCSize = 8; break;
    }
    // Check if it's safe to fold the load. If the size of the object is
    // narrower than the load width, then it's not.
    if (Size < RCSize)
      return NULL;
    // Change to CMPXXri r, 0 first.
    MI->setDesc(get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return NULL;

  SmallVector<MachineOperand,4> MOs;
  MOs.push_back(MachineOperand::CreateFI(FrameIndex));
  return foldMemoryOperandImpl(MF, MI, Ops[0], MOs, Size, Alignment);
}

MachineInstr* X86InstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                  MachineInstr *MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                                  MachineInstr *LoadMI) const {
  // Check switch flag
  if (NoFusing) return NULL;

  if (!MF.getFunction()->hasFnAttr(Attribute::OptimizeForSize))
    switch (MI->getOpcode()) {
    case X86::CVTSD2SSrr:
    case X86::Int_CVTSD2SSrr:
    case X86::CVTSS2SDrr:
    case X86::Int_CVTSS2SDrr:
    case X86::RCPSSr:
    case X86::RCPSSr_Int:
    case X86::ROUNDSDr:
    case X86::ROUNDSSr:
    case X86::RSQRTSSr:
    case X86::RSQRTSSr_Int:
    case X86::SQRTSSr:
    case X86::SQRTSSr_Int:
      return 0;
    }

  // Determine the alignment of the load.
  unsigned Alignment = 0;
  if (LoadMI->hasOneMemOperand())
    Alignment = (*LoadMI->memoperands_begin())->getAlignment();
  else
    switch (LoadMI->getOpcode()) {
    case X86::AVX_SET0PSY:
    case X86::AVX_SET0PDY:
      Alignment = 32;
      break;
    case X86::V_SET0PS:
    case X86::V_SET0PD:
    case X86::V_SET0PI:
    case X86::V_SETALLONES:
    case X86::AVX_SET0PS:
    case X86::AVX_SET0PD:
    case X86::AVX_SET0PI:
      Alignment = 16;
      break;
    case X86::FsFLD0SD:
    case X86::VFsFLD0SD:
      Alignment = 8;
      break;
    case X86::FsFLD0SS:
    case X86::VFsFLD0SS:
      Alignment = 4;
      break;
    default:
      return 0;
    }
  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    switch (MI->getOpcode()) {
    default: return NULL;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri8; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri8; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri8; break;
    }
    // Change to CMPXXri r, 0 first.
    MI->setDesc(get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return NULL;

  // Make sure the subregisters match.
  // Otherwise we risk changing the size of the load.
  if (LoadMI->getOperand(0).getSubReg() != MI->getOperand(Ops[0]).getSubReg())
    return NULL;

  SmallVector<MachineOperand,X86::AddrNumOperands> MOs;
  switch (LoadMI->getOpcode()) {
  case X86::V_SET0PS:
  case X86::V_SET0PD:
  case X86::V_SET0PI:
  case X86::V_SETALLONES:
  case X86::AVX_SET0PS:
  case X86::AVX_SET0PD:
  case X86::AVX_SET0PI:
  case X86::AVX_SET0PSY:
  case X86::AVX_SET0PDY:
  case X86::FsFLD0SD:
  case X86::FsFLD0SS: {
    // Folding a V_SET0P? or V_SETALLONES as a load, to ease register pressure.
    // Create a constant-pool entry and operands to load from it.

    // Medium and large mode can't fold loads this way.
    if (TM.getCodeModel() != CodeModel::Small &&
        TM.getCodeModel() != CodeModel::Kernel)
      return NULL;

    // x86-32 PIC requires a PIC base register for constant pools.
    unsigned PICBase = 0;
    if (TM.getRelocationModel() == Reloc::PIC_) {
      if (TM.getSubtarget<X86Subtarget>().is64Bit())
        PICBase = X86::RIP;
      else
        // FIXME: PICBase = getGlobalBaseReg(&MF);
        // This doesn't work for several reasons.
        // 1. GlobalBaseReg may have been spilled.
        // 2. It may not be live at MI.
        return NULL;
    }

    // Create a constant-pool entry.
    MachineConstantPool &MCP = *MF.getConstantPool();
    const Type *Ty;
    unsigned Opc = LoadMI->getOpcode();
    if (Opc == X86::FsFLD0SS || Opc == X86::VFsFLD0SS)
      Ty = Type::getFloatTy(MF.getFunction()->getContext());
    else if (Opc == X86::FsFLD0SD || Opc == X86::VFsFLD0SD)
      Ty = Type::getDoubleTy(MF.getFunction()->getContext());
    else if (Opc == X86::AVX_SET0PSY || Opc == X86::AVX_SET0PDY)
      Ty = VectorType::get(Type::getFloatTy(MF.getFunction()->getContext()), 8);
    else
      Ty = VectorType::get(Type::getInt32Ty(MF.getFunction()->getContext()), 4);
    const Constant *C = LoadMI->getOpcode() == X86::V_SETALLONES ?
                    Constant::getAllOnesValue(Ty) :
                    Constant::getNullValue(Ty);
    unsigned CPI = MCP.getConstantPoolIndex(C, Alignment);

    // Create operands to load from the constant pool entry.
    MOs.push_back(MachineOperand::CreateReg(PICBase, false));
    MOs.push_back(MachineOperand::CreateImm(1));
    MOs.push_back(MachineOperand::CreateReg(0, false));
    MOs.push_back(MachineOperand::CreateCPI(CPI, 0));
    MOs.push_back(MachineOperand::CreateReg(0, false));
    break;
  }
  default: {
    // Folding a normal load. Just copy the load's address operands.
    unsigned NumOps = LoadMI->getDesc().getNumOperands();
    for (unsigned i = NumOps - X86::AddrNumOperands; i != NumOps; ++i)
      MOs.push_back(LoadMI->getOperand(i));
    break;
  }
  }
  return foldMemoryOperandImpl(MF, MI, Ops[0], MOs, 0, Alignment);
}


bool X86InstrInfo::canFoldMemoryOperand(const MachineInstr *MI,
                                  const SmallVectorImpl<unsigned> &Ops) const {
  // Check switch flag
  if (NoFusing) return 0;

  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    switch (MI->getOpcode()) {
    default: return false;
    case X86::TEST8rr:
    case X86::TEST16rr:
    case X86::TEST32rr:
    case X86::TEST64rr:
      return true;
    case X86::ADD32ri:
      // FIXME: AsmPrinter doesn't know how to handle
      // X86II::MO_GOT_ABSOLUTE_ADDRESS after folding.
      if (MI->getOperand(2).getTargetFlags() == X86II::MO_GOT_ABSOLUTE_ADDRESS)
        return false;
      break;
    }
  }

  if (Ops.size() != 1)
    return false;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  unsigned NumOps = MI->getDesc().getNumOperands();
  bool isTwoAddr = NumOps > 1 &&
    MI->getDesc().getOperandConstraint(1, MCOI::TIED_TO) != -1;

  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  const DenseMap<unsigned, std::pair<unsigned,unsigned> > *OpcodeTablePtr = 0;
  if (isTwoAddr && NumOps >= 2 && OpNum < 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2Addr;
  } else if (OpNum == 0) { // If operand 0
    switch (Opc) {
    case X86::MOV8r0:
    case X86::MOV16r0:
    case X86::MOV32r0:
    case X86::MOV64r0: return true;
    default: break;
    }
    OpcodeTablePtr = &RegOp2MemOpTable0;
  } else if (OpNum == 1) {
    OpcodeTablePtr = &RegOp2MemOpTable1;
  } else if (OpNum == 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2;
  }

  if (OpcodeTablePtr && OpcodeTablePtr->count(Opc))
    return true;
  return TargetInstrInfoImpl::canFoldMemoryOperand(MI, Ops);
}

bool X86InstrInfo::unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                                unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                                SmallVectorImpl<MachineInstr*> &NewMIs) const {
  DenseMap<unsigned, std::pair<unsigned,unsigned> >::const_iterator I =
    MemOp2RegOpTable.find(MI->getOpcode());
  if (I == MemOp2RegOpTable.end())
    return false;
  unsigned Opc = I->second.first;
  unsigned Index = I->second.second & 0xf;
  bool FoldedLoad = I->second.second & (1 << 4);
  bool FoldedStore = I->second.second & (1 << 5);
  if (UnfoldLoad && !FoldedLoad)
    return false;
  UnfoldLoad &= FoldedLoad;
  if (UnfoldStore && !FoldedStore)
    return false;
  UnfoldStore &= FoldedStore;

  const MCInstrDesc &MCID = get(Opc);
  const TargetRegisterClass *RC = getRegClass(MCID, Index, &RI);
  if (!MI->hasOneMemOperand() &&
      RC == &X86::VR128RegClass &&
      !TM.getSubtarget<X86Subtarget>().isUnalignedMemAccessFast())
    // Without memoperands, loadRegFromAddr and storeRegToStackSlot will
    // conservatively assume the address is unaligned. That's bad for
    // performance.
    return false;
  SmallVector<MachineOperand, X86::AddrNumOperands> AddrOps;
  SmallVector<MachineOperand,2> BeforeOps;
  SmallVector<MachineOperand,2> AfterOps;
  SmallVector<MachineOperand,4> ImpOps;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &Op = MI->getOperand(i);
    if (i >= Index && i < Index + X86::AddrNumOperands)
      AddrOps.push_back(Op);
    else if (Op.isReg() && Op.isImplicit())
      ImpOps.push_back(Op);
    else if (i < Index)
      BeforeOps.push_back(Op);
    else if (i > Index)
      AfterOps.push_back(Op);
  }

  // Emit the load instruction.
  if (UnfoldLoad) {
    std::pair<MachineInstr::mmo_iterator,
              MachineInstr::mmo_iterator> MMOs =
      MF.extractLoadMemRefs(MI->memoperands_begin(),
                            MI->memoperands_end());
    loadRegFromAddr(MF, Reg, AddrOps, RC, MMOs.first, MMOs.second, NewMIs);
    if (UnfoldStore) {
      // Address operands cannot be marked isKill.
      for (unsigned i = 1; i != 1 + X86::AddrNumOperands; ++i) {
        MachineOperand &MO = NewMIs[0]->getOperand(i);
        if (MO.isReg())
          MO.setIsKill(false);
      }
    }
  }

  // Emit the data processing instruction.
  MachineInstr *DataMI = MF.CreateMachineInstr(MCID, MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(DataMI);

  if (FoldedStore)
    MIB.addReg(Reg, RegState::Define);
  for (unsigned i = 0, e = BeforeOps.size(); i != e; ++i)
    MIB.addOperand(BeforeOps[i]);
  if (FoldedLoad)
    MIB.addReg(Reg);
  for (unsigned i = 0, e = AfterOps.size(); i != e; ++i)
    MIB.addOperand(AfterOps[i]);
  for (unsigned i = 0, e = ImpOps.size(); i != e; ++i) {
    MachineOperand &MO = ImpOps[i];
    MIB.addReg(MO.getReg(),
               getDefRegState(MO.isDef()) |
               RegState::Implicit |
               getKillRegState(MO.isKill()) |
               getDeadRegState(MO.isDead()) |
               getUndefRegState(MO.isUndef()));
  }
  // Change CMP32ri r, 0 back to TEST32rr r, r, etc.
  unsigned NewOpc = 0;
  switch (DataMI->getOpcode()) {
  default: break;
  case X86::CMP64ri32:
  case X86::CMP64ri8:
  case X86::CMP32ri:
  case X86::CMP32ri8:
  case X86::CMP16ri:
  case X86::CMP16ri8:
  case X86::CMP8ri: {
    MachineOperand &MO0 = DataMI->getOperand(0);
    MachineOperand &MO1 = DataMI->getOperand(1);
    if (MO1.getImm() == 0) {
      switch (DataMI->getOpcode()) {
      default: break;
      case X86::CMP64ri8:
      case X86::CMP64ri32: NewOpc = X86::TEST64rr; break;
      case X86::CMP32ri8:
      case X86::CMP32ri:   NewOpc = X86::TEST32rr; break;
      case X86::CMP16ri8:
      case X86::CMP16ri:   NewOpc = X86::TEST16rr; break;
      case X86::CMP8ri:    NewOpc = X86::TEST8rr; break;
      }
      DataMI->setDesc(get(NewOpc));
      MO1.ChangeToRegister(MO0.getReg(), false);
    }
  }
  }
  NewMIs.push_back(DataMI);

  // Emit the store instruction.
  if (UnfoldStore) {
    const TargetRegisterClass *DstRC = getRegClass(MCID, 0, &RI);
    std::pair<MachineInstr::mmo_iterator,
              MachineInstr::mmo_iterator> MMOs =
      MF.extractStoreMemRefs(MI->memoperands_begin(),
                             MI->memoperands_end());
    storeRegToAddr(MF, Reg, true, AddrOps, DstRC, MMOs.first, MMOs.second, NewMIs);
  }

  return true;
}

bool
X86InstrInfo::unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                                  SmallVectorImpl<SDNode*> &NewNodes) const {
  if (!N->isMachineOpcode())
    return false;

  DenseMap<unsigned, std::pair<unsigned,unsigned> >::const_iterator I =
    MemOp2RegOpTable.find(N->getMachineOpcode());
  if (I == MemOp2RegOpTable.end())
    return false;
  unsigned Opc = I->second.first;
  unsigned Index = I->second.second & 0xf;
  bool FoldedLoad = I->second.second & (1 << 4);
  bool FoldedStore = I->second.second & (1 << 5);
  const MCInstrDesc &MCID = get(Opc);
  const TargetRegisterClass *RC = getRegClass(MCID, Index, &RI);
  unsigned NumDefs = MCID.NumDefs;
  std::vector<SDValue> AddrOps;
  std::vector<SDValue> BeforeOps;
  std::vector<SDValue> AfterOps;
  DebugLoc dl = N->getDebugLoc();
  unsigned NumOps = N->getNumOperands();
  for (unsigned i = 0; i != NumOps-1; ++i) {
    SDValue Op = N->getOperand(i);
    if (i >= Index-NumDefs && i < Index-NumDefs + X86::AddrNumOperands)
      AddrOps.push_back(Op);
    else if (i < Index-NumDefs)
      BeforeOps.push_back(Op);
    else if (i > Index-NumDefs)
      AfterOps.push_back(Op);
  }
  SDValue Chain = N->getOperand(NumOps-1);
  AddrOps.push_back(Chain);

  // Emit the load instruction.
  SDNode *Load = 0;
  MachineFunction &MF = DAG.getMachineFunction();
  if (FoldedLoad) {
    EVT VT = *RC->vt_begin();
    std::pair<MachineInstr::mmo_iterator,
              MachineInstr::mmo_iterator> MMOs =
      MF.extractLoadMemRefs(cast<MachineSDNode>(N)->memoperands_begin(),
                            cast<MachineSDNode>(N)->memoperands_end());
    if (!(*MMOs.first) &&
        RC == &X86::VR128RegClass &&
        !TM.getSubtarget<X86Subtarget>().isUnalignedMemAccessFast())
      // Do not introduce a slow unaligned load.
      return false;
    bool isAligned = (*MMOs.first) && (*MMOs.first)->getAlignment() >= 16;
    Load = DAG.getMachineNode(getLoadRegOpcode(0, RC, isAligned, TM), dl,
                              VT, MVT::Other, &AddrOps[0], AddrOps.size());
    NewNodes.push_back(Load);

    // Preserve memory reference information.
    cast<MachineSDNode>(Load)->setMemRefs(MMOs.first, MMOs.second);
  }

  // Emit the data processing instruction.
  std::vector<EVT> VTs;
  const TargetRegisterClass *DstRC = 0;
  if (MCID.getNumDefs() > 0) {
    DstRC = getRegClass(MCID, 0, &RI);
    VTs.push_back(*DstRC->vt_begin());
  }
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
    EVT VT = N->getValueType(i);
    if (VT != MVT::Other && i >= (unsigned)MCID.getNumDefs())
      VTs.push_back(VT);
  }
  if (Load)
    BeforeOps.push_back(SDValue(Load, 0));
  std::copy(AfterOps.begin(), AfterOps.end(), std::back_inserter(BeforeOps));
  SDNode *NewNode= DAG.getMachineNode(Opc, dl, VTs, &BeforeOps[0],
                                      BeforeOps.size());
  NewNodes.push_back(NewNode);

  // Emit the store instruction.
  if (FoldedStore) {
    AddrOps.pop_back();
    AddrOps.push_back(SDValue(NewNode, 0));
    AddrOps.push_back(Chain);
    std::pair<MachineInstr::mmo_iterator,
              MachineInstr::mmo_iterator> MMOs =
      MF.extractStoreMemRefs(cast<MachineSDNode>(N)->memoperands_begin(),
                             cast<MachineSDNode>(N)->memoperands_end());
    if (!(*MMOs.first) &&
        RC == &X86::VR128RegClass &&
        !TM.getSubtarget<X86Subtarget>().isUnalignedMemAccessFast())
      // Do not introduce a slow unaligned store.
      return false;
    bool isAligned = (*MMOs.first) && (*MMOs.first)->getAlignment() >= 16;
    SDNode *Store = DAG.getMachineNode(getStoreRegOpcode(0, DstRC,
                                                         isAligned, TM),
                                       dl, MVT::Other,
                                       &AddrOps[0], AddrOps.size());
    NewNodes.push_back(Store);

    // Preserve memory reference information.
    cast<MachineSDNode>(Load)->setMemRefs(MMOs.first, MMOs.second);
  }

  return true;
}

unsigned X86InstrInfo::getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore,
                                      unsigned *LoadRegIndex) const {
  DenseMap<unsigned, std::pair<unsigned,unsigned> >::const_iterator I =
    MemOp2RegOpTable.find(Opc);
  if (I == MemOp2RegOpTable.end())
    return 0;
  bool FoldedLoad = I->second.second & (1 << 4);
  bool FoldedStore = I->second.second & (1 << 5);
  if (UnfoldLoad && !FoldedLoad)
    return 0;
  if (UnfoldStore && !FoldedStore)
    return 0;
  if (LoadRegIndex)
    *LoadRegIndex = I->second.second & 0xf;
  return I->second.first;
}

bool
X86InstrInfo::areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                                     int64_t &Offset1, int64_t &Offset2) const {
  if (!Load1->isMachineOpcode() || !Load2->isMachineOpcode())
    return false;
  unsigned Opc1 = Load1->getMachineOpcode();
  unsigned Opc2 = Load2->getMachineOpcode();
  switch (Opc1) {
  default: return false;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV32rm:
  case X86::MOV64rm:
  case X86::LD_Fp32m:
  case X86::LD_Fp64m:
  case X86::LD_Fp80m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
  case X86::FsMOVAPSrm:
  case X86::FsMOVAPDrm:
  case X86::MOVAPSrm:
  case X86::MOVUPSrm:
  case X86::MOVAPDrm:
  case X86::MOVDQArm:
  case X86::MOVDQUrm:
    break;
  }
  switch (Opc2) {
  default: return false;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV32rm:
  case X86::MOV64rm:
  case X86::LD_Fp32m:
  case X86::LD_Fp64m:
  case X86::LD_Fp80m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
  case X86::FsMOVAPSrm:
  case X86::FsMOVAPDrm:
  case X86::MOVAPSrm:
  case X86::MOVUPSrm:
  case X86::MOVAPDrm:
  case X86::MOVDQArm:
  case X86::MOVDQUrm:
    break;
  }

  // Check if chain operands and base addresses match.
  if (Load1->getOperand(0) != Load2->getOperand(0) ||
      Load1->getOperand(5) != Load2->getOperand(5))
    return false;
  // Segment operands should match as well.
  if (Load1->getOperand(4) != Load2->getOperand(4))
    return false;
  // Scale should be 1, Index should be Reg0.
  if (Load1->getOperand(1) == Load2->getOperand(1) &&
      Load1->getOperand(2) == Load2->getOperand(2)) {
    if (cast<ConstantSDNode>(Load1->getOperand(1))->getZExtValue() != 1)
      return false;

    // Now let's examine the displacements.
    if (isa<ConstantSDNode>(Load1->getOperand(3)) &&
        isa<ConstantSDNode>(Load2->getOperand(3))) {
      Offset1 = cast<ConstantSDNode>(Load1->getOperand(3))->getSExtValue();
      Offset2 = cast<ConstantSDNode>(Load2->getOperand(3))->getSExtValue();
      return true;
    }
  }
  return false;
}

bool X86InstrInfo::shouldScheduleLoadsNear(SDNode *Load1, SDNode *Load2,
                                           int64_t Offset1, int64_t Offset2,
                                           unsigned NumLoads) const {
  assert(Offset2 > Offset1);
  if ((Offset2 - Offset1) / 8 > 64)
    return false;

  unsigned Opc1 = Load1->getMachineOpcode();
  unsigned Opc2 = Load2->getMachineOpcode();
  if (Opc1 != Opc2)
    return false;  // FIXME: overly conservative?

  switch (Opc1) {
  default: break;
  case X86::LD_Fp32m:
  case X86::LD_Fp64m:
  case X86::LD_Fp80m:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
    return false;
  }

  EVT VT = Load1->getValueType(0);
  switch (VT.getSimpleVT().SimpleTy) {
  default:
    // XMM registers. In 64-bit mode we can be a bit more aggressive since we
    // have 16 of them to play with.
    if (TM.getSubtargetImpl()->is64Bit()) {
      if (NumLoads >= 3)
        return false;
    } else if (NumLoads) {
      return false;
    }
    break;
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
  case MVT::f32:
  case MVT::f64:
    if (NumLoads)
      return false;
    break;
  }

  return true;
}


bool X86InstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid X86 branch condition!");
  X86::CondCode CC = static_cast<X86::CondCode>(Cond[0].getImm());
  if (CC == X86::COND_NE_OR_P || CC == X86::COND_NP_OR_E)
    return true;
  Cond[0].setImm(GetOppositeBranchCondition(CC));
  return false;
}

bool X86InstrInfo::
isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const {
  // FIXME: Return false for x87 stack register classes for now. We can't
  // allow any loads of these registers before FpGet_ST0_80.
  return !(RC == &X86::CCRRegClass || RC == &X86::RFP32RegClass ||
           RC == &X86::RFP64RegClass || RC == &X86::RFP80RegClass);
}


/// isX86_64ExtendedReg - Is the MachineOperand a x86-64 extended (r8 or higher)
/// register?  e.g. r8, xmm8, xmm13, etc.
bool X86InstrInfo::isX86_64ExtendedReg(unsigned RegNo) {
  switch (RegNo) {
  default: break;
  case X86::R8:    case X86::R9:    case X86::R10:   case X86::R11:
  case X86::R12:   case X86::R13:   case X86::R14:   case X86::R15:
  case X86::R8D:   case X86::R9D:   case X86::R10D:  case X86::R11D:
  case X86::R12D:  case X86::R13D:  case X86::R14D:  case X86::R15D:
  case X86::R8W:   case X86::R9W:   case X86::R10W:  case X86::R11W:
  case X86::R12W:  case X86::R13W:  case X86::R14W:  case X86::R15W:
  case X86::R8B:   case X86::R9B:   case X86::R10B:  case X86::R11B:
  case X86::R12B:  case X86::R13B:  case X86::R14B:  case X86::R15B:
  case X86::XMM8:  case X86::XMM9:  case X86::XMM10: case X86::XMM11:
  case X86::XMM12: case X86::XMM13: case X86::XMM14: case X86::XMM15:
  case X86::YMM8:  case X86::YMM9:  case X86::YMM10: case X86::YMM11:
  case X86::YMM12: case X86::YMM13: case X86::YMM14: case X86::YMM15:
  case X86::CR8:   case X86::CR9:   case X86::CR10:  case X86::CR11:
  case X86::CR12:  case X86::CR13:  case X86::CR14:  case X86::CR15:
    return true;
  }
  return false;
}

/// getGlobalBaseReg - Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
/// TODO: Eliminate this and move the code to X86MachineFunctionInfo.
///
unsigned X86InstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  assert(!TM.getSubtarget<X86Subtarget>().is64Bit() &&
         "X86-64 PIC uses RIP relative addressing");

  X86MachineFunctionInfo *X86FI = MF->getInfo<X86MachineFunctionInfo>();
  unsigned GlobalBaseReg = X86FI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // Create the register. The code to initialize it is inserted
  // later, by the CGBR pass (below).
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  GlobalBaseReg = RegInfo.createVirtualRegister(X86::GR32RegisterClass);
  X86FI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}

// These are the replaceable SSE instructions. Some of these have Int variants
// that we don't include here. We don't want to replace instructions selected
// by intrinsics.
static const unsigned ReplaceableInstrs[][3] = {
  //PackedSingle     PackedDouble    PackedInt
  { X86::MOVAPSmr,   X86::MOVAPDmr,  X86::MOVDQAmr  },
  { X86::MOVAPSrm,   X86::MOVAPDrm,  X86::MOVDQArm  },
  { X86::MOVAPSrr,   X86::MOVAPDrr,  X86::MOVDQArr  },
  { X86::MOVUPSmr,   X86::MOVUPDmr,  X86::MOVDQUmr  },
  { X86::MOVUPSrm,   X86::MOVUPDrm,  X86::MOVDQUrm  },
  { X86::MOVNTPSmr,  X86::MOVNTPDmr, X86::MOVNTDQmr },
  { X86::ANDNPSrm,   X86::ANDNPDrm,  X86::PANDNrm   },
  { X86::ANDNPSrr,   X86::ANDNPDrr,  X86::PANDNrr   },
  { X86::ANDPSrm,    X86::ANDPDrm,   X86::PANDrm    },
  { X86::ANDPSrr,    X86::ANDPDrr,   X86::PANDrr    },
  { X86::ORPSrm,     X86::ORPDrm,    X86::PORrm     },
  { X86::ORPSrr,     X86::ORPDrr,    X86::PORrr     },
  { X86::V_SET0PS,   X86::V_SET0PD,  X86::V_SET0PI  },
  { X86::XORPSrm,    X86::XORPDrm,   X86::PXORrm    },
  { X86::XORPSrr,    X86::XORPDrr,   X86::PXORrr    },
  // AVX 128-bit support
  { X86::VMOVAPSmr,  X86::VMOVAPDmr,  X86::VMOVDQAmr  },
  { X86::VMOVAPSrm,  X86::VMOVAPDrm,  X86::VMOVDQArm  },
  { X86::VMOVAPSrr,  X86::VMOVAPDrr,  X86::VMOVDQArr  },
  { X86::VMOVUPSmr,  X86::VMOVUPDmr,  X86::VMOVDQUmr  },
  { X86::VMOVUPSrm,  X86::VMOVUPDrm,  X86::VMOVDQUrm  },
  { X86::VMOVNTPSmr, X86::VMOVNTPDmr, X86::VMOVNTDQmr },
  { X86::VANDNPSrm,  X86::VANDNPDrm,  X86::VPANDNrm   },
  { X86::VANDNPSrr,  X86::VANDNPDrr,  X86::VPANDNrr   },
  { X86::VANDPSrm,   X86::VANDPDrm,   X86::VPANDrm    },
  { X86::VANDPSrr,   X86::VANDPDrr,   X86::VPANDrr    },
  { X86::VORPSrm,    X86::VORPDrm,    X86::VPORrm     },
  { X86::VORPSrr,    X86::VORPDrr,    X86::VPORrr     },
  { X86::AVX_SET0PS, X86::AVX_SET0PD, X86::AVX_SET0PI },
  { X86::VXORPSrm,   X86::VXORPDrm,   X86::VPXORrm    },
  { X86::VXORPSrr,   X86::VXORPDrr,   X86::VPXORrr    },
};

// FIXME: Some shuffle and unpack instructions have equivalents in different
// domains, but they require a bit more work than just switching opcodes.

static const unsigned *lookup(unsigned opcode, unsigned domain) {
  for (unsigned i = 0, e = array_lengthof(ReplaceableInstrs); i != e; ++i)
    if (ReplaceableInstrs[i][domain-1] == opcode)
      return ReplaceableInstrs[i];
  return 0;
}

std::pair<uint16_t, uint16_t>
X86InstrInfo::GetSSEDomain(const MachineInstr *MI) const {
  uint16_t domain = (MI->getDesc().TSFlags >> X86II::SSEDomainShift) & 3;
  return std::make_pair(domain,
                        domain && lookup(MI->getOpcode(), domain) ? 0xe : 0);
}

void X86InstrInfo::SetSSEDomain(MachineInstr *MI, unsigned Domain) const {
  assert(Domain>0 && Domain<4 && "Invalid execution domain");
  uint16_t dom = (MI->getDesc().TSFlags >> X86II::SSEDomainShift) & 3;
  assert(dom && "Not an SSE instruction");
  const unsigned *table = lookup(MI->getOpcode(), dom);
  assert(table && "Cannot change domain");
  MI->setDesc(get(table[Domain-1]));
}

/// getNoopForMachoTarget - Return the noop instruction to use for a noop.
void X86InstrInfo::getNoopForMachoTarget(MCInst &NopInst) const {
  NopInst.setOpcode(X86::NOOP);
}

bool X86InstrInfo::isHighLatencyDef(int opc) const {
  switch (opc) {
  default: return false;
  case X86::DIVSDrm:
  case X86::DIVSDrm_Int:
  case X86::DIVSDrr:
  case X86::DIVSDrr_Int:
  case X86::DIVSSrm:
  case X86::DIVSSrm_Int:
  case X86::DIVSSrr:
  case X86::DIVSSrr_Int:
  case X86::SQRTPDm:
  case X86::SQRTPDm_Int:
  case X86::SQRTPDr:
  case X86::SQRTPDr_Int:
  case X86::SQRTPSm:
  case X86::SQRTPSm_Int:
  case X86::SQRTPSr:
  case X86::SQRTPSr_Int:
  case X86::SQRTSDm:
  case X86::SQRTSDm_Int:
  case X86::SQRTSDr:
  case X86::SQRTSDr_Int:
  case X86::SQRTSSm:
  case X86::SQRTSSm_Int:
  case X86::SQRTSSr:
  case X86::SQRTSSr_Int:
    return true;
  }
}

bool X86InstrInfo::
hasHighOperandLatency(const InstrItineraryData *ItinData,
                      const MachineRegisterInfo *MRI,
                      const MachineInstr *DefMI, unsigned DefIdx,
                      const MachineInstr *UseMI, unsigned UseIdx) const {
  return isHighLatencyDef(DefMI->getOpcode());
}

namespace {
  /// CGBR - Create Global Base Reg pass. This initializes the PIC
  /// global base register for x86-32.
  struct CGBR : public MachineFunctionPass {
    static char ID;
    CGBR() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      const X86TargetMachine *TM =
        static_cast<const X86TargetMachine *>(&MF.getTarget());

      assert(!TM->getSubtarget<X86Subtarget>().is64Bit() &&
             "X86-64 PIC uses RIP relative addressing");

      // Only emit a global base reg in PIC mode.
      if (TM->getRelocationModel() != Reloc::PIC_)
        return false;

      X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
      unsigned GlobalBaseReg = X86FI->getGlobalBaseReg();

      // If we didn't need a GlobalBaseReg, don't insert code.
      if (GlobalBaseReg == 0)
        return false;

      // Insert the set of GlobalBaseReg into the first MBB of the function
      MachineBasicBlock &FirstMBB = MF.front();
      MachineBasicBlock::iterator MBBI = FirstMBB.begin();
      DebugLoc DL = FirstMBB.findDebugLoc(MBBI);
      MachineRegisterInfo &RegInfo = MF.getRegInfo();
      const X86InstrInfo *TII = TM->getInstrInfo();

      unsigned PC;
      if (TM->getSubtarget<X86Subtarget>().isPICStyleGOT())
        PC = RegInfo.createVirtualRegister(X86::GR32RegisterClass);
      else
        PC = GlobalBaseReg;

      // Operand of MovePCtoStack is completely ignored by asm printer. It's
      // only used in JIT code emission as displacement to pc.
      BuildMI(FirstMBB, MBBI, DL, TII->get(X86::MOVPC32r), PC).addImm(0);

      // If we're using vanilla 'GOT' PIC style, we should use relative addressing
      // not to pc, but to _GLOBAL_OFFSET_TABLE_ external.
      if (TM->getSubtarget<X86Subtarget>().isPICStyleGOT()) {
        // Generate addl $__GLOBAL_OFFSET_TABLE_ + [.-piclabel], %some_register
        BuildMI(FirstMBB, MBBI, DL, TII->get(X86::ADD32ri), GlobalBaseReg)
          .addReg(PC).addExternalSymbol("_GLOBAL_OFFSET_TABLE_",
                                        X86II::MO_GOT_ABSOLUTE_ADDRESS);
      }

      return true;
    }

    virtual const char *getPassName() const {
      return "X86 PIC Global Base Reg Initialization";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };
}

char CGBR::ID = 0;
FunctionPass*
llvm::createGlobalBaseRegPass() { return new CGBR(); }

//===-- PPCMCCodeEmitter.cpp - Convert PPC code to machine code -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPCMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mccodeemitter"
#include "PPC.h"
#include "PPCRegisterInfo.h"
#include "PPCFixupKinds.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {
class PPCMCCodeEmitter : public MCCodeEmitter {
  PPCMCCodeEmitter(const PPCMCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const PPCMCCodeEmitter &);   // DO NOT IMPLEMENT
  const TargetMachine &TM;
  MCContext &Ctx;
  
public:
  PPCMCCodeEmitter(TargetMachine &tm, MCContext &ctx)
    : TM(tm), Ctx(ctx) {
  }
  
  ~PPCMCCodeEmitter() {}
  
  unsigned getNumFixupKinds() const { return PPC::NumTargetFixupKinds; }
  
  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    const static MCFixupKindInfo Infos[] = {
      // name                     offset  bits  flags
      { "fixup_ppc_br24",         6,      24,   MCFixupKindInfo::FKF_IsPCRel }
#if 0
      { "fixup_arm_vfp_pcrel_12", 3,      8,    MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_arm_branch",       1,      24,   MCFixupKindInfo::FKF_IsPCRel },
#endif
    };
    
    if (Kind < FirstTargetFixupKind)
      return MCCodeEmitter::getFixupKindInfo(Kind);
    
    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }

  unsigned getDirectBrEncoding(const MCInst &MI, unsigned OpNo,
                               SmallVectorImpl<MCFixup> &Fixups) const;

  unsigned getCondBrEncoding(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups) const;

  unsigned get_crbitm_encoding(const MCInst &MI, unsigned OpNo,
                               SmallVectorImpl<MCFixup> &Fixups) const;

  /// getMachineOpValue - Return binary encoding of operand. If the machine
  /// operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI,const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups) const;
  
  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  unsigned getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups) const;
  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const {
    unsigned Bits = getBinaryCodeForInstr(MI, Fixups);
    
    // Output the constant in big endian byte order.
    for (unsigned i = 0; i != 4; ++i) {
      OS << (char)(Bits >> 24);
      Bits <<= 8;
    }
    
    ++MCNumEmitted;  // Keep track of the # of mi's emitted.
  }
  
};
  
} // end anonymous namespace
  
MCCodeEmitter *llvm::createPPCMCCodeEmitter(const Target &, TargetMachine &TM,
                                            MCContext &Ctx) {
  return new PPCMCCodeEmitter(TM, Ctx);
}

unsigned PPCMCCodeEmitter::
getDirectBrEncoding(const MCInst &MI, unsigned OpNo,
                    SmallVectorImpl<MCFixup> &Fixups) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isReg() || MO.isImm()) return getMachineOpValue(MI, MO, Fixups);
  
  // Add a fixup for the branch target.
  Fixups.push_back(MCFixup::Create(0, MO.getExpr(),
                                   (MCFixupKind)PPC::fixup_ppc_br24));
  return 0;
}

unsigned PPCMCCodeEmitter::getCondBrEncoding(const MCInst &MI, unsigned OpNo,
                                     SmallVectorImpl<MCFixup> &Fixups) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isReg() || MO.isImm()) return getMachineOpValue(MI, MO, Fixups);

  
  
  return 0;
}


unsigned PPCMCCodeEmitter::
get_crbitm_encoding(const MCInst &MI, unsigned OpNo,
                    SmallVectorImpl<MCFixup> &Fixups) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  assert((MI.getOpcode() == PPC::MTCRF || MI.getOpcode() == PPC::MFOCRF) &&
         (MO.getReg() >= PPC::CR0 && MO.getReg() <= PPC::CR7));
  return 0x80 >> PPCRegisterInfo::getRegisterNumbering(MO.getReg());
}


unsigned PPCMCCodeEmitter::
getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  if (MO.isReg()) {
    assert(MI.getOpcode() != PPC::MTCRF && MI.getOpcode() != PPC::MFOCRF);
    return PPCRegisterInfo::getRegisterNumbering(MO.getReg());
  }
  
  if (MO.isImm())
    return MO.getImm();
  
  // FIXME.
  return 0;
}


#include "PPCGenMCCodeEmitter.inc"

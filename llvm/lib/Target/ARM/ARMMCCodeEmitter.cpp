//===-- ARM/ARMMCCodeEmitter.cpp - Convert ARM code to machine code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARMMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-emitter"
#include "ARM.h"
#include "ARMInstrInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
class ARMMCCodeEmitter : public MCCodeEmitter {
  ARMMCCodeEmitter(const ARMMCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const ARMMCCodeEmitter &); // DO NOT IMPLEMENT
  const TargetMachine &TM;
  const TargetInstrInfo &TII;
  MCContext &Ctx;

public:
  ARMMCCodeEmitter(TargetMachine &tm, MCContext &ctx)
    : TM(tm), TII(*TM.getInstrInfo()), Ctx(ctx) {
    assert(0 && "ARMMCCodeEmitter::ARMMCCodeEmitter() not yet implemented.");
  }

  ~ARMMCCodeEmitter() {}

  unsigned getNumFixupKinds() const {
    assert(0 && "ARMMCCodeEmitter::getNumFixupKinds() not yet implemented.");
    return 0;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    static MCFixupKindInfo rtn;
    assert(0 && "ARMMCCodeEmitter::getFixupKindInfo() not yet implemented.");
    return rtn;
  }

  static unsigned GetARMRegNum(const MCOperand &MO) {
    // FIXME: getARMRegisterNumbering() is sufficient?
    assert(0 && "ARMMCCodeEmitter::GetARMRegNum() not yet implemented.");
    return 0;
  }

  void EmitByte(unsigned char C, unsigned &CurByte, raw_ostream &OS) const {
    OS << (char)C;
    ++CurByte;
  }

  void EmitConstant(uint64_t Val, unsigned Size, unsigned &CurByte,
                    raw_ostream &OS) const {
    // Output the constant in little endian byte order.
    for (unsigned i = 0; i != Size; ++i) {
      EmitByte(Val & 255, CurByte, OS);
      Val >>= 8;
    }
  }

  void EmitImmediate(const MCOperand &Disp,
                     unsigned ImmSize, MCFixupKind FixupKind,
                     unsigned &CurByte, raw_ostream &OS,
                     SmallVectorImpl<MCFixup> &Fixups,
                     int ImmOffset = 0) const;

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;

  void EmitOpcodePrefix(uint64_t TSFlags, unsigned &CurByte, int MemOperand,
                        const MCInst &MI, const TargetInstrDesc &Desc,
                        raw_ostream &OS) const;
};

} // end anonymous namespace


MCCodeEmitter *llvm::createARMMCCodeEmitter(const Target &,
                                             TargetMachine &TM,
                                             MCContext &Ctx) {
  return new ARMMCCodeEmitter(TM, Ctx);
}

void ARMMCCodeEmitter::
EmitImmediate(const MCOperand &DispOp, unsigned Size, MCFixupKind FixupKind,
              unsigned &CurByte, raw_ostream &OS,
              SmallVectorImpl<MCFixup> &Fixups, int ImmOffset) const {
  assert(0 && "ARMMCCodeEmitter::EmitImmediate() not yet implemented.");
}

/// EmitOpcodePrefix - Emit all instruction prefixes prior to the opcode.
///
/// MemOperand is the operand # of the start of a memory operand if present.  If
/// Not present, it is -1.
void ARMMCCodeEmitter::EmitOpcodePrefix(uint64_t TSFlags, unsigned &CurByte,
                                        int MemOperand, const MCInst &MI,
                                        const TargetInstrDesc &Desc,
                                        raw_ostream &OS) const {
  assert(0 && "ARMMCCodeEmitter::EmitOpcodePrefix() not yet implemented.");
}

void ARMMCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  assert(0 && "ARMMCCodeEmitter::EncodeInstruction() not yet implemented.");
}

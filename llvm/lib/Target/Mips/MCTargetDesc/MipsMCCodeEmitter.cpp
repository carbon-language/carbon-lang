//===-- MipsMCCodeEmitter.cpp - Convert Mips Code to Machine Code ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MipsMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//
//
#define DEBUG_TYPE "mccodeemitter"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MCTargetDesc/MipsFixupKinds.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"

#define GET_INSTRMAP_INFO
#include "MipsGenInstrInfo.inc"

using namespace llvm;

namespace {
class MipsMCCodeEmitter : public MCCodeEmitter {
  MipsMCCodeEmitter(const MipsMCCodeEmitter &) LLVM_DELETED_FUNCTION;
  void operator=(const MipsMCCodeEmitter &) LLVM_DELETED_FUNCTION;
  const MCInstrInfo &MCII;
  MCContext &Ctx;
  const MCSubtargetInfo &STI;
  bool IsLittleEndian;
  bool IsMicroMips;

public:
  MipsMCCodeEmitter(const MCInstrInfo &mcii, MCContext &Ctx_,
                    const MCSubtargetInfo &sti, bool IsLittle) :
    MCII(mcii), Ctx(Ctx_), STI (sti), IsLittleEndian(IsLittle) {
      IsMicroMips = STI.getFeatureBits() & Mips::FeatureMicroMips;
    }

  ~MipsMCCodeEmitter() {}

  void EmitByte(unsigned char C, raw_ostream &OS) const {
    OS << (char)C;
  }

  void EmitInstruction(uint64_t Val, unsigned Size, raw_ostream &OS) const {
    // Output the instruction encoding in little endian byte order.
    // Little-endian byte ordering:
    //   mips32r2:   4 | 3 | 2 | 1
    //   microMIPS:  2 | 1 | 4 | 3
    if (IsLittleEndian && Size == 4 && IsMicroMips) {
      EmitInstruction(Val>>16, 2, OS);
      EmitInstruction(Val, 2, OS);
    } else {
      for (unsigned i = 0; i < Size; ++i) {
        unsigned Shift = IsLittleEndian ? i * 8 : (Size - 1 - i) * 8;
        EmitByte((Val >> Shift) & 0xff, OS);
      }
    }
  }

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups) const;

  // getBranchJumpOpValue - Return binary encoding of the jump
  // target operand. If the machine operand requires relocation,
  // record the relocation and return zero.
   unsigned getJumpTargetOpValue(const MCInst &MI, unsigned OpNo,
                                 SmallVectorImpl<MCFixup> &Fixups) const;

  // getBranchJumpOpValueMM - Return binary encoding of the microMIPS jump
  // target operand. If the machine operand requires relocation,
  // record the relocation and return zero.
  unsigned getJumpTargetOpValueMM(const MCInst &MI, unsigned OpNo,
                                  SmallVectorImpl<MCFixup> &Fixups) const;

   // getBranchTargetOpValue - Return binary encoding of the branch
   // target operand. If the machine operand requires relocation,
   // record the relocation and return zero.
  unsigned getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                                  SmallVectorImpl<MCFixup> &Fixups) const;

  // getBranchTargetOpValue - Return binary encoding of the microMIPS branch
  // target operand. If the machine operand requires relocation,
  // record the relocation and return zero.
  unsigned getBranchTargetOpValueMM(const MCInst &MI, unsigned OpNo,
                                    SmallVectorImpl<MCFixup> &Fixups) const;

   // getMachineOpValue - Return binary encoding of operand. If the machin
   // operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI,const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups) const;

  unsigned getMSAMemEncoding(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups) const;

  unsigned getMemEncoding(const MCInst &MI, unsigned OpNo,
                          SmallVectorImpl<MCFixup> &Fixups) const;
  unsigned getMemEncodingMMImm12(const MCInst &MI, unsigned OpNo,
                                 SmallVectorImpl<MCFixup> &Fixups) const;
  unsigned getSizeExtEncoding(const MCInst &MI, unsigned OpNo,
                              SmallVectorImpl<MCFixup> &Fixups) const;
  unsigned getSizeInsEncoding(const MCInst &MI, unsigned OpNo,
                              SmallVectorImpl<MCFixup> &Fixups) const;

  // getLSAImmEncoding - Return binary encoding of LSA immediate.
  unsigned getLSAImmEncoding(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups) const;

  unsigned
  getExprOpValue(const MCExpr *Expr,SmallVectorImpl<MCFixup> &Fixups) const;

}; // class MipsMCCodeEmitter
}  // namespace

MCCodeEmitter *llvm::createMipsMCCodeEmitterEB(const MCInstrInfo &MCII,
                                               const MCRegisterInfo &MRI,
                                               const MCSubtargetInfo &STI,
                                               MCContext &Ctx)
{
  return new MipsMCCodeEmitter(MCII, Ctx, STI, false);
}

MCCodeEmitter *llvm::createMipsMCCodeEmitterEL(const MCInstrInfo &MCII,
                                               const MCRegisterInfo &MRI,
                                               const MCSubtargetInfo &STI,
                                               MCContext &Ctx)
{
  return new MipsMCCodeEmitter(MCII, Ctx, STI, true);
}


// If the D<shift> instruction has a shift amount that is greater
// than 31 (checked in calling routine), lower it to a D<shift>32 instruction
static void LowerLargeShift(MCInst& Inst) {

  assert(Inst.getNumOperands() == 3 && "Invalid no. of operands for shift!");
  assert(Inst.getOperand(2).isImm());

  int64_t Shift = Inst.getOperand(2).getImm();
  if (Shift <= 31)
    return; // Do nothing
  Shift -= 32;

  // saminus32
  Inst.getOperand(2).setImm(Shift);

  switch (Inst.getOpcode()) {
  default:
    // Calling function is not synchronized
    llvm_unreachable("Unexpected shift instruction");
  case Mips::DSLL:
    Inst.setOpcode(Mips::DSLL32);
    return;
  case Mips::DSRL:
    Inst.setOpcode(Mips::DSRL32);
    return;
  case Mips::DSRA:
    Inst.setOpcode(Mips::DSRA32);
    return;
  case Mips::DROTR:
    Inst.setOpcode(Mips::DROTR32);
    return;
  }
}

// Pick a DEXT or DINS instruction variant based on the pos and size operands
static void LowerDextDins(MCInst& InstIn) {
  int Opcode = InstIn.getOpcode();

  if (Opcode == Mips::DEXT)
    assert(InstIn.getNumOperands() == 4 &&
           "Invalid no. of machine operands for DEXT!");
  else // Only DEXT and DINS are possible
    assert(InstIn.getNumOperands() == 5 &&
           "Invalid no. of machine operands for DINS!");

  assert(InstIn.getOperand(2).isImm());
  int64_t pos = InstIn.getOperand(2).getImm();
  assert(InstIn.getOperand(3).isImm());
  int64_t size = InstIn.getOperand(3).getImm();

  if (size <= 32) {
    if (pos < 32)  // DEXT/DINS, do nothing
      return;
    // DEXTU/DINSU
    InstIn.getOperand(2).setImm(pos - 32);
    InstIn.setOpcode((Opcode == Mips::DEXT) ? Mips::DEXTU : Mips::DINSU);
    return;
  }
  // DEXTM/DINSM
  assert(pos < 32 && "DEXT/DINS cannot have both size and pos > 32");
  InstIn.getOperand(3).setImm(size - 32);
  InstIn.setOpcode((Opcode == Mips::DEXT) ? Mips::DEXTM : Mips::DINSM);
  return;
}

/// EncodeInstruction - Emit the instruction.
/// Size the instruction with Desc.getSize().
void MipsMCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const
{

  // Non-pseudo instructions that get changed for direct object
  // only based on operand values.
  // If this list of instructions get much longer we will move
  // the check to a function call. Until then, this is more efficient.
  MCInst TmpInst = MI;
  switch (MI.getOpcode()) {
  // If shift amount is >= 32 it the inst needs to be lowered further
  case Mips::DSLL:
  case Mips::DSRL:
  case Mips::DSRA:
  case Mips::DROTR:
    LowerLargeShift(TmpInst);
    break;
    // Double extract instruction is chosen by pos and size operands
  case Mips::DEXT:
  case Mips::DINS:
    LowerDextDins(TmpInst);
  }

  unsigned long N = Fixups.size();
  uint32_t Binary = getBinaryCodeForInstr(TmpInst, Fixups);

  // Check for unimplemented opcodes.
  // Unfortunately in MIPS both NOP and SLL will come in with Binary == 0
  // so we have to special check for them.
  unsigned Opcode = TmpInst.getOpcode();
  if ((Opcode != Mips::NOP) && (Opcode != Mips::SLL) && !Binary)
    llvm_unreachable("unimplemented opcode in EncodeInstruction()");

  if (STI.getFeatureBits() & Mips::FeatureMicroMips) {
    int NewOpcode = Mips::Std2MicroMips (Opcode, Mips::Arch_micromips);
    if (NewOpcode != -1) {
      if (Fixups.size() > N)
        Fixups.pop_back();
      Opcode = NewOpcode;
      TmpInst.setOpcode (NewOpcode);
      Binary = getBinaryCodeForInstr(TmpInst, Fixups);
    }
  }

  const MCInstrDesc &Desc = MCII.get(TmpInst.getOpcode());

  // Get byte count of instruction
  unsigned Size = Desc.getSize();
  if (!Size)
    llvm_unreachable("Desc.getSize() returns 0");

  EmitInstruction(Binary, Size, OS);
}

/// getBranchTargetOpValue - Return binary encoding of the branch
/// target operand. If the machine operand requires relocation,
/// record the relocation and return zero.
unsigned MipsMCCodeEmitter::
getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                       SmallVectorImpl<MCFixup> &Fixups) const {

  const MCOperand &MO = MI.getOperand(OpNo);

  // If the destination is an immediate, divide by 4.
  if (MO.isImm()) return MO.getImm() >> 2;

  assert(MO.isExpr() &&
         "getBranchTargetOpValue expects only expressions or immediates");

  const MCExpr *Expr = MO.getExpr();
  Fixups.push_back(MCFixup::Create(0, Expr,
                                   MCFixupKind(Mips::fixup_Mips_PC16)));
  return 0;
}

/// getBranchTargetOpValue - Return binary encoding of the microMIPS branch
/// target operand. If the machine operand requires relocation,
/// record the relocation and return zero.
unsigned MipsMCCodeEmitter::
getBranchTargetOpValueMM(const MCInst &MI, unsigned OpNo,
                         SmallVectorImpl<MCFixup> &Fixups) const {

  const MCOperand &MO = MI.getOperand(OpNo);

  // If the destination is an immediate, divide by 2.
  if (MO.isImm()) return MO.getImm() >> 1;

  assert(MO.isExpr() &&
         "getBranchTargetOpValueMM expects only expressions or immediates");

  const MCExpr *Expr = MO.getExpr();
  Fixups.push_back(MCFixup::Create(0, Expr,
                   MCFixupKind(Mips::
                               fixup_MICROMIPS_PC16_S1)));
  return 0;
}

/// getJumpTargetOpValue - Return binary encoding of the jump
/// target operand. If the machine operand requires relocation,
/// record the relocation and return zero.
unsigned MipsMCCodeEmitter::
getJumpTargetOpValue(const MCInst &MI, unsigned OpNo,
                     SmallVectorImpl<MCFixup> &Fixups) const {

  const MCOperand &MO = MI.getOperand(OpNo);
  // If the destination is an immediate, divide by 4.
  if (MO.isImm()) return MO.getImm()>>2;

  assert(MO.isExpr() &&
         "getJumpTargetOpValue expects only expressions or an immediate");

  const MCExpr *Expr = MO.getExpr();
  Fixups.push_back(MCFixup::Create(0, Expr,
                                   MCFixupKind(Mips::fixup_Mips_26)));
  return 0;
}

unsigned MipsMCCodeEmitter::
getJumpTargetOpValueMM(const MCInst &MI, unsigned OpNo,
                       SmallVectorImpl<MCFixup> &Fixups) const {

  const MCOperand &MO = MI.getOperand(OpNo);
  // If the destination is an immediate, divide by 2.
  if (MO.isImm()) return MO.getImm() >> 1;

  assert(MO.isExpr() &&
         "getJumpTargetOpValueMM expects only expressions or an immediate");

  const MCExpr *Expr = MO.getExpr();
  Fixups.push_back(MCFixup::Create(0, Expr,
                                   MCFixupKind(Mips::fixup_MICROMIPS_26_S1)));
  return 0;
}

unsigned MipsMCCodeEmitter::
getExprOpValue(const MCExpr *Expr,SmallVectorImpl<MCFixup> &Fixups) const {
  int64_t Res;

  if (Expr->EvaluateAsAbsolute(Res))
    return Res;

  MCExpr::ExprKind Kind = Expr->getKind();
  if (Kind == MCExpr::Constant) {
    return cast<MCConstantExpr>(Expr)->getValue();
  }

  if (Kind == MCExpr::Binary) {
    unsigned Res = getExprOpValue(cast<MCBinaryExpr>(Expr)->getLHS(), Fixups);
    Res += getExprOpValue(cast<MCBinaryExpr>(Expr)->getRHS(), Fixups);
    return Res;
  }
  if (Kind == MCExpr::SymbolRef) {
  Mips::Fixups FixupKind = Mips::Fixups(0);

  switch(cast<MCSymbolRefExpr>(Expr)->getKind()) {
  default: llvm_unreachable("Unknown fixup kind!");
    break;
  case MCSymbolRefExpr::VK_Mips_GPOFF_HI :
    FixupKind = Mips::fixup_Mips_GPOFF_HI;
    break;
  case MCSymbolRefExpr::VK_Mips_GPOFF_LO :
    FixupKind = Mips::fixup_Mips_GPOFF_LO;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_PAGE :
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_GOT_PAGE
                            : Mips::fixup_Mips_GOT_PAGE;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_OFST :
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_GOT_OFST
                            : Mips::fixup_Mips_GOT_OFST;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_DISP :
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_GOT_DISP
                            : Mips::fixup_Mips_GOT_DISP;
    break;
  case MCSymbolRefExpr::VK_Mips_GPREL:
    FixupKind = Mips::fixup_Mips_GPREL16;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_CALL:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_CALL16
                            : Mips::fixup_Mips_CALL16;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT16:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_GOT16
                            : Mips::fixup_Mips_GOT_Global;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_GOT16
                            : Mips::fixup_Mips_GOT_Local;
    break;
  case MCSymbolRefExpr::VK_Mips_ABS_HI:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_HI16
                            : Mips::fixup_Mips_HI16;
    break;
  case MCSymbolRefExpr::VK_Mips_ABS_LO:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_LO16
                            : Mips::fixup_Mips_LO16;
    break;
  case MCSymbolRefExpr::VK_Mips_TLSGD:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_TLS_GD
                            : Mips::fixup_Mips_TLSGD;
    break;
  case MCSymbolRefExpr::VK_Mips_TLSLDM:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_TLS_LDM
                            : Mips::fixup_Mips_TLSLDM;
    break;
  case MCSymbolRefExpr::VK_Mips_DTPREL_HI:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_TLS_DTPREL_HI16
                            : Mips::fixup_Mips_DTPREL_HI;
    break;
  case MCSymbolRefExpr::VK_Mips_DTPREL_LO:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_TLS_DTPREL_LO16
                            : Mips::fixup_Mips_DTPREL_LO;
    break;
  case MCSymbolRefExpr::VK_Mips_GOTTPREL:
    FixupKind = Mips::fixup_Mips_GOTTPREL;
    break;
  case MCSymbolRefExpr::VK_Mips_TPREL_HI:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_TLS_TPREL_HI16
                            : Mips::fixup_Mips_TPREL_HI;
    break;
  case MCSymbolRefExpr::VK_Mips_TPREL_LO:
    FixupKind = IsMicroMips ? Mips::fixup_MICROMIPS_TLS_TPREL_LO16
                            : Mips::fixup_Mips_TPREL_LO;
    break;
  case MCSymbolRefExpr::VK_Mips_HIGHER:
    FixupKind = Mips::fixup_Mips_HIGHER;
    break;
  case MCSymbolRefExpr::VK_Mips_HIGHEST:
    FixupKind = Mips::fixup_Mips_HIGHEST;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_HI16:
    FixupKind = Mips::fixup_Mips_GOT_HI16;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_LO16:
    FixupKind = Mips::fixup_Mips_GOT_LO16;
    break;
  case MCSymbolRefExpr::VK_Mips_CALL_HI16:
    FixupKind = Mips::fixup_Mips_CALL_HI16;
    break;
  case MCSymbolRefExpr::VK_Mips_CALL_LO16:
    FixupKind = Mips::fixup_Mips_CALL_LO16;
    break;
  } // switch

    Fixups.push_back(MCFixup::Create(0, Expr, MCFixupKind(FixupKind)));
    return 0;
  }
  return 0;
}

/// getMachineOpValue - Return binary encoding of operand. If the machine
/// operand requires relocation, record the relocation and return zero.
unsigned MipsMCCodeEmitter::
getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  if (MO.isReg()) {
    unsigned Reg = MO.getReg();
    unsigned RegNo = Ctx.getRegisterInfo()->getEncodingValue(Reg);
    return RegNo;
  } else if (MO.isImm()) {
    return static_cast<unsigned>(MO.getImm());
  } else if (MO.isFPImm()) {
    return static_cast<unsigned>(APFloat(MO.getFPImm())
        .bitcastToAPInt().getHiBits(32).getLimitedValue());
  }
  // MO must be an Expr.
  assert(MO.isExpr());
  return getExprOpValue(MO.getExpr(),Fixups);
}

/// getMSAMemEncoding - Return binary encoding of memory operand for LD/ST
/// instructions.
unsigned
MipsMCCodeEmitter::getMSAMemEncoding(const MCInst &MI, unsigned OpNo,
                                     SmallVectorImpl<MCFixup> &Fixups) const {
  // Base register is encoded in bits 20-16, offset is encoded in bits 15-0.
  assert(MI.getOperand(OpNo).isReg());
  unsigned RegBits = getMachineOpValue(MI, MI.getOperand(OpNo),Fixups) << 16;
  unsigned OffBits = getMachineOpValue(MI, MI.getOperand(OpNo+1), Fixups);

  // The immediate field of an LD/ST instruction is scaled which means it must
  // be divided (when encoding) by the size (in bytes) of the instructions'
  // data format.
  // .b - 1 byte
  // .h - 2 bytes
  // .w - 4 bytes
  // .d - 8 bytes
  switch(MI.getOpcode())
  {
  default:
    assert (0 && "Unexpected instruction");
    break;
  case Mips::LD_B:
  case Mips::ST_B:
    // We don't need to scale the offset in this case
    break;
  case Mips::LD_H:
  case Mips::ST_H:
    OffBits >>= 1;
    break;
  case Mips::LD_W:
  case Mips::ST_W:
    OffBits >>= 2;
    break;
  case Mips::LD_D:
  case Mips::ST_D:
    OffBits >>= 3;
    break;
  }

  return (OffBits & 0xFFFF) | RegBits;
}

/// getMemEncoding - Return binary encoding of memory related operand.
/// If the offset operand requires relocation, record the relocation.
unsigned
MipsMCCodeEmitter::getMemEncoding(const MCInst &MI, unsigned OpNo,
                                  SmallVectorImpl<MCFixup> &Fixups) const {
  // Base register is encoded in bits 20-16, offset is encoded in bits 15-0.
  assert(MI.getOperand(OpNo).isReg());
  unsigned RegBits = getMachineOpValue(MI, MI.getOperand(OpNo),Fixups) << 16;
  unsigned OffBits = getMachineOpValue(MI, MI.getOperand(OpNo+1), Fixups);

  return (OffBits & 0xFFFF) | RegBits;
}

unsigned MipsMCCodeEmitter::
getMemEncodingMMImm12(const MCInst &MI, unsigned OpNo,
                      SmallVectorImpl<MCFixup> &Fixups) const {
  // Base register is encoded in bits 20-16, offset is encoded in bits 11-0.
  assert(MI.getOperand(OpNo).isReg());
  unsigned RegBits = getMachineOpValue(MI, MI.getOperand(OpNo), Fixups) << 16;
  unsigned OffBits = getMachineOpValue(MI, MI.getOperand(OpNo+1), Fixups);

  return (OffBits & 0x0FFF) | RegBits;
}

unsigned
MipsMCCodeEmitter::getSizeExtEncoding(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups) const {
  assert(MI.getOperand(OpNo).isImm());
  unsigned SizeEncoding = getMachineOpValue(MI, MI.getOperand(OpNo), Fixups);
  return SizeEncoding - 1;
}

// FIXME: should be called getMSBEncoding
//
unsigned
MipsMCCodeEmitter::getSizeInsEncoding(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups) const {
  assert(MI.getOperand(OpNo-1).isImm());
  assert(MI.getOperand(OpNo).isImm());
  unsigned Position = getMachineOpValue(MI, MI.getOperand(OpNo-1), Fixups);
  unsigned Size = getMachineOpValue(MI, MI.getOperand(OpNo), Fixups);

  return Position + Size - 1;
}

unsigned
MipsMCCodeEmitter::getLSAImmEncoding(const MCInst &MI, unsigned OpNo,
                                     SmallVectorImpl<MCFixup> &Fixups) const {
  assert(MI.getOperand(OpNo).isImm());
  // The immediate is encoded as 'immediate - 1'.
  return getMachineOpValue(MI, MI.getOperand(OpNo), Fixups) - 1;
}

#include "MipsGenMCCodeEmitter.inc"


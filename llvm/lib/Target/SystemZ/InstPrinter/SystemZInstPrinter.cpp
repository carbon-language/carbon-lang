//===-- SystemZInstPrinter.cpp - Convert SystemZ MCInst to assembly syntax ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZInstPrinter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "SystemZGenAsmWriter.inc"

void SystemZInstPrinter::printAddress(unsigned Base, int64_t Disp,
                                      unsigned Index, raw_ostream &O) {
  O << Disp;
  if (Base) {
    O << '(';
    if (Index)
      O << '%' << getRegisterName(Index) << ',';
    O << '%' << getRegisterName(Base) << ')';
  } else
    assert(!Index && "Shouldn't have an index without a base");
}

void SystemZInstPrinter::printOperand(const MCOperand &MO, raw_ostream &O) {
  if (MO.isReg())
    O << '%' << getRegisterName(MO.getReg());
  else if (MO.isImm())
    O << MO.getImm();
  else if (MO.isExpr())
    O << *MO.getExpr();
  else
    llvm_unreachable("Invalid operand");
}

void SystemZInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                   StringRef Annot,
                                   const MCSubtargetInfo &STI) {
  printInstruction(MI, O);
  printAnnotation(O, Annot);
}

void SystemZInstPrinter::printRegName(raw_ostream &O, unsigned RegNo) const {
  O << '%' << getRegisterName(RegNo);
}

void SystemZInstPrinter::printU4ImmOperand(const MCInst *MI, int OpNum,
                                           raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isUInt<4>(Value) && "Invalid u4imm argument");
  O << Value;
}

void SystemZInstPrinter::printU6ImmOperand(const MCInst *MI, int OpNum,
                                           raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isUInt<6>(Value) && "Invalid u6imm argument");
  O << Value;
}

void SystemZInstPrinter::printS8ImmOperand(const MCInst *MI, int OpNum,
                                           raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isInt<8>(Value) && "Invalid s8imm argument");
  O << Value;
}

void SystemZInstPrinter::printU8ImmOperand(const MCInst *MI, int OpNum,
                                           raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isUInt<8>(Value) && "Invalid u8imm argument");
  O << Value;
}

void SystemZInstPrinter::printS16ImmOperand(const MCInst *MI, int OpNum,
                                            raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isInt<16>(Value) && "Invalid s16imm argument");
  O << Value;
}

void SystemZInstPrinter::printU16ImmOperand(const MCInst *MI, int OpNum,
                                            raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isUInt<16>(Value) && "Invalid u16imm argument");
  O << Value;
}

void SystemZInstPrinter::printS32ImmOperand(const MCInst *MI, int OpNum,
                                            raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isInt<32>(Value) && "Invalid s32imm argument");
  O << Value;
}

void SystemZInstPrinter::printU32ImmOperand(const MCInst *MI, int OpNum,
                                            raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isUInt<32>(Value) && "Invalid u32imm argument");
  O << Value;
}

void SystemZInstPrinter::printAccessRegOperand(const MCInst *MI, int OpNum,
                                               raw_ostream &O) {
  uint64_t Value = MI->getOperand(OpNum).getImm();
  assert(Value < 16 && "Invalid access register number");
  O << "%a" << (unsigned int)Value;
}

void SystemZInstPrinter::printPCRelOperand(const MCInst *MI, int OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isImm()) {
    O << "0x";
    O.write_hex(MO.getImm());
  } else
    O << *MO.getExpr();
}

void SystemZInstPrinter::printPCRelTLSOperand(const MCInst *MI, int OpNum,
                                              raw_ostream &O) {
  // Output the PC-relative operand.
  printPCRelOperand(MI, OpNum, O);

  // Output the TLS marker if present.
  if ((unsigned)OpNum + 1 < MI->getNumOperands()) {
    const MCOperand &MO = MI->getOperand(OpNum + 1);
    const MCSymbolRefExpr &refExp = cast<MCSymbolRefExpr>(*MO.getExpr());
    switch (refExp.getKind()) {
      case MCSymbolRefExpr::VK_TLSGD:
        O << ":tls_gdcall:";
        break;
      case MCSymbolRefExpr::VK_TLSLDM:
        O << ":tls_ldcall:";
        break;
      default:
        llvm_unreachable("Unexpected symbol kind");
    }
    O << refExp.getSymbol().getName();
  }
}

void SystemZInstPrinter::printOperand(const MCInst *MI, int OpNum,
                                      raw_ostream &O) {
  printOperand(MI->getOperand(OpNum), O);
}

void SystemZInstPrinter::printBDAddrOperand(const MCInst *MI, int OpNum,
                                            raw_ostream &O) {
  printAddress(MI->getOperand(OpNum).getReg(),
               MI->getOperand(OpNum + 1).getImm(), 0, O);
}

void SystemZInstPrinter::printBDXAddrOperand(const MCInst *MI, int OpNum,
                                             raw_ostream &O) {
  printAddress(MI->getOperand(OpNum).getReg(),
               MI->getOperand(OpNum + 1).getImm(),
               MI->getOperand(OpNum + 2).getReg(), O);
}

void SystemZInstPrinter::printBDLAddrOperand(const MCInst *MI, int OpNum,
                                             raw_ostream &O) {
  unsigned Base = MI->getOperand(OpNum).getReg();
  uint64_t Disp = MI->getOperand(OpNum + 1).getImm();
  uint64_t Length = MI->getOperand(OpNum + 2).getImm();
  O << Disp << '(' << Length;
  if (Base)
    O << ",%" << getRegisterName(Base);
  O << ')';
}

void SystemZInstPrinter::printCond4Operand(const MCInst *MI, int OpNum,
                                           raw_ostream &O) {
  static const char *const CondNames[] = {
    "o", "h", "nle", "l", "nhe", "lh", "ne",
    "e", "nlh", "he", "nl", "le", "nh", "no"
  };
  uint64_t Imm = MI->getOperand(OpNum).getImm();
  assert(Imm > 0 && Imm < 15 && "Invalid condition");
  O << CondNames[Imm - 1];
}

//===-- HexagonInstPrinter.h - Convert Hexagon MCInst to assembly syntax --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Hexagon MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONINSTPRINTER_H
#define HEXAGONINSTPRINTER_H

#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"

namespace llvm {
  class HexagonMCInst;

  class HexagonInstPrinter : public MCInstPrinter {
  public:
    explicit HexagonInstPrinter(const MCAsmInfo &MAI,
                                const MCInstrInfo &MII,
                                const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI), MII(MII) {}

    void printInst(const MCInst *MI, raw_ostream &O, StringRef Annot) override;
    void printInst(const HexagonMCInst *MI, raw_ostream &O, StringRef Annot);
    virtual StringRef getOpcodeName(unsigned Opcode) const;
    void printInstruction(const MCInst *MI, raw_ostream &O);
    StringRef getRegName(unsigned RegNo) const;
    static const char *getRegisterName(unsigned RegNo);

    void printOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O) const;
    void printImmOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O) const;
    void printExtOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O) const;
    void printUnsignedImmOperand(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) const;
    void printNegImmOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printNOneImmOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printMEMriOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printFrameIndexOperand(const MCInst *MI, unsigned OpNo,
                                raw_ostream &O) const;
    void printBranchOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printCallOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printAbsAddrOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printPredicateOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printGlobalOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printJumpTable(const MCInst *MI, unsigned OpNo, raw_ostream &O) const;

    void printConstantPool(const MCInst *MI, unsigned OpNo,
                           raw_ostream &O) const;

    void printSymbolHi(const MCInst *MI, unsigned OpNo, raw_ostream &O) const
      { printSymbol(MI, OpNo, O, true); }
    void printSymbolLo(const MCInst *MI, unsigned OpNo, raw_ostream &O) const
      { printSymbol(MI, OpNo, O, false); }

    const MCInstrInfo &getMII() const {
      return MII;
    }

  protected:
    void printSymbol(const MCInst *MI, unsigned OpNo, raw_ostream &O, bool hi)
           const;

    static const char PacketPadding;

  private:
    const MCInstrInfo &MII;

  };

} // end namespace llvm

#endif

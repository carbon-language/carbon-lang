//===-- llvm/MC/MCParsedAsmOperand.h - Asm Parser Operand -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMOPERAND_H
#define LLVM_MC_MCASMOPERAND_H

namespace llvm {
class SMLoc;
class raw_ostream;

/// MCParsedAsmOperand - This abstract class represents a source-level assembly
/// instruction operand.  It should be subclassed by target-specific code.  This
/// base class is used by target-independent clients and is the interface
/// between parsing an asm instruction and recognizing it.
class MCParsedAsmOperand {
public:
  MCParsedAsmOperand() {}
  virtual ~MCParsedAsmOperand() {}

  /// isToken - Is this a token operand?
  virtual bool isToken() const = 0;
  /// isImm - Is this an immediate operand?
  virtual bool isImm() const = 0;
  /// isReg - Is this a register operand?
  virtual bool isReg() const = 0;
  virtual unsigned getReg() const = 0;

  /// isMem - Is this a memory operand?
  virtual bool isMem() const = 0;

  /// getStartLoc - Get the location of the first token of this operand.
  virtual SMLoc getStartLoc() const = 0;
  /// getEndLoc - Get the location of the last token of this operand.
  virtual SMLoc getEndLoc() const = 0;

  /// print - Print a debug representation of the operand to the given stream.
  virtual void print(raw_ostream &OS) const = 0;
  /// dump - Print to the debug stream.
  virtual void dump() const;
};

//===----------------------------------------------------------------------===//
// Debugging Support

inline raw_ostream& operator<<(raw_ostream &OS, const MCParsedAsmOperand &MO) {
  MO.print(OS);
  return OS;
}

} // end namespace llvm.

#endif

//===--- lib/CodeGen/DwarfPrinter.h - Dwarf Printer -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Emit general DWARF directives.
// 
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DWARFPRINTER_H__
#define CODEGEN_ASMPRINTER_DWARFPRINTER_H__

#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormattedStream.h"
#include <vector>

namespace llvm {
class AsmPrinter;
class MachineFunction;
class MachineModuleInfo;
class Module;
class MCAsmInfo;
class TargetData;
class TargetRegisterInfo;
class GlobalValue;
class MCSymbol;
class Twine;

class DwarfPrinter {
protected:
  ~DwarfPrinter() {}

  //===-------------------------------------------------------------==---===//
  // Core attributes used by the DWARF printer.
  //

  /// O - Stream to .s file.
  raw_ostream &O;

  /// Asm - Target of Dwarf emission.
  AsmPrinter *Asm;

  /// MAI - Target asm information.
  const MCAsmInfo *MAI;

  /// TD - Target data.
  const TargetData *TD;

  /// RI - Register Information.
  const TargetRegisterInfo *RI;

  /// M - Current module.
  Module *M;

  /// MF - Current machine function.
  const MachineFunction *MF;

  /// MMI - Collected machine module information.
  MachineModuleInfo *MMI;

  /// SubprogramCount - The running count of functions being compiled.
  unsigned SubprogramCount;

  DwarfPrinter(raw_ostream &OS, AsmPrinter *A, const MCAsmInfo *T);
public:
  
  //===------------------------------------------------------------------===//
  // Accessors.
  //
  const AsmPrinter *getAsm() const { return Asm; }
  MachineModuleInfo *getMMI() const { return MMI; }
  const MCAsmInfo *getMCAsmInfo() const { return MAI; }
  const TargetData *getTargetData() const { return TD; }

  /// getDWLabel - Return the MCSymbol corresponding to the assembler temporary
  /// label with the specified stem and unique ID.
  MCSymbol *getDWLabel(const char *Name, unsigned ID) const;
  
  /// getTempLabel - Return an assembler temporary label with the specified
  /// name.
  MCSymbol *getTempLabel(const char *Name) const;

  /// SizeOfEncodedValue - Return the size of the encoding in bytes.
  unsigned SizeOfEncodedValue(unsigned Encoding) const;

  /// EmitEncodingByte - Emit a .byte 42 directive that corresponds to an
  /// encoding.  If verbose assembly output is enabled, we output comments
  /// describing the encoding.  Desc is a string saying what the encoding is
  /// specifying (e.g. "LSDA").
  void EmitEncodingByte(unsigned Val, const char *Desc);
  
  /// EmitCFAByte - Emit a .byte 42 directive for a DW_CFA_xxx value.
  void EmitCFAByte(unsigned Val);
  
  
  /// EmitSLEB128 - emit the specified signed leb128 value.
  void EmitSLEB128(int Value, const char *Desc) const;

  /// EmitULEB128 - emit the specified unsigned leb128 value.
  void EmitULEB128(unsigned Value, const char *Desc = 0,
                   unsigned PadTo = 0) const;

  
  /// EmitReference - Emit a reference to a label.
  ///
  void EmitReference(const MCSymbol *Sym, unsigned Encoding) const;
  void EmitReference(const GlobalValue *GV, unsigned Encoding) const;

  /// EmitDifference - Emit the difference between two labels.
  void EmitDifference(const MCSymbol *LabelHi, const MCSymbol *LabelLo,
                      bool IsSmall = false);

  /// EmitSectionOffset - Emit Label-Section or use a special purpose directive
  /// to emit a section offset if the target has one.
  void EmitSectionOffset(const MCSymbol *Label, const MCSymbol *Section,
                         bool IsSmall = false, bool isEH = false);
  
  /// EmitFrameMoves - Emit frame instructions to describe the layout of the
  /// frame.
  void EmitFrameMoves(MCSymbol *BaseLabel,
                      const std::vector<MachineMove> &Moves, bool isEH);
};

} // end llvm namespace

#endif

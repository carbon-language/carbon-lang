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

#include "DwarfLabel.h"
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
class MCSymbol;

class DwarfPrinter {
protected:
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
  MachineFunction *MF;

  /// MMI - Collected machine module information.
  MachineModuleInfo *MMI;

  /// SubprogramCount - The running count of functions being compiled.
  unsigned SubprogramCount;

  /// Flavor - A unique string indicating what dwarf producer this is, used to
  /// unique labels.
  const char * const Flavor;

  /// SetCounter - A unique number for each '.set' directive.
  unsigned SetCounter;

  DwarfPrinter(raw_ostream &OS, AsmPrinter *A, const MCAsmInfo *T,
               const char *flavor);
public:
  
  //===------------------------------------------------------------------===//
  // Accessors.
  //
  const AsmPrinter *getAsm() const { return Asm; }
  MachineModuleInfo *getMMI() const { return MMI; }
  const MCAsmInfo *getMCAsmInfo() const { return MAI; }
  const TargetData *getTargetData() const { return TD; }

  void PrintRelDirective(bool Force32Bit = false,
                         bool isInSection = false) const;

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
  void EmitULEB128(unsigned Value, const char *Desc = 0) const;

  
  /// PrintLabelName - Print label name in form used by Dwarf writer.
  ///
  void PrintLabelName(const DWLabel &Label) const {
    PrintLabelName(Label.getTag(), Label.getNumber());
  }
  void PrintLabelName(const char *Tag, unsigned Number) const;
  void PrintLabelName(const char *Tag, unsigned Number,
                      const char *Suffix) const;

  /// EmitLabel - Emit location label for internal use by Dwarf.
  ///
  void EmitLabel(const DWLabel &Label) const {
    EmitLabel(Label.getTag(), Label.getNumber());
  }
  void EmitLabel(const char *Tag, unsigned Number) const;

  /// EmitReference - Emit a reference to a label.
  ///
  void EmitReference(const DWLabel &Label, bool IsPCRelative = false,
                     bool Force32Bit = false) const {
    EmitReference(Label.getTag(), Label.getNumber(),
                  IsPCRelative, Force32Bit);
  }
  void EmitReference(const char *Tag, unsigned Number,
                     bool IsPCRelative = false,
                     bool Force32Bit = false) const;
  void EmitReference(const std::string &Name, bool IsPCRelative = false,
                     bool Force32Bit = false) const;
  void EmitReference(const MCSymbol *Sym, bool IsPCRelative = false,
                     bool Force32Bit = false) const;

  /// EmitDifference - Emit the difference between two labels.  Some
  /// assemblers do not behave with absolute expressions with data directives,
  /// so there is an option (needsSet) to use an intermediary set expression.
  void EmitDifference(const DWLabel &LabelHi, const DWLabel &LabelLo,
                      bool IsSmall = false) {
    EmitDifference(LabelHi.getTag(), LabelHi.getNumber(),
                   LabelLo.getTag(), LabelLo.getNumber(),
                   IsSmall);
  }
  void EmitDifference(const char *TagHi, unsigned NumberHi,
                      const char *TagLo, unsigned NumberLo,
                      bool IsSmall = false);

  void EmitSectionOffset(const char* Label, const char* Section,
                         unsigned LabelNumber, unsigned SectionNumber,
                         bool IsSmall = false, bool isEH = false,
                         bool useSet = true);

  /// EmitFrameMoves - Emit frame instructions to describe the layout of the
  /// frame.
  void EmitFrameMoves(const char *BaseLabel, unsigned BaseLabelID,
                      const std::vector<MachineMove> &Moves, bool isEH);
};

} // end llvm namespace

#endif

//===-- MCInstPrinter.h - Convert an MCInst to target assembly syntax -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTPRINTER_H
#define LLVM_MC_MCINSTPRINTER_H

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Format.h"

namespace llvm {
class MCInst;
class raw_ostream;
class MCAsmInfo;
class MCInstrInfo;
class MCRegisterInfo;
class StringRef;

namespace HexStyle {
    enum Style {
        C,          ///< 0xff
        Asm         ///< 0ffh
    };
}

/// MCInstPrinter - This is an instance of a target assembly language printer
/// that converts an MCInst to valid target assembly syntax.
class MCInstPrinter {
protected:
  /// CommentStream - a stream that comments can be emitted to if desired.
  /// Each comment must end with a newline.  This will be null if verbose
  /// assembly emission is disable.
  raw_ostream *CommentStream;
  const MCAsmInfo &MAI;
  const MCInstrInfo &MII;
  const MCRegisterInfo &MRI;

  /// The current set of available features.
  uint64_t AvailableFeatures;

  /// True if we are printing marked up assembly.
  bool UseMarkup;

  /// True if we are printing immediates as hex.
  bool PrintImmHex;

  /// Which style to use for printing hexadecimal values.
  HexStyle::Style PrintHexStyle;

  /// Utility function for printing annotations.
  void printAnnotation(raw_ostream &OS, StringRef Annot);
public:
  MCInstPrinter(const MCAsmInfo &mai, const MCInstrInfo &mii,
                const MCRegisterInfo &mri)
    : CommentStream(0), MAI(mai), MII(mii), MRI(mri), AvailableFeatures(0),
      UseMarkup(0), PrintImmHex(0), PrintHexStyle(HexStyle::C) {}

  virtual ~MCInstPrinter();

  /// setCommentStream - Specify a stream to emit comments to.
  void setCommentStream(raw_ostream &OS) { CommentStream = &OS; }

  /// printInst - Print the specified MCInst to the specified raw_ostream.
  ///
  virtual void printInst(const MCInst *MI, raw_ostream &OS,
                         StringRef Annot) = 0;

  /// getOpcodeName - Return the name of the specified opcode enum (e.g.
  /// "MOV32ri") or empty if we can't resolve it.
  StringRef getOpcodeName(unsigned Opcode) const;

  /// printRegName - Print the assembler register name.
  virtual void printRegName(raw_ostream &OS, unsigned RegNo) const;

  uint64_t getAvailableFeatures() const { return AvailableFeatures; }
  void setAvailableFeatures(uint64_t Value) { AvailableFeatures = Value; }

  bool getUseMarkup() const { return UseMarkup; }
  void setUseMarkup(bool Value) { UseMarkup = Value; }

  /// Utility functions to make adding mark ups simpler.
  StringRef markup(StringRef s) const;
  StringRef markup(StringRef a, StringRef b) const;

  bool getPrintImmHex() const { return PrintImmHex; }
  void setPrintImmHex(bool Value) { PrintImmHex = Value; }

  HexStyle::Style getPrintHexStyleHex() const { return PrintHexStyle; }
  void setPrintImmHex(HexStyle::Style Value) { PrintHexStyle = Value; }

  /// Utility function to print immediates in decimal or hex.
  format_object1<int64_t> formatImm(const int64_t Value) const { return PrintImmHex ? formatHex(Value) : formatDec(Value); }

  /// Utility functions to print decimal/hexadecimal values.
  format_object1<int64_t> formatDec(const int64_t Value) const;
  format_object1<int64_t> formatHex(const int64_t Value) const;
  format_object1<uint64_t> formatHex(const uint64_t Value) const;
};

} // namespace llvm

#endif

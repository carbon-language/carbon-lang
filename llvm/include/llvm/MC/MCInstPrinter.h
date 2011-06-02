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

namespace llvm {
class MCInst;
class raw_ostream;
class MCAsmInfo;
class StringRef;

/// MCInstPrinter - This is an instance of a target assembly language printer
/// that converts an MCInst to valid target assembly syntax.
class MCInstPrinter {
protected:
  /// CommentStream - a stream that comments can be emitted to if desired.
  /// Each comment must end with a newline.  This will be null if verbose
  /// assembly emission is disable.
  raw_ostream *CommentStream;
  const MCAsmInfo &MAI;

  /// The current set of available features.
  unsigned AvailableFeatures;
public:
  MCInstPrinter(const MCAsmInfo &mai)
    : CommentStream(0), MAI(mai), AvailableFeatures(0) {}

  virtual ~MCInstPrinter();

  /// setCommentStream - Specify a stream to emit comments to.
  void setCommentStream(raw_ostream &OS) { CommentStream = &OS; }

  /// printInst - Print the specified MCInst to the specified raw_ostream.
  ///
  virtual void printInst(const MCInst *MI, raw_ostream &OS) = 0;

  /// getOpcodeName - Return the name of the specified opcode enum (e.g.
  /// "MOV32ri") or empty if we can't resolve it.
  virtual StringRef getOpcodeName(unsigned Opcode) const;

  /// printRegName - Print the assembler register name.
  virtual void printRegName(raw_ostream &OS, unsigned RegNo) const;

  unsigned getAvailableFeatures() const { return AvailableFeatures; }
  void setAvailableFeatures(unsigned Value) { AvailableFeatures = Value; }
};

} // namespace llvm

#endif

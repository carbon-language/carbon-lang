//===-- llvm/MC/MCCodeEmitter.h - Instruction Encoding ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCODEEMITTER_H
#define LLVM_MC_MCCODEEMITTER_H

#include "llvm/MC/MCFixup.h"

#include <cassert>

namespace llvm {
class MCExpr;
class MCInst;
class raw_ostream;
template<typename T> class SmallVectorImpl;

/// MCFixupKindInfo - Target independent information on a fixup kind.
struct MCFixupKindInfo {
  /// A target specific name for the fixup kind. The names will be unique for
  /// distinct kinds on any given target.
  const char *Name;

  /// The bit offset to write the relocation into.
  //
  // FIXME: These two fields are under-specified and not general enough, but it
  // is covers many things, and is enough to let the AsmStreamer pretty-print
  // the encoding.
  unsigned TargetOffset;

  /// The number of bits written by this fixup. The bits are assumed to be
  /// contiguous.
  unsigned TargetSize;
};

/// MCCodeEmitter - Generic instruction encoding interface.
class MCCodeEmitter {
private:
  MCCodeEmitter(const MCCodeEmitter &);   // DO NOT IMPLEMENT
  void operator=(const MCCodeEmitter &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  MCCodeEmitter();

public:
  virtual ~MCCodeEmitter();

  /// @name Target Independent Fixup Information
  /// @{

  /// getNumFixupKinds - Get the number of target specific fixup kinds.
  virtual unsigned getNumFixupKinds() const = 0;

  /// getFixupKindInfo - Get information on a fixup kind.
  virtual const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const;

  /// @}

  /// EncodeInstruction - Encode the given \arg Inst to bytes on the output
  /// stream \arg OS.
  virtual void EncodeInstruction(const MCInst &Inst, raw_ostream &OS,
                                 SmallVectorImpl<MCFixup> &Fixups) const = 0;
};

} // End llvm namespace

#endif

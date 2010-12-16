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

/// MCCodeEmitter - Generic instruction encoding interface.
class MCCodeEmitter {
private:
  MCCodeEmitter(const MCCodeEmitter &);   // DO NOT IMPLEMENT
  void operator=(const MCCodeEmitter &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  MCCodeEmitter();

public:
  virtual ~MCCodeEmitter();

  /// EncodeInstruction - Encode the given \arg Inst to bytes on the output
  /// stream \arg OS.
  virtual void EncodeInstruction(const MCInst &Inst, raw_ostream &OS,
                                 SmallVectorImpl<MCFixup> &Fixups) const = 0;
};

} // End llvm namespace

#endif

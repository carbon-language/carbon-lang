//===-- llvm/MC/MachObjectWriter.h - Mach-O File Writer ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MACHOBJECTWRITER_H
#define LLVM_MC_MACHOBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

namespace llvm {
class MCAsmFixup;
class MCAssembler;
class MCFragment;
class MCValue;
class raw_ostream;

class MachObjectWriter : public MCObjectWriter {
  void *Impl;

public:
  MachObjectWriter(raw_ostream &OS, bool Is64Bit, bool IsLittleEndian = true);
  virtual ~MachObjectWriter();

  virtual void ExecutePostLayoutBinding(MCAssembler &Asm);

  virtual void RecordRelocation(const MCAssembler &Asm,
                                const MCAsmLayout &Layout,
                                const MCFragment *Fragment,
                                const MCAsmFixup &Fixup, MCValue Target,
                                uint64_t &FixedValue);

  virtual void WriteObject(const MCAssembler &Asm, const MCAsmLayout &Layout);
};

} // End llvm namespace

#endif

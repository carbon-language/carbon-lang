//===-- llvm/MC/ELFObjectWriter.h - ELF File Writer ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_ELFOBJECTWRITER_H
#define LLVM_MC_ELFOBJECTWRITER_H

#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

namespace llvm {
class MCAsmFixup;
class MCAssembler;
class MCFragment;
class MCValue;
class raw_ostream;

class ELFObjectWriter : public MCObjectWriter {
  void *Impl;

public:
  ELFObjectWriter(raw_ostream &OS, bool Is64Bit, Triple::OSType OSType, 
                  uint16_t EMachine, bool IsLittleEndian = true,
                  bool HasRelocationAddend = true);

  virtual ~ELFObjectWriter();

  virtual void ExecutePostLayoutBinding(MCAssembler &Asm);

  virtual void RecordRelocation(const MCAssembler &Asm,
                                const MCAsmLayout &Layout,
                                const MCFragment *Fragment,
                                const MCFixup &Fixup, MCValue Target,
                                uint64_t &FixedValue);

  virtual bool IsFixupFullyResolved(const MCAssembler &Asm,
                                    const MCValue Target,
                                    bool IsPCRel,
                                    const MCFragment *DF) const;

  virtual void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout);
};

} // End llvm namespace

#endif

//===-- llvm/MC/WinCOFFObjectWriter.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of a Win32 COFF object file writer.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "WinCOFFObjectWriter"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
using namespace llvm;

namespace {

  class WinCOFFObjectWriter : public MCObjectWriter {
  public:
    WinCOFFObjectWriter(raw_ostream &OS);

    // MCObjectWriter interface implementation.

    void ExecutePostLayoutBinding(MCAssembler &Asm);

    void RecordRelocation(const MCAssembler &Asm,
                          const MCAsmLayout &Layout,
                          const MCFragment *Fragment,
                          const MCFixup &Fixup,
                          MCValue Target,
                          uint64_t &FixedValue);

    void WriteObject(const MCAssembler &Asm, const MCAsmLayout &Layout);
  };
}

WinCOFFObjectWriter::WinCOFFObjectWriter(raw_ostream &OS)
                                : MCObjectWriter(OS, true) {
}

////////////////////////////////////////////////////////////////////////////////
// MCObjectWriter interface implementations

void WinCOFFObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm) {
}

void WinCOFFObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           const MCFragment *Fragment,
                                           const MCFixup &Fixup,
                                           MCValue Target,
                                           uint64_t &FixedValue) {
}

void WinCOFFObjectWriter::WriteObject(const MCAssembler &Asm,
                                      const MCAsmLayout &Layout) {
}

//------------------------------------------------------------------------------
// WinCOFFObjectWriter factory function

namespace llvm {
  MCObjectWriter *createWinCOFFObjectWriter(raw_ostream &OS) {
    return new WinCOFFObjectWriter(OS);
  }
}

//===-- llvm/MC/MCMachObjectWriter.h - Mach Object Writer -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCMACHOBJECTWRITER_H
#define LLVM_MC_MCMACHOBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"

namespace llvm {

class MCMachObjectTargetWriter {
protected:
  MCMachObjectTargetWriter();

public:
  virtual ~MCMachObjectTargetWriter();
};

/// \brief Construct a new Mach-O writer instance.
///
/// This routine takes ownership of the target writer subclass.
///
/// \param MOTW - The target specific Mach-O writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
MCObjectWriter *createMachObjectWriter(MCMachObjectTargetWriter *MOTW,
                                       raw_ostream &OS, bool is64Bit,
                                       uint32_t CPUType, uint32_t CPUSubtype,
                                       bool IsLittleEndian);

} // End llvm namespace

#endif

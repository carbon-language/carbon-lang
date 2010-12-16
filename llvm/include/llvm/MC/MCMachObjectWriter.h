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
#include "llvm/Support/DataTypes.h"

namespace llvm {

class MCMachObjectTargetWriter {
  const unsigned Is64Bit : 1;
  const uint32_t CPUType;
  const uint32_t CPUSubtype;

protected:
  MCMachObjectTargetWriter(bool Is64Bit_, uint32_t CPUType_,
                           uint32_t CPUSubtype_);

public:
  virtual ~MCMachObjectTargetWriter();

  /// @name Accessors
  /// @{

  bool is64Bit() const { return Is64Bit; }
  uint32_t getCPUType() const { return CPUType; }
  uint32_t getCPUSubtype() const { return CPUSubtype; }

  /// @}
};

/// \brief Construct a new Mach-O writer instance.
///
/// This routine takes ownership of the target writer subclass.
///
/// \param MOTW - The target specific Mach-O writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
MCObjectWriter *createMachObjectWriter(MCMachObjectTargetWriter *MOTW,
                                       raw_ostream &OS, bool IsLittleEndian);

} // End llvm namespace

#endif

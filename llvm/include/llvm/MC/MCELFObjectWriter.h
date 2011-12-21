//===-- llvm/MC/MCELFObjectWriter.h - ELF Object Writer ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELFOBJECTWRITER_H
#define LLVM_MC_MCELFOBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCELFObjectTargetWriter {
  const Triple::OSType OSType;
  const uint16_t EMachine;
  const unsigned HasRelocationAddend : 1;
  const unsigned Is64Bit : 1;
protected:
  MCELFObjectTargetWriter(bool Is64Bit_, Triple::OSType OSType_,
                          uint16_t EMachine_,  bool HasRelocationAddend_);

public:
  virtual ~MCELFObjectTargetWriter();

  /// @name Accessors
  /// @{
  Triple::OSType getOSType() { return OSType; }
  uint16_t getEMachine() { return EMachine; }
  bool hasRelocationAddend() { return HasRelocationAddend; }
  bool is64Bit() const { return Is64Bit; }
  /// @}
};

/// \brief Construct a new ELF writer instance.
///
/// \param MOTW - The target specific ELF writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
MCObjectWriter *createELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                      raw_ostream &OS, bool IsLittleEndian);
} // End llvm namespace

#endif

//===-- PPCMCTargetDesc.h - PowerPC Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides PowerPC specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCTARGETDESC_H
#define LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCTARGETDESC_H

// GCC #defines PPC on Linux but we use it as our namespace name
#undef PPC

#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class Target;
class StringRef;
class raw_ostream;

extern Target ThePPC32Target;
extern Target ThePPC64Target;
extern Target ThePPC64LETarget;

MCCodeEmitter *createPPCMCCodeEmitter(const MCInstrInfo &MCII,
                                      const MCRegisterInfo &MRI,
                                      MCContext &Ctx);

MCAsmBackend *createPPCAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                  StringRef TT, StringRef CPU);

/// createPPCELFObjectWriter - Construct an PPC ELF object writer.
MCObjectWriter *createPPCELFObjectWriter(raw_ostream &OS,
                                         bool Is64Bit,
                                         bool IsLittleEndian,
                                         uint8_t OSABI);
/// createPPCELFObjectWriter - Construct a PPC Mach-O object writer.
MCObjectWriter *createPPCMachObjectWriter(raw_ostream &OS, bool Is64Bit,
                                          uint32_t CPUType,
                                          uint32_t CPUSubtype);
} // End llvm namespace

// Generated files will use "namespace PPC". To avoid symbol clash,
// undefine PPC here. PPC may be predefined on some hosts.
#undef PPC

// Defines symbolic names for PowerPC registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "PPCGenRegisterInfo.inc"

// Defines symbolic names for the PowerPC instructions.
//
#define GET_INSTRINFO_ENUM
#include "PPCGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "PPCGenSubtargetInfo.inc"

#endif

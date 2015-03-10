//===-- ARMMCTargetDesc.h - ARM Target Descriptions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides ARM specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCTARGETDESC_H
#define LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
class formatted_raw_ostream;
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCInstPrinter;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCStreamer;
class MCRelocationInfo;
class MCTargetStreamer;
class StringRef;
class Target;
class raw_ostream;

extern Target TheARMLETarget, TheThumbLETarget;
extern Target TheARMBETarget, TheThumbBETarget;

namespace ARM_MC {
  std::string ParseARMTriple(StringRef TT, StringRef CPU);

  /// createARMMCSubtargetInfo - Create a ARM MCSubtargetInfo instance.
  /// This is exposed so Asm parser, etc. do not need to go through
  /// TargetRegistry.
  MCSubtargetInfo *createARMMCSubtargetInfo(StringRef TT, StringRef CPU,
                                            StringRef FS);
}

MCStreamer *createMCAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                                bool isVerboseAsm, bool useDwarfDirectory,
                                MCInstPrinter *InstPrint, MCCodeEmitter *CE,
                                MCAsmBackend *TAB, bool ShowInst);

MCTargetStreamer *createARMNullTargetStreamer(MCStreamer &S);

MCCodeEmitter *createARMLEMCCodeEmitter(const MCInstrInfo &MCII,
                                        const MCRegisterInfo &MRI,
                                        MCContext &Ctx);

MCCodeEmitter *createARMBEMCCodeEmitter(const MCInstrInfo &MCII,
                                        const MCRegisterInfo &MRI,
                                        MCContext &Ctx);

MCAsmBackend *createARMAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                  StringRef TT, StringRef CPU,
                                  bool IsLittleEndian);

MCAsmBackend *createARMLEAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                  StringRef TT, StringRef CPU);

MCAsmBackend *createARMBEAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                  StringRef TT, StringRef CPU);

MCAsmBackend *createThumbLEAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                      StringRef TT, StringRef CPU);

MCAsmBackend *createThumbBEAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                      StringRef TT, StringRef CPU);

/// createARMWinCOFFStreamer - Construct a PE/COFF machine code streamer which
/// will generate a PE/COFF object file.
MCStreamer *createARMWinCOFFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     MCCodeEmitter &Emitter, raw_ostream &OS);

/// createARMELFObjectWriter - Construct an ELF Mach-O object writer.
MCObjectWriter *createARMELFObjectWriter(raw_ostream &OS,
                                         uint8_t OSABI,
                                         bool IsLittleEndian);

/// createARMMachObjectWriter - Construct an ARM Mach-O object writer.
MCObjectWriter *createARMMachObjectWriter(raw_ostream &OS,
                                          bool Is64Bit,
                                          uint32_t CPUType,
                                          uint32_t CPUSubtype);

/// createARMWinCOFFObjectWriter - Construct an ARM PE/COFF object writer.
MCObjectWriter *createARMWinCOFFObjectWriter(raw_ostream &OS, bool Is64Bit);

/// createARMMachORelocationInfo - Construct ARM Mach-O relocation info.
MCRelocationInfo *createARMMachORelocationInfo(MCContext &Ctx);
} // End llvm namespace

// Defines symbolic names for ARM registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "ARMGenRegisterInfo.inc"

// Defines symbolic names for the ARM instructions.
//
#define GET_INSTRINFO_ENUM
#include "ARMGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "ARMGenSubtargetInfo.inc"

#endif

//===-- BPFMCTargetDesc.h - BPF Target Descriptions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides BPF specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_MCTARGETDESC_BPFMCTARGETDESC_H
#define LLVM_LIB_TARGET_BPF_MCTARGETDESC_BPFMCTARGETDESC_H

#include "llvm/Config/config.h"
#include "llvm/Support/DataTypes.h"

#include <memory>

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectTargetWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class Target;

MCCodeEmitter *createBPFMCCodeEmitter(const MCInstrInfo &MCII,
                                      const MCRegisterInfo &MRI,
                                      MCContext &Ctx);
MCCodeEmitter *createBPFbeMCCodeEmitter(const MCInstrInfo &MCII,
                                        const MCRegisterInfo &MRI,
                                        MCContext &Ctx);

MCAsmBackend *createBPFAsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                  const MCRegisterInfo &MRI,
                                  const MCTargetOptions &Options);
MCAsmBackend *createBPFbeAsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                    const MCRegisterInfo &MRI,
                                    const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter> createBPFELFObjectWriter(uint8_t OSABI);
}

// Defines symbolic names for BPF registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "BPFGenRegisterInfo.inc"

// Defines symbolic names for the BPF instructions.
//
#define GET_INSTRINFO_ENUM
#include "BPFGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "BPFGenSubtargetInfo.inc"

#endif

//===-- ARM64ELFStreamer.h - ELF Streamer for ARM64 -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF streamer information for the ARM64 backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AARCH64_ELF_STREAMER_H
#define LLVM_AARCH64_ELF_STREAMER_H

#include "llvm/MC/MCELFStreamer.h"

namespace llvm {

MCELFStreamer *createARM64ELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                      raw_ostream &OS, MCCodeEmitter *Emitter,
                                      bool RelaxAll, bool NoExecStack);
}

#endif // ARM64_ELF_STREAMER_H

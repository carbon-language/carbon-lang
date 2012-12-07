//===-- ARMELFStreamer.h - ELF Streamer for ARM ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF streamer information for the ARM backend.
//
//===----------------------------------------------------------------------===//

#ifndef ARM_ELF_STREAMER_H
#define ARM_ELF_STREAMER_H

#include "llvm/MC/MCELFStreamer.h"

namespace llvm {

  MCELFStreamer* createARMELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                      raw_ostream &OS, MCCodeEmitter *Emitter,
                                      bool RelaxAll, bool NoExecStack,
                                      bool IsThumb);
}

#endif // ARM_ELF_STREAMER_H

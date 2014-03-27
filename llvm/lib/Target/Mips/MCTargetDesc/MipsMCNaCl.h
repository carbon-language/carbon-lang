//===-- MipsMCNaCl.h - NaCl-related declarations --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSMCNACL_H
#define MIPSMCNACL_H

#include "llvm/MC/MCELFStreamer.h"

namespace llvm {

// Log2 of the NaCl MIPS sandbox's instruction bundle size.
static const unsigned MIPS_NACL_BUNDLE_ALIGN = 4u;

bool isBasePlusOffsetMemoryAccess(unsigned Opcode, unsigned *AddrIdx,
                                  bool *IsStore = NULL);
bool baseRegNeedsLoadStoreMask(unsigned Reg);

// This function creates an MCELFStreamer for Mips NaCl.
MCELFStreamer *createMipsNaClELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                         raw_ostream &OS,
                                         MCCodeEmitter *Emitter,
                                         const MCSubtargetInfo &STI,
                                         bool RelaxAll, bool NoExecStack);

}

#endif

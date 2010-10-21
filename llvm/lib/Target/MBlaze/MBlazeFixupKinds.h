//===-- MBlaze/MBlazeFixupKinds.h - MBlaze Fixup Entries --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MBLAZE_MBLAZEFIXUPKINDS_H
#define LLVM_MBLAZE_MBLAZEFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace MBlaze {
enum Fixups {
  reloc_pcrel_4byte = FirstTargetFixupKind,  // 32-bit pcrel, e.g. a brlid
  reloc_pcrel_2byte                          // 16-bit pcrel, e.g. beqid
};
}
}

#endif

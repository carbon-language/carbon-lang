//===-- MipsABIFlagsSection.cpp - Mips ELF ABI Flags Section ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsABIFlagsSection.h"

using namespace llvm;

StringRef MipsABIFlagsSection::getFpABIString(Val_GNU_MIPS_ABI Value,
                                              bool Is32BitAbi) {
  switch (Value) {
  case MipsABIFlagsSection::Val_GNU_MIPS_ABI_FP_XX:
    return "xx";
  case MipsABIFlagsSection::Val_GNU_MIPS_ABI_FP_64:
    return "64";
  case MipsABIFlagsSection::Val_GNU_MIPS_ABI_FP_DOUBLE:
    if (Is32BitAbi)
      return "32";
    return "64";
  default:
    llvm_unreachable("unsupported fp abi value");
  }
}

namespace llvm {
MCStreamer &operator<<(MCStreamer &OS, MipsABIFlagsSection &ABIFlagsSection) {
  // Write out a Elf_Internal_ABIFlags_v0 struct
  OS.EmitIntValue(ABIFlagsSection.getVersion(), 2);         // version
  OS.EmitIntValue(ABIFlagsSection.getISALevel(), 1);        // isa_level
  OS.EmitIntValue(ABIFlagsSection.getISARevision(), 1);     // isa_rev
  OS.EmitIntValue(ABIFlagsSection.getGPRSize(), 1);         // gpr_size
  OS.EmitIntValue(ABIFlagsSection.getCPR1Size(), 1);        // cpr1_size
  OS.EmitIntValue(ABIFlagsSection.getCPR2Size(), 1);        // cpr2_size
  OS.EmitIntValue(ABIFlagsSection.getFpABI(), 1);           // fp_abi
  OS.EmitIntValue(ABIFlagsSection.getISAExtensionSet(), 4); // isa_ext
  OS.EmitIntValue(ABIFlagsSection.getASESet(), 4);          // ases
  OS.EmitIntValue(ABIFlagsSection.getFlags1(), 4);          // flags1
  OS.EmitIntValue(ABIFlagsSection.getFlags2(), 4);          // flags2
  return OS;
}
}

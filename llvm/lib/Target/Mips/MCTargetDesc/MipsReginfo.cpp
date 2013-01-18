//===-- MipsReginfo.cpp - Registerinfo handling  --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// .reginfo
//    Elf32_Word ri_gprmask
//    Elf32_Word ri_cprmask[4]
//    Elf32_Word ri_gp_value
//
// .MIPS.options - N64
//    Elf64_Byte    kind (ODK_REGINFO)
//    Elf64_Byte    size (40 bytes)
//    Elf64_Section section (0)
//    Elf64_Word    info (unused)
//    Elf64_Word    ri_gprmask ()
//    Elf64_Word    ri_pad ()
//    Elf64_Word[4] ri_cprmask ()
//    Elf64_Addr    ri_gp_value ()
//
// .MIPS.options - N32
//    Elf32_Byte    kind (ODK_REGINFO)
//    Elf32_Byte    size (36 bytes)
//    Elf32_Section section (0)
//    Elf32_Word    info (unused)
//    Elf32_Word    ri_gprmask ()
//    Elf32_Word    ri_pad ()
//    Elf32_Word[4] ri_cprmask ()
//    Elf32_Addr    ri_gp_value ()
//
//===----------------------------------------------------------------------===//
#include "MCTargetDesc/MipsReginfo.h"
#include "MipsSubtarget.h"
#include "MipsTargetObjectFile.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

// Integrated assembler version
void
MipsReginfo::emitMipsReginfoSectionCG(MCStreamer &OS,
    const TargetLoweringObjectFile &TLOF,
    const MipsSubtarget &MST) const
{

  if (OS.hasRawTextSupport())
    return;

  const MipsTargetObjectFile &TLOFELF =
      static_cast<const MipsTargetObjectFile &>(TLOF);
  OS.SwitchSection(TLOFELF.getReginfoSection());

  // .reginfo
  if (MST.isABI_O32()) {
    OS.EmitIntValue(0, 4); // ri_gprmask
    OS.EmitIntValue(0, 4); // ri_cpr[0]mask
    OS.EmitIntValue(0, 4); // ri_cpr[1]mask
    OS.EmitIntValue(0, 4); // ri_cpr[2]mask
    OS.EmitIntValue(0, 4); // ri_cpr[3]mask
    OS.EmitIntValue(0, 4); // ri_gp_value
  }
  // .MIPS.options
  else if (MST.isABI_N64()) {
    OS.EmitIntValue(1, 1); // kind
    OS.EmitIntValue(40, 1); // size
    OS.EmitIntValue(0, 2); // section
    OS.EmitIntValue(0, 4); // info
    OS.EmitIntValue(0, 4); // ri_gprmask
    OS.EmitIntValue(0, 4); // pad
    OS.EmitIntValue(0, 4); // ri_cpr[0]mask
    OS.EmitIntValue(0, 4); // ri_cpr[1]mask
    OS.EmitIntValue(0, 4); // ri_cpr[2]mask
    OS.EmitIntValue(0, 4); // ri_cpr[3]mask
    OS.EmitIntValue(0, 8); // ri_gp_value
  }
  else llvm_unreachable("Unsupported abi for reginfo");
}


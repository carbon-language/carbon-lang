//===-- ARMTargetAsmInfo.cpp - ARM asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ARMTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ARMTargetAsmInfo.h"
#include "ARMTargetMachine.h"
#include <cstring>
#include <cctype>
using namespace llvm;

ARMTargetAsmInfo::ARMTargetAsmInfo(const ARMTargetMachine &TM) {
  const ARMSubtarget *Subtarget = &TM.getSubtarget<ARMSubtarget>();
  if (Subtarget->isTargetDarwin()) {
    GlobalPrefix = "_";
    PrivateGlobalPrefix = "L";
    BSSSection = 0;                       // no BSS section.
    ZeroFillDirective = "\t.zerofill\t";  // Uses .zerofill
    SetDirective = "\t.set";
    WeakRefDirective = "\t.weak_reference\t";
    HiddenDirective = "\t.private_extern\t";
    ProtectedDirective = NULL;
    JumpTableDataSection = ".const";
    CStringSection = "\t.cstring";
    FourByteConstantSection = "\t.literal4\n";
    EightByteConstantSection = "\t.literal8\n";
    ReadOnlySection = "\t.const\n";
    HasDotTypeDotSizeDirective = false;
    if (TM.getRelocationModel() == Reloc::Static) {
      StaticCtorsSection = ".constructor";
      StaticDtorsSection = ".destructor";
    } else {
      StaticCtorsSection = ".mod_init_func";
      StaticDtorsSection = ".mod_term_func";
    }
    
    // In non-PIC modes, emit a special label before jump tables so that the
    // linker can perform more accurate dead code stripping.
    if (TM.getRelocationModel() != Reloc::PIC_) {
      // Emit a local label that is preserved until the linker runs.
      JumpTableSpecialLabelPrefix = "l";
    }
    
    NeedsSet = true;
    DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
    DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
    DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
    DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
    DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
    DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
    DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
    DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
    DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
    DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
    DwarfMacInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";
  } else {
    PrivateGlobalPrefix = ".L";
    WeakRefDirective = "\t.weak\t";
    if (Subtarget->isAAPCS_ABI()) {
      StaticCtorsSection = "\t.section .init_array,\"aw\",%init_array";
      StaticDtorsSection = "\t.section .fini_array,\"aw\",%fini_array";
    } else {
      StaticCtorsSection = "\t.section .ctors,\"aw\",%progbits";
      StaticDtorsSection = "\t.section .dtors,\"aw\",%progbits";
    }
    TLSDataSection = "\t.section .tdata,\"awT\",%progbits";
    TLSBSSSection = "\t.section .tbss,\"awT\",%nobits";
  }

  ZeroDirective = "\t.space\t";
  AlignmentIsInBytes = false;
  Data64bitsDirective = 0;
  CommentString = "@";
  DataSection = "\t.data";
  ConstantPoolSection = "\t.text\n";
  COMMDirectiveTakesAlignment = false;
  InlineAsmStart = "@ InlineAsm Start";
  InlineAsmEnd = "@ InlineAsm End";
  LCOMMDirective = "\t.lcomm\t";
  isThumb = Subtarget->isThumb();
}

/// ARM-specific version of TargetAsmInfo::getInlineAsmLength.
unsigned ARMTargetAsmInfo::getInlineAsmLength(const char *Str) const {
  // Count the number of bytes in the asm.
  bool atInsnStart = true;
  unsigned Length = 0;
  for (; *Str; ++Str) {
    if (atInsnStart) {
      // Skip whitespace
      while (*Str && isspace(*Str) && *Str != '\n')
        Str++;
      // Skip label
      for (const char* p = Str; *p && !isspace(*p); p++)
        if (*p == ':') {
          Str = p+1;
          break;
        }
      // Ignore everything from comment char(s) to EOL
      if (strncmp(Str, CommentString, strlen(CommentString))==-0)
        atInsnStart = false;
      else {
        // An instruction
        atInsnStart = false;
        if (isThumb) {
          // BL and BLX <non-reg> are 4 bytes, all others 2.
          const char*p = Str;
          if ((*Str=='b' || *Str=='B') &&
              (*(Str+1)=='l' || *(Str+1)=='L')) {
            if (*(Str+2)=='x' || *(Str+2)=='X') {
              const char* p = Str+3;
              while (*p && isspace(*p))
                p++;
              if (*p == 'r' || *p=='R')
                Length += 2;    // BLX reg
              else
                Length += 4;    // BLX non-reg
            }
            else
              Length += 4;    // BL
          } else
            Length += 2;    // Thumb anything else
        }
        else
          Length += 4;    // ARM
      }
    }
    if (*Str == '\n' || *Str == SeparatorChar)
      atInsnStart = true;
  }
  return Length;
}


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
  Subtarget = &TM.getSubtarget<ARMSubtarget>();
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
    NeedsSet = false;
    HasLEB128 = true;
    AbsoluteDebugSectionOffsets = true;
    ReadOnlySection = "\t.section\t.rodata\n";
    PrivateGlobalPrefix = ".L";
    WeakRefDirective = "\t.weak\t";
    DwarfRequiresFrameSection = false;
    DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"\",%progbits";
    DwarfInfoSection =    "\t.section\t.debug_info,\"\",%progbits";
    DwarfLineSection =    "\t.section\t.debug_line,\"\",%progbits";
    DwarfFrameSection =   "\t.section\t.debug_frame,\"\",%progbits";
    DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"\",%progbits";
    DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"\",%progbits";
    DwarfStrSection =     "\t.section\t.debug_str,\"\",%progbits";
    DwarfLocSection =     "\t.section\t.debug_loc,\"\",%progbits";
    DwarfARangesSection = "\t.section\t.debug_aranges,\"\",%progbits";
    DwarfRangesSection =  "\t.section\t.debug_ranges,\"\",%progbits";
    DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"\",%progbits";

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
}

/// Count the number of comma-separated arguments.
/// Do not try to detect errors.
unsigned ARMTargetAsmInfo::countArguments(const char* p) const {
  unsigned count = 0;
  while (*p && isspace(*p) && *p != '\n')
    p++;
  count++;
  while (*p && *p!='\n' && 
         strncmp(p, CommentString, strlen(CommentString))!=0) {
    if (*p==',')
      count++;
    p++;
  }
  return count;
}

/// Count the length of a string enclosed in quote characters.
/// Do not try to detect errors.
unsigned ARMTargetAsmInfo::countString(const char* p) const {
  unsigned count = 0;
  while (*p && isspace(*p) && *p!='\n')
    p++;
  if (!*p || *p != '\"')
    return count;
  while (*++p && *p != '\"')
    count++;
  return count;
}

/// ARM-specific version of TargetAsmInfo::getInlineAsmLength.
unsigned ARMTargetAsmInfo::getInlineAsmLength(const char *Str) const {
  // Count the number of bytes in the asm.
  bool atInsnStart = true;
  bool inTextSection = true;
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
          while (*Str && isspace(*Str) && *Str != '\n')
            Str++;
          break;
        }
      // Ignore everything from comment char(s) to EOL
      if (strncmp(Str, CommentString, strlen(CommentString))==-0)
        atInsnStart = false;
      // FIXME do something like the following for non-Darwin
      else if (*Str == '.' && Subtarget->isTargetDarwin()) {
        // Directive.
        atInsnStart = false;
        // Some change the section, but don't generate code.
        if (strncasecmp(Str, ".literal4", strlen(".literal4"))==0 ||
            strncasecmp(Str, ".literal8", strlen(".literal8"))==0 ||
            strncasecmp(Str, ".const", strlen(".const"))==0 ||
            strncasecmp(Str, ".constructor", strlen(".constructor"))==0 ||
            strncasecmp(Str, ".cstring", strlen(".cstring"))==0 ||
            strncasecmp(Str, ".data", strlen(".data"))==0 ||
            strncasecmp(Str, ".destructor", strlen(".destructor"))==0 ||
            strncasecmp(Str, ".fvmlib_init0", strlen(".fvmlib_init0"))==0 ||
            strncasecmp(Str, ".fvmlib_init1", strlen(".fvmlib_init1"))==0 ||
            strncasecmp(Str, ".mod_init_func", strlen(".mod_init_func"))==0 ||
            strncasecmp(Str, ".mod_term_func", strlen(".mod_term_func"))==0 ||
            strncasecmp(Str, ".picsymbol_stub", strlen(".picsymbol_stub"))==0 ||
            strncasecmp(Str, ".symbol_stub", strlen(".symbol_stub"))==0 ||
            strncasecmp(Str, ".static_data", strlen(".static_data"))==0 ||
            strncasecmp(Str, ".section", strlen(".section"))==0 ||
            strncasecmp(Str, ".lazy_symbol_pointer", strlen(".lazy_symbol_pointer"))==0 ||
            strncasecmp(Str, ".non_lazy_symbol_pointer", strlen(".non_lazy_symbol_pointer"))==0 ||
            strncasecmp(Str, ".dyld", strlen(".dyld"))==0 ||
            strncasecmp(Str, ".const_data", strlen(".const_data"))==0 ||
            strncasecmp(Str, ".objc", strlen(".objc"))==0 ||       //// many directives
            strncasecmp(Str, ".static_const", strlen(".static_const"))==0)
          inTextSection=false;
        else if (strncasecmp(Str, ".text", strlen(".text"))==0)
          inTextSection = true;
        // Some can't really be handled without implementing significant pieces
        // of an assembler.  Others require dynamic adjustment of block sizes in
        // AdjustBBOffsetsAfter; it's a big compile-time speed hit to check every
        // instruction in there, and none of these are currently used in the kernel.
        else if (strncasecmp(Str, ".macro", strlen(".macro"))==0 ||
                 strncasecmp(Str, ".if", strlen(".if"))==0 ||
                 strncasecmp(Str, ".align", strlen(".align"))==0 ||
                 strncasecmp(Str, ".fill", strlen(".fill"))==0 ||
                 strncasecmp(Str, ".space", strlen(".space"))==0 ||
                 strncasecmp(Str, ".zerofill", strlen(".zerofill"))==0 ||
                 strncasecmp(Str, ".p2align", strlen(".p2align"))==0 ||
                 strncasecmp(Str, ".p2alignw", strlen(".p2alignw"))==0 ||
                 strncasecmp(Str, ".p2alignl", strlen(".p2alignl"))==0 ||
                 strncasecmp(Str, ".align32", strlen(".p2align32"))==0 ||
                 strncasecmp(Str, ".include", strlen(".include"))==0)
          cerr << "Directive " << Str << " in asm may lead to invalid offsets for" <<
                   " constant pools (the assembler will tell you if this happens).\n";
        // Some generate code, but this is only interesting in the text section.
        else if (inTextSection) {
          if (strncasecmp(Str, ".long", strlen(".long"))==0)
            Length += 4*countArguments(Str+strlen(".long"));
          else if (strncasecmp(Str, ".short", strlen(".short"))==0)
            Length += 2*countArguments(Str+strlen(".short"));
          else if (strncasecmp(Str, ".byte", strlen(".byte"))==0)
            Length += 1*countArguments(Str+strlen(".byte"));
          else if (strncasecmp(Str, ".single", strlen(".single"))==0)
            Length += 4*countArguments(Str+strlen(".single"));
          else if (strncasecmp(Str, ".double", strlen(".double"))==0)
            Length += 8*countArguments(Str+strlen(".double"));
          else if (strncasecmp(Str, ".quad", strlen(".quad"))==0)
            Length += 16*countArguments(Str+strlen(".quad"));
          else if (strncasecmp(Str, ".ascii", strlen(".ascii"))==0)
            Length += countString(Str+strlen(".ascii"));
          else if (strncasecmp(Str, ".asciz", strlen(".asciz"))==0)
            Length += countString(Str+strlen(".asciz"))+1;
        }
      } else if (inTextSection) {
        // An instruction
        atInsnStart = false;
        if (Subtarget->isThumb()) {
          // BL and BLX <non-reg> are 4 bytes, all others 2.
          if (strncasecmp(Str, "blx", strlen("blx"))==0) {
            const char* p = Str+3;
            while (*p && isspace(*p))
              p++;
            if (*p == 'r' || *p=='R')
              Length += 2;    // BLX reg
            else
              Length += 4;    // BLX non-reg
          } else if (strncasecmp(Str, "bl", strlen("bl"))==0)
            Length += 4;    // BL
          else
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

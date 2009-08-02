//===- ARMInstrInfo.cpp - ARM Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

ARMInstrInfo::ARMInstrInfo(const ARMSubtarget &STI)
  : RI(*this, STI), Subtarget(STI) {
}

unsigned ARMInstrInfo::getUnindexedOpcode(unsigned Opc) const {
  switch (Opc) {
  default: break;
  case ARM::LDR_PRE:
  case ARM::LDR_POST:
    return ARM::LDR;
  case ARM::LDRH_PRE:
  case ARM::LDRH_POST:
    return ARM::LDRH;
  case ARM::LDRB_PRE:
  case ARM::LDRB_POST:
    return ARM::LDRB;
  case ARM::LDRSH_PRE:
  case ARM::LDRSH_POST:
    return ARM::LDRSH;
  case ARM::LDRSB_PRE:
  case ARM::LDRSB_POST:
    return ARM::LDRSB;
  case ARM::STR_PRE:
  case ARM::STR_POST:
    return ARM::STR;
  case ARM::STRH_PRE:
  case ARM::STRH_POST:
    return ARM::STRH;
  case ARM::STRB_PRE:
  case ARM::STRB_POST:
    return ARM::STRB;
  }

  return 0;
}

bool ARMInstrInfo::BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;

  switch (MBB.back().getOpcode()) {
  case ARM::BX_RET:   // Return.
  case ARM::LDM_RET:
  case ARM::B:
  case ARM::BR_JTr:   // Jumptable branch.
  case ARM::BR_JTm:   // Jumptable branch through mem.
  case ARM::BR_JTadd: // Jumptable branch add to pc.
    return true;
  default:
    break;
  }

  return false;
}

void ARMInstrInfo::
reMaterialize(MachineBasicBlock &MBB,
              MachineBasicBlock::iterator I,
              unsigned DestReg, unsigned SubIdx,
              const MachineInstr *Orig) const {
  DebugLoc dl = Orig->getDebugLoc();
  if (Orig->getOpcode() == ARM::MOVi2pieces) {
    RI.emitLoadConstPool(MBB, I, dl,
                         DestReg, SubIdx,
                         Orig->getOperand(1).getImm(),
                         (ARMCC::CondCodes)Orig->getOperand(2).getImm(),
                         Orig->getOperand(3).getReg());
    return;
  }

  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MI->getOperand(0).setReg(DestReg);
  MBB.insert(I, MI);
}

/// Count the number of comma-separated arguments.
/// Do not try to detect errors.
static unsigned countArguments(const char* p,
                               const TargetAsmInfo &TAI) {
  unsigned count = 0;
  while (*p && isspace(*p) && *p != '\n')
    p++;
  count++;
  while (*p && *p!='\n' &&
         strncmp(p, TAI.getCommentString(),
                 strlen(TAI.getCommentString())) != 0) {
    if (*p==',')
      count++;
    p++;
  }
  return count;
}

/// Count the length of a string enclosed in quote characters.
/// Do not try to detect errors.
static unsigned countString(const char *p) {
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
unsigned ARMInstrInfo::getInlineAsmLength(const char *s,
                                          const TargetAsmInfo &TAI) const {
  // Make a lowercase-folded version of s for counting purposes.
  char *q, *s_copy = (char *)malloc(strlen(s) + 1);
  strcpy(s_copy, s);
  for (q=s_copy; *q; q++)
    *q = tolower(*q);
  const char *Str = s_copy;

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
      
      if (*Str == 0) break;
      
      // Ignore everything from comment char(s) to EOL
      if (strncmp(Str, TAI.getCommentString(),
                  strlen(TAI.getCommentString())) == 0)
        atInsnStart = false;
      // FIXME do something like the following for non-Darwin
      else if (*Str == '.' && Subtarget.isTargetDarwin()) {
        // Directive.
        atInsnStart = false;

        // Some change the section, but don't generate code.
        if (strncmp(Str, ".literal4", strlen(".literal4"))==0 ||
            strncmp(Str, ".literal8", strlen(".literal8"))==0 ||
            strncmp(Str, ".const", strlen(".const"))==0 ||
            strncmp(Str, ".constructor", strlen(".constructor"))==0 ||
            strncmp(Str, ".cstring", strlen(".cstring"))==0 ||
            strncmp(Str, ".data", strlen(".data"))==0 ||
            strncmp(Str, ".destructor", strlen(".destructor"))==0 ||
            strncmp(Str, ".fvmlib_init0", strlen(".fvmlib_init0"))==0 ||
            strncmp(Str, ".fvmlib_init1", strlen(".fvmlib_init1"))==0 ||
            strncmp(Str, ".mod_init_func", strlen(".mod_init_func"))==0 ||
            strncmp(Str, ".mod_term_func", strlen(".mod_term_func"))==0 ||
            strncmp(Str, ".picsymbol_stub", strlen(".picsymbol_stub"))==0 ||
            strncmp(Str, ".symbol_stub", strlen(".symbol_stub"))==0 ||
            strncmp(Str, ".static_data", strlen(".static_data"))==0 ||
            strncmp(Str, ".section", strlen(".section"))==0 ||
            strncmp(Str, ".lazy_symbol_pointer", strlen(".lazy_symbol_pointer"))==0 ||
            strncmp(Str, ".non_lazy_symbol_pointer", strlen(".non_lazy_symbol_pointer"))==0 ||
            strncmp(Str, ".dyld", strlen(".dyld"))==0 ||
            strncmp(Str, ".const_data", strlen(".const_data"))==0 ||
            strncmp(Str, ".objc", strlen(".objc"))==0 ||       //// many directives
            strncmp(Str, ".static_const", strlen(".static_const"))==0)
          inTextSection=false;
        else if (strncmp(Str, ".text", strlen(".text"))==0)
          inTextSection = true;
        // Some can't really be handled without implementing significant pieces
        // of an assembler.  Others require dynamic adjustment of block sizes in
        // AdjustBBOffsetsAfter; it's a big compile-time speed hit to check every
        // instruction in there, and none of these are currently used in the kernel.
        else if (strncmp(Str, ".macro", strlen(".macro"))==0 ||
                 strncmp(Str, ".if", strlen(".if"))==0 ||
                 strncmp(Str, ".align", strlen(".align"))==0 ||
                 strncmp(Str, ".fill", strlen(".fill"))==0 ||
                 strncmp(Str, ".space", strlen(".space"))==0 ||
                 strncmp(Str, ".zerofill", strlen(".zerofill"))==0 ||
                 strncmp(Str, ".p2align", strlen(".p2align"))==0 ||
                 strncmp(Str, ".p2alignw", strlen(".p2alignw"))==0 ||
                 strncmp(Str, ".p2alignl", strlen(".p2alignl"))==0 ||
                 strncmp(Str, ".align32", strlen(".p2align32"))==0 ||
                 strncmp(Str, ".include", strlen(".include"))==0)
          cerr << "Directive " << Str << " in asm may lead to invalid offsets for" <<
                   " constant pools (the assembler will tell you if this happens).\n";
        // Some generate code, but this is only interesting in the text section.
        else if (inTextSection) {
          if (strncmp(Str, ".long", strlen(".long"))==0)
            Length += 4*countArguments(Str+strlen(".long"), TAI);
          else if (strncmp(Str, ".short", strlen(".short"))==0)
            Length += 2*countArguments(Str+strlen(".short"), TAI);
          else if (strncmp(Str, ".byte", strlen(".byte"))==0)
            Length += 1*countArguments(Str+strlen(".byte"), TAI);
          else if (strncmp(Str, ".single", strlen(".single"))==0)
            Length += 4*countArguments(Str+strlen(".single"), TAI);
          else if (strncmp(Str, ".double", strlen(".double"))==0)
            Length += 8*countArguments(Str+strlen(".double"), TAI);
          else if (strncmp(Str, ".quad", strlen(".quad"))==0)
            Length += 16*countArguments(Str+strlen(".quad"), TAI);
          else if (strncmp(Str, ".ascii", strlen(".ascii"))==0)
            Length += countString(Str+strlen(".ascii"));
          else if (strncmp(Str, ".asciz", strlen(".asciz"))==0)
            Length += countString(Str+strlen(".asciz"))+1;
        }
      } else if (inTextSection) {
        // An instruction
        atInsnStart = false;
        if (Subtarget.isThumb()) {  // FIXME thumb2
          // BL and BLX <non-reg> are 4 bytes, all others 2.
          if (strncmp(Str, "blx", strlen("blx"))==0) {
            const char* p = Str+3;
            while (*p && isspace(*p))
              p++;
            if (*p == 'r' || *p=='R')
              Length += 2;    // BLX reg
            else
              Length += 4;    // BLX non-reg
          } else if (strncmp(Str, "bl", strlen("bl"))==0)
            Length += 4;    // BL
          else
            Length += 2;    // Thumb anything else
        }
        else
          Length += 4;    // ARM
      }
    }
    if (*Str == '\n' || *Str == TAI.getSeparatorChar())
      atInsnStart = true;
  }
  free(s_copy);
  return Length;
}


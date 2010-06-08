/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- AssemblyParser.hpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Disassemble the prologue instructions in functions, create a profile
// of stack movements and register saves performed therein.

#ifndef __ASSEMBLY_PARSER_HPP
#define __ASSEMBLY_PARSER_HPP

#if defined (SUPPORT_REMOTE_UNWINDING)

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <vector>

#include "libunwind.h"
#include "RemoteProcInfo.hpp"
#include "Registers.hpp"
#include "FileAbstraction.hpp"
#include "AddressSpace.hpp"
#include "RemoteUnwindProfile.h"

namespace lldb_private
{

// Analyze the instructions in an x86_64/i386 function prologue, fill out an RemoteUnwindProfile.

class AssemblyParse_x86 {
public:
    AssemblyParse_x86 (RemoteProcInfo& procinfo, unw_accessors_t *acc, unw_addr_space_t as, void *arg) : fArg(arg), fAccessors(acc), fAs(as), fRemoteProcInfo(procinfo) {
        fRegisterMap = fRemoteProcInfo.getRegisterMap();
        if (fRemoteProcInfo.getTargetArch() == UNW_TARGET_X86_64) {
            fStackPointerRegnum = UNW_X86_64_RSP;
            fFramePointerRegnum = UNW_X86_64_RBP;
            fWordSize = 8;
        } else {
            fStackPointerRegnum = UNW_X86_ESP;
            fFramePointerRegnum = UNW_X86_EBP;
            fWordSize = 4;
        }
    }

    uint32_t extract_4_LE (uint8_t *b) {
        uint32_t v = 0;
        for (int i = 3; i >= 0; i--)
            v = (v << 8) | b[i];
        return v;
    }

    bool push_rbp_pattern_p ();
    bool push_0_pattern_p ();
    bool mov_rsp_rbp_pattern_p ();
    bool sub_rsp_pattern_p (int *amount);
    bool push_reg_p (int *regno);
    bool mov_reg_to_local_stack_frame_p (int *regno, int *rbp_offset);
    bool ret_pattern_p ();
    bool profileFunction (uint64_t start, uint64_t end, RemoteUnwindProfile& profile);

private:

    void *fArg;
    uint8_t*           fCurInsnByteBuf;
    int                fCurInsnSize;
    RemoteProcInfo&    fRemoteProcInfo;
    RemoteRegisterMap  *fRegisterMap;
    unw_accessors_t    *fAccessors;
    unw_addr_space_t   fAs;
    int                fWordSize;
    int                fStackPointerRegnum;
    int                fFramePointerRegnum;
};

// Macro to detect if this is a REX mode prefix byte. 
#define REX_W_PREFIX_P(opcode) (((opcode) & (~0x5)) == 0x48)

// The high bit which should be added to the source register number (the "R" bit)
#define REX_W_SRCREG(opcode) (((opcode) & 0x4) >> 2)

// The high bit which should be added to the destination register number (the "B" bit)
#define REX_W_DSTREG(opcode) ((opcode) & 0x1)

// pushq %rbp [0x55]
bool AssemblyParse_x86::push_rbp_pattern_p () {
    uint8_t *p = fCurInsnByteBuf;
    if (*p == 0x55)
      return true;
    return false;
}

// pushq $0 ; the first instruction in start() [0x6a 0x00]
bool AssemblyParse_x86::push_0_pattern_p ()
{
    uint8_t *p = fCurInsnByteBuf;
    if (*p == 0x6a && *(p + 1) == 0x0)
        return true;
    return false;
}

// movq %rsp, %rbp [0x48 0x8b 0xec] or [0x48 0x89 0xe5]
// movl %esp, %ebp [0x8b 0xec] or [0x89 0xe5]
bool AssemblyParse_x86::mov_rsp_rbp_pattern_p () {
    uint8_t *p = fCurInsnByteBuf;
    if (fWordSize == 8 && *p == 0x48)
      p++;
    if (*(p) == 0x8b && *(p + 1) == 0xec)
        return true;
    if (*(p) == 0x89 && *(p + 1) == 0xe5)
        return true;
    return false;
}

// subq $0x20, %rsp 
bool AssemblyParse_x86::sub_rsp_pattern_p (int *amount) {
    uint8_t *p = fCurInsnByteBuf;
    if (fWordSize == 8 && *p == 0x48)
      p++;
    // 8-bit immediate operand
    if (*p == 0x83 && *(p + 1) == 0xec) {
        *amount = (int8_t) *(p + 2);
        return true;
    }
    // 32-bit immediate operand
    if (*p == 0x81 && *(p + 1) == 0xec) {
        *amount = (int32_t) extract_4_LE (p + 2);
        return true;
    }
    // Not handled:  [0x83 0xc4] for imm8 with neg values
    // [0x81 0xc4] for imm32 with neg values
    return false;
}

// pushq %rbx
// pushl $ebx
bool AssemblyParse_x86::push_reg_p (int *regno) {
    uint8_t *p = fCurInsnByteBuf;
    int regno_prefix_bit = 0;
    // If we have a rex prefix byte, check to see if a B bit is set
    if (fWordSize == 8 && *p == 0x41) {
        regno_prefix_bit = 1 << 3;
        p++;
    }
    if (*p >= 0x50 && *p <= 0x57) {
        int r = (*p - 0x50) | regno_prefix_bit;
        if (fRegisterMap->machine_regno_to_unwind_regno (r, *regno) == true) {
            return true;
        }
    }
    return false;
}

// Look for an instruction sequence storing a nonvolatile register
// on to the stack frame.

//  movq %rax, -0x10(%rbp) [0x48 0x89 0x45 0xf0]
//  movl %eax, -0xc(%ebp)  [0x89 0x45 0xf4]
bool AssemblyParse_x86::mov_reg_to_local_stack_frame_p (int *regno, int *rbp_offset) {
    uint8_t *p = fCurInsnByteBuf;
    int src_reg_prefix_bit = 0;
    int target_reg_prefix_bit = 0;

    if (fWordSize == 8 && REX_W_PREFIX_P (*p)) {
        src_reg_prefix_bit = REX_W_SRCREG (*p) << 3;
        target_reg_prefix_bit = REX_W_DSTREG (*p) << 3;
        if (target_reg_prefix_bit == 1) {
            // rbp/ebp don't need a prefix bit - we know this isn't the
            // reg we care about.
            return false;
        }
        p++;
    }

    if (*p == 0x89) {
        /* Mask off the 3-5 bits which indicate the destination register
           if this is a ModR/M byte.  */
        int opcode_destreg_masked_out = *(p + 1) & (~0x38);

        /* Is this a ModR/M byte with Mod bits 01 and R/M bits 101 
           and three bits between them, e.g. 01nnn101
           We're looking for a destination of ebp-disp8 or ebp-disp32.   */
        int immsize;
        if (opcode_destreg_masked_out == 0x45)
          immsize = 2;
        else if (opcode_destreg_masked_out == 0x85)
          immsize = 4;
        else
          return false;

        int offset = 0;
        if (immsize == 2)
          offset = (int8_t) *(p + 2);
        if (immsize == 4)
          offset = (uint32_t) extract_4_LE (p + 2);
        if (offset > 0)
          return false;

        int savedreg = ((*(p + 1) >> 3) & 0x7) | src_reg_prefix_bit;
        if (fRegisterMap->machine_regno_to_unwind_regno (savedreg, *regno) == true) {
            *rbp_offset = offset > 0 ? offset : -offset;
            return true;
        }
    }
    return false;
}

// ret [0xc9] or [0xc2 imm8] or [0xca imm8]
bool AssemblyParse_x86::ret_pattern_p () {
    uint8_t *p = fCurInsnByteBuf;
    if (*p == 0xc9 || *p == 0xc2 || *p == 0xca || *p == 0xc3)
        return true;
    return false;
}

bool AssemblyParse_x86::profileFunction (uint64_t start, uint64_t end, RemoteUnwindProfile& profile) {
    if (start == -1 || end == 0)
        return false;

    profile.fStart = start;
    profile.fEnd = end;
    profile.fRegSizes[RemoteUnwindProfile::kGeneralPurposeRegister] = fWordSize;
    profile.fRegSizes[RemoteUnwindProfile::kFloatingPointRegister] = 8;
    profile.fRegSizes[RemoteUnwindProfile::kVectorRegister] = 16;

    // On function entry, the CFA is rsp+fWordSize

    RemoteUnwindProfile::CFALocation initial_cfaloc;
    initial_cfaloc.regno = fStackPointerRegnum;
    initial_cfaloc.offset = fWordSize;
    profile.cfa[start] = initial_cfaloc;

    // The return address is at CFA - fWordSize
    // CFA doesn't change value during the lifetime of the function (hence "C")
    // so the returnAddress is the same for the duration of the function.

    profile.returnAddress.regno = 0;
    profile.returnAddress.location = RemoteUnwindProfile::kRegisterOffsetFromCFA;
    profile.returnAddress.value = -fWordSize;
    profile.returnAddress.adj = 0;
    profile.returnAddress.type = RemoteUnwindProfile::kGeneralPurposeRegister;

    // The caller's rsp has the same value as the CFA at all points during 
    // this function's lifetime.  

    RemoteUnwindProfile::SavedReg rsp_loc;
    rsp_loc.regno = fStackPointerRegnum;
    rsp_loc.location = RemoteUnwindProfile::kRegisterIsCFA;
    rsp_loc.value = 0;
    rsp_loc.adj = 0;
    rsp_loc.type = RemoteUnwindProfile::kGeneralPurposeRegister;
    profile.saved_registers[start].push_back(rsp_loc);
    profile.fRegistersSaved[fStackPointerRegnum] = 1;

    int non_prologue_insn_count = 0;
    int insn_count = 0;
    uint64_t cur_addr = start;
    uint64_t first_insn_past_prologue = start;
    int push_rbp_seen = 0;
    int current_cfa_register = fStackPointerRegnum;
    int sp_adjustments = 0;

    while (cur_addr < end && non_prologue_insn_count < 10)
    {
        int offset, regno;
        uint64_t next_addr;
        insn_count++;
        int is_prologue_insn = 0;

        if (fAccessors->instruction_length (fAs, cur_addr, &fCurInsnSize, fArg) != 0) {
            /* An error parsing the instruction; stop scanning.  */
            break;
        }
        fCurInsnByteBuf = (uint8_t *) malloc (fCurInsnSize);
        if (fRemoteProcInfo.getBytes (cur_addr, fCurInsnSize, fCurInsnByteBuf, fArg) == 0)
          return false;
        next_addr = cur_addr + fCurInsnSize;

        // start () opens with a 'push $0x0' which is in the saved ip slot on the stack -
        // so we know to stop backtracing here.  We need to ignore this instruction.
        if (push_0_pattern_p () && push_rbp_seen == 0 && insn_count == 1)
        {
            cur_addr = next_addr;
            first_insn_past_prologue = next_addr;
            continue;
        }

        if (push_rbp_pattern_p () && push_rbp_seen == 0)
          {
            if (current_cfa_register == fStackPointerRegnum) {
                sp_adjustments -= fWordSize;
                RemoteUnwindProfile::CFALocation cfaloc;
                cfaloc.regno = fStackPointerRegnum;
                cfaloc.offset = abs (sp_adjustments - fWordSize);
                profile.cfa[next_addr] = cfaloc;
            }

            RemoteUnwindProfile::SavedReg sreg;
            sreg.regno = fFramePointerRegnum;
            sreg.location = RemoteUnwindProfile::kRegisterOffsetFromCFA;
            sreg.value = sp_adjustments - fWordSize;
            sreg.adj = 0;
            sreg.type = RemoteUnwindProfile::kGeneralPurposeRegister;
            profile.saved_registers[next_addr].push_back(sreg);

            push_rbp_seen = 1;
            profile.fRegistersSaved[fFramePointerRegnum] = 1;
            is_prologue_insn = 1;
            goto next_iteration;
          }
        if (mov_rsp_rbp_pattern_p ()) {
            RemoteUnwindProfile::CFALocation cfaloc;
            cfaloc.regno = fFramePointerRegnum;
            cfaloc.offset = abs (sp_adjustments - fWordSize);
            profile.cfa[next_addr] = cfaloc;
            current_cfa_register = fFramePointerRegnum;
            is_prologue_insn = 1;
            goto next_iteration;
        }
        if (ret_pattern_p ()) {
            break;
        }
        if (sub_rsp_pattern_p (&offset)) {
            sp_adjustments -= offset;
            if (current_cfa_register == fStackPointerRegnum) {
               RemoteUnwindProfile::CFALocation cfaloc;
               cfaloc.regno = fStackPointerRegnum;
               cfaloc.offset = abs (sp_adjustments - fWordSize);
               profile.cfa[next_addr] = cfaloc;
            }
            is_prologue_insn = 1;
        }
        if (push_reg_p (&regno)) {
            sp_adjustments -= fWordSize;
            if (current_cfa_register == fStackPointerRegnum) {
                RemoteUnwindProfile::CFALocation cfaloc;
                cfaloc.regno = fStackPointerRegnum;
                cfaloc.offset = abs (sp_adjustments - fWordSize);
                profile.cfa[next_addr] = cfaloc;
                is_prologue_insn = 1;
            }
            if (fRegisterMap->nonvolatile_reg_p (regno) && profile.fRegistersSaved[regno] == 0) {
                RemoteUnwindProfile::SavedReg sreg;
                sreg.regno = regno;
                sreg.location = RemoteUnwindProfile::kRegisterOffsetFromCFA;
                sreg.value = sp_adjustments - fWordSize;
                sreg.adj = 0;
                sreg.type = RemoteUnwindProfile::kGeneralPurposeRegister;
                profile.saved_registers[next_addr].push_back(sreg);
                profile.fRegistersSaved[regno] = 1;
                is_prologue_insn = 1;
            }
        }
        if (mov_reg_to_local_stack_frame_p (&regno, &offset) 
            && fRegisterMap->nonvolatile_reg_p (regno)
            && profile.fRegistersSaved[regno] == 0) {
                RemoteUnwindProfile::SavedReg sreg;
                sreg.regno = regno;
                sreg.location = RemoteUnwindProfile::kRegisterOffsetFromCFA;
                sreg.value = offset - fWordSize;
                sreg.adj = 0;
                sreg.type = RemoteUnwindProfile::kGeneralPurposeRegister;
                profile.saved_registers[next_addr].push_back(sreg);
                profile.fRegistersSaved[regno] = 1;
                is_prologue_insn = 1;
        }
next_iteration:
        if (is_prologue_insn) {
            first_insn_past_prologue = next_addr;
            non_prologue_insn_count = 0;
        }
        cur_addr = next_addr;
        non_prologue_insn_count++;
    }
    profile.fFirstInsnPastPrologue = first_insn_past_prologue;
    return true;
}




bool AssemblyParse (RemoteProcInfo *procinfo, unw_accessors_t *acc, unw_addr_space_t as, uint64_t start, uint64_t end, RemoteUnwindProfile &profile, void *arg) {
    if (procinfo->getTargetArch() == UNW_TARGET_X86_64 || procinfo->getTargetArch() == UNW_TARGET_I386) {
        AssemblyParse_x86 parser(*procinfo, acc, as, arg);
        return parser.profileFunction (start, end, profile);
    } else {
        ABORT("Only x86_64 and i386 assembly parsing supported at this time");
        return false;
    }
}

}; // namespace lldb_private

#endif // SUPPORT_REMOTE_UNWINDING
#endif  //ASSEMBLY_PARSER_HPP

//===-- ARM_DWARF_Registers.c -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM_DWARF_Registers.h"

const char *
GetARMDWARFRegisterName (unsigned reg_num)
{
    switch (reg_num)
    {
        case dwarf_r0:  return "r0";
        case dwarf_r1:  return "r1";
        case dwarf_r2:  return "r2";
        case dwarf_r3:  return "r3";
        case dwarf_r4:  return "r4";
        case dwarf_r5:  return "r5";
        case dwarf_r6:  return "r6";
        case dwarf_r7:  return "r7";
        case dwarf_r8:  return "r8";
        case dwarf_r9:  return "r9";        
        case dwarf_r10: return "r10";
        case dwarf_r11: return "r11";
        case dwarf_r12: return "r12";
        case dwarf_sp:  return "sp";
        case dwarf_lr:  return "lr";
        case dwarf_pc:  return "pc";
        case dwarf_cpsr:return "cpsr";
            
        case dwarf_s0:  return "s0";
        case dwarf_s1:  return "s1";
        case dwarf_s2:  return "s2";
        case dwarf_s3:  return "s3";
        case dwarf_s4:  return "s4";
        case dwarf_s5:  return "s5";
        case dwarf_s6:  return "s6";
        case dwarf_s7:  return "s7";
        case dwarf_s8:  return "s8";
        case dwarf_s9:  return "s9";
        case dwarf_s10: return "s10";
        case dwarf_s11: return "s11";
        case dwarf_s12: return "s12";
        case dwarf_s13: return "s13";
        case dwarf_s14: return "s14";
        case dwarf_s15: return "s15";
        case dwarf_s16: return "s16";
        case dwarf_s17: return "s17";
        case dwarf_s18: return "s18";
        case dwarf_s19: return "s19";
        case dwarf_s20: return "s20";
        case dwarf_s21: return "s21";
        case dwarf_s22: return "s22";
        case dwarf_s23: return "s23";
        case dwarf_s24: return "s24";
        case dwarf_s25: return "s25";
        case dwarf_s26: return "s26";
        case dwarf_s27: return "s27";
        case dwarf_s28: return "s28";
        case dwarf_s29: return "s29";
        case dwarf_s30: return "s30";
        case dwarf_s31: return "s31";
            
        // FPA Registers 0-7
        case dwarf_f0:  return "f0";
        case dwarf_f1:  return "f1";
        case dwarf_f2:  return "f2";
        case dwarf_f3:  return "f3";
        case dwarf_f4:  return "f4";
        case dwarf_f5:  return "f5";
        case dwarf_f6:  return "f6";
        case dwarf_f7:  return "f7";
            
        // Intel wireless MMX general purpose registers 0–7
        // XScale accumulator register 0–7 (they do overlap with wCGR0 - wCGR7)
        case dwarf_wCGR0: return "wCGR0/ACC0";   
        case dwarf_wCGR1: return "wCGR1/ACC1";
        case dwarf_wCGR2: return "wCGR2/ACC2";
        case dwarf_wCGR3: return "wCGR3/ACC3";
        case dwarf_wCGR4: return "wCGR4/ACC4";
        case dwarf_wCGR5: return "wCGR5/ACC5";
        case dwarf_wCGR6: return "wCGR6/ACC6";
        case dwarf_wCGR7: return "wCGR7/ACC7";
            
        // Intel wireless MMX data registers 0–15
        case dwarf_wR0:   return "wR0";
        case dwarf_wR1:   return "wR1";
        case dwarf_wR2:   return "wR2";
        case dwarf_wR3:   return "wR3";
        case dwarf_wR4:   return "wR4";
        case dwarf_wR5:   return "wR5";
        case dwarf_wR6:   return "wR6";
        case dwarf_wR7:   return "wR7";
        case dwarf_wR8:   return "wR8";
        case dwarf_wR9:   return "wR9";
        case dwarf_wR10:  return "wR10";
        case dwarf_wR11:  return "wR11";
        case dwarf_wR12:  return "wR12";
        case dwarf_wR13:  return "wR13";
        case dwarf_wR14:  return "wR14";
        case dwarf_wR15:  return "wR15";
            
        case dwarf_spsr:        return "spsr";
        case dwarf_spsr_fiq:    return "spsr_fiq";
        case dwarf_spsr_irq:    return "spsr_irq";
        case dwarf_spsr_abt:    return "spsr_abt";
        case dwarf_spsr_und:    return "spsr_und";
        case dwarf_spsr_svc:    return "spsr_svc";
            
        case dwarf_r8_usr:      return "r8_usr";
        case dwarf_r9_usr:      return "r9_usr";
        case dwarf_r10_usr:     return "r10_usr";
        case dwarf_r11_usr:     return "r11_usr";
        case dwarf_r12_usr:     return "r12_usr";
        case dwarf_r13_usr:     return "r13_usr";
        case dwarf_r14_usr:     return "r14_usr";
        case dwarf_r8_fiq:      return "r8_fiq";
        case dwarf_r9_fiq:      return "r9_fiq";
        case dwarf_r10_fiq:     return "r10_fiq";
        case dwarf_r11_fiq:     return "r11_fiq";
        case dwarf_r12_fiq:     return "r12_fiq";
        case dwarf_r13_fiq:     return "r13_fiq";
        case dwarf_r14_fiq:     return "r14_fiq";
        case dwarf_r13_irq:     return "r13_irq";
        case dwarf_r14_irq:     return "r14_irq";
        case dwarf_r13_abt:     return "r13_abt";
        case dwarf_r14_abt:     return "r14_abt";
        case dwarf_r13_und:     return "r13_und";
        case dwarf_r14_und:     return "r14_und";
        case dwarf_r13_svc:     return "r13_svc";
        case dwarf_r14_svc:     return "r14_svc";
            
        // Intel wireless MMX control register in co-processor 0–7
        case dwarf_wC0:         return "wC0";
        case dwarf_wC1:         return "wC1";
        case dwarf_wC2:         return "wC2";
        case dwarf_wC3:         return "wC3";
        case dwarf_wC4:         return "wC4";
        case dwarf_wC5:         return "wC5";
        case dwarf_wC6:         return "wC6";
        case dwarf_wC7:         return "wC7";
            
        // VFP-v3/Neon
        case dwarf_d0:          return "d0";
        case dwarf_d1:          return "d1";
        case dwarf_d2:          return "d2";
        case dwarf_d3:          return "d3";
        case dwarf_d4:          return "d4";
        case dwarf_d5:          return "d5";
        case dwarf_d6:          return "d6";
        case dwarf_d7:          return "d7";
        case dwarf_d8:          return "d8";
        case dwarf_d9:          return "d9";
        case dwarf_d10:         return "d10";
        case dwarf_d11:         return "d11";
        case dwarf_d12:         return "d12";
        case dwarf_d13:         return "d13";
        case dwarf_d14:         return "d14";
        case dwarf_d15:         return "d15";
        case dwarf_d16:         return "d16";
        case dwarf_d17:         return "d17";
        case dwarf_d18:         return "d18";
        case dwarf_d19:         return "d19";
        case dwarf_d20:         return "d20";
        case dwarf_d21:         return "d21";
        case dwarf_d22:         return "d22";
        case dwarf_d23:         return "d23";
        case dwarf_d24:         return "d24";
        case dwarf_d25:         return "d25";
        case dwarf_d26:         return "d26";
        case dwarf_d27:         return "d27";
        case dwarf_d28:         return "d28";
        case dwarf_d29:         return "d29";
        case dwarf_d30:         return "d30";
        case dwarf_d31:         return "d31";
    }
    return 0;
}


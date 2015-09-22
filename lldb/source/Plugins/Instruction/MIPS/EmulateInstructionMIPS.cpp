//===-- EmulateInstructionMIPS.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EmulateInstructionMIPS.h"

#include <stdlib.h>

#include "llvm-c/Disassembler.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Opcode.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/UnwindPlan.h"

#include "llvm/ADT/STLExtras.h"

#include "Plugins/Process/Utility/InstructionUtils.h"
#include "Plugins/Process/Utility/RegisterContext_mips.h"  //mips32 has same registers nos as mips64

using namespace lldb;
using namespace lldb_private;

#define UInt(x) ((uint64_t)x)
#define integer int64_t


//----------------------------------------------------------------------
//
// EmulateInstructionMIPS implementation
//
//----------------------------------------------------------------------

#ifdef __mips__
extern "C" {
    void LLVMInitializeMipsTargetInfo ();
    void LLVMInitializeMipsTarget ();
    void LLVMInitializeMipsAsmPrinter ();
    void LLVMInitializeMipsTargetMC ();
    void LLVMInitializeMipsDisassembler ();
}
#endif

EmulateInstructionMIPS::EmulateInstructionMIPS (const lldb_private::ArchSpec &arch) :
    EmulateInstruction (arch)
{
    /* Create instance of llvm::MCDisassembler */
    std::string Error;
    llvm::Triple triple = arch.GetTriple();
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget (triple.getTriple(), Error);

    /*
     * If we fail to get the target then we haven't registered it. The SystemInitializerCommon 
     * does not initialize targets, MCs and disassemblers. However we need the MCDisassembler 
     * to decode the instructions so that the decoding complexity stays with LLVM. 
     * Initialize the MIPS targets and disassemblers.
    */
#ifdef __mips__
    if (!target)
    {
        LLVMInitializeMipsTargetInfo ();
        LLVMInitializeMipsTarget ();
        LLVMInitializeMipsAsmPrinter ();
        LLVMInitializeMipsTargetMC ();
        LLVMInitializeMipsDisassembler ();
        target = llvm::TargetRegistry::lookupTarget (triple.getTriple(), Error);
    }
#endif

    assert (target);

    llvm::StringRef cpu;

    switch (arch.GetCore())
    {
        case ArchSpec::eCore_mips32:
        case ArchSpec::eCore_mips32el:
            cpu = "mips32"; break;
        case ArchSpec::eCore_mips32r2:
        case ArchSpec::eCore_mips32r2el:
            cpu = "mips32r2"; break;
        case ArchSpec::eCore_mips32r3:
        case ArchSpec::eCore_mips32r3el:
            cpu = "mips32r3"; break;
        case ArchSpec::eCore_mips32r5:
        case ArchSpec::eCore_mips32r5el:
            cpu = "mips32r5"; break;
        case ArchSpec::eCore_mips32r6:
        case ArchSpec::eCore_mips32r6el:
            cpu = "mips32r6"; break;
        case ArchSpec::eCore_mips64:
        case ArchSpec::eCore_mips64el:
            cpu = "mips64"; break;
        case ArchSpec::eCore_mips64r2:
        case ArchSpec::eCore_mips64r2el:
            cpu = "mips64r2"; break;
        case ArchSpec::eCore_mips64r3:
        case ArchSpec::eCore_mips64r3el:
            cpu = "mips64r3"; break;
        case ArchSpec::eCore_mips64r5:
        case ArchSpec::eCore_mips64r5el:
            cpu = "mips64r5"; break;
        case ArchSpec::eCore_mips64r6:
        case ArchSpec::eCore_mips64r6el:
            cpu = "mips64r6"; break;
        default:
            cpu = "generic"; break;
    }

    std::string features = "";
    uint32_t arch_flags = arch.GetFlags ();
    if (arch_flags & ArchSpec::eMIPSAse_msa)
        features += "+msa,";
    if (arch_flags & ArchSpec::eMIPSAse_dsp)
        features += "+dsp,";
    if (arch_flags & ArchSpec::eMIPSAse_dspr2)
        features += "+dspr2,";
    if (arch_flags & ArchSpec::eMIPSAse_mips16)
        features += "+mips16,";
    if (arch_flags & ArchSpec::eMIPSAse_micromips)
        features += "+micromips,";

    m_reg_info.reset (target->createMCRegInfo (triple.getTriple()));
    assert (m_reg_info.get());

    m_insn_info.reset (target->createMCInstrInfo());
    assert (m_insn_info.get());

    m_asm_info.reset (target->createMCAsmInfo (*m_reg_info, triple.getTriple()));
    m_subtype_info.reset (target->createMCSubtargetInfo (triple.getTriple(), cpu, features));
    assert (m_asm_info.get() && m_subtype_info.get());

    m_context.reset (new llvm::MCContext (m_asm_info.get(), m_reg_info.get(), nullptr));
    assert (m_context.get());

    m_disasm.reset (target->createMCDisassembler (*m_subtype_info, *m_context));
    assert (m_disasm.get());
}

void
EmulateInstructionMIPS::Initialize ()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic (),
                                   GetPluginDescriptionStatic (),
                                   CreateInstance);
}

void
EmulateInstructionMIPS::Terminate ()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

ConstString
EmulateInstructionMIPS::GetPluginNameStatic ()
{
    ConstString g_plugin_name ("lldb.emulate-instruction.mips32");
    return g_plugin_name;
}

lldb_private::ConstString
EmulateInstructionMIPS::GetPluginName()
{
    static ConstString g_plugin_name ("EmulateInstructionMIPS");
    return g_plugin_name;
}

const char *
EmulateInstructionMIPS::GetPluginDescriptionStatic ()
{
    return "Emulate instructions for the MIPS32 architecture.";
}

EmulateInstruction *
EmulateInstructionMIPS::CreateInstance (const ArchSpec &arch, InstructionType inst_type)
{
    if (EmulateInstructionMIPS::SupportsEmulatingInstructionsOfTypeStatic(inst_type))
    {
        if (arch.GetTriple().getArch() == llvm::Triple::mips
            || arch.GetTriple().getArch() == llvm::Triple::mipsel)
        {
            std::auto_ptr<EmulateInstructionMIPS> emulate_insn_ap (new EmulateInstructionMIPS (arch));
            if (emulate_insn_ap.get())
                return emulate_insn_ap.release();
        }
    }
    
    return NULL;
}

bool
EmulateInstructionMIPS::SetTargetTriple (const ArchSpec &arch)
{
    if (arch.GetTriple().getArch () == llvm::Triple::mips
        || arch.GetTriple().getArch () == llvm::Triple::mipsel)
        return true;
    return false;
}

const char *
EmulateInstructionMIPS::GetRegisterName (unsigned reg_num, bool alternate_name)
{
    if (alternate_name)
    {
        switch (reg_num)
        {
            case dwarf_sp_mips:          return "r29"; 
            case dwarf_r30_mips:         return "r30"; 
            case dwarf_ra_mips:          return "r31";
            case dwarf_f0_mips:          return "f0";
            case dwarf_f1_mips:          return "f1";
            case dwarf_f2_mips:          return "f2";
            case dwarf_f3_mips:          return "f3";
            case dwarf_f4_mips:          return "f4";
            case dwarf_f5_mips:          return "f5";
            case dwarf_f6_mips:          return "f6";
            case dwarf_f7_mips:          return "f7";
            case dwarf_f8_mips:          return "f8";
            case dwarf_f9_mips:          return "f9";
            case dwarf_f10_mips:         return "f10";
            case dwarf_f11_mips:         return "f11";
            case dwarf_f12_mips:         return "f12";
            case dwarf_f13_mips:         return "f13";
            case dwarf_f14_mips:         return "f14";
            case dwarf_f15_mips:         return "f15";
            case dwarf_f16_mips:         return "f16";
            case dwarf_f17_mips:         return "f17";
            case dwarf_f18_mips:         return "f18";
            case dwarf_f19_mips:         return "f19";
            case dwarf_f20_mips:         return "f20";
            case dwarf_f21_mips:         return "f21";
            case dwarf_f22_mips:         return "f22";
            case dwarf_f23_mips:         return "f23";
            case dwarf_f24_mips:         return "f24";
            case dwarf_f25_mips:         return "f25";
            case dwarf_f26_mips:         return "f26";
            case dwarf_f27_mips:         return "f27";
            case dwarf_f28_mips:         return "f28";
            case dwarf_f29_mips:         return "f29";
            case dwarf_f30_mips:         return "f30";
            case dwarf_f31_mips:         return "f31";
            case dwarf_w0_mips:          return "w0";
            case dwarf_w1_mips:          return "w1";
            case dwarf_w2_mips:          return "w2";
            case dwarf_w3_mips:          return "w3";
            case dwarf_w4_mips:          return "w4";
            case dwarf_w5_mips:          return "w5";
            case dwarf_w6_mips:          return "w6";
            case dwarf_w7_mips:          return "w7";
            case dwarf_w8_mips:          return "w8";
            case dwarf_w9_mips:          return "w9";
            case dwarf_w10_mips:         return "w10";
            case dwarf_w11_mips:         return "w11";
            case dwarf_w12_mips:         return "w12";
            case dwarf_w13_mips:         return "w13";
            case dwarf_w14_mips:         return "w14";
            case dwarf_w15_mips:         return "w15";
            case dwarf_w16_mips:         return "w16";
            case dwarf_w17_mips:         return "w17";
            case dwarf_w18_mips:         return "w18";
            case dwarf_w19_mips:         return "w19";
            case dwarf_w20_mips:         return "w20";
            case dwarf_w21_mips:         return "w21";
            case dwarf_w22_mips:         return "w22";
            case dwarf_w23_mips:         return "w23";
            case dwarf_w24_mips:         return "w24";
            case dwarf_w25_mips:         return "w25";
            case dwarf_w26_mips:         return "w26";
            case dwarf_w27_mips:         return "w27";
            case dwarf_w28_mips:         return "w28";
            case dwarf_w29_mips:         return "w29";
            case dwarf_w30_mips:         return "w30";
            case dwarf_w31_mips:         return "w31";
            case dwarf_mir_mips:         return "mir";
            case dwarf_mcsr_mips:        return "mcsr";
            case dwarf_config5_mips:     return "config5";
            default:
                break;
        }
        return nullptr;
    }

    switch (reg_num)
    {
        case dwarf_zero_mips:         return "r0";
        case dwarf_r1_mips:           return "r1";
        case dwarf_r2_mips:           return "r2";
        case dwarf_r3_mips:           return "r3";
        case dwarf_r4_mips:           return "r4";
        case dwarf_r5_mips:           return "r5";
        case dwarf_r6_mips:           return "r6";
        case dwarf_r7_mips:           return "r7";
        case dwarf_r8_mips:           return "r8";
        case dwarf_r9_mips:           return "r9";
        case dwarf_r10_mips:          return "r10";
        case dwarf_r11_mips:          return "r11";
        case dwarf_r12_mips:          return "r12";
        case dwarf_r13_mips:          return "r13";
        case dwarf_r14_mips:          return "r14";
        case dwarf_r15_mips:          return "r15";
        case dwarf_r16_mips:          return "r16";
        case dwarf_r17_mips:          return "r17";
        case dwarf_r18_mips:          return "r18";
        case dwarf_r19_mips:          return "r19";
        case dwarf_r20_mips:          return "r20";
        case dwarf_r21_mips:          return "r21";
        case dwarf_r22_mips:          return "r22";
        case dwarf_r23_mips:          return "r23";
        case dwarf_r24_mips:          return "r24";
        case dwarf_r25_mips:          return "r25";
        case dwarf_r26_mips:          return "r26";
        case dwarf_r27_mips:          return "r27";
        case dwarf_gp_mips:           return "gp";
        case dwarf_sp_mips:           return "sp";
        case dwarf_r30_mips:          return "fp";
        case dwarf_ra_mips:           return "ra";
        case dwarf_sr_mips:           return "sr";
        case dwarf_lo_mips:           return "lo";
        case dwarf_hi_mips:           return "hi";
        case dwarf_bad_mips:          return "bad";
        case dwarf_cause_mips:        return "cause";
        case dwarf_pc_mips:           return "pc";
        case dwarf_f0_mips:           return "f0";
        case dwarf_f1_mips:           return "f1";
        case dwarf_f2_mips:           return "f2";
        case dwarf_f3_mips:           return "f3";
        case dwarf_f4_mips:           return "f4";
        case dwarf_f5_mips:           return "f5";
        case dwarf_f6_mips:           return "f6";
        case dwarf_f7_mips:           return "f7";
        case dwarf_f8_mips:           return "f8";
        case dwarf_f9_mips:           return "f9";
        case dwarf_f10_mips:          return "f10";
        case dwarf_f11_mips:          return "f11";
        case dwarf_f12_mips:          return "f12";
        case dwarf_f13_mips:          return "f13";
        case dwarf_f14_mips:          return "f14";
        case dwarf_f15_mips:          return "f15";
        case dwarf_f16_mips:          return "f16";
        case dwarf_f17_mips:          return "f17";
        case dwarf_f18_mips:          return "f18";
        case dwarf_f19_mips:          return "f19";
        case dwarf_f20_mips:          return "f20";
        case dwarf_f21_mips:          return "f21";
        case dwarf_f22_mips:          return "f22";
        case dwarf_f23_mips:          return "f23";
        case dwarf_f24_mips:          return "f24";
        case dwarf_f25_mips:          return "f25";
        case dwarf_f26_mips:          return "f26";
        case dwarf_f27_mips:          return "f27";
        case dwarf_f28_mips:          return "f28";
        case dwarf_f29_mips:          return "f29";
        case dwarf_f30_mips:          return "f30";
        case dwarf_f31_mips:          return "f31";
        case dwarf_fcsr_mips:         return "fcsr";
        case dwarf_fir_mips:          return "fir";
        case dwarf_w0_mips:           return "w0";
        case dwarf_w1_mips:           return "w1";
        case dwarf_w2_mips:           return "w2";
        case dwarf_w3_mips:           return "w3";
        case dwarf_w4_mips:           return "w4";
        case dwarf_w5_mips:           return "w5";
        case dwarf_w6_mips:           return "w6";
        case dwarf_w7_mips:           return "w7";
        case dwarf_w8_mips:           return "w8";
        case dwarf_w9_mips:           return "w9";
        case dwarf_w10_mips:          return "w10";
        case dwarf_w11_mips:          return "w11";
        case dwarf_w12_mips:          return "w12";
        case dwarf_w13_mips:          return "w13";
        case dwarf_w14_mips:          return "w14";
        case dwarf_w15_mips:          return "w15";
        case dwarf_w16_mips:          return "w16";
        case dwarf_w17_mips:          return "w17";
        case dwarf_w18_mips:          return "w18";
        case dwarf_w19_mips:          return "w19";
        case dwarf_w20_mips:          return "w20";
        case dwarf_w21_mips:          return "w21";
        case dwarf_w22_mips:          return "w22";
        case dwarf_w23_mips:          return "w23";
        case dwarf_w24_mips:          return "w24";
        case dwarf_w25_mips:          return "w25";
        case dwarf_w26_mips:          return "w26";
        case dwarf_w27_mips:          return "w27";
        case dwarf_w28_mips:          return "w28";
        case dwarf_w29_mips:          return "w29";
        case dwarf_w30_mips:          return "w30";
        case dwarf_w31_mips:          return "w31";
        case dwarf_mcsr_mips:         return "mcsr";
        case dwarf_mir_mips:          return "mir";
        case dwarf_config5_mips:      return "config5";
    }
    return nullptr;
}

bool
EmulateInstructionMIPS::GetRegisterInfo (RegisterKind reg_kind, uint32_t reg_num, RegisterInfo &reg_info)
{
    if (reg_kind == eRegisterKindGeneric)
    {
        switch (reg_num)
        {
            case LLDB_REGNUM_GENERIC_PC:    reg_kind = eRegisterKindDWARF; reg_num = dwarf_pc_mips; break;
            case LLDB_REGNUM_GENERIC_SP:    reg_kind = eRegisterKindDWARF; reg_num = dwarf_sp_mips; break;
            case LLDB_REGNUM_GENERIC_FP:    reg_kind = eRegisterKindDWARF; reg_num = dwarf_r30_mips; break;
            case LLDB_REGNUM_GENERIC_RA:    reg_kind = eRegisterKindDWARF; reg_num = dwarf_ra_mips; break;
            case LLDB_REGNUM_GENERIC_FLAGS: reg_kind = eRegisterKindDWARF; reg_num = dwarf_sr_mips; break;
            default:
                return false;
        }
    }

    if (reg_kind == eRegisterKindDWARF)
    {
       ::memset (&reg_info, 0, sizeof(RegisterInfo));
       ::memset (reg_info.kinds, LLDB_INVALID_REGNUM, sizeof(reg_info.kinds));

       if (reg_num == dwarf_sr_mips || reg_num == dwarf_fcsr_mips || reg_num == dwarf_fir_mips || reg_num == dwarf_mcsr_mips || reg_num == dwarf_mir_mips || reg_num == dwarf_config5_mips)
       {
           reg_info.byte_size = 4;
           reg_info.format = eFormatHex;
           reg_info.encoding = eEncodingUint;
       }
       else if ((int)reg_num >= dwarf_zero_mips && (int)reg_num <= dwarf_f31_mips)
       {
           reg_info.byte_size = 4;
           reg_info.format = eFormatHex;
           reg_info.encoding = eEncodingUint;
       }
       else if ((int)reg_num >= dwarf_w0_mips && (int)reg_num <= dwarf_w31_mips)
       {
           reg_info.byte_size = 16;
           reg_info.format = eFormatVectorOfUInt8;
           reg_info.encoding = eEncodingVector;
       }
       else
       {
           return false;
       }

       reg_info.name = GetRegisterName (reg_num, false);
       reg_info.alt_name = GetRegisterName (reg_num, true);
       reg_info.kinds[eRegisterKindDWARF] = reg_num;

       switch (reg_num)
       {
           case dwarf_r30_mips: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP; break;
           case dwarf_ra_mips: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_RA; break;
           case dwarf_sp_mips: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP; break;
           case dwarf_pc_mips: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC; break;
           case dwarf_sr_mips: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS; break;
           default: break;
       }
       return true;
    }
    return false;
}

EmulateInstructionMIPS::MipsOpcode*
EmulateInstructionMIPS::GetOpcodeForInstruction (const char *op_name)
{
    static EmulateInstructionMIPS::MipsOpcode
    g_opcodes[] = 
    {
        //----------------------------------------------------------------------
        // Prologue/Epilogue instructions
        //----------------------------------------------------------------------
        { "ADDiu",      &EmulateInstructionMIPS::Emulate_ADDiu,       "ADDIU rt,rs,immediate"    },
        { "SW",         &EmulateInstructionMIPS::Emulate_SW,          "SW rt,offset(rs)"         },
        { "LW",         &EmulateInstructionMIPS::Emulate_LW,          "LW rt,offset(base)"       },

        //----------------------------------------------------------------------
        // Load/Store  instructions
        //----------------------------------------------------------------------
        /* Following list of emulated instructions are required by implementation of hardware watchpoint
           for MIPS in lldb. As we just need the address accessed by instructions, we have generalised 
           all these instructions in 2 functions depending on their addressing modes */

        { "LB",         &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LB    rt, offset(base)" },
        { "LBE",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LBE   rt, offset(base)" },
        { "LBU",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LBU   rt, offset(base)" },
        { "LBUE",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LBUE  rt, offset(base)" },
        { "LDC1",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LDC1  ft, offset(base)" },
        { "LD",         &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LD    rt, offset(base)" },
        { "LDL",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LDL   rt, offset(base)" },
        { "LDR",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LDR   rt, offset(base)" },
        { "LLD",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LLD   rt, offset(base)" },
        { "LDC2",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LDC2  rt, offset(base)" },
        { "LDXC1",      &EmulateInstructionMIPS::Emulate_LDST_Reg,          "LDXC1 fd, index (base)" },
        { "LH",         &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LH    rt, offset(base)" },
        { "LHE",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LHE   rt, offset(base)" },
        { "LHU",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LHU   rt, offset(base)" },
        { "LHUE",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LHUE  rt, offset(base)" },
        { "LL",         &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LL    rt, offset(base)" },
        { "LLE",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LLE   rt, offset(base)" },
        { "LUXC1",      &EmulateInstructionMIPS::Emulate_LDST_Reg,          "LUXC1 fd, index (base)" },
        { "LW",         &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LW    rt, offset(base)" },
        { "LWC1",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LWC1  ft, offset(base)" },
        { "LWC2",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LWC2  rt, offset(base)" },
        { "LWE",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LWE   rt, offset(base)" },
        { "LWL",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LWL   rt, offset(base)" },
        { "LWLE",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LWLE  rt, offset(base)" },
        { "LWR",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LWR   rt, offset(base)" },
        { "LWRE",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "LWRE  rt, offset(base)" },
        { "LWXC1",      &EmulateInstructionMIPS::Emulate_LDST_Reg,          "LWXC1 fd, index (base)" },

        { "SB",         &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SB    rt, offset(base)" },
        { "SBE",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SBE   rt, offset(base)" },
        { "SC",         &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SC    rt, offset(base)" },
        { "SCE",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SCE   rt, offset(base)" },
        { "SCD",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SCD   rt, offset(base)" },
        { "SD",         &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SD    rt, offset(base)" },
        { "SDL",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SDL   rt, offset(base)" },
        { "SDR",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SDR   rt, offset(base)" },
        { "SDC1",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SDC1  ft, offset(base)" },
        { "SDC2",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SDC2  rt, offset(base)" },
        { "SDXC1",      &EmulateInstructionMIPS::Emulate_LDST_Reg,          "SDXC1 fs, index(base)"  },
        { "SH",         &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SH    rt, offset(base)" },
        { "SHE",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SHE   rt, offset(base)" },
        { "SUXC1",      &EmulateInstructionMIPS::Emulate_LDST_Reg,          "SUXC1 fs, index (base)" },
        { "SWC1",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SWC1  ft, offset(base)" },
        { "SWC2",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SWC2  rt, offset(base)" },
        { "SWE",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SWE   rt, offset(base)" },
        { "SWL",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SWL   rt, offset(base)" },
        { "SWLE",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SWLE  rt, offset(base)" },
        { "SWR",        &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SWR   rt, offset(base)" },
        { "SWRE",       &EmulateInstructionMIPS::Emulate_LDST_Imm,          "SWRE  rt, offset(base)" },
        { "SWXC1",      &EmulateInstructionMIPS::Emulate_LDST_Reg,          "SWXC1 fs, index (base)" },

        //----------------------------------------------------------------------
        // Branch instructions
        //----------------------------------------------------------------------
        { "BEQ",        &EmulateInstructionMIPS::Emulate_BEQ,         "BEQ rs,rt,offset"          },
        { "BNE",        &EmulateInstructionMIPS::Emulate_BNE,         "BNE rs,rt,offset"          },
        { "BEQL",       &EmulateInstructionMIPS::Emulate_BEQL,        "BEQL rs,rt,offset"         },
        { "BNEL",       &EmulateInstructionMIPS::Emulate_BNEL,        "BNEL rs,rt,offset"         },
        { "BGEZALL",    &EmulateInstructionMIPS::Emulate_BGEZALL,     "BGEZALL rt,offset"         },
        { "BAL",        &EmulateInstructionMIPS::Emulate_BAL,         "BAL offset"                },
        { "BGEZAL",     &EmulateInstructionMIPS::Emulate_BGEZAL,      "BGEZAL rs,offset"          },
        { "BALC",       &EmulateInstructionMIPS::Emulate_BALC,        "BALC offset"               },
        { "BC",         &EmulateInstructionMIPS::Emulate_BC,          "BC offset"                 },
        { "BGEZ",       &EmulateInstructionMIPS::Emulate_BGEZ,        "BGEZ rs,offset"            },
        { "BLEZALC",    &EmulateInstructionMIPS::Emulate_BLEZALC,     "BLEZALC rs,offset"         },
        { "BGEZALC",    &EmulateInstructionMIPS::Emulate_BGEZALC,     "BGEZALC rs,offset"         },
        { "BLTZALC",    &EmulateInstructionMIPS::Emulate_BLTZALC,     "BLTZALC rs,offset"         },
        { "BGTZALC",    &EmulateInstructionMIPS::Emulate_BGTZALC,     "BGTZALC rs,offset"         },
        { "BEQZALC",    &EmulateInstructionMIPS::Emulate_BEQZALC,     "BEQZALC rs,offset"         },
        { "BNEZALC",    &EmulateInstructionMIPS::Emulate_BNEZALC,     "BNEZALC rs,offset"         },
        { "BEQC",       &EmulateInstructionMIPS::Emulate_BEQC,        "BEQC rs,rt,offset"         },
        { "BNEC",       &EmulateInstructionMIPS::Emulate_BNEC,        "BNEC rs,rt,offset"         },
        { "BLTC",       &EmulateInstructionMIPS::Emulate_BLTC,        "BLTC rs,rt,offset"         },
        { "BGEC",       &EmulateInstructionMIPS::Emulate_BGEC,        "BGEC rs,rt,offset"         },
        { "BLTUC",      &EmulateInstructionMIPS::Emulate_BLTUC,       "BLTUC rs,rt,offset"        },
        { "BGEUC",      &EmulateInstructionMIPS::Emulate_BGEUC,       "BGEUC rs,rt,offset"        },
        { "BLTZC",      &EmulateInstructionMIPS::Emulate_BLTZC,       "BLTZC rt,offset"           },
        { "BLEZC",      &EmulateInstructionMIPS::Emulate_BLEZC,       "BLEZC rt,offset"           },
        { "BGEZC",      &EmulateInstructionMIPS::Emulate_BGEZC,       "BGEZC rt,offset"           },
        { "BGTZC",      &EmulateInstructionMIPS::Emulate_BGTZC,       "BGTZC rt,offset"           },
        { "BEQZC",      &EmulateInstructionMIPS::Emulate_BEQZC,       "BEQZC rt,offset"           },
        { "BNEZC",      &EmulateInstructionMIPS::Emulate_BNEZC,       "BNEZC rt,offset"           },
        { "BGEZL",      &EmulateInstructionMIPS::Emulate_BGEZL,       "BGEZL rt,offset"           },
        { "BGTZ",       &EmulateInstructionMIPS::Emulate_BGTZ,        "BGTZ rt,offset"            },
        { "BGTZL",      &EmulateInstructionMIPS::Emulate_BGTZL,       "BGTZL rt,offset"           },
        { "BLEZ",       &EmulateInstructionMIPS::Emulate_BLEZ,        "BLEZ rt,offset"            },
        { "BLEZL",      &EmulateInstructionMIPS::Emulate_BLEZL,       "BLEZL rt,offset"           },
        { "BLTZ",       &EmulateInstructionMIPS::Emulate_BLTZ,        "BLTZ rt,offset"            },
        { "BLTZAL",     &EmulateInstructionMIPS::Emulate_BLTZAL,      "BLTZAL rt,offset"          },
        { "BLTZALL",    &EmulateInstructionMIPS::Emulate_BLTZALL,     "BLTZALL rt,offset"         },
        { "BLTZL",      &EmulateInstructionMIPS::Emulate_BLTZL,       "BLTZL rt,offset"           },
        { "BOVC",       &EmulateInstructionMIPS::Emulate_BOVC,        "BOVC rs,rt,offset"         },
        { "BNVC",       &EmulateInstructionMIPS::Emulate_BNVC,        "BNVC rs,rt,offset"         },
        { "J",          &EmulateInstructionMIPS::Emulate_J,           "J target"                  },
        { "JAL",        &EmulateInstructionMIPS::Emulate_JAL,         "JAL target"                },
        { "JALX",       &EmulateInstructionMIPS::Emulate_JAL,         "JALX target"               },
        { "JALR",       &EmulateInstructionMIPS::Emulate_JALR,        "JALR target"               },
        { "JALR_HB",    &EmulateInstructionMIPS::Emulate_JALR,        "JALR.HB target"            },
        { "JIALC",      &EmulateInstructionMIPS::Emulate_JIALC,       "JIALC rt,offset"           },
        { "JIC",        &EmulateInstructionMIPS::Emulate_JIC,         "JIC rt,offset"             },
        { "JR",         &EmulateInstructionMIPS::Emulate_JR,          "JR target"                 },
        { "JR_HB",      &EmulateInstructionMIPS::Emulate_JR,          "JR.HB target"              },
        { "BC1F",       &EmulateInstructionMIPS::Emulate_BC1F,        "BC1F cc, offset"           },
        { "BC1T",       &EmulateInstructionMIPS::Emulate_BC1T,        "BC1T cc, offset"           },
        { "BC1FL",      &EmulateInstructionMIPS::Emulate_BC1FL,       "BC1FL cc, offset"          },
        { "BC1TL",      &EmulateInstructionMIPS::Emulate_BC1TL,       "BC1TL cc, offset"          },
        { "BC1EQZ",     &EmulateInstructionMIPS::Emulate_BC1EQZ,      "BC1EQZ ft, offset"         },
        { "BC1NEZ",     &EmulateInstructionMIPS::Emulate_BC1NEZ,      "BC1NEZ ft, offset"         },
        { "BC1ANY2F",   &EmulateInstructionMIPS::Emulate_BC1ANY2F,    "BC1ANY2F cc, offset"       },
        { "BC1ANY2T",   &EmulateInstructionMIPS::Emulate_BC1ANY2T,    "BC1ANY2T cc, offset"       },
        { "BC1ANY4F",   &EmulateInstructionMIPS::Emulate_BC1ANY4F,    "BC1ANY4F cc, offset"       },
        { "BC1ANY4T",   &EmulateInstructionMIPS::Emulate_BC1ANY4T,    "BC1ANY4T cc, offset"       },
        { "BNZ_B",      &EmulateInstructionMIPS::Emulate_BNZB,        "BNZ.b wt,s16"              },
        { "BNZ_H",      &EmulateInstructionMIPS::Emulate_BNZH,        "BNZ.h wt,s16"              },
        { "BNZ_W",      &EmulateInstructionMIPS::Emulate_BNZW,        "BNZ.w wt,s16"              },
        { "BNZ_D",      &EmulateInstructionMIPS::Emulate_BNZD,        "BNZ.d wt,s16"              },
        { "BZ_B",       &EmulateInstructionMIPS::Emulate_BZB,         "BZ.b wt,s16"               },
        { "BZ_H",       &EmulateInstructionMIPS::Emulate_BZH,         "BZ.h wt,s16"               },
        { "BZ_W",       &EmulateInstructionMIPS::Emulate_BZW,         "BZ.w wt,s16"               },
        { "BZ_D",       &EmulateInstructionMIPS::Emulate_BZD,         "BZ.d wt,s16"               },
        { "BNZ_V",      &EmulateInstructionMIPS::Emulate_BNZV,        "BNZ.V wt,s16"              },
        { "BZ_V",       &EmulateInstructionMIPS::Emulate_BZV,         "BZ.V wt,s16"               },
    };

    static const size_t k_num_mips_opcodes = llvm::array_lengthof(g_opcodes);

    for (size_t i = 0; i < k_num_mips_opcodes; ++i)
    {
        if (! strcasecmp (g_opcodes[i].op_name, op_name))
            return &g_opcodes[i];
    }

    return NULL;
}

bool 
EmulateInstructionMIPS::ReadInstruction ()
{
    bool success = false;
    m_addr = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_ADDRESS, &success);
    if (success)
    {
        Context read_inst_context;
        read_inst_context.type = eContextReadOpcode;
        read_inst_context.SetNoArgs ();
        m_opcode.SetOpcode32 (ReadMemoryUnsigned (read_inst_context, m_addr, 4, 0, &success), GetByteOrder());
    }
    if (!success)
        m_addr = LLDB_INVALID_ADDRESS;
    return success;
}

bool
EmulateInstructionMIPS::EvaluateInstruction (uint32_t evaluate_options)
{
    bool success = false;
    llvm::MCInst mc_insn;
    uint64_t insn_size;
    DataExtractor data;

    /* Keep the complexity of the decode logic with the llvm::MCDisassembler class. */
    if (m_opcode.GetData (data))
    {
        llvm::MCDisassembler::DecodeStatus decode_status;
        llvm::ArrayRef<uint8_t> raw_insn (data.GetDataStart(), data.GetByteSize());
        decode_status = m_disasm->getInstruction (mc_insn, insn_size, raw_insn, m_addr, llvm::nulls(), llvm::nulls());
        if (decode_status != llvm::MCDisassembler::Success)
            return false;
    }

    /*
     * mc_insn.getOpcode() returns decoded opcode. However to make use
     * of llvm::Mips::<insn> we would need "MipsGenInstrInfo.inc".
    */
    const char *op_name = m_insn_info->getName (mc_insn.getOpcode ());

    if (op_name == NULL)
        return false;

    /*
     * Decoding has been done already. Just get the call-back function
     * and emulate the instruction.
    */
    MipsOpcode *opcode_data = GetOpcodeForInstruction (op_name);

    if (opcode_data == NULL)
        return false;

    uint64_t old_pc = 0, new_pc = 0;
    const bool auto_advance_pc = evaluate_options & eEmulateInstructionOptionAutoAdvancePC;

    if (auto_advance_pc)
    {
        old_pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
        if (!success)
            return false;
    }

    /* emulate instruction */
    success = (this->*opcode_data->callback) (mc_insn);
    if (!success)
        return false;

    if (auto_advance_pc)
    {
        new_pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
        if (!success)
            return false;

        /* If we haven't changed the PC, change it here */
        if (old_pc == new_pc)
        {
            new_pc += 4;
            Context context;
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, new_pc))
                return false;
        }
    }

    return true;
}

bool
EmulateInstructionMIPS::CreateFunctionEntryUnwind (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    UnwindPlan::RowSP row(new UnwindPlan::Row);
    const bool can_replace = false;

    // Our previous Call Frame Address is the stack pointer
    row->GetCFAValue().SetIsRegisterPlusOffset(dwarf_sp_mips, 0);

    // Our previous PC is in the RA
    row->SetRegisterLocationToRegister(dwarf_pc_mips, dwarf_ra_mips, can_replace);

    unwind_plan.AppendRow (row);

    // All other registers are the same.
    unwind_plan.SetSourceName ("EmulateInstructionMIPS");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolYes);

    return true;
}

bool
EmulateInstructionMIPS::nonvolatile_reg_p (uint32_t regnum)
{
    switch (regnum)
    {
        case dwarf_r16_mips:
        case dwarf_r17_mips:
        case dwarf_r18_mips:
        case dwarf_r19_mips:
        case dwarf_r20_mips:
        case dwarf_r21_mips:
        case dwarf_r22_mips:
        case dwarf_r23_mips:
        case dwarf_gp_mips:
        case dwarf_sp_mips:
        case dwarf_r30_mips:
        case dwarf_ra_mips:
            return true;
        default:
            return false;
    }
    return false;
}

bool
EmulateInstructionMIPS::Emulate_ADDiu (llvm::MCInst& insn)
{
    bool success = false;
    const uint32_t imm16 = insn.getOperand(2).getImm();
    uint32_t imm = SignedBits(imm16, 15, 0);
    uint64_t result;
    uint32_t src, dst;

    dst = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    src = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    /* Check if this is addiu sp,<src>,imm16 */
    if (dst == dwarf_sp_mips)
    {
        /* read <src> register */
        uint64_t src_opd_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + src, 0, &success);
        if (!success)
            return false;

        result = src_opd_val + imm;

        Context context;
        RegisterInfo reg_info_sp;
        if (GetRegisterInfo (eRegisterKindDWARF, dwarf_sp_mips, reg_info_sp))
            context.SetRegisterPlusOffset (reg_info_sp, imm);

        /* We are allocating bytes on stack */
        context.type = eContextAdjustStackPointer;

        WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_sp_mips, result);
    }
    
    return true;
}

bool
EmulateInstructionMIPS::Emulate_SW (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t imm16 = insn.getOperand(2).getImm();
    uint32_t imm = SignedBits(imm16, 15, 0);
    uint32_t src, base;
    int32_t address;
    Context bad_vaddr_context;

    RegisterInfo reg_info_base;

    src = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    base = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    if (!GetRegisterInfo (eRegisterKindDWARF, dwarf_zero_mips64 + base, reg_info_base))
        return false;

    /* read base register */
    address = (int32_t)ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips64 + base, 0, &success);
    if (!success)
        return false;

    /* destination address */
    address = address + imm;

    /* Set the bad_vaddr register with base address used in the instruction */
    bad_vaddr_context.type = eContextInvalid;
    WriteRegisterUnsigned (bad_vaddr_context, eRegisterKindDWARF, dwarf_bad_mips64, address);

    /* We look for sp based non-volatile register stores */
    if (base == dwarf_sp_mips && nonvolatile_reg_p (src))
    {

        RegisterInfo reg_info_src;

        if (!GetRegisterInfo (eRegisterKindDWARF, dwarf_zero_mips + src, reg_info_src))
            return false;

        Context context;
        RegisterValue data_src;
        context.type = eContextPushRegisterOnStack;
        context.SetRegisterToRegisterPlusOffset (reg_info_src, reg_info_base, 0);

        uint8_t buffer [RegisterValue::kMaxRegisterByteSize];
        Error error;

        if (!ReadRegister (&reg_info_base, data_src))
            return false;

        if (data_src.GetAsMemoryData (&reg_info_src, buffer, reg_info_src.byte_size, eByteOrderLittle, error) == 0)
            return false;

        if (!WriteMemory (context, address, buffer, reg_info_src.byte_size))
            return false;

        return true;
    }

    return false;
}

bool
EmulateInstructionMIPS::Emulate_LW (llvm::MCInst& insn)
{
  bool success =false;
    uint32_t src, base;
    int32_t imm, address;
    Context bad_vaddr_context;

    src = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    base = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    imm = insn.getOperand(2).getImm();

    RegisterInfo reg_info_base;
    if (!GetRegisterInfo (eRegisterKindDWARF, dwarf_zero_mips64 + base, reg_info_base))
        return false;

    /* read base register */
    address = (int32_t)ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips64 + base, 0, &success);
    if (!success)
        return false;

    /* destination address */
    address = address + imm;

    /* Set the bad_vaddr register with base address used in the instruction */
    bad_vaddr_context.type = eContextInvalid;
    WriteRegisterUnsigned (bad_vaddr_context, eRegisterKindDWARF, dwarf_bad_mips64, address);

    if (base == dwarf_sp_mips && nonvolatile_reg_p (src))
    {
        RegisterValue data_src;
        RegisterInfo reg_info_src;

        if (!GetRegisterInfo (eRegisterKindDWARF, dwarf_zero_mips + src, reg_info_src))
            return false;

        Context context;
        context.type = eContextRegisterLoad;

        if (!WriteRegister (context, &reg_info_src, data_src))
            return false;

        return true;
    }

    return false;
}

bool
EmulateInstructionMIPS::Emulate_BEQ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target, rs_val, rt_val;

    /*
     * BEQ rs, rt, offset
     *      condition <- (GPR[rs] = GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val == rt_val)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BNE (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target, rs_val, rt_val;

    /*
     * BNE rs, rt, offset
     *      condition <- (GPR[rs] != GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val != rt_val)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BEQL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target, rs_val, rt_val;

    /*
     * BEQL rs, rt, offset
     *      condition <- (GPR[rs] = GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val == rt_val)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BNEL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target, rs_val, rt_val;

    /*
     * BNEL rs, rt, offset
     *      condition <- (GPR[rs] != GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val != rt_val)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGEZL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target; 
    int32_t rs_val;

    /*
     * BGEZL rs, offset
     *      condition <- (GPR[rs] >= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLTZL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BLTZL rs, offset
     *      condition <- (GPR[rs] < 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val < 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGTZL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BGTZL rs, offset
     *      condition <- (GPR[rs] > 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val > 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLEZL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BLEZL rs, offset
     *      condition <- (GPR[rs] <= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val <= 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGTZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BGTZ rs, offset
     *      condition <- (GPR[rs] > 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val > 0)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLEZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target; 
    int32_t rs_val;

    /*
     * BLEZ rs, offset
     *      condition <- (GPR[rs] <= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val <= 0)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLTZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BLTZ rs, offset
     *      condition <- (GPR[rs] < 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val < 0)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGEZALL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BGEZALL rt, offset
     *      condition <- (GPR[rs] >= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BAL (llvm::MCInst& insn)
{
    bool success = false;
    int32_t offset, pc, target;

    /*
     * BAL offset
     *      offset = sign_ext (offset << 2)
     *      RA = PC + 8
     *      PC = PC + offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    target = pc + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BALC (llvm::MCInst& insn)
{
    bool success = false;
    int32_t offset, pc, target;

    /* 
     * BALC offset
     *      offset = sign_ext (offset << 2)
     *      RA = PC + 4
     *      PC = PC + 4 + offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    target = pc + 4 + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGEZAL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BGEZAL rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] >= 0)
     *      if condition then     
     *          RA = PC + 8
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if ((int32_t) rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 8;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLTZAL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BLTZAL rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] < 0)
     *      if condition then     
     *          RA = PC + 8
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if ((int32_t) rs_val < 0)
        target = pc + offset;
    else
        target = pc + 8;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLTZALL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BLTZALL rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] < 0)
     *      if condition then     
     *          RA = PC + 8
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val < 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 8))
        return false;

    return true;
}


bool
EmulateInstructionMIPS::Emulate_BLEZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BLEZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] <= 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val <= 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGEZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BGEZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] >= 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLTZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BLTZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] < 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val < 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGTZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BGTZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] > 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val > 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BEQZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target, rs_val;

    /*
     * BEQZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] == 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val == 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BNEZALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target, rs_val;

    /*
     * BNEZALC rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] != 0)
     *      if condition then     
     *          RA = PC + 4
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val != 0)
        target = pc + offset;
    else
        target = pc + 4;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGEZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target, rs_val;

    /*
     * BGEZ rs,offset
     *      offset = sign_ext (offset << 2)
     *      condition <- (GPR[rs] >= 0)
     *      if condition then     
     *          PC = PC + offset
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (rs_val >= 0)
        target = pc + offset;
    else
        target = pc + 8;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC (llvm::MCInst& insn)
{
    bool success = false;
    int32_t offset, pc, target;

    /* 
     * BC offset
     *      offset = sign_ext (offset << 2)
     *      PC = PC + 4 + offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    target = pc + 4 + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BEQC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target, rs_val, rt_val;

    /*
     * BEQC rs, rt, offset
     *      condition <- (GPR[rs] = GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val == rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BNEC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target, rs_val, rt_val;

    /*
     * BNEC rs, rt, offset
     *      condition <- (GPR[rs] != GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val != rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLTC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target;
    int32_t rs_val, rt_val;

    /*
     * BLTC rs, rt, offset
     *      condition <- (GPR[rs] < GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val < rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGEC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target;
    int32_t rs_val, rt_val;

    /*
     * BGEC rs, rt, offset
     *      condition <- (GPR[rs] > GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val > rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLTUC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target;
    uint32_t rs_val, rt_val;

    /*
     * BLTUC rs, rt, offset
     *      condition <- (GPR[rs] < GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val < rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGEUC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target;
    uint32_t rs_val, rt_val;

    /*
     * BGEUC rs, rt, offset
     *      condition <- (GPR[rs] > GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (rs_val > rt_val)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLTZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BLTZC rs, offset
     *      condition <- (GPR[rs] < 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val < 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BLEZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BLEZC rs, offset
     *      condition <- (GPR[rs] <= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val <= 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGEZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BGEZC rs, offset
     *      condition <- (GPR[rs] >= 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val >= 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BGTZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    int32_t rs_val;

    /*
     * BGTZC rs, offset
     *      condition <- (GPR[rs] > 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val > 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BEQZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    uint32_t rs_val;

    /*
     * BEQZC rs, offset
     *      condition <- (GPR[rs] = 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val == 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BNEZC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    int32_t offset, pc, target;
    uint32_t rs_val;

    /*
     * BNEZC rs, offset
     *      condition <- (GPR[rs] != 0)
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    if (rs_val != 0)
        target = pc + 4 + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

static int
IsAdd64bitOverflow (int32_t a, int32_t b)
{
  int32_t r = (uint32_t) a + (uint32_t) b;
  return (a < 0 && b < 0 && r >= 0) || (a >= 0 && b >= 0 && r < 0);
}

bool
EmulateInstructionMIPS::Emulate_BOVC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target;
    int32_t rs_val, rt_val;

    /*
     * BOVC rs, rt, offset
     *      condition <- overflow(GPR[rs] + GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (IsAdd64bitOverflow (rs_val, rt_val))
        target = pc + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BNVC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    int32_t offset, pc, target;
    int32_t rs_val, rt_val;

    /*
     * BNVC rs, rt, offset
     *      condition <- overflow(GPR[rs] + GPR[rt])
     *      if condition then
     *          PC = PC + sign_ext (offset << 2)
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rt = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());
    offset = insn.getOperand(2).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    if (! IsAdd64bitOverflow (rs_val, rt_val))
        target = pc + offset;
    else
        target = pc + 4;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_J (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t offset, pc;

    /* 
     * J offset
     *      offset = sign_ext (offset << 2)
     *      PC = PC[63-28] | offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    /* This is a PC-region branch and not PC-relative */
    pc = (pc & 0xF0000000UL) | offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, pc))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_JAL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t offset, target, pc;

    /* 
     * JAL offset
     *      offset = sign_ext (offset << 2)
     *      PC = PC[63-28] | offset
    */
    offset = insn.getOperand(0).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    /* This is a PC-region branch and not PC-relative */
    target = (pc & 0xF0000000UL) | offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_JALR (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs, rt;
    uint32_t pc, rs_val;

    /* 
     * JALR rt, rs
     *      GPR[rt] = PC + 8
     *      PC = GPR[rs]
    */
    rt = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    rs = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, rs_val))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_zero_mips + rt, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_JIALC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rt;
    int32_t target, offset, pc, rt_val;

    /* 
     * JIALC rt, offset
     *      offset = sign_ext (offset)
     *      PC = GPR[rt] + offset
     *      RA = PC + 4
    */
    rt = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    target = rt_val + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_ra_mips, pc + 4))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_JIC (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rt;
    int32_t target, offset, rt_val;

    /* 
     * JIC rt, offset
     *      offset = sign_ext (offset)
     *      PC = GPR[rt] + offset
    */
    rt = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();

    rt_val = (int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rt, 0, &success);
    if (!success)
        return false;

    target = rt_val + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_JR (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t rs;
    uint32_t rs_val;

    /* 
     * JR rs
     *      PC = GPR[rs]
    */
    rs = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());

    rs_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + rs, 0, &success);
    if (!success)
        return false;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, rs_val))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1F (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int32_t target, pc, offset;
    
    /*
     * BC1F cc, offset
     *  condition <- (FPConditionCode(cc) == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    fcsr = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_fcsr_mips, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);
    
    if ((fcsr & (1 << cc)) == 0)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1T (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int32_t target, pc, offset;
    
    /*
     * BC1T cc, offset
     *  condition <- (FPConditionCode(cc) != 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    fcsr = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_fcsr_mips, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);
    
    if ((fcsr & (1 << cc)) != 0)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1FL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int32_t target, pc, offset;
    
    /*
     * BC1F cc, offset
     *  condition <- (FPConditionCode(cc) == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    fcsr = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_fcsr_mips, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);
    
    if ((fcsr & (1 << cc)) == 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1TL (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int32_t target, pc, offset;
    
    /*
     * BC1T cc, offset
     *  condition <- (FPConditionCode(cc) != 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    fcsr = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_fcsr_mips, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);
    
    if ((fcsr & (1 << cc)) != 0)
        target = pc + offset;
    else
        target = pc + 8;    /* skip delay slot */
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1EQZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t ft;
    uint32_t ft_val;
    int32_t target, pc, offset;
    
    /*
     * BC1EQZ ft, offset
     *  condition <- (FPR[ft].bit0 == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + 4 + offset
    */
    ft = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    ft_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + ft, 0, &success);
    if (!success)
        return false;

    if ((ft_val & 1) == 0)
        target = pc + 4 + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1NEZ (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t ft;
    uint32_t ft_val;
    int32_t target, pc, offset;
    
    /*
     * BC1NEZ ft, offset
     *  condition <- (FPR[ft].bit0 != 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + 4 + offset
    */
    ft = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    ft_val = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + ft, 0, &success);
    if (!success)
        return false;

    if ((ft_val & 1) != 0)
        target = pc + 4 + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1ANY2F (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int32_t target, pc, offset;
    
    /*
     * BC1ANY2F cc, offset
     *  condition <- (FPConditionCode(cc) == 0 
     *                  || FPConditionCode(cc+1) == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    fcsr = (uint32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_fcsr_mips, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

    /* if any one bit is 0 */
    if (((fcsr >> cc) & 3) != 3)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1ANY2T (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int32_t target, pc, offset;
    
    /*
     * BC1ANY2T cc, offset
     *  condition <- (FPConditionCode(cc) == 1 
     *                  || FPConditionCode(cc+1) == 1)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    fcsr = (uint32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_fcsr_mips, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

    /* if any one bit is 1 */
    if (((fcsr >> cc) & 3) != 0)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1ANY4F (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int32_t target, pc, offset;
    
    /*
     * BC1ANY4F cc, offset
     *  condition <- (FPConditionCode(cc) == 0 
     *                  || FPConditionCode(cc+1) == 0)
     *                  || FPConditionCode(cc+2) == 0)
     *                  || FPConditionCode(cc+3) == 0)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    fcsr = (uint32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_fcsr_mips, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

    /* if any one bit is 0 */
    if (((fcsr >> cc) & 0xf) != 0xf)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BC1ANY4T (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t cc, fcsr;
    int32_t target, pc, offset;
    
    /*
     * BC1ANY4T cc, offset
     *  condition <- (FPConditionCode(cc) == 1 
     *                  || FPConditionCode(cc+1) == 1)
     *                  || FPConditionCode(cc+2) == 1)
     *                  || FPConditionCode(cc+3) == 1)
     *      if condition then
     *          offset = sign_ext (offset)
     *          PC = PC + offset
    */
    cc = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    offset = insn.getOperand(1).getImm();
    
    pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    fcsr = (uint32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_fcsr_mips, 0, &success);
    if (!success)
        return false;

    /* fcsr[23], fcsr[25-31] are vaild condition bits */
    fcsr = ((fcsr >> 24) & 0xfe) | ((fcsr >> 23) & 0x01);

    /* if any one bit is 1 */
    if (((fcsr >> cc) & 0xf) != 0)
        target = pc + offset;
    else
        target = pc + 8;
    
    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BNZB (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_DF(insn, 1, true);
}

bool
EmulateInstructionMIPS::Emulate_BNZH (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_DF(insn, 2, true);
}

bool
EmulateInstructionMIPS::Emulate_BNZW (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_DF(insn, 4, true);
}

bool
EmulateInstructionMIPS::Emulate_BNZD (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_DF(insn, 8, true);
}

bool
EmulateInstructionMIPS::Emulate_BZB (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_DF(insn, 1, false);
}

bool
EmulateInstructionMIPS::Emulate_BZH (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_DF(insn, 2, false);
}

bool
EmulateInstructionMIPS::Emulate_BZW (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_DF(insn, 4, false);
}

bool
EmulateInstructionMIPS::Emulate_BZD (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_DF(insn, 8, false);
}

bool
EmulateInstructionMIPS::Emulate_MSA_Branch_DF (llvm::MCInst& insn, int element_byte_size, bool bnz)
{
    bool success = false, branch_hit = true;
    int32_t target = 0;
    RegisterValue reg_value;
    uint8_t * ptr = NULL;

    uint32_t wt = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    int32_t offset = insn.getOperand(1).getImm();

    int32_t pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    if (ReadRegister (eRegisterKindDWARF, dwarf_w0_mips + wt, reg_value))
        ptr = (uint8_t *)reg_value.GetBytes();
    else
        return false;

    for(int i = 0; i < 16 / element_byte_size; i++)
    {
        switch(element_byte_size)
        {
            case 1:
                if((*ptr == 0 && bnz) || (*ptr != 0 && !bnz) )
                    branch_hit = false;
                break;
            case 2:
                if((*(uint16_t *)ptr == 0 && bnz) || (*(uint16_t *)ptr != 0 && !bnz))
                    branch_hit = false;
                break;
            case 4:
                if((*(uint32_t *)ptr == 0 && bnz) || (*(uint32_t *)ptr != 0 && !bnz))
                    branch_hit = false;
                break;
            case 8:
                if((*(uint64_t *)ptr == 0 && bnz) || (*(uint64_t *)ptr != 0 && !bnz))
                    branch_hit = false;
                break;
        }
        if(!branch_hit)
           break;
        ptr = ptr + element_byte_size;
    }

    if(branch_hit)
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_BNZV (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_V (insn, true);
}

bool
EmulateInstructionMIPS::Emulate_BZV (llvm::MCInst& insn)
{
    return Emulate_MSA_Branch_V (insn, false);
}

bool
EmulateInstructionMIPS::Emulate_MSA_Branch_V (llvm::MCInst& insn, bool bnz)
{
    bool success = false;
    int32_t target = 0;
    llvm::APInt wr_val = llvm::APInt::getNullValue(128);
    llvm::APInt fail_value = llvm::APInt::getMaxValue(128);
    llvm::APInt zero_value = llvm::APInt::getNullValue(128);
    RegisterValue reg_value;

    uint32_t wt = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    int32_t offset = insn.getOperand(1).getImm();

    int32_t pc = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_pc_mips, 0, &success);
    if (!success)
        return false;

    if (ReadRegister (eRegisterKindDWARF, dwarf_w0_mips + wt, reg_value))
        wr_val = reg_value.GetAsUInt128(fail_value);
    else
        return false;

    if((llvm::APInt::isSameValue(zero_value, wr_val) && !bnz) || (!llvm::APInt::isSameValue(zero_value, wr_val) && bnz))
        target = pc + offset;
    else
        target = pc + 8;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_pc_mips, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS::Emulate_LDST_Imm (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t base;
    int32_t imm, address;
    Context bad_vaddr_context;

    uint32_t num_operands = insn.getNumOperands();
    base = m_reg_info->getEncodingValue (insn.getOperand(num_operands-2).getReg());
    imm = insn.getOperand(num_operands-1).getImm();

    RegisterInfo reg_info_base;
    if (!GetRegisterInfo (eRegisterKindDWARF, dwarf_zero_mips + base, reg_info_base))
        return false;

    /* read base register */
    address =(int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + base, 0, &success);
    if (!success)
        return false;

    /* destination address */
    address = address + imm;

    /* Set the bad_vaddr register with base address used in the instruction */
    bad_vaddr_context.type = eContextInvalid;
    WriteRegisterUnsigned (bad_vaddr_context, eRegisterKindDWARF, dwarf_bad_mips, address);

    return true;
}

bool
EmulateInstructionMIPS::Emulate_LDST_Reg (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t base, index;
    int32_t address, index_address;
    Context bad_vaddr_context;

    uint32_t num_operands = insn.getNumOperands();
    base = m_reg_info->getEncodingValue (insn.getOperand(num_operands-2).getReg());
    index = m_reg_info->getEncodingValue (insn.getOperand(num_operands-1).getReg());

    RegisterInfo reg_info_base, reg_info_index;
    if (!GetRegisterInfo (eRegisterKindDWARF, dwarf_zero_mips + base, reg_info_base))
        return false;

    if (!GetRegisterInfo (eRegisterKindDWARF, dwarf_zero_mips + index, reg_info_index))
        return false;

    /* read base register */
    address =(int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + base, 0, &success);
    if (!success)
        return false;

    /* read index register */
    index_address =(int32_t) ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_zero_mips + index, 0, &success);
    if (!success)
        return false;

    /* destination address */
    address = address + index_address;

    /* Set the bad_vaddr register with base address used in the instruction */
    bad_vaddr_context.type = eContextInvalid;
    WriteRegisterUnsigned (bad_vaddr_context, eRegisterKindDWARF, dwarf_bad_mips, address);

    return true;
}

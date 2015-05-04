//===-- EmulateInstructionMIPS64.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EmulateInstructionMIPS64.h"

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
#include "Plugins/Process/Utility/RegisterContext_mips64.h"

using namespace lldb;
using namespace lldb_private;

#define UInt(x) ((uint64_t)x)
#define integer int64_t


//----------------------------------------------------------------------
//
// EmulateInstructionMIPS64 implementation
//
//----------------------------------------------------------------------

EmulateInstructionMIPS64::EmulateInstructionMIPS64 (const lldb_private::ArchSpec &arch) :
    EmulateInstruction (arch)
{
    /* Create instance of llvm::MCDisassembler */
    std::string Error;
    llvm::Triple triple = arch.GetTriple();
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget (triple.getTriple(), Error);
    assert (target);

    m_reg_info.reset (target->createMCRegInfo (triple.getTriple()));
    assert (m_reg_info.get());

    m_insn_info.reset (target->createMCInstrInfo());
    assert (m_insn_info.get());

    m_asm_info.reset (target->createMCAsmInfo (*m_reg_info, triple.getTriple()));
    m_subtype_info.reset (target->createMCSubtargetInfo (triple.getTriple(), "", ""));
    assert (m_asm_info.get() && m_subtype_info.get());

    m_context.reset (new llvm::MCContext (m_asm_info.get(), m_reg_info.get(), nullptr));
    assert (m_context.get());

    m_disasm.reset (target->createMCDisassembler (*m_subtype_info, *m_context));
    assert (m_disasm.get());
}

void
EmulateInstructionMIPS64::Initialize ()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic (),
                                   GetPluginDescriptionStatic (),
                                   CreateInstance);
}

void
EmulateInstructionMIPS64::Terminate ()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

ConstString
EmulateInstructionMIPS64::GetPluginNameStatic ()
{
    ConstString g_plugin_name ("lldb.emulate-instruction.mips64");
    return g_plugin_name;
}

lldb_private::ConstString
EmulateInstructionMIPS64::GetPluginName()
{
    static ConstString g_plugin_name ("EmulateInstructionMIPS64");
    return g_plugin_name;
}

const char *
EmulateInstructionMIPS64::GetPluginDescriptionStatic ()
{
    return "Emulate instructions for the MIPS64 architecture.";
}

EmulateInstruction *
EmulateInstructionMIPS64::CreateInstance (const ArchSpec &arch, InstructionType inst_type)
{
    if (EmulateInstructionMIPS64::SupportsEmulatingInstructionsOfTypeStatic(inst_type))
    {
        if (arch.GetTriple().getArch() == llvm::Triple::mips64 
            || arch.GetTriple().getArch() == llvm::Triple::mips64el)
        {
            std::auto_ptr<EmulateInstructionMIPS64> emulate_insn_ap (new EmulateInstructionMIPS64 (arch));
            if (emulate_insn_ap.get())
                return emulate_insn_ap.release();
        }
    }
    
    return NULL;
}

bool
EmulateInstructionMIPS64::SetTargetTriple (const ArchSpec &arch)
{
    if (arch.GetTriple().getArch () == llvm::Triple::mips64
        || arch.GetTriple().getArch () == llvm::Triple::mips64el)
        return true;
    return false;
}

const char *
EmulateInstructionMIPS64::GetRegisterName (unsigned reg_num, bool alternate_name)
{
    if (alternate_name)
    {
        switch (reg_num)
        {
            case gcc_dwarf_sp_mips64: return "r29"; 
            case gcc_dwarf_r30_mips64: return "r30"; 
            case gcc_dwarf_ra_mips64: return "r31";
            default:
                break;
        }
        return nullptr;
    }

    switch (reg_num)
    {
        case gcc_dwarf_zero_mips64:     return "r0";
        case gcc_dwarf_r1_mips64:       return "r1";
        case gcc_dwarf_r2_mips64:       return "r2";
        case gcc_dwarf_r3_mips64:       return "r3";
        case gcc_dwarf_r4_mips64:       return "r4";
        case gcc_dwarf_r5_mips64:       return "r5";
        case gcc_dwarf_r6_mips64:       return "r6";
        case gcc_dwarf_r7_mips64:       return "r7";
        case gcc_dwarf_r8_mips64:       return "r8";
        case gcc_dwarf_r9_mips64:       return "r9";
        case gcc_dwarf_r10_mips64:      return "r10";
        case gcc_dwarf_r11_mips64:      return "r11";
        case gcc_dwarf_r12_mips64:      return "r12";
        case gcc_dwarf_r13_mips64:      return "r13";
        case gcc_dwarf_r14_mips64:      return "r14";
        case gcc_dwarf_r15_mips64:      return "r15";
        case gcc_dwarf_r16_mips64:      return "r16";
        case gcc_dwarf_r17_mips64:      return "r17";
        case gcc_dwarf_r18_mips64:      return "r18";
        case gcc_dwarf_r19_mips64:      return "r19";
        case gcc_dwarf_r20_mips64:      return "r20";
        case gcc_dwarf_r21_mips64:      return "r21";
        case gcc_dwarf_r22_mips64:      return "r22";
        case gcc_dwarf_r23_mips64:      return "r23";
        case gcc_dwarf_r24_mips64:      return "r24";
        case gcc_dwarf_r25_mips64:      return "r25";
        case gcc_dwarf_r26_mips64:      return "r26";
        case gcc_dwarf_r27_mips64:      return "r27";
        case gcc_dwarf_gp_mips64:       return "gp";
        case gcc_dwarf_sp_mips64:       return "sp";
        case gcc_dwarf_r30_mips64:      return "fp";
        case gcc_dwarf_ra_mips64:       return "ra";
        case gcc_dwarf_sr_mips64:       return "sr";
        case gcc_dwarf_lo_mips64:       return "lo";
        case gcc_dwarf_hi_mips64:       return "hi";
        case gcc_dwarf_bad_mips64:      return "bad";
        case gcc_dwarf_cause_mips64:    return "cause";
        case gcc_dwarf_pc_mips64:       return "pc";

    }
    return nullptr;
}

bool
EmulateInstructionMIPS64::GetRegisterInfo (RegisterKind reg_kind, uint32_t reg_num, RegisterInfo &reg_info)
{
    if (reg_kind == eRegisterKindGeneric)
    {
        switch (reg_num)
        {
            case LLDB_REGNUM_GENERIC_PC:    reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_pc_mips64; break;
            case LLDB_REGNUM_GENERIC_SP:    reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_sp_mips64; break;
            case LLDB_REGNUM_GENERIC_FP:    reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_r30_mips64; break;
            case LLDB_REGNUM_GENERIC_RA:    reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_ra_mips64; break;
            case LLDB_REGNUM_GENERIC_FLAGS: reg_kind = eRegisterKindDWARF; reg_num = gcc_dwarf_sr_mips64; break;
                 return true;
            default:
                return false;
        }
    }

    if (reg_kind == eRegisterKindDWARF)
    {
       ::memset (&reg_info, 0, sizeof(RegisterInfo));
       ::memset (reg_info.kinds, LLDB_INVALID_REGNUM, sizeof(reg_info.kinds));

       if (reg_num == gcc_dwarf_sr_mips64)
       {
           reg_info.byte_size = 4;
           reg_info.format = eFormatHex;
           reg_info.encoding = eEncodingUint;
       }
       else if ((int)reg_num >= gcc_dwarf_zero_mips64 && (int)reg_num <= gcc_dwarf_pc_mips64)
       {
           reg_info.byte_size = 8;
           reg_info.format = eFormatHex;
           reg_info.encoding = eEncodingUint;
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
           case gcc_dwarf_r30_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP; break;
           case gcc_dwarf_ra_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_RA; break;
           case gcc_dwarf_sp_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP; break;
           case gcc_dwarf_pc_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC; break;
           case gcc_dwarf_sr_mips64: reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS; break;
           default: break;
       }
       return true;
    }
    return false;
}

EmulateInstructionMIPS64::MipsOpcode*
EmulateInstructionMIPS64::GetOpcodeForInstruction (const char *op_name)
{
    static EmulateInstructionMIPS64::MipsOpcode
    g_opcodes[] = 
    {
        //----------------------------------------------------------------------
        // Prologue/Epilogue instructions
        //----------------------------------------------------------------------
        { "DADDiu",     &EmulateInstructionMIPS64::Emulate_DADDiu,      "DADDIU rt,rs,immediate"    },
        { "SD",         &EmulateInstructionMIPS64::Emulate_SD,          "SD rt,offset(rs)"          },
        { "LD",         &EmulateInstructionMIPS64::Emulate_LD,          "LD rt,offset(base)"        },

        //----------------------------------------------------------------------
        // Branch instructions
        //----------------------------------------------------------------------
        { "B",          &EmulateInstructionMIPS64::Emulate_B,           "B offset"                  },
        { "BAL",        &EmulateInstructionMIPS64::Emulate_BAL,         "BAL offset"                },
        { "BALC",       &EmulateInstructionMIPS64::Emulate_BALC,        "BALC offset"               },
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
EmulateInstructionMIPS64::ReadInstruction ()
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
EmulateInstructionMIPS64::EvaluateInstruction (uint32_t evaluate_options)
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
        old_pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
        if (!success)
            return false;
    }

    /* emulate instruction */
    success = (this->*opcode_data->callback) (mc_insn);
    if (!success)
        return false;

    if (auto_advance_pc)
    {
        new_pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
        if (!success)
            return false;

        /* If we haven't changed the PC, change it here */
        if (old_pc == new_pc)
        {
            new_pc += 4;
            Context context;
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, new_pc))
                return false;
        }
    }

    return true;
}

bool
EmulateInstructionMIPS64::CreateFunctionEntryUnwind (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    UnwindPlan::RowSP row(new UnwindPlan::Row);
    const bool can_replace = false;

    // Our previous Call Frame Address is the stack pointer
    row->GetCFAValue().SetIsRegisterPlusOffset(gcc_dwarf_sp_mips64, 0);

    // Our previous PC is in the RA
    row->SetRegisterLocationToRegister(gcc_dwarf_pc_mips64, gcc_dwarf_ra_mips64, can_replace);

    unwind_plan.AppendRow (row);

    // All other registers are the same.
    unwind_plan.SetSourceName ("EmulateInstructionMIPS64");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolYes);

    return true;
}

bool
EmulateInstructionMIPS64::nonvolatile_reg_p (uint64_t regnum)
{
    switch (regnum)
    {
        case gcc_dwarf_r16_mips64:
        case gcc_dwarf_r17_mips64:
        case gcc_dwarf_r18_mips64:
        case gcc_dwarf_r19_mips64:
        case gcc_dwarf_r20_mips64:
        case gcc_dwarf_r21_mips64:
        case gcc_dwarf_r22_mips64:
        case gcc_dwarf_r23_mips64:
        case gcc_dwarf_gp_mips64:
        case gcc_dwarf_sp_mips64:
        case gcc_dwarf_r30_mips64:
        case gcc_dwarf_ra_mips64:
            return true;
        default:
            return false;
    }
    return false;
}

bool
EmulateInstructionMIPS64::Emulate_DADDiu (llvm::MCInst& insn)
{
    bool success = false;
    const uint32_t imm16 = insn.getOperand(2).getImm();
    uint64_t imm = SignedBits(imm16, 15, 0);
    uint64_t result;
    uint32_t src, dst;

    dst = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    src = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    /* Check if this is daddiu sp,<src>,imm16 */
    if (dst == gcc_dwarf_sp_mips64)
    {
        /* read <src> register */
        uint64_t src_opd_val = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + src, 0, &success);
        if (!success)
            return false;

        result = src_opd_val + imm;

        Context context;
        RegisterInfo reg_info_sp;
        if (GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_sp_mips64, reg_info_sp))
            context.SetRegisterPlusOffset (reg_info_sp, imm);

        /* We are allocating bytes on stack */
        context.type = eContextAdjustStackPointer;

        WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_sp_mips64, result);
    }
    
    return true;
}

bool
EmulateInstructionMIPS64::Emulate_SD (llvm::MCInst& insn)
{
    bool success = false;
    uint32_t imm16 = insn.getOperand(2).getImm();
    uint64_t imm = SignedBits(imm16, 15, 0);
    uint32_t src, base;

    src = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    base = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    /* We look for sp based non-volatile register stores */
    if (base == gcc_dwarf_sp_mips64 && nonvolatile_reg_p (src))
    {
        uint64_t address;
        RegisterInfo reg_info_base;
        RegisterInfo reg_info_src;

        if (!GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + base, reg_info_base)
            || !GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + src, reg_info_src))
            return false;

        /* read SP */
        address = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + base, 0, &success);
        if (!success)
            return false;

        /* destination address */
        address = address + imm;

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
EmulateInstructionMIPS64::Emulate_LD (llvm::MCInst& insn)
{
    uint32_t src, base;

    src = m_reg_info->getEncodingValue (insn.getOperand(0).getReg());
    base = m_reg_info->getEncodingValue (insn.getOperand(1).getReg());

    if (base == gcc_dwarf_sp_mips64 && nonvolatile_reg_p (src))
    {
        RegisterValue data_src;
        RegisterInfo reg_info_src;

        if (!GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + src, reg_info_src))
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
EmulateInstructionMIPS64::Emulate_B (llvm::MCInst& insn)
{
    bool success = false;
    uint64_t offset, pc, target;

    /*
     * B offset
     *      PC = PC + sign_ext (offset << 2)
    */
    offset = insn.getOperand(0).getImm() << 2;
    offset = SignedBits (offset, 17, 0);

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    target = pc + offset;

    Context context;
    context.type = eContextRelativeBranchImmediate;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BAL (llvm::MCInst& insn)
{
    bool success = false;
    uint64_t offset, pc, target;

    /*
     * BAL offset
     *      offset = sign_ext (offset << 2)
     *      RA = PC + 8
     *      PC = PC + offset
    */
    offset = insn.getOperand(0).getImm() << 2;
    offset = SignedBits (offset, 17, 0);

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    target = pc + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 8))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_BALC (llvm::MCInst& insn)
{
    bool success = false;
    uint64_t offset, pc, target;

    /* 
     * BALC offset
     *      offset = sign_ext (offset << 2)
     *      RA = PC + 4
     *      PC = PC + 4 + offset
    */
    offset = insn.getOperand(0).getImm() << 2;
    offset = SignedBits (offset, 17, 0);

    pc = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_pc_mips64, 0, &success);
    if (!success)
        return false;

    target = pc + 4 + offset;

    Context context;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_pc_mips64, target))
        return false;

    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_ra_mips64, pc + 4))
        return false;

    return true;
}

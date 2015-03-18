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

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/UnwindPlan.h"

#include "llvm/ADT/STLExtras.h"
//#include "llvm/Support/MathExtras.h" // for SignExtend32 template function

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
        if (arch.GetTriple().getArch() == llvm::Triple::mips64)
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
    if (arch.GetTriple().getArch () == llvm::Triple::mips64)
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

            default: return false;
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
       else if (reg_num >= gcc_dwarf_zero_mips64 && reg_num <= gcc_dwarf_pc_mips64)
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

EmulateInstructionMIPS64::Opcode*
EmulateInstructionMIPS64::GetOpcodeForInstruction (const uint32_t opcode)
{
    static EmulateInstructionMIPS64::Opcode 
    g_opcodes[] = 
    {
        //----------------------------------------------------------------------
        // Prologue/Epilogue instructions
        //----------------------------------------------------------------------

        // stack adjustment
        { 0xffff0000, 0x67bd0000, &EmulateInstructionMIPS64::Emulate_addsp_imm, "DADDIU rt, rs, immediate" },

        // store register
        { 0xfc000000, 0xfc000000, &EmulateInstructionMIPS64::Emulate_store, "SD rt, offset(Rn)" },

        // Load register
        { 0xfc000000, 0xdc000000, &EmulateInstructionMIPS64::Emulate_load, "LD rt, offset(base)" },

    };
    static const size_t k_num_mips_opcodes = llvm::array_lengthof(g_opcodes);

    for (size_t i=0; i<k_num_mips_opcodes; ++i)
    {
        if ((g_opcodes[i].mask & opcode) == g_opcodes[i].value)
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
    uint32_t opcode = m_opcode.GetOpcode32();

    if (GetByteOrder() == eByteOrderBig)
       opcode = llvm::ByteSwap_32(opcode);

    Opcode *opcode_data = GetOpcodeForInstruction(opcode);
    bool success = false;

    if (opcode_data == NULL)
        return false;

    // Call the Emulate... function.
    success = (this->*opcode_data->callback) (opcode);  
    if (!success)
        return false;
        
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
EmulateInstructionMIPS64::Emulate_addsp_imm (const uint32_t opcode)
{

    /* Get immediate operand */
    const uint32_t imm16 = Bits32(opcode, 15, 0);
    
    bool success = false;
    uint64_t result;
    uint64_t imm = SignedBits(imm16, 15, 0);

    uint64_t operand1 = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_sp_mips64, 0, &success);
    uint64_t operand2 = imm;

    result = operand1 + operand2;
    
    Context context;
    RegisterInfo reg_info_Rn;
    if (GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_sp_mips64, reg_info_Rn))
        context.SetRegisterPlusOffset (reg_info_Rn, imm);

    /* We are allocating bytes on stack */
    context.type = eContextAdjustStackPointer;

    WriteRegisterUnsigned (context, eRegisterKindDWARF, gcc_dwarf_sp_mips64, result);
    
    return true;
}

bool
EmulateInstructionMIPS64::Emulate_store (const uint32_t opcode)
{
    uint32_t imm16 = Bits32(opcode, 15, 0);
    uint32_t Rt = Bits32(opcode, 20, 16);
    uint32_t base = Bits32(opcode, 25, 21);
    uint64_t imm = SignedBits(imm16, 15, 0);
    uint64_t address;

    integer n = UInt(base);
    integer t = UInt(Rt);

    RegisterValue data_Rt;

    RegisterInfo reg_info_base;
    RegisterInfo reg_info_Rt;

    if (!GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + n, reg_info_base))
        return false;

    if (!GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + t, reg_info_Rt))
        return false;

    bool success = false;
    Context context_t;

    /* We look for sp based non-volatile register stores */
    if (n == 29 && nonvolatile_reg_p(t))
        address = ReadRegisterUnsigned (eRegisterKindDWARF, gcc_dwarf_sp_mips64, 0, &success);
    else
        return false;

    /* Calculate address to store */
    address = address + imm;

    context_t.type = eContextPushRegisterOnStack;
    context_t.SetRegisterToRegisterPlusOffset (reg_info_Rt, reg_info_base, 0);

    uint8_t buffer [RegisterValue::kMaxRegisterByteSize];
    Error error;

    if (!ReadRegister (&reg_info_Rt, data_Rt))
        return false;

    if (data_Rt.GetAsMemoryData(&reg_info_Rt, buffer, reg_info_Rt.byte_size, eByteOrderLittle, error) == 0)
        return false;

    if (!WriteMemory(context_t, address, buffer, reg_info_Rt.byte_size))
        return false;

    return true;
}

bool
EmulateInstructionMIPS64::Emulate_load (const uint32_t opcode)
{
    uint32_t Rt = Bits32(opcode, 20, 16);
    uint32_t base = Bits32(opcode, 25, 21);

    integer n = UInt(base);
    integer t = UInt(Rt);
   
    RegisterValue data_Rt;
    RegisterInfo reg_info_Rt;

    if (!GetRegisterInfo (eRegisterKindDWARF, gcc_dwarf_zero_mips64 + t, reg_info_Rt))
        return false;

    Context context_t;

    /* We are looking for "saved register" being restored from stack */
    if (!n == 29 || !nonvolatile_reg_p(t))
        return false;

    context_t.type = eContextRegisterLoad;

    if (!WriteRegister (context_t, &reg_info_Rt, data_Rt))
        return false;

    return true;
}

//===-- EmulateInstructionARM64.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EmulateInstructionARM64.h"

#include <stdlib.h>

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/UnwindPlan.h"

#include "Plugins/Process/Utility/ARMDefines.h"
#include "Plugins/Process/Utility/ARMUtils.h"
#include "Utility/ARM64_DWARF_Registers.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h" // for SignExtend32 template function
                                     // and CountTrailingZeros_32 function

#include "Plugins/Process/Utility/InstructionUtils.h"

using namespace lldb;
using namespace lldb_private;

#define No_VFP  0
#define VFPv1   (1u << 1)
#define VFPv2   (1u << 2)
#define VFPv3   (1u << 3)
#define AdvancedSIMD (1u << 4)

#define VFPv1_ABOVE (VFPv1 | VFPv2 | VFPv3 | AdvancedSIMD)
#define VFPv2_ABOVE (VFPv2 | VFPv3 | AdvancedSIMD)
#define VFPv2v3     (VFPv2 | VFPv3)

#define UInt(x) ((uint64_t)x)
#define SInt(x) ((int64_t)x)
#define bit bool
#define boolean bool
#define integer int64_t

static inline bool
IsZero(uint64_t x)
{
    return x == 0;
}

static inline uint64_t
NOT(uint64_t x)
{
    return ~x;
}

#if 0
// LSL_C() 
// =======
static inline uint64_t
LSL_C (uint64_t x, integer shift, bool &carry_out)
{
    assert (shift >= 0); 
    uint64_t result = x << shift;
    carry_out = ((1ull << (64-1)) >> (shift - 1)) != 0;
    return result;
}
#endif

// LSL()
// =====

static inline uint64_t
LSL(uint64_t x, integer shift)
{
    if (shift == 0)
        return x;
    return x << shift;
}

// AddWithCarry()
// ===============
static inline uint64_t
AddWithCarry (uint32_t N, uint64_t x, uint64_t y, bit carry_in, EmulateInstructionARM64::ProcState &proc_state)
{
    uint64_t unsigned_sum = UInt(x) + UInt(y) + UInt(carry_in);
    int64_t signed_sum = SInt(x) + SInt(y) + UInt(carry_in);
    uint64_t result = unsigned_sum;
    if (N < 64)
        result = Bits64 (result, N-1, 0);
    proc_state.N = Bit64(result, N-1);
    proc_state.Z = IsZero(result);
    proc_state.C = UInt(result) == unsigned_sum;
    proc_state.V = SInt(result) == signed_sum;
    return result;
}

// ConstrainUnpredictable()
// ========================

EmulateInstructionARM64::ConstraintType
ConstrainUnpredictable (EmulateInstructionARM64::Unpredictable which)
{
    EmulateInstructionARM64::ConstraintType result = EmulateInstructionARM64::Constraint_UNKNOWN;
    switch (which)
    {
        case EmulateInstructionARM64::Unpredictable_WBOVERLAP:
        case EmulateInstructionARM64::Unpredictable_LDPOVERLAP:
            // TODO: don't know what to really do here? Pseudo code says:
            // set result to one of above Constraint behaviours or UNDEFINED
            break;
    }
    return result;
}



//----------------------------------------------------------------------
//
// EmulateInstructionARM implementation
//
//----------------------------------------------------------------------

void
EmulateInstructionARM64::Initialize ()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic (),
                                   GetPluginDescriptionStatic (),
                                   CreateInstance);
}

void
EmulateInstructionARM64::Terminate ()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

ConstString
EmulateInstructionARM64::GetPluginNameStatic ()
{
    ConstString g_plugin_name ("lldb.emulate-instruction.arm64");
    return g_plugin_name;
}

lldb_private::ConstString
EmulateInstructionARM64::GetPluginName()
{
    static ConstString g_plugin_name ("EmulateInstructionARM64");
    return g_plugin_name;
}

const char *
EmulateInstructionARM64::GetPluginDescriptionStatic ()
{
    return "Emulate instructions for the ARM64 architecture.";
}

EmulateInstruction *
EmulateInstructionARM64::CreateInstance (const ArchSpec &arch, InstructionType inst_type)
{
    if (EmulateInstructionARM64::SupportsEmulatingInstructionsOfTypeStatic(inst_type))
    {
        if (arch.GetTriple().getArch() == llvm::Triple::aarch64)
        {
            std::auto_ptr<EmulateInstructionARM64> emulate_insn_ap (new EmulateInstructionARM64 (arch));
            if (emulate_insn_ap.get())
                return emulate_insn_ap.release();
        }
    }
    
    return NULL;
}

bool
EmulateInstructionARM64::SetTargetTriple (const ArchSpec &arch)
{
    if (arch.GetTriple().getArch () == llvm::Triple::arm)
        return true;
    else if (arch.GetTriple().getArch () == llvm::Triple::thumb)
        return true;
       
    return false;
}
    
bool
EmulateInstructionARM64::GetRegisterInfo (RegisterKind reg_kind, uint32_t reg_num, RegisterInfo &reg_info)
{
    if (reg_kind == eRegisterKindGeneric)
    {
        switch (reg_num)
        {
            case LLDB_REGNUM_GENERIC_PC:    reg_kind = eRegisterKindDWARF; reg_num = arm64_dwarf::pc; break;
            case LLDB_REGNUM_GENERIC_SP:    reg_kind = eRegisterKindDWARF; reg_num = arm64_dwarf::sp; break;
            case LLDB_REGNUM_GENERIC_FP:    reg_kind = eRegisterKindDWARF; reg_num = arm64_dwarf::fp; break;
            case LLDB_REGNUM_GENERIC_RA:    reg_kind = eRegisterKindDWARF; reg_num = arm64_dwarf::lr; break;
            case LLDB_REGNUM_GENERIC_FLAGS: 
                // There is no DWARF register number for the CPSR right now...
                reg_info.name = "cpsr";
                reg_info.alt_name = NULL;
                reg_info.byte_size = 4;
                reg_info.byte_offset = 0;
                reg_info.encoding = eEncodingUint;
                reg_info.format = eFormatHex;
                for (uint32_t i=0; i<lldb::kNumRegisterKinds; ++i)
                    reg_info.kinds[reg_kind] = LLDB_INVALID_REGNUM;
                reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS;
                return true;
                
            default: return false;
        }
    }
    
    if (reg_kind == eRegisterKindDWARF)
        return arm64_dwarf::GetRegisterInfo(reg_num, reg_info);
    return false;
}

EmulateInstructionARM64::Opcode*
EmulateInstructionARM64::GetOpcodeForInstruction (const uint32_t opcode)
{
    static EmulateInstructionARM64::Opcode 
    g_opcodes[] = 
    {
        //----------------------------------------------------------------------
        // Prologue instructions
        //----------------------------------------------------------------------

        // push register(s)
        { 0xff000000, 0xd1000000, No_VFP, &EmulateInstructionARM64::Emulate_addsub_imm, "SUB  <Xd|SP>, <Xn|SP>, #<imm> {, <shift>}" },
        { 0xff000000, 0xf1000000, No_VFP, &EmulateInstructionARM64::Emulate_addsub_imm, "SUBS  <Xd>, <Xn|SP>, #<imm> {, <shift>}" },
        { 0xff000000, 0x91000000, No_VFP, &EmulateInstructionARM64::Emulate_addsub_imm, "ADD  <Xd|SP>, <Xn|SP>, #<imm> {, <shift>}" },
        { 0xff000000, 0xb1000000, No_VFP, &EmulateInstructionARM64::Emulate_addsub_imm, "ADDS  <Xd>, <Xn|SP>, #<imm> {, <shift>}" },


        { 0xff000000, 0x51000000, No_VFP, &EmulateInstructionARM64::Emulate_addsub_imm, "SUB  <Wd|WSP>, <Wn|WSP>, #<imm> {, <shift>}" },
        { 0xff000000, 0x71000000, No_VFP, &EmulateInstructionARM64::Emulate_addsub_imm, "SUBS  <Wd>, <Wn|WSP>, #<imm> {, <shift>}" },
        { 0xff000000, 0x11000000, No_VFP, &EmulateInstructionARM64::Emulate_addsub_imm, "ADD  <Wd|WSP>, <Wn|WSP>, #<imm> {, <shift>}" },
        { 0xff000000, 0x31000000, No_VFP, &EmulateInstructionARM64::Emulate_addsub_imm, "ADDS  <Wd>, <Wn|WSP>, #<imm> {, <shift>}" },
        
        { 0xffc00000, 0x29000000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_off, "STP  <Wt>, <Wt2>, [<Xn|SP>{, #<imm>}]" },
        { 0xffc00000, 0xa9000000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_off, "STP  <Xt>, <Xt2>, [<Xn|SP>{, #<imm>}]" },
        { 0xffc00000, 0x2d000000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_off, "STP  <St>, <St2>, [<Xn|SP>{, #<imm>}]" },
        { 0xffc00000, 0x6d000000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_off, "STP  <Dt>, <Dt2>, [<Xn|SP>{, #<imm>}]" },
        { 0xffc00000, 0xad000000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_off, "STP  <Qt>, <Qt2>, [<Xn|SP>{, #<imm>}]" },

        { 0xffc00000, 0xad800000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_pre, "STP  <Qt>, <Qt2>, [<Xn|SP>, #<imm>]!" },
        { 0xffc00000, 0x2d800000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_pre, "STP  <St>, <St2>, [<Xn|SP>, #<imm>]!" },
        { 0xffc00000, 0x29800000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_pre, "STP  <Wt>, <Wt2>, [<Xn|SP>, #<imm>]!" },
        { 0xffc00000, 0x6d800000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_pre, "STP  <Dt>, <Dt2>, [<Xn|SP>, #<imm>]!" },
        { 0xffc00000, 0xa9800000, No_VFP, &EmulateInstructionARM64::Emulate_ldstpair_pre, "STP  <Xt>, <Xt2>, [<Xn|SP>, #<imm>]!" },

    };
    static const size_t k_num_arm_opcodes = llvm::array_lengthof(g_opcodes);
                  
    for (size_t i=0; i<k_num_arm_opcodes; ++i)
    {
        if ((g_opcodes[i].mask & opcode) == g_opcodes[i].value)
            return &g_opcodes[i];
    }
    return NULL;
}

bool 
EmulateInstructionARM64::ReadInstruction ()
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
EmulateInstructionARM64::EvaluateInstruction (uint32_t evaluate_options)
{
    const uint32_t opcode = m_opcode.GetOpcode32();
    Opcode *opcode_data = GetOpcodeForInstruction(opcode);
    if (opcode_data == NULL)
        return false;
    
    //printf ("opcode template for 0x%8.8x: %s\n", opcode, opcode_data->name);
    const bool auto_advance_pc = evaluate_options & eEmulateInstructionOptionAutoAdvancePC;
    m_ignore_conditions = evaluate_options & eEmulateInstructionOptionIgnoreConditions;
                 
    bool success = false;
//    if (m_opcode_cpsr == 0 || m_ignore_conditions == false)
//    {
//        m_opcode_cpsr = ReadRegisterUnsigned (eRegisterKindGeneric,         // use eRegisterKindDWARF is we ever get a cpsr DWARF register number
//                                              LLDB_REGNUM_GENERIC_FLAGS,    // use arm64_dwarf::cpsr if we ever get one
//                                              0,
//                                              &success);
//    }

    // Only return false if we are unable to read the CPSR if we care about conditions
    if (success == false && m_ignore_conditions == false)
        return false;
    
    uint32_t orig_pc_value = 0;
    if (auto_advance_pc)
    {
        orig_pc_value = ReadRegisterUnsigned (eRegisterKindDWARF, arm64_dwarf::pc, 0, &success);
        if (!success)
            return false;
    }
    
    // Call the Emulate... function.
    success = (this->*opcode_data->callback) (opcode);  
    if (!success)
        return false;
        
    if (auto_advance_pc)
    {
        uint32_t new_pc_value = ReadRegisterUnsigned (eRegisterKindDWARF, arm64_dwarf::pc, 0, &success);
        if (!success)
            return false;
            
        if (auto_advance_pc && (new_pc_value == orig_pc_value))
        {
            EmulateInstruction::Context context;
            context.type = eContextAdvancePC;
            context.SetNoArgs();
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, arm64_dwarf::pc, orig_pc_value + 4))
                return false;
        }
    }
    return true;
}

bool
EmulateInstructionARM64::CreateFunctionEntryUnwind (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    UnwindPlan::RowSP row(new UnwindPlan::Row);
    const bool can_replace = false;

    // Our previous Call Frame Address is the stack pointer
    row->SetCFARegister (arm64_dwarf::sp);
    
    // Our previous PC is in the LR
    row->SetRegisterLocationToRegister(arm64_dwarf::pc, arm64_dwarf::lr, can_replace);

    unwind_plan.AppendRow (row);
    
    // All other registers are the same.
    
    unwind_plan.SetSourceName ("EmulateInstructionARM64");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolYes);
    return true;
}



bool
EmulateInstructionARM64::Emulate_addsub_imm (const uint32_t opcode)
{
    // integer d = UInt(Rd);
    // integer n = UInt(Rn);
    // integer datasize = if sf == 1 then 64 else 32;
    // boolean sub_op = (op == 1);
    // boolean setflags = (S == 1);
    // bits(datasize) imm;
    //
    // case shift of
    //     when '00' imm = ZeroExtend(imm12, datasize);
    //     when '01' imm = ZeroExtend(imm12 : Zeros(12), datasize);
    //    when '1x' UNDEFINED;
    //
    //
    // bits(datasize) result;
    // bits(datasize) operand1 = if n == 31 then SP[] else X[n];
    // bits(datasize) operand2 = imm;
    // bits(4) nzcv;
    // bit carry_in;
    //
    // if sub_op then
    //     operand2 = NOT(operand2);
    //     carry_in = 1;
    // else
    //     carry_in = 0;
    //
    // (result, nzcv) = AddWithCarry(operand1, operand2, carry_in);
    //
    // if setflags then 
    //     PSTATE.NZCV = nzcv;
    //
    // if d == 31 && !setflags then
    //     SP[] = result;
    // else
    //     X[d] = result;
    
    const uint32_t sf = Bit32(opcode, 31);
    const uint32_t op = Bit32(opcode, 30);
    const uint32_t S = Bit32(opcode, 29);
    const uint32_t shift = Bits32(opcode, 23, 22);
    const uint32_t imm12 = Bits32(opcode, 21, 10);
    const uint32_t Rn = Bits32(opcode, 9, 5);
    const uint32_t Rd = Bits32(opcode, 4, 0);
    
    bool success = false;
    
    const uint32_t d = UInt(Rd);
    const uint32_t n = UInt(Rn);
    const uint32_t datasize = (sf == 1) ? 64 : 32;
    boolean sub_op = op == 1;
    boolean setflags = S == 1;
    uint64_t imm;

    switch (shift)
    {
        case 0: imm = imm12; break;
        case 1: imm = imm12 << 12; break;
        default: return false;  // UNDEFINED;
    }
    uint64_t result;
    uint64_t operand1 = ReadRegisterUnsigned (eRegisterKindDWARF, arm64_dwarf::x0 + n, 0, &success);
    uint64_t operand2 = imm;
    bit carry_in;
    
    if (sub_op)
    {
        operand2 = NOT(operand2);
        carry_in = 1;
        imm = -imm; // For the Register plug offset context below
    }
    else
    {
        carry_in = 0;
    }
    
    ProcState proc_state;
    
    result = AddWithCarry (datasize, operand1, operand2, carry_in, proc_state);

    if (setflags)
    {
        m_emulated_pstate.N = proc_state.N;
        m_emulated_pstate.Z = proc_state.Z;
        m_emulated_pstate.C = proc_state.C;
        m_emulated_pstate.V = proc_state.V;
    }
    
    Context context;
    RegisterInfo reg_info_Rn;
    if (arm64_dwarf::GetRegisterInfo (n, reg_info_Rn))
        context.SetRegisterPlusOffset (reg_info_Rn, imm);

    if ((n == arm64_dwarf::sp || n == arm64_dwarf::fp) &&
        d == arm64_dwarf::sp &&
        !setflags)
    {
        context.type = EmulateInstruction::eContextAdjustStackPointer;
    }
    else if (d == arm64_dwarf::fp &&
             n == arm64_dwarf::sp &&
             !setflags)
    {
        context.type = EmulateInstruction::eContextSetFramePointer;
    }
    else
    {
        context.type = EmulateInstruction::eContextImmediate;
    }
    WriteRegisterUnsigned (context, eRegisterKindDWARF, arm64_dwarf::x0 + d, result);
    
    return false;
}

bool
EmulateInstructionARM64::Emulate_ldstpair_off (const uint32_t opcode)
{
    return Emulate_ldstpair (opcode, AddrMode_OFF);
}


bool
EmulateInstructionARM64::Emulate_ldstpair_pre (const uint32_t opcode)
{
    return Emulate_ldstpair (opcode, AddrMode_PRE);
}

bool
EmulateInstructionARM64::Emulate_ldstpair (const uint32_t opcode, AddrMode a_mode)
{
    uint32_t opc = Bits32(opcode, 31, 30);
    uint32_t V = Bit32(opcode, 26);
    uint32_t L = Bit32(opcode, 22);
    uint32_t imm7 = Bits32(opcode, 21, 15);
    uint32_t Rt2 = Bits32(opcode, 14, 10);
    uint32_t Rn = Bits32(opcode, 9, 5);
    uint32_t Rt = Bits32(opcode, 4, 0);
    
    integer n = UInt(Rn);
    integer t = UInt(Rt);
    integer t2 = UInt(Rt2);
    uint64_t idx;
    
    MemOp memop = L == 1 ? MemOp_LOAD : MemOp_STORE;
    boolean vector = (V == 1);
    //AccType acctype = AccType_NORMAL;
    boolean is_signed = false;
    boolean wback = a_mode != AddrMode_OFF;
    boolean wb_unknown = false;
    boolean rt_unknown = false;
    integer scale;
    integer size;
    
    if (opc == 3)
        return false; // UNDEFINED
    
    if (vector) 
    {
        scale = 2 + UInt(opc);
    }
    else
    {
        scale = (opc & 2) ? 3 : 2;
        is_signed = (opc & 1) != 0;
        if (is_signed && memop == MemOp_STORE)
            return false; // UNDEFINED
    }
    
    if (!vector && wback && ((t == n) || (t2 == n)))
    {
        switch (ConstrainUnpredictable(Unpredictable_WBOVERLAP))
        {
            case Constraint_UNKNOWN:
                wb_unknown = true;  // writeback is UNKNOWN
                break;
                
            case Constraint_SUPPRESSWB:
                wback = false;      // writeback is suppressed
                break;
                
            case Constraint_NOP:
                memop = MemOp_NOP;  // do nothing
                wback = false;
                break;

            case Constraint_NONE:
                break;
        }
    }

    if (memop == MemOp_LOAD && t == t2)
    {
        switch (ConstrainUnpredictable(Unpredictable_LDPOVERLAP))
        {
            case Constraint_UNKNOWN:
                rt_unknown = true;  // result is UNKNOWN
                break;
                
            case Constraint_NOP:
                memop = MemOp_NOP;  // do nothing 
                wback = false;
                break;
                
            default:
                break;
        }
    }
    
    idx = LSL(llvm::SignExtend64<7>(imm7), scale);
    size = (integer)1 << scale;
    uint64_t datasize = size * 8;
    uint64_t address;
    uint64_t wb_address;
    
    RegisterValue data_Rt;
    RegisterValue data_Rt2;
    
    //    if (vector)
    //        CheckFPEnabled(false);
    
    RegisterInfo reg_info_base;
    RegisterInfo reg_info_Rt;
    RegisterInfo reg_info_Rt2;
    if (!GetRegisterInfo (eRegisterKindDWARF, arm64_dwarf::x0 + n, reg_info_base))
        return false;
    
    if (vector)
    {
        if (!GetRegisterInfo (eRegisterKindDWARF, arm64_dwarf::v0 + n, reg_info_Rt))
            return false;
        if (!GetRegisterInfo (eRegisterKindDWARF, arm64_dwarf::v0 + n, reg_info_Rt2))
            return false;
    }
    else
    {
        if (!GetRegisterInfo (eRegisterKindDWARF, arm64_dwarf::x0 + t, reg_info_Rt))
            return false;
        if (!GetRegisterInfo (eRegisterKindDWARF, arm64_dwarf::x0 + t2, reg_info_Rt2))
            return false;
    }
    
    bool success = false;
    if (n == 31)
    {
        //CheckSPAlignment();
        address = ReadRegisterUnsigned (eRegisterKindDWARF, arm64_dwarf::sp, 0, &success);
    }
    else
        address = ReadRegisterUnsigned (eRegisterKindDWARF, arm64_dwarf::x0 + n, 0, &success);
    
    wb_address = address + idx;
    if (a_mode != AddrMode_POST)
        address = wb_address;
    
    Context context_t;
    Context context_t2;
    
    if (n == 31 || n == 29) // if this store is based off of the sp or fp register
    {
        context_t.type = eContextPushRegisterOnStack;
        context_t2.type = eContextPushRegisterOnStack;
    }
    else
    {
        context_t.type = eContextRegisterPlusOffset;
        context_t2.type = eContextRegisterPlusOffset;
    }
    context_t.SetRegisterToRegisterPlusOffset (reg_info_Rt, reg_info_base, 0);
    context_t2.SetRegisterToRegisterPlusOffset (reg_info_Rt2, reg_info_base, size);
    uint8_t buffer [RegisterValue::kMaxRegisterByteSize];
    Error error;
    
    switch (memop)
    {
        case MemOp_STORE:
        {
            if (!ReadRegister (&reg_info_Rt, data_Rt))
                return false;
            
            if (data_Rt.GetAsMemoryData(&reg_info_Rt, buffer, reg_info_Rt.byte_size, eByteOrderLittle, error) == 0)
                return false;
            
            if (!WriteMemory(context_t, address + 0, buffer, reg_info_Rt.byte_size))
                return false;
            
            if (!ReadRegister (&reg_info_Rt2, data_Rt2))
                return false;
            
            if (data_Rt2.GetAsMemoryData(&reg_info_Rt2, buffer, reg_info_Rt2.byte_size, eByteOrderLittle, error) == 0)
                return false;
            
            if (!WriteMemory(context_t2, address + size, buffer, reg_info_Rt2.byte_size))
                return false;
        }
            break;
            
        case MemOp_LOAD:
        {
            if (rt_unknown)
                memset (buffer, 'U', reg_info_Rt.byte_size);
            else
            {
                if (!ReadMemory (context_t, address, buffer, reg_info_Rt.byte_size))
                    return false;
            }
            
            if (data_Rt.SetFromMemoryData(&reg_info_Rt, buffer, reg_info_Rt.byte_size, eByteOrderLittle, error) == 0)
                return false;
            
            if (!vector && is_signed && !data_Rt.SignExtend (datasize))
                return false;
            
            if (!WriteRegister (context_t, &reg_info_Rt, data_Rt))
                return false;
            
            if (!rt_unknown)
            {
                if (!ReadMemory (context_t2, address + size, buffer, reg_info_Rt2.byte_size))
                    return false;
            }
            
            if (data_Rt2.SetFromMemoryData(&reg_info_Rt2, buffer, reg_info_Rt2.byte_size, eByteOrderLittle, error) == 0)
                return false;
            
            if (!vector && is_signed && !data_Rt2.SignExtend (datasize))
                return false;
            
            if (!WriteRegister (context_t2, &reg_info_Rt2, data_Rt2))
                return false;
        }
            break;
            
        default:
            break;
    }
    
    if (wback)
    {
        if (wb_unknown)
            wb_address = LLDB_INVALID_ADDRESS;
        Context context;
        context.SetImmediateSigned (idx);
        if (n == 31)
            context.type = eContextAdjustStackPointer;
        else
            context.type = eContextAdjustBaseRegister;
        WriteRegisterUnsigned (context, &reg_info_base, wb_address);
    }    
    return true;
}

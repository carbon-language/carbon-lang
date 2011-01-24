//===-- EmulateInstructionARM.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EmulateInstructionARM.h"
#include "ARMUtils.h"

using namespace lldb;
using namespace lldb_private;

// ARM constants used during decoding
#define REG_RD          0
#define LDM_REGLIST     1
#define PC_REG          15
#define PC_REGLIST_BIT  0x8000

#define ARMv4     (1u << 0)
#define ARMv4T    (1u << 1)
#define ARMv5T    (1u << 2)
#define ARMv5TE   (1u << 3)
#define ARMv5TEJ  (1u << 4)
#define ARMv6     (1u << 5)
#define ARMv6K    (1u << 6)
#define ARMv6T2   (1u << 7)
#define ARMv7     (1u << 8)
#define ARMv8     (1u << 8)
#define ARMvAll   (0xffffffffu)

typedef enum
{
    eEncodingA1,
    eEncodingA2,
    eEncodingA3,
    eEncodingA4,
    eEncodingA5,
    eEncodingT1,
    eEncodingT2,
    eEncodingT3,
    eEncodingT4,
    eEncodingT5,
} ARMEncoding;

typedef enum
{
    eSize16,
    eSize32
} ARMInstrSize;

// Typedef for the callback function used during the emulation.
// Pass along (ARMEncoding)encoding as the callback data.
typedef bool (*EmulateCallback) (EmulateInstructionARM *emulator, ARMEncoding encoding);
    
typedef struct
{
    uint32_t mask;
    uint32_t value;
    uint32_t variants;
    ARMEncoding encoding;
    ARMInstrSize size;
    EmulateCallback callback;
    const char *name;
}  ARMOpcode;

static bool 
EmulateARMPushEncoding (EmulateInstructionARM *emulator, ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations(); 
        NullCheckIfThumbEE(13); 
        address = SP - 4*BitCount(registers);

        for (i = 0 to 14)
        {
            if (registers<i> == ’1’)
            {
                if i == 13 && i != LowestSetBit(registers) // Only possible for encoding A1 
                    MemA[address,4] = bits(32) UNKNOWN;
                else 
                    MemA[address,4] = R[i];
                address = address + 4;
            }
        }

        if (registers<15> == ’1’) // Only possible for encoding A1 or A2 
            MemA[address,4] = PCStoreValue();
        
        SP = SP - 4*BitCount(registers);
    }
#endif

    bool success = false;
    const uint32_t opcode = emulator->OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (emulator->ConditionPassed())
    {
        const uint32_t addr_byte_size = emulator->GetAddressByteSize();
        const addr_t sp = emulator->ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t registers = 0;
        uint32_t t; // UInt(Rt)
        switch (encoding) {
        case eEncodingT2:
            // Ignore bits 15 & 13.
            registers = EmulateInstruction::UnsignedBits (opcode, 15, 0) & ~0xa000;
            // if BitCount(registers) < 2 then UNPREDICTABLE;
            if (BitCount(registers) < 2)
                return false;
            break;
        case eEncodingT3:
            t = EmulateInstruction::UnsignedBits (opcode, 15, 12);
            // if BadReg(t) then UNPREDICTABLE;
            if (BadReg(t))
                return false;
            registers = (1u << t);
            break;
        case eEncodingA1:
            registers = EmulateInstruction::UnsignedBits (opcode, 15, 0);
            break;
        case eEncodingA2:
            t = EmulateInstruction::UnsignedBits (opcode, 15, 12);
            // if t == 13 then UNPREDICTABLE;
            if (t == dwarf_sp)
                return false;
            registers = (1u << t);
            break;
        }
        addr_t sp_offset = addr_byte_size * EmulateInstruction::BitCount (registers);
        addr_t addr = sp - sp_offset;
        uint32_t i;
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextPushRegisterOnStack, eRegisterKindDWARF, 0, 0 };
        for (i=0; i<15; ++i)
        {
            if (EmulateInstruction::BitIsSet (registers, 1u << i))
            {
                context.arg1 = dwarf_r0 + i;    // arg1 in the context is the DWARF register number
                context.arg2 = addr - sp;       // arg2 in the context is the stack pointer offset
                uint32_t reg_value = emulator->ReadRegisterUnsigned(eRegisterKindDWARF, context.arg1, 0, &success);
                if (!success)
                    return false;
                if (!emulator->WriteMemoryUnsigned (context, addr, reg_value, addr_byte_size))
                    return false;
                addr += addr_byte_size;
            }
        }
        
        if (EmulateInstruction::BitIsSet (registers, 1u << 15))
        {
            context.arg1 = dwarf_pc;    // arg1 in the context is the DWARF register number
            context.arg2 = addr - sp;   // arg2 in the context is the stack pointer offset
            const uint32_t pc = emulator->ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
            if (!success)
                return false;
            if (!emulator->WriteMemoryUnsigned (context, addr, pc + 8, addr_byte_size))
                return false;
        }
        
        context.type = EmulateInstruction::eContextAdjustStackPointer;
        context.arg0 = eRegisterKindGeneric;
        context.arg1 = LLDB_REGNUM_GENERIC_SP;
        context.arg2 = sp_offset;
    
        if (!emulator->WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, sp - sp_offset))
            return false;
    }
    return true;
}

static ARMOpcode g_arm_opcodes[] =
{
    { 0xffff0000, 0xe8ad0000, ARMv6T2|ARMv7, eEncodingT2, eSize32, EmulateARMPushEncoding,
      "PUSH<c>.W <registers> ; <registers> contains more than one register" },
    { 0xffff0fff, 0xf84d0d04, ARMv6T2|ARMv7, eEncodingT3, eSize32, EmulateARMPushEncoding,
      "PUSH<c>.W <registers> ; <registers> contains one register, <Rt>" },
    { 0x0fff0000, 0x092d0000, ARMvAll,       eEncodingA1, eSize32, EmulateARMPushEncoding,
      "PUSH<c> <registers> ; <registers> contains more than one register" },
    { 0x0fff0fff, 0x052d0004, ARMvAll,       eEncodingA2, eSize32, EmulateARMPushEncoding,
      "PUSH<c> <registers> ; <registers> contains one register, <Rt>" }
};

static const size_t k_num_arm_opcodes = sizeof(g_arm_opcodes)/sizeof(ARMOpcode);

bool 
EmulateInstructionARM::ReadInstruction ()
{
    bool success = false;
    m_inst_cpsr = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS, 0, &success);
    if (success)
    {
        addr_t pc = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_ADDRESS, &success);
        if (success)
        {
            Context read_inst_context = {eContextReadOpcode, 0, 0};
            if (m_inst_cpsr & MASK_CPSR_T)
            {
                m_inst_mode = eModeThumb;
                uint32_t thumb_opcode = ReadMemoryUnsigned(read_inst_context, pc, 2, 0, &success);
                
                if (success)
                {
                    if ((m_inst.opcode.inst16 & 0xe000) != 0xe000 || ((m_inst.opcode.inst16 & 0x1800u) == 0))
                    {
                        m_inst.opcode_type = eOpcode16;
                        m_inst.opcode.inst16 = thumb_opcode;
                    }
                    else
                    {
                        m_inst.opcode_type = eOpcode32;
                        m_inst.opcode.inst32 = (thumb_opcode << 16) | ReadMemoryUnsigned(read_inst_context, pc + 2, 2, 0, &success);
                    }
                }
            }
            else
            {
                m_inst_mode = eModeARM;
                m_inst.opcode_type = eOpcode32;
                m_inst.opcode.inst32 = ReadMemoryUnsigned(read_inst_context, pc, 4, 0, &success);
            }
        }
    }
    if (!success)
    {
        m_inst_mode = eModeInvalid;
        m_inst_pc = LLDB_INVALID_ADDRESS;
    }
    return success;
}

uint32_t
EmulateInstructionARM::CurrentCond ()
{
    switch (m_inst_mode)
    {
    default:
    case eModeInvalid:
        break;

    case eModeARM:
        return UnsignedBits(m_inst.opcode.inst32, 31, 28);
    
    case eModeThumb:
        return 0x0000000Eu; // Return always for now, we need to handl IT instructions later
    }
    return UINT32_MAX;  // Return invalid value
}
bool
EmulateInstructionARM::ConditionPassed ()
{
    if (m_inst_cpsr == 0)
        return false;

    const uint32_t cond = CurrentCond ();
    
    if (cond == UINT32_MAX)
        return false;
    
    bool result = false;
    switch (UnsignedBits(cond, 3, 1))
    {
    case 0: result = (m_inst_cpsr & MASK_CPSR_Z) != 0; break;
    case 1: result = (m_inst_cpsr & MASK_CPSR_C) != 0; break;
    case 2: result = (m_inst_cpsr & MASK_CPSR_N) != 0; break;
    case 3: result = (m_inst_cpsr & MASK_CPSR_V) != 0; break;
    case 4: result = ((m_inst_cpsr & MASK_CPSR_C) != 0) && ((m_inst_cpsr & MASK_CPSR_Z) == 0); break;
    case 5: 
        {
            bool n = (m_inst_cpsr & MASK_CPSR_N);
            bool v = (m_inst_cpsr & MASK_CPSR_V);
            result = n == v;
        }
        break;
    case 6: 
        {
            bool n = (m_inst_cpsr & MASK_CPSR_N);
            bool v = (m_inst_cpsr & MASK_CPSR_V);
            result = n == v && ((m_inst_cpsr & MASK_CPSR_Z) == 0);
        }
        break;
    case 7: 
        result = true; 
        break;
    }

    if (cond & 1)
        result = !result;
    return result;
}


bool
EmulateInstructionARM::EvaluateInstruction ()
{
    return false;
}

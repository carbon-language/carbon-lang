//===-- EmulateInstructionARM.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EmulateInstructionARM.h"
#include "lldb/Core/ConstString.h"

#include "Plugins/Process/Utility/ARMDefines.h"
#include "Plugins/Process/Utility/ARMUtils.h"
#include "Utility/ARM_DWARF_Registers.h"

#include "llvm/Support/MathExtras.h" // for SignExtend32 template function
                                     // and CountTrailingZeros_32 function

using namespace lldb;
using namespace lldb_private;

static inline uint32_t Align(uint32_t val, uint32_t alignment)
{
    return alignment * (val / alignment);
}

//----------------------------------------------------------------------
//
// ITSession implementation
//
//----------------------------------------------------------------------

// A8.6.50
// Valid return values are {1, 2, 3, 4}, with 0 signifying an error condition.
static unsigned short CountITSize(unsigned ITMask) {
    // First count the trailing zeros of the IT mask.
    unsigned TZ = llvm::CountTrailingZeros_32(ITMask);
    if (TZ > 3)
    {
        printf("Encoding error: IT Mask '0000'\n");
        return 0;
    }
    return (4 - TZ);
}

// Init ITState.  Note that at least one bit is always 1 in mask.
bool ITSession::InitIT(unsigned short bits7_0)
{
    ITCounter = CountITSize(Bits32(bits7_0, 3, 0));
    if (ITCounter == 0)
        return false;

    // A8.6.50 IT
    unsigned short FirstCond = Bits32(bits7_0, 7, 4);
    if (FirstCond == 0xF)
    {
        printf("Encoding error: IT FirstCond '1111'\n");
        return false;
    }
    if (FirstCond == 0xE && ITCounter != 1)
    {
        printf("Encoding error: IT FirstCond '1110' && Mask != '1000'\n");
        return false;
    }

    ITState = bits7_0;
    return true;
}

// Update ITState if necessary.
void ITSession::ITAdvance()
{
    assert(ITCounter);
    --ITCounter;
    if (ITCounter == 0)
        ITState = 0;
    else
    {
        unsigned short NewITState4_0 = Bits32(ITState, 4, 0) << 1;
        SetBits32(ITState, 4, 0, NewITState4_0);
    }
}

// Return true if we're inside an IT Block.
bool ITSession::InITBlock()
{
    return ITCounter != 0;
}

// Return true if we're the last instruction inside an IT Block.
bool ITSession::LastInITBlock()
{
    return ITCounter == 1;
}

// Get condition bits for the current thumb instruction.
uint32_t ITSession::GetCond()
{
    if (InITBlock())
        return Bits32(ITState, 7, 4);
    else
        return COND_AL;
}

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
#define ARMv8     (1u << 9)
#define ARMvAll   (0xffffffffu)

#define ARMV4T_ABOVE  (ARMv4T|ARMv5T|ARMv5TE|ARMv5TEJ|ARMv6|ARMv6K|ARMv6T2|ARMv7|ARMv8)
#define ARMV5_ABOVE   (ARMv5T|ARMv5TE|ARMv5TEJ|ARMv6|ARMv6K|ARMv6T2|ARMv7|ARMv8)
#define ARMV6T2_ABOVE (ARMv6T2|ARMv7|ARMv8)

//----------------------------------------------------------------------
//
// EmulateInstructionARM implementation
//
//----------------------------------------------------------------------

void
EmulateInstructionARM::Initialize ()
{
}

void
EmulateInstructionARM::Terminate ()
{
}

// Write "bits (32) UNKNOWN" to register n.  Helper function for many ARM instructions.
bool
EmulateInstructionARM::WriteBits32Unknown (int n)
{
    EmulateInstruction::Context context = { EmulateInstruction::eContextWriteRegisterRandomBits, 
                                            eRegisterKindDWARF, 
                                            dwarf_r0 + n, 
                                            0 };

    bool success;
    uint32_t data = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + n, 0, &success);
                  
    if (!success)
        return false;
   
    if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + n, data))
        return false;
    
    return true;
}

// Push Multiple Registers stores multiple registers to the stack, storing to
// consecutive memory locations ending just below the address in SP, and updates
// SP to point to the start of the stored data.
bool 
EmulateInstructionARM::EmulatePush (ARMEncoding encoding)
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
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const uint32_t addr_byte_size = GetAddressByteSize();
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t registers = 0;
        uint32_t Rt; // the source register
        switch (encoding) {
        case eEncodingT1:
            registers = Bits32(opcode, 7, 0);
            // The M bit represents LR.
            if (Bit32(opcode, 8))
                registers |= (1u << 14);
            // if BitCount(registers) < 1 then UNPREDICTABLE;
            if (BitCount(registers) < 1)
                return false;
            break;
        case eEncodingT2:
            // Ignore bits 15 & 13.
            registers = Bits32(opcode, 15, 0) & ~0xa000;
            // if BitCount(registers) < 2 then UNPREDICTABLE;
            if (BitCount(registers) < 2)
                return false;
            break;
        case eEncodingT3:
            Rt = Bits32(opcode, 15, 12);
            // if BadReg(t) then UNPREDICTABLE;
            if (BadReg(Rt))
                return false;
            registers = (1u << Rt);
            break;
        case eEncodingA1:
            registers = Bits32(opcode, 15, 0);
            // Instead of return false, let's handle the following case as well,
            // which amounts to pushing one reg onto the full descending stacks.
            // if BitCount(register_list) < 2 then SEE STMDB / STMFD;
            break;
        case eEncodingA2:
            Rt = Bits32(opcode, 15, 12);
            // if t == 13 then UNPREDICTABLE;
            if (Rt == dwarf_sp)
                return false;
            registers = (1u << Rt);
            break;
        default:
            return false;
        }
        addr_t sp_offset = addr_byte_size * BitCount (registers);
        addr_t addr = sp - sp_offset;
        uint32_t i;
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextPushRegisterOnStack, eRegisterKindDWARF, 0, 0 };
        for (i=0; i<15; ++i)
        {
            if (BitIsSet (registers, i))
            {
                context.arg1 = dwarf_r0 + i;    // arg1 in the context is the DWARF register number
                context.arg2 = addr - sp;       // arg2 in the context is the stack pointer offset
                uint32_t reg_value = ReadRegisterUnsigned(eRegisterKindDWARF, context.arg1, 0, &success);
                if (!success)
                    return false;
                if (!WriteMemoryUnsigned (context, addr, reg_value, addr_byte_size))
                    return false;
                addr += addr_byte_size;
            }
        }
        
        if (BitIsSet (registers, 15))
        {
            context.arg1 = dwarf_pc;    // arg1 in the context is the DWARF register number
            context.arg2 = addr - sp;   // arg2 in the context is the stack pointer offset
            const uint32_t pc = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
            if (!success)
                return false;
            if (!WriteMemoryUnsigned (context, addr, pc + 8, addr_byte_size))
                return false;
        }
        
        context.type = EmulateInstruction::eContextAdjustStackPointer;
        context.arg0 = eRegisterKindGeneric;
        context.arg1 = LLDB_REGNUM_GENERIC_SP;
        context.arg2 = -sp_offset;
    
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, sp - sp_offset))
            return false;
    }
    return true;
}

// Pop Multiple Registers loads multiple registers from the stack, loading from
// consecutive memory locations staring at the address in SP, and updates
// SP to point just above the loaded data.
bool 
EmulateInstructionARM::EmulatePop (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations(); NullCheckIfThumbEE(13);
        address = SP;
        for i = 0 to 14
            if registers<i> == ‘1’ then
                R[i} = if UnalignedAllowed then MemU[address,4] else MemA[address,4]; address = address + 4;
        if registers<15> == ‘1’ then
            if UnalignedAllowed then
                LoadWritePC(MemU[address,4]);
            else 
                LoadWritePC(MemA[address,4]);
        if registers<13> == ‘0’ then SP = SP + 4*BitCount(registers);
        if registers<13> == ‘1’ then SP = bits(32) UNKNOWN;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const uint32_t addr_byte_size = GetAddressByteSize();
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t registers = 0;
        uint32_t Rt; // the destination register
        switch (encoding) {
        case eEncodingT1:
            registers = Bits32(opcode, 7, 0);
            // The P bit represents PC.
            if (Bit32(opcode, 8))
                registers |= (1u << 15);
            // if BitCount(registers) < 1 then UNPREDICTABLE;
            if (BitCount(registers) < 1)
                return false;
            break;
        case eEncodingT2:
            // Ignore bit 13.
            registers = Bits32(opcode, 15, 0) & ~0x2000;
            // if BitCount(registers) < 2 || (P == '1' && M == '1') then UNPREDICTABLE;
            if (BitCount(registers) < 2 || (Bit32(opcode, 15) && Bit32(opcode, 14)))
                return false;
            break;
        case eEncodingT3:
            Rt = Bits32(opcode, 15, 12);
            // if t == 13 || (t == 15 && InITBlock() && !LastInITBlock()) then UNPREDICTABLE;
            if (Rt == dwarf_sp)
                return false;
            registers = (1u << Rt);
            break;
        case eEncodingA1:
            registers = Bits32(opcode, 15, 0);
            // Instead of return false, let's handle the following case as well,
            // which amounts to popping one reg from the full descending stacks.
            // if BitCount(register_list) < 2 then SEE LDM / LDMIA / LDMFD;

            // if registers<13> == ‘1’ && ArchVersion() >= 7 then UNPREDICTABLE;
            if (Bit32(opcode, 13) && ArchVersion() >= ARMv7)
                return false;
            break;
        case eEncodingA2:
            Rt = Bits32(opcode, 15, 12);
            // if t == 13 then UNPREDICTABLE;
            if (Rt == dwarf_sp)
                return false;
            registers = (1u << Rt);
            break;
        default:
            return false;
        }
        addr_t sp_offset = addr_byte_size * BitCount (registers);
        addr_t addr = sp;
        uint32_t i, data;
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextPopRegisterOffStack, eRegisterKindDWARF, 0, 0 };
        for (i=0; i<15; ++i)
        {
            if (BitIsSet (registers, i))
            {
                context.arg1 = dwarf_r0 + i;    // arg1 in the context is the DWARF register number
                context.arg2 = addr - sp;       // arg2 in the context is the stack pointer offset
                data = ReadMemoryUnsigned(context, addr, 4, 0, &success);
                if (!success)
                    return false;    
                if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, context.arg1, data))
                    return false;
                addr += addr_byte_size;
            }
        }
        
        if (BitIsSet (registers, 15))
        {
            context.arg1 = dwarf_pc;    // arg1 in the context is the DWARF register number
            context.arg2 = addr - sp;   // arg2 in the context is the stack pointer offset
            data = ReadMemoryUnsigned(context, addr, 4, 0, &success);
            if (!success)
                return false;
            // In ARMv5T and above, this is an interworking branch.
            if (!LoadWritePC(context, data))
                return false;
            addr += addr_byte_size;
        }
        
        context.type = EmulateInstruction::eContextAdjustStackPointer;
        context.arg0 = eRegisterKindGeneric;
        context.arg1 = LLDB_REGNUM_GENERIC_SP;
        context.arg2 = sp_offset;
    
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, sp + sp_offset))
            return false;
    }
    return true;
}

// Set r7 or ip to point to saved value residing within the stack.
// ADD (SP plus immediate)
bool
EmulateInstructionARM::EmulateAddRdSPImmediate (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        (result, carry, overflow) = AddWithCarry(SP, imm32, ‘0’);
        if d == 15 then
           ALUWritePC(result); // setflags is always FALSE here
        else
            R[d] = result;
            if setflags then
                APSR.N = result<31>;
                APSR.Z = IsZeroBit(result);
                APSR.C = carry;
                APSR.V = overflow;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t Rd; // the destination register
        uint32_t imm32;
        switch (encoding) {
        case eEncodingT1:
            Rd = 7;
            imm32 = Bits32(opcode, 7, 0) << 2; // imm32 = ZeroExtend(imm8:'00', 32)
            break;
        case eEncodingA1:
            Rd = Bits32(opcode, 15, 12);
            imm32 = ARMExpandImm(opcode); // imm32 = ARMExpandImm(imm12)
            break;
        default:
            return false;
        }
        addr_t sp_offset = imm32;
        addr_t addr = sp + sp_offset; // a pointer to the stack area
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindGeneric,
                                                LLDB_REGNUM_GENERIC_SP,
                                                sp_offset };
    
        if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + Rd, addr))
            return false;
    }
    return true;
}

// Set r7 or ip to the current stack pointer.
// MOV (register)
bool
EmulateInstructionARM::EmulateMovRdSP (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        result = R[m];
        if d == 15 then
            ALUWritePC(result); // setflags is always FALSE here
        else
            R[d] = result;
            if setflags then
                APSR.N = result<31>;
                APSR.Z = IsZeroBit(result);
                // APSR.C unchanged
                // APSR.V unchanged
    }
#endif

    bool success = false;
    //const uint32_t opcode = OpcodeAsUnsigned (&success);
    //if (!success)
    //    return false;

    if (ConditionPassed())
    {
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t Rd; // the destination register
        switch (encoding) {
        case eEncodingT1:
            Rd = 7;
            break;
        case eEncodingA1:
            Rd = 12;
            break;
        default:
            return false;
        }
        EmulateInstruction::Context context = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindGeneric,
                                                LLDB_REGNUM_GENERIC_SP,
                                                0 };
    
        if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + Rd, sp))
            return false;
    }
    return true;
}

// Move from high register (r8-r15) to low register (r0-r7).
// MOV (register)
bool
EmulateInstructionARM::EmulateMovLowHigh (ARMEncoding encoding)
{
    return EmulateMovRdRm (encoding);
}

// Move from register to register.
// MOV (register)
bool
EmulateInstructionARM::EmulateMovRdRm (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        result = R[m];
        if d == 15 then
            ALUWritePC(result); // setflags is always FALSE here
        else
            R[d] = result;
            if setflags then
                APSR.N = result<31>;
                APSR.Z = IsZeroBit(result);
                // APSR.C unchanged
                // APSR.V unchanged
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        uint32_t Rm; // the source register
        uint32_t Rd; // the destination register
        bool setflags;
        switch (encoding) {
        case eEncodingT1:
            Rm = Bits32(opcode, 6, 3);
            Rd = Bit32(opcode, 7) << 3 | Bits32(opcode, 2, 1);
            setflags = false;
            break;
        case eEncodingT2:
            Rm = Bits32(opcode, 5, 3);
            Rd = Bits32(opcode, 2, 1);
            setflags = true;
            break;
        default:
            return false;
        }
        uint32_t reg_value = ReadRegisterUnsigned(eRegisterKindDWARF, dwarf_r0 + Rm, 0, &success);
        if (!success)
            return false;
        
        // The context specifies that Rm is to be moved into Rd.
        EmulateInstruction::Context context = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindDWARF,
                                                dwarf_r0 + Rm,
                                                0 };
    
        if (Rd == 15)
        {
            if (!ALUWritePC (context, reg_value))
                return false;
        }
        else
        {
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + Rd, reg_value))
                return false;
            if (setflags)
            {
                m_new_inst_cpsr = m_inst_cpsr;
                SetBit32(m_new_inst_cpsr, CPSR_N, Bit32(reg_value, CPSR_N));
                SetBit32(m_new_inst_cpsr, CPSR_Z, reg_value == 0 ? 1 : 0);
                if (m_new_inst_cpsr != m_inst_cpsr)
                {
                    if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS, m_new_inst_cpsr))
                        return false;
                }
            }
        }
    }
    return true;
}

// PC relative immediate load into register, possibly followed by ADD (SP plus register).
// LDR (literal)
bool
EmulateInstructionARM::EmulateLDRRtPCRelative (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations(); NullCheckIfThumbEE(15);
        base = Align(PC,4);
        address = if add then (base + imm32) else (base - imm32);
        data = MemU[address,4];
        if t == 15 then
            if address<1:0> == ‘00’ then LoadWritePC(data); else UNPREDICTABLE;
        elsif UnalignedSupport() || address<1:0> = ‘00’ then
            R[t] = data;
        else // Can only apply before ARMv7
            if CurrentInstrSet() == InstrSet_ARM then
                R[t] = ROR(data, 8*UInt(address<1:0>));
            else
                R[t] = bits(32) UNKNOWN;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const uint32_t pc = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
        if (!success)
            return false;

        // PC relative immediate load context
        EmulateInstruction::Context context = {EmulateInstruction::eContextRegisterPlusOffset,
                                               eRegisterKindGeneric,
                                               LLDB_REGNUM_GENERIC_PC,
                                               0};
        uint32_t Rt;    // the destination register
        uint32_t imm32; // immediate offset from the PC
        bool add;       // +imm32 or -imm32?
        addr_t base;    // the base address
        addr_t address; // the PC relative address
        uint32_t data;  // the literal data value from the PC relative load
        switch (encoding) {
        case eEncodingT1:
            Rt = Bits32(opcode, 10, 8);
            imm32 = Bits32(opcode, 7, 0) << 2; // imm32 = ZeroExtend(imm8:'00', 32);
            add = true;
            base = Align(pc + 4, 4);
            context.arg2 = 4 + imm32;
            break;
        case eEncodingT2:
            Rt = Bits32(opcode, 15, 12);
            imm32 = Bits32(opcode, 11, 0) << 2; // imm32 = ZeroExtend(imm12, 32);
            add = BitIsSet(opcode, 23);
            if (Rt == 15
                && m_it_session.InITBlock()
                && !m_it_session.LastInITBlock())
                return false;
            base = Align(pc + 4, 4);
            context.arg2 = 4 + imm32;
            break;
        default:
            return false;
        }

        if (add)
            address = base + imm32;
        else
            address = base - imm32;
        data = ReadMemoryUnsigned(context, address, 4, 0, &success);
        if (!success)
            return false;    

        if (Rt == 15)
        {
            if (Bits32(address, 1, 0) == 0)
            {
                // In ARMv5T and above, this is an interworking branch.
                if (!LoadWritePC(context, data))
                    return false;
            }
            else
                return false;
        }
        else if (UnalignedSupport() || Bits32(address, 1, 0) == 0)
        {
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + Rt, data))
                return false;
        }
        else // We don't handle ARM for now.
            return false;

        if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + Rt, data))
            return false;
    }
    return true;
}

// An add operation to adjust the SP.
// ADD (SP plus immediate)
bool
EmulateInstructionARM::EmulateAddSPImmediate (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        (result, carry, overflow) = AddWithCarry(SP, imm32, ‘0’);
        if d == 15 then // Can only occur for ARM encoding
            ALUWritePC(result); // setflags is always FALSE here
        else
            R[d] = result;
            if setflags then
                APSR.N = result<31>;
                APSR.Z = IsZeroBit(result);
                APSR.C = carry;
                APSR.V = overflow;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t imm32; // the immediate operand
        switch (encoding) {
        case eEncodingT2:
            imm32 = ThumbImmScaled(opcode); // imm32 = ZeroExtend(imm7:'00', 32)
            break;
        default:
            return false;
        }
        addr_t sp_offset = imm32;
        addr_t addr = sp + sp_offset; // the adjusted stack pointer value
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextAdjustStackPointer,
                                                eRegisterKindGeneric,
                                                LLDB_REGNUM_GENERIC_SP,
                                                sp_offset };
    
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, addr))
            return false;
    }
    return true;
}

// An add operation to adjust the SP.
// ADD (SP plus register)
bool
EmulateInstructionARM::EmulateAddSPRm (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        shifted = Shift(R[m], shift_t, shift_n, APSR.C);
        (result, carry, overflow) = AddWithCarry(SP, shifted, ‘0’);
        if d == 15 then
            ALUWritePC(result); // setflags is always FALSE here
        else
            R[d] = result;
            if setflags then
                APSR.N = result<31>;
                APSR.Z = IsZeroBit(result);
                APSR.C = carry;
                APSR.V = overflow;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t Rm; // the second operand
        switch (encoding) {
        case eEncodingT2:
            Rm = Bits32(opcode, 6, 3);
            break;
        default:
            return false;
        }
        int32_t reg_value = ReadRegisterUnsigned(eRegisterKindDWARF, dwarf_r0 + Rm, 0, &success);
        if (!success)
            return false;

        addr_t addr = (int32_t)sp + reg_value; // the adjusted stack pointer value
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextAdjustStackPointer,
                                                eRegisterKindGeneric,
                                                LLDB_REGNUM_GENERIC_SP,
                                                reg_value };
    
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, addr))
            return false;
    }
    return true;
}

// Branch with Link and Exchange Instruction Sets (immediate) calls a subroutine
// at a PC-relative address, and changes instruction set from ARM to Thumb, or
// from Thumb to ARM.
// BLX (immediate)
bool
EmulateInstructionARM::EmulateBLXImmediate (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        if CurrentInstrSet() == InstrSet_ARM then
            LR = PC - 4;
        else
            LR = PC<31:1> : '1';
        if targetInstrSet == InstrSet_ARM then
            targetAddress = Align(PC,4) + imm32;
        else
            targetAddress = PC + imm32;
        SelectInstrSet(targetInstrSet);
        BranchWritePC(targetAddress);
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        EmulateInstruction::Context context = { EmulateInstruction::eContextRelativeBranchImmediate, 0, 0, 0};
        const uint32_t pc = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
        if (!success)
            return false;
        addr_t lr; // next instruction address
        addr_t target; // target address
        int32_t imm32; // PC-relative offset
        switch (encoding) {
        case eEncodingT1:
            {
            lr = (pc + 4) | 1u; // return address
            uint32_t S = Bit32(opcode, 26);
            uint32_t imm10 = Bits32(opcode, 25, 16);
            uint32_t J1 = Bit32(opcode, 13);
            uint32_t J2 = Bit32(opcode, 11);
            uint32_t imm11 = Bits32(opcode, 10, 0);
            uint32_t I1 = !(J1 ^ S);
            uint32_t I2 = !(J2 ^ S);
            uint32_t imm25 = (S << 24) | (I1 << 23) | (I2 << 22) | (imm10 << 12) | (imm11 << 1);
            imm32 = llvm::SignExtend32<25>(imm25);
            target = pc + 4 + imm32;
            context.arg1 = 4 + imm32;  // signed offset
            context.arg2 = eModeThumb; // target instruction set
            break;
            }
        case eEncodingT2:
            {
            lr = (pc + 4) | 1u; // return address
            uint32_t S = Bit32(opcode, 26);
            uint32_t imm10H = Bits32(opcode, 25, 16);
            uint32_t J1 = Bit32(opcode, 13);
            uint32_t J2 = Bit32(opcode, 11);
            uint32_t imm10L = Bits32(opcode, 10, 1);
            uint32_t I1 = !(J1 ^ S);
            uint32_t I2 = !(J2 ^ S);
            uint32_t imm25 = (S << 24) | (I1 << 23) | (I2 << 22) | (imm10H << 12) | (imm10L << 2);
            imm32 = llvm::SignExtend32<25>(imm25);
            target = Align(pc + 4, 4) + imm32;
            context.arg1 = 4 + imm32; // signed offset
            context.arg2 = eModeARM;  // target instruction set
            break;
            }
        case eEncodingA1:
            lr = pc + 4; // return address
            imm32 = llvm::SignExtend32<26>(Bits32(opcode, 23, 0) << 2);
            target = Align(pc + 8, 4) + imm32;
            context.arg1 = 8 + imm32; // signed offset
            context.arg2 = eModeARM;  // target instruction set
            break;
        case eEncodingA2:
            lr = pc + 4; // return address
            imm32 = llvm::SignExtend32<26>(Bits32(opcode, 23, 0) << 2 | Bits32(opcode, 24, 24) << 1);
            target = pc + 8 + imm32;
            context.arg1 = 8 + imm32;  // signed offset
            context.arg2 = eModeThumb; // target instruction set
            break;
        default:
            return false;
        }
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA, lr))
            return false;
        if (!BranchWritePC(context, target))
            return false;
    }
    return true;
}

// Branch with Link and Exchange (register) calls a subroutine at an address and
// instruction set specified by a register.
// BLX (register)
bool
EmulateInstructionARM::EmulateBLXRm (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        target = R[m];
        if CurrentInstrSet() == InstrSet_ARM then
            next_instr_addr = PC - 4;
            LR = next_instr_addr;
        else
            next_instr_addr = PC - 2;
            LR = next_instr_addr<31:1> : ‘1’;
        BXWritePC(target);
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        EmulateInstruction::Context context = { EmulateInstruction::eContextAbsoluteBranchRegister, 0, 0, 0};
        const uint32_t pc = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
        addr_t lr; // next instruction address
        addr_t target; // target address
        if (!success)
            return false;
        uint32_t Rm; // the register with the target address
        switch (encoding) {
        case eEncodingT1:
            lr = (pc + 2) | 1u; // return address
            Rm = Bits32(opcode, 6, 3);
            // if m == 15 then UNPREDICTABLE;
            if (Rm == 15)
                return false;
            target = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + Rm, 0, &success);
            break;
        case eEncodingA1:
            lr = pc + 4; // return address
            Rm = Bits32(opcode, 3, 0);
            // if m == 15 then UNPREDICTABLE;
            if (Rm == 15)
                return false;
            target = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + Rm, 0, &success);
            break;
        default:
            return false;
        }
        context.arg0 = eRegisterKindDWARF;
        context.arg1 = dwarf_r0 + Rm;
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA, lr))
            return false;
        if (!BXWritePC(context, target))
            return false;
    }
    return true;
}

// Set r7 to point to some ip offset.
// SUB (immediate)
bool
EmulateInstructionARM::EmulateSubR7IPImmediate (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        (result, carry, overflow) = AddWithCarry(SP, NOT(imm32), ‘1’);
        if d == 15 then // Can only occur for ARM encoding
           ALUWritePC(result); // setflags is always FALSE here
        else
            R[d] = result;
            if setflags then
                APSR.N = result<31>;
                APSR.Z = IsZeroBit(result);
                APSR.C = carry;
                APSR.V = overflow;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const addr_t ip = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r12, 0, &success);
        if (!success)
            return false;
        uint32_t imm32;
        switch (encoding) {
        case eEncodingA1:
            imm32 = ARMExpandImm(opcode); // imm32 = ARMExpandImm(imm12)
            break;
        default:
            return false;
        }
        addr_t ip_offset = imm32;
        addr_t addr = ip - ip_offset; // the adjusted ip value
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindDWARF,
                                                dwarf_r12,
                                                -ip_offset };
    
        if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r7, addr))
            return false;
    }
    return true;
}

// Set ip to point to some stack offset.
// SUB (SP minus immediate)
bool
EmulateInstructionARM::EmulateSubIPSPImmediate (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        (result, carry, overflow) = AddWithCarry(SP, NOT(imm32), ‘1’);
        if d == 15 then // Can only occur for ARM encoding
           ALUWritePC(result); // setflags is always FALSE here
        else
            R[d] = result;
            if setflags then
                APSR.N = result<31>;
                APSR.Z = IsZeroBit(result);
                APSR.C = carry;
                APSR.V = overflow;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t imm32;
        switch (encoding) {
        case eEncodingA1:
            imm32 = ARMExpandImm(opcode); // imm32 = ARMExpandImm(imm12)
            break;
        default:
            return false;
        }
        addr_t sp_offset = imm32;
        addr_t addr = sp - sp_offset; // the adjusted stack pointer value
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindGeneric,
                                                LLDB_REGNUM_GENERIC_SP,
                                                -sp_offset };
    
        if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r12, addr))
            return false;
    }
    return true;
}

// A sub operation to adjust the SP -- allocate space for local storage.
bool
EmulateInstructionARM::EmulateSubSPImmdiate (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        (result, carry, overflow) = AddWithCarry(SP, NOT(imm32), ‘1’);
        if d == 15 then // Can only occur for ARM encoding
           ALUWritePC(result); // setflags is always FALSE here
        else
            R[d] = result;
            if setflags then
                APSR.N = result<31>;
                APSR.Z = IsZeroBit(result);
                APSR.C = carry;
                APSR.V = overflow;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t imm32;
        switch (encoding) {
        case eEncodingT1:
            imm32 = ThumbImmScaled(opcode); // imm32 = ZeroExtend(imm7:'00', 32)
        case eEncodingT2:
            imm32 = ThumbExpandImm(opcode); // imm32 = ThumbExpandImm(i:imm3:imm8)
            break;
        case eEncodingT3:
            imm32 = ThumbImm12(opcode); // imm32 = ZeroExtend(i:imm3:imm8, 32)
            break;
        case eEncodingA1:
            imm32 = ARMExpandImm(opcode); // imm32 = ARMExpandImm(imm12)
            break;
        default:
            return false;
        }
        addr_t sp_offset = imm32;
        addr_t addr = sp - sp_offset; // the adjusted stack pointer value
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextAdjustStackPointer,
                                                eRegisterKindGeneric,
                                                LLDB_REGNUM_GENERIC_SP,
                                                -sp_offset };
    
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, addr))
            return false;
    }
    return true;
}

// A store operation to the stack that also updates the SP.
bool
EmulateInstructionARM::EmulateSTRRtSP (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        offset_addr = if add then (R[n] + imm32) else (R[n] - imm32);
        address = if index then offset_addr else R[n];
        MemU[address,4] = if t == 15 then PCStoreValue() else R[t];
        if wback then R[n] = offset_addr;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const uint32_t addr_byte_size = GetAddressByteSize();
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        uint32_t Rt; // the source register
        uint32_t imm12;
        switch (encoding) {
        case eEncodingA1:
            Rt = Bits32(opcode, 15, 12);
            imm12 = Bits32(opcode, 11, 0);
            break;
        default:
            return false;
        }
        addr_t sp_offset = imm12;
        addr_t addr = sp - sp_offset;
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextPushRegisterOnStack, eRegisterKindDWARF, 0, 0 };
        if (Rt != 15)
        {
            context.arg1 = dwarf_r0 + Rt;    // arg1 in the context is the DWARF register number
            context.arg2 = addr - sp;        // arg2 in the context is the stack pointer offset
            uint32_t reg_value = ReadRegisterUnsigned(eRegisterKindDWARF, context.arg1, 0, &success);
            if (!success)
                return false;
            if (!WriteMemoryUnsigned (context, addr, reg_value, addr_byte_size))
                return false;
        }
        else
        {
            context.arg1 = dwarf_pc;    // arg1 in the context is the DWARF register number
            context.arg2 = addr - sp;   // arg2 in the context is the stack pointer offset
            const uint32_t pc = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
            if (!success)
                return false;
            if (!WriteMemoryUnsigned (context, addr, pc + 8, addr_byte_size))
                return false;
        }
        
        context.type = EmulateInstruction::eContextAdjustStackPointer;
        context.arg0 = eRegisterKindGeneric;
        context.arg1 = LLDB_REGNUM_GENERIC_SP;
        context.arg2 = -sp_offset;
    
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, sp - sp_offset))
            return false;
    }
    return true;
}

// Vector Push stores multiple extension registers to the stack.
// It also updates SP to point to the start of the stored data.
bool 
EmulateInstructionARM::EmulateVPUSH (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations(); CheckVFPEnabled(TRUE); NullCheckIfThumbEE(13);
        address = SP - imm32;
        SP = SP - imm32;
        if single_regs then
            for r = 0 to regs-1
                MemA[address,4] = S[d+r]; address = address+4;
        else
            for r = 0 to regs-1
                // Store as two word-aligned words in the correct order for current endianness.
                MemA[address,4] = if BigEndian() then D[d+r]<63:32> else D[d+r]<31:0>;
                MemA[address+4,4] = if BigEndian() then D[d+r]<31:0> else D[d+r]<63:32>;
                address = address+8;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const uint32_t addr_byte_size = GetAddressByteSize();
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        bool single_regs;
        uint32_t d;     // UInt(D:Vd) or UInt(Vd:D) starting register
        uint32_t imm32; // stack offset
        uint32_t regs;  // number of registers
        switch (encoding) {
        case eEncodingT1:
        case eEncodingA1:
            single_regs = false;
            d = Bit32(opcode, 22) << 4 | Bits32(opcode, 15, 12);
            imm32 = Bits32(opcode, 7, 0) * addr_byte_size;
            // If UInt(imm8) is odd, see "FSTMX".
            regs = Bits32(opcode, 7, 0) / 2;
            // if regs == 0 || regs > 16 || (d+regs) > 32 then UNPREDICTABLE;
            if (regs == 0 || regs > 16 || (d + regs) > 32)
                return false;
            break;
        case eEncodingT2:
        case eEncodingA2:
            single_regs = true;
            d = Bits32(opcode, 15, 12) << 1 | Bit32(opcode, 22);
            imm32 = Bits32(opcode, 7, 0) * addr_byte_size;
            regs = Bits32(opcode, 7, 0);
            // if regs == 0 || regs > 16 || (d+regs) > 32 then UNPREDICTABLE;
            if (regs == 0 || regs > 16 || (d + regs) > 32)
                return false;
            break;
        default:
            return false;
        }
        uint32_t start_reg = single_regs ? dwarf_s0 : dwarf_d0;
        uint32_t reg_byte_size = single_regs ? addr_byte_size : addr_byte_size * 2;
        addr_t sp_offset = imm32;
        addr_t addr = sp - sp_offset;
        uint32_t i;
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextPushRegisterOnStack, eRegisterKindDWARF, 0, 0 };
        for (i=d; i<regs; ++i)
        {
            context.arg1 = start_reg + i;    // arg1 in the context is the DWARF register number
            context.arg2 = addr - sp;        // arg2 in the context is the stack pointer offset
            // uint64_t to accommodate 64-bit registers.
            uint64_t reg_value = ReadRegisterUnsigned(eRegisterKindDWARF, context.arg1, 0, &success);
            if (!success)
                return false;
            if (!WriteMemoryUnsigned (context, addr, reg_value, reg_byte_size))
                return false;
            addr += reg_byte_size;
        }
        
        context.type = EmulateInstruction::eContextAdjustStackPointer;
        context.arg0 = eRegisterKindGeneric;
        context.arg1 = LLDB_REGNUM_GENERIC_SP;
        context.arg2 = -sp_offset;
    
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, sp - sp_offset))
            return false;
    }
    return true;
}

// Vector Pop loads multiple extension registers from the stack.
// It also updates SP to point just above the loaded data.
bool 
EmulateInstructionARM::EmulateVPOP (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations(); CheckVFPEnabled(TRUE); NullCheckIfThumbEE(13);
        address = SP;
        SP = SP + imm32;
        if single_regs then
            for r = 0 to regs-1
                S[d+r] = MemA[address,4]; address = address+4;
        else
            for r = 0 to regs-1
                word1 = MemA[address,4]; word2 = MemA[address+4,4]; address = address+8;
                // Combine the word-aligned words in the correct order for current endianness.
                D[d+r] = if BigEndian() then word1:word2 else word2:word1;
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const uint32_t addr_byte_size = GetAddressByteSize();
        const addr_t sp = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, 0, &success);
        if (!success)
            return false;
        bool single_regs;
        uint32_t d;     // UInt(D:Vd) or UInt(Vd:D) starting register
        uint32_t imm32; // stack offset
        uint32_t regs;  // number of registers
        switch (encoding) {
        case eEncodingT1:
        case eEncodingA1:
            single_regs = false;
            d = Bit32(opcode, 22) << 4 | Bits32(opcode, 15, 12);
            imm32 = Bits32(opcode, 7, 0) * addr_byte_size;
            // If UInt(imm8) is odd, see "FLDMX".
            regs = Bits32(opcode, 7, 0) / 2;
            // if regs == 0 || regs > 16 || (d+regs) > 32 then UNPREDICTABLE;
            if (regs == 0 || regs > 16 || (d + regs) > 32)
                return false;
            break;
        case eEncodingT2:
        case eEncodingA2:
            single_regs = true;
            d = Bits32(opcode, 15, 12) << 1 | Bit32(opcode, 22);
            imm32 = Bits32(opcode, 7, 0) * addr_byte_size;
            regs = Bits32(opcode, 7, 0);
            // if regs == 0 || regs > 16 || (d+regs) > 32 then UNPREDICTABLE;
            if (regs == 0 || regs > 16 || (d + regs) > 32)
                return false;
            break;
        default:
            return false;
        }
        uint32_t start_reg = single_regs ? dwarf_s0 : dwarf_d0;
        uint32_t reg_byte_size = single_regs ? addr_byte_size : addr_byte_size * 2;
        addr_t sp_offset = imm32;
        addr_t addr = sp;
        uint32_t i;
        uint64_t data; // uint64_t to accomodate 64-bit registers.
        
        EmulateInstruction::Context context = { EmulateInstruction::eContextPopRegisterOffStack, eRegisterKindDWARF, 0, 0 };
        for (i=d; i<regs; ++i)
        {
            context.arg1 = start_reg + i;    // arg1 in the context is the DWARF register number
            context.arg2 = addr - sp;        // arg2 in the context is the stack pointer offset
            data = ReadMemoryUnsigned(context, addr, reg_byte_size, 0, &success);
            if (!success)
                return false;    
            if (!WriteRegisterUnsigned(context, eRegisterKindDWARF, context.arg1, data))
                return false;
            addr += reg_byte_size;
        }
        
        context.type = EmulateInstruction::eContextAdjustStackPointer;
        context.arg0 = eRegisterKindGeneric;
        context.arg1 = LLDB_REGNUM_GENERIC_SP;
        context.arg2 = sp_offset;
    
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, sp + sp_offset))
            return false;
    }
    return true;
}

// SVC (previously SWI)
bool
EmulateInstructionARM::EmulateSVC (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        CallSupervisor();
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        const uint32_t pc = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
        addr_t lr; // next instruction address
        if (!success)
            return false;
        uint32_t imm32; // the immediate constant
        uint32_t mode;  // ARM or Thumb mode
        switch (encoding) {
        case eEncodingT1:
            lr = (pc + 2) | 1u; // return address
            imm32 = Bits32(opcode, 7, 0);
            mode = eModeThumb;
            break;
        case eEncodingA1:
            lr = pc + 4; // return address
            imm32 = Bits32(opcode, 23, 0);
            mode = eModeARM;
            break;
        default:
            return false;
        }
        EmulateInstruction::Context context = { EmulateInstruction::eContextSupervisorCall, mode, imm32, 0};
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA, lr))
            return false;
    }
    return true;
}

// If Then makes up to four following instructions (the IT block) conditional.
bool
EmulateInstructionARM::EmulateIT (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    EncodingSpecificOperations();
    ITSTATE.IT<7:0> = firstcond:mask;
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    m_it_session.InitIT(Bits32(opcode, 7, 0));
    return true;
}

// Branch causes a branch to a target address.
bool
EmulateInstructionARM::EmulateB (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations();
        BranchWritePC(PC + imm32);
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        EmulateInstruction::Context context = { EmulateInstruction::eContextRelativeBranchImmediate, 0, 0, 0};
        const uint32_t pc = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
        if (!success)
            return false;
        addr_t target; // target address
        int32_t imm32; // PC-relative offset
        switch (encoding) {
        case eEncodingT1:
            // The 'cond' field is handled in EmulateInstructionARM::CurrentCond().
            imm32 = llvm::SignExtend32<9>(Bits32(opcode, 7, 0) << 1);
            target = pc + 4 + imm32;
            context.arg1 = 4 + imm32;  // signed offset
            context.arg2 = eModeThumb; // target instruction set
            break;
        case eEncodingT2:
            imm32 = llvm::SignExtend32<12>(Bits32(opcode, 10, 0));
            target = pc + 4 + imm32;
            context.arg1 = 4 + imm32;  // signed offset
            context.arg2 = eModeThumb; // target instruction set
            break;
        case eEncodingT3:
            // The 'cond' field is handled in EmulateInstructionARM::CurrentCond().
            {
            uint32_t S = Bit32(opcode, 26);
            uint32_t imm6 = Bits32(opcode, 21, 16);
            uint32_t J1 = Bit32(opcode, 13);
            uint32_t J2 = Bit32(opcode, 11);
            uint32_t imm11 = Bits32(opcode, 10, 0);
            uint32_t imm21 = (S << 20) | (J2 << 19) | (J1 << 18) | (imm6 << 12) | (imm11 << 1);
            imm32 = llvm::SignExtend32<21>(imm21);
            target = pc + 4 + imm32;
            context.arg1 = eModeThumb; // target instruction set
            context.arg2 = 4 + imm32;  // signed offset
            break;
            }
        case eEncodingT4:
            {
            uint32_t S = Bit32(opcode, 26);
            uint32_t imm10 = Bits32(opcode, 25, 16);
            uint32_t J1 = Bit32(opcode, 13);
            uint32_t J2 = Bit32(opcode, 11);
            uint32_t imm11 = Bits32(opcode, 10, 0);
            uint32_t I1 = !(J1 ^ S);
            uint32_t I2 = !(J2 ^ S);
            uint32_t imm25 = (S << 24) | (I1 << 23) | (I2 << 22) | (imm10 << 12) | (imm11 << 1);
            imm32 = llvm::SignExtend32<25>(imm25);
            target = pc + 4 + imm32;
            context.arg1 = eModeThumb; // target instruction set
            context.arg2 = 4 + imm32;  // signed offset
            break;
            }
        case eEncodingA1:
            imm32 = llvm::SignExtend32<26>(Bits32(opcode, 23, 0) << 2);
            target = pc + 8 + imm32;
            context.arg1 = eModeARM; // target instruction set
            context.arg2 = 8 + imm32;  // signed offset
            break;
        default:
            return false;
        }
        if (!BranchWritePC(context, target))
            return false;
    }
    return true;
}

// Compare and Branch on Nonzero and Compare and Branch on Zero compare the value in a register with
// zero and conditionally branch forward a constant value.  They do not affect the condition flags.
// CBNZ, CBZ
bool
EmulateInstructionARM::EmulateCB (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    EncodingSpecificOperations();
    if nonzero ^ IsZero(R[n]) then
        BranchWritePC(PC + imm32);
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    // Read the register value from the operand register Rn.
    uint32_t reg_val = ReadRegisterUnsigned(eRegisterKindDWARF, dwarf_r0 + Bits32(opcode, 2, 0), 0, &success);
    if (!success)
        return false;
                  
    EmulateInstruction::Context context = { EmulateInstruction::eContextRelativeBranchImmediate, 0, 0, 0};
    const uint32_t pc = ReadRegisterUnsigned(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
    if (!success)
        return false;

    addr_t target;  // target address
    uint32_t imm32; // PC-relative offset to branch forward
    bool nonzero;
    switch (encoding) {
    case eEncodingT1:
        imm32 = Bit32(opcode, 9) << 6 | Bits32(opcode, 7, 3) << 1;
        nonzero = BitIsSet(opcode, 11);
        target = pc + 4 + imm32;
        context.arg1 = 4 + imm32;  // signed offset
        context.arg2 = eModeThumb; // target instruction set
        break;
    default:
        return false;
    }
    if (nonzero ^ (reg_val == 0))
        if (!BranchWritePC(context, target))
            return false;

    return true;
}

// ADD <Rdn>, <Rm>
// where <Rdn> the destination register is also the first operand register
// and <Rm> is the second operand register.
bool
EmulateInstructionARM::EmulateAddRdnRm (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if ConditionPassed() then
        EncodingSpecificOperations();
        shifted = Shift(R[m], shift_t, shift_n, APSR.C);
        (result, carry, overflow) = AddWithCarry(R[n], shifted, '0');
        if d == 15 then
            ALUWritePC(result); // setflags is always FALSE here
        else
            R[d] = result;
            if setflags then
                APSR.N = result<31>;
                APSR.Z = IsZeroBit(result);
                APSR.C = carry;
                APSR.V = overflow;
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        uint32_t Rd, Rn, Rm;
        //bool setflags = false;
        switch (encoding)
        {
        case eEncodingT2:
            // setflags = FALSE
            Rd = Rn = Bit32(opcode, 7) << 3 | Bits32(opcode, 2, 0);
            Rm = Bits32(opcode, 6, 3);
            if (Rn == 15 && Rm == 15)
                return false;
            break;
        default:
            return false;
        }

        int32_t result, val1, val2;
        // Read the first operand.
        if (Rn == 15)
            val1 = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
        else
            val1 = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + Rn, 0, &success);
        if (!success)
            return false;

        // Read the second operand.
        if (Rm == 15)
            val2 = ReadRegisterUnsigned (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, 0, &success);
        else
            val2 = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + Rm, 0, &success);
        if (!success)
            return false;

        result = val1 + val2;
        EmulateInstruction::Context context = { EmulateInstruction::eContextImmediate,
                                                result,
                                                0,
                                                0 };
    
        if (Rd == 15)
        {
            if (!ALUWritePC (context, result))
                return false;
        }
        else
        {
            if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, result))
                return false;
        }
    }
    return true;
}

bool
EmulateInstructionARM::EmulateCmpRnImm (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if ConditionPassed() then
        EncodingSpecificOperations();
        (result, carry, overflow) = AddWithCarry(R[n], NOT(imm32), '1');
        APSR.N = result<31>;
        APSR.Z = IsZeroBit(result);
        APSR.C = carry;
        APSR.V = overflow;
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    uint32_t Rn; // the first operand
    uint32_t imm32; // the immediate value to be compared with
    switch (encoding) {
    case eEncodingT1:
        Rn = Bits32(opcode, 10, 8);
        imm32 = Bits32(opcode, 7, 0);
        break;
    default:
        return false;
    }
    // Read the register value from the operand register Rn.
    uint32_t reg_val = ReadRegisterUnsigned(eRegisterKindDWARF, dwarf_r0 + Rn, 0, &success);
    if (!success)
        return false;
                  
    EmulateInstruction::Context context = { EmulateInstruction::eContextImmediate, 0, 0, 0};
    AddWithCarryResult res = AddWithCarry(reg_val, ~imm32, 1);
    m_new_inst_cpsr = m_inst_cpsr;
    SetBit32(m_new_inst_cpsr, CPSR_N, Bit32(res.result, CPSR_N));
    SetBit32(m_new_inst_cpsr, CPSR_Z, res.result == 0 ? 1 : 0);
    SetBit32(m_new_inst_cpsr, CPSR_C, res.carry_out);
    SetBit32(m_new_inst_cpsr, CPSR_V, res.overflow);
    if (m_new_inst_cpsr != m_inst_cpsr)
    {
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS, m_new_inst_cpsr))
            return false;
    }
    return true;
}

// LDM loads multiple registers from consecutive memory locations, using an
// address from a base register.  Optionally the address just above the highest of those locations
// can be written back to the base register.
bool
EmulateInstructionARM::EmulateLDM (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if ConditionPassed()
        EncodingSpecificOperations(); NullCheckIfThumbEE (n);
        address = R[n];
                  
        for i = 0 to 14
            if registers<i> == '1' then
                R[i] = MemA[address, 4]; address = address + 4;
        if registers<15> == '1' then
            LoadWritePC (MemA[address, 4]);
                  
        if wback && registers<n> == '0' then R[n] = R[n] + 4 * BitCount (registers);
        if wback && registers<n> == '1' then R[n] = bits(32) UNKNOWN; // Only possible for encoding A1

#endif
            
    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;
            
    if (ConditionPassed())
    {
        uint32_t n;
        uint32_t registers = 0;
        bool wback;
        const uint32_t addr_byte_size = GetAddressByteSize();
        switch (encoding)
        {
            case eEncodingT1:
                n = Bits32 (opcode, 10, 8);
                registers = Bits32 (opcode, 7, 0);
                wback = BitIsClear (registers, n);
                // if BitCount(registers) < 1 then UNPREDICTABLE;
                if (BitCount(registers) < 1)
                    return false;
                break;
            case eEncodingT2:
                n = Bits32 (opcode, 19, 16);
                registers = Bits32 (opcode, 15, 0);
                wback = BitIsSet (opcode, 21);
                if ((n == 15)
                    || (BitCount (registers) < 2)
                    || (BitIsSet (opcode, 14) && BitIsSet (opcode, 15)))
                    return false;
                if (BitIsSet (registers, 15)
                    && m_it_session.InITBlock()
                    && !m_it_session.LastInITBlock())
                    return false;
                if (wback
                    && BitIsSet (registers, n))
                    return false;
                break;
            case eEncodingA1:
                n = Bits32 (opcode, 19, 16);
                registers = Bits32 (opcode, 15, 0);
                wback = BitIsSet (opcode, 21);
                if ((n == 15)
                    || (BitCount (registers) < 1))
                    return false;
                break;
            default:
                return false;
        }
        
        int32_t offset = 0;
        const addr_t base_address = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + n, 0, &success);
        if (!success)
            return false;

        EmulateInstruction::Context context = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindDWARF,
                                                dwarf_r0 + n,
                                                offset };
                  
        for (int i = 0; i < 14; ++i)
        {
            if (BitIsSet (registers, i))
            {
                context.type = EmulateInstruction::eContextRegisterPlusOffset;
                context.arg2 = offset;
                if (wback && (n == 13)) // Pop Instruction
                    context.type = EmulateInstruction::eContextPopRegisterOffStack;

                // R[i] = MemA [address, 4]; address = address + 4;
                uint32_t data = ReadMemoryUnsigned (context, base_address + offset, addr_byte_size, 0, &success);
                if (!success)
                    return false;
                  
                if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + i, data))
                    return false;

                offset += addr_byte_size;
            }
        }
                
        if (BitIsSet (registers, 15))
        {
            //LoadWritePC (MemA [address, 4]);
            context.type = EmulateInstruction::eContextRegisterPlusOffset;
            context.arg2 = offset;
            uint32_t data = ReadMemoryUnsigned (context, base_address + offset, addr_byte_size, 0, &success);
            if (!success)
                return false;
            // In ARMv5T and above, this is an interworking branch.
            if (!LoadWritePC(context, data))
                return false;
        }
                             
        if (wback && BitIsClear (registers, n))
        {
            addr_t offset = addr_byte_size * BitCount (registers);
            context.type = EmulateInstruction::eContextRegisterPlusOffset;
            context.arg2 = offset;
                
            // R[n] = R[n] + 4 * BitCount (registers)
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + n, base_address + offset))
                return false;
        }
        if (wback && BitIsSet (registers, n))
            // R[n] bits(32) UNKNOWN;
            return WriteBits32Unknown (n);
    }
    return true;
}
                
// LDMDA loads multiple registers from consecutive memory locations using an address from a base registers.
// The consecutive memorty locations end at this address and the address just below the lowest of those locations
// can optionally be written back tot he base registers.
bool
EmulateInstructionARM::EmulateLDMDA (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if ConditionPassed() then 
        EncodingSpecificOperations(); 
        address = R[n] - 4*BitCount(registers) + 4;
                  
        for i = 0 to 14 
            if registers<i> == ’1’ then
                  R[i] = MemA[address,4]; address = address + 4; 
                  
        if registers<15> == ’1’ then
            LoadWritePC(MemA[address,4]);
                  
        if wback && registers<n> == ’0’ then R[n] = R[n] - 4*BitCount(registers); 
        if wback && registers<n> == ’1’ then R[n] = bits(32) UNKNOWN;
#endif
                  
    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;
                  
    if (ConditionPassed())
    {
        uint32_t n;
        uint32_t registers = 0;
        bool wback;
        const uint32_t addr_byte_size = GetAddressByteSize();
                  
        // EncodingSpecificOperations(); 
        switch (encoding)
        {
            case eEncodingA1:
                // n = UInt(Rn); registers = register_list; wback = (W == ’1’);
                n = Bits32 (opcode, 19, 16);
                registers = Bits32 (opcode, 15, 0);
                wback = BitIsSet (opcode, 21);
                  
                // if n == 15 || BitCount(registers) < 1 then UNPREDICTABLE;
                if ((n == 15) || (BitCount (registers) < 1))
                    return false;
                  
                break;

            default:
                return false;
        }
        // address = R[n] - 4*BitCount(registers) + 4;
                  
        int32_t offset = 0;
        addr_t address = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + n, 0, &success);
                  
        if (!success)
            return false;
            
        address = address - (addr_byte_size * BitCount (registers)) + addr_byte_size;
                                                        
        EmulateInstruction::Context context = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindDWARF,
                                                dwarf_r0 + n,
                                                offset };
                  
        // for i = 0 to 14 
        for (int i = 0; i < 14; ++i)
        {
            // if registers<i> == ’1’ then
            if (BitIsSet (registers, i))
            {
                  // R[i] = MemA[address,4]; address = address + 4; 
                  context.arg2 = offset;
                  uint32_t data = ReadMemoryUnsigned (context, address + offset, addr_byte_size, 0, &success);
                  if (!success)
                      return false;
                  if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + i, data))
                      return false;
                  offset += addr_byte_size;
            }
        }
                  
        // if registers<15> == ’1’ then
        //     LoadWritePC(MemA[address,4]);
        if (BitIsSet (registers, 15))
        {
            context.arg2 = offset;
            uint32_t data = ReadMemoryUnsigned (context, address + offset, addr_byte_size, 0, &success);
            if (!success)
                return false;
            // In ARMv5T and above, this is an interworking branch.
            if (!LoadWritePC(context, data))
                return false;
        }
                  
        // if wback && registers<n> == ’0’ then R[n] = R[n] - 4*BitCount(registers); 
        if (wback && BitIsClear (registers, n))
        {
            context.arg2 = offset;      
            addr_t addr = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + n, 0, &success);
            if (!success)
                return false;
            addr = addr - (addr_byte_size * BitCount (registers));
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + n, addr))
                return false;
        }
                  
        // if wback && registers<n> == ’1’ then R[n] = bits(32) UNKNOWN;
        if (wback && BitIsSet (registers, n))
            return WriteBits32Unknown (n);
    }
    return true;
}
  
// LDMDB loads multiple registers from consecutive memory locations using an address from a base register.  The 
// consecutive memory lcoations end just below this address, and the address of the lowest of those locations can 
// be optionally written back to the base register.
bool
EmulateInstructionARM::EmulateLDMDB (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if ConditionPassed() then 
        EncodingSpecificOperations(); NullCheckIfThumbEE(n); 
        address = R[n] - 4*BitCount(registers);
                  
        for i = 0 to 14 
            if registers<i> == ’1’ then
                  R[i] = MemA[address,4]; address = address + 4; 
        if registers<15> == ’1’ then
                  LoadWritePC(MemA[address,4]);
                  
        if wback && registers<n> == ’0’ then R[n] = R[n] - 4*BitCount(registers); 
        if wback && registers<n> == ’1’ then R[n] = bits(32) UNKNOWN; // Only possible for encoding A1
#endif
                  
    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;
                  
    if (ConditionPassed())
    {
        uint32_t n;
        uint32_t registers = 0;
        bool wback;
        const uint32_t addr_byte_size = GetAddressByteSize();
        switch (encoding)
        {
            case eEncodingT1:
                // n = UInt(Rn); registers = P:M:’0’:register_list; wback = (W == ’1’);
                n = Bits32 (opcode, 19, 16);
                registers = Bits32 (opcode, 15, 0);
                wback = BitIsSet (opcode, 21);

                // if n == 15 || BitCount(registers) < 2 || (P == ’1’ && M == ’1’) then UNPREDICTABLE;
                if ((n == 15)
                    || (BitCount (registers) < 2)
                    || (BitIsSet (opcode, 14) && BitIsSet (opcode, 15)))
                    return false;

                // if registers<15> == ’1’ && InITBlock() && !LastInITBlock() then UNPREDICTABLE;
                if (BitIsSet (registers, 15)
                    && m_it_session.InITBlock()
                    && !m_it_session.LastInITBlock())
                    return false;

                // if wback && registers<n> == ’1’ then UNPREDICTABLE;
                if (wback && BitIsSet (registers, n))
                    return false;
                  
                break;
                  
            case eEncodingA1:
                // n = UInt(Rn); registers = register_list; wback = (W == ’1’);
                n = Bits32 (opcode, 19, 16);
                registers = Bits32 (opcode, 15, 0);
                wback = BitIsSet (opcode, 21);
                  
                // if n == 15 || BitCount(registers) < 1 then UNPREDICTABLE;
                if ((n == 15) || (BitCount (registers) < 1))
                    return false;
                  
                break;
                  
            default:
                return false;
        }
                  
        // address = R[n] - 4*BitCount(registers);
                  
        int32_t offset = 0;
        addr_t address = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + n, 0, &success);
                  
        if (!success)
            return false;
                  
        address = address - (addr_byte_size * BitCount (registers));
        EmulateInstruction::Context context = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindDWARF,
                                                dwarf_r0 + n,
                                                offset };
                  
        for (int i = 0; i < 14; ++i)
        {
            if (BitIsSet (registers, i))
            {
                // R[i] = MemA[address,4]; address = address + 4;
                context.arg2 = offset;
                uint32_t data = ReadMemoryUnsigned (context, address + offset, addr_byte_size, 0, &success);
                if (!success)
                    return false;
                  
                if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + i, data))
                    return false;
                  
                offset += addr_byte_size;
            }
        }
                  
        // if registers<15> == ’1’ then
        //     LoadWritePC(MemA[address,4]);
        if (BitIsSet (registers, 15))
        {
            context.arg2 = offset;
            uint32_t data = ReadMemoryUnsigned (context, address + offset, addr_byte_size, 0, &success);
            if (!success)
                return false;
            // In ARMv5T and above, this is an interworking branch.
            if (!LoadWritePC(context, data))
                return false;
        }
                  
        // if wback && registers<n> == ’0’ then R[n] = R[n] - 4*BitCount(registers);
        if (wback && BitIsClear (registers, n))
        {
            context.arg2 = offset;      
            addr_t addr = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + n, 0, &success);
            if (!success)
                return false;
            addr = addr - (addr_byte_size * BitCount (registers));
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + n, addr))
                return false;
        }
                  
        // if wback && registers<n> == ’1’ then R[n] = bits(32) UNKNOWN; // Only possible for encoding A1
        if (wback && BitIsSet (registers, n))
            return WriteBits32Unknown (n);
    }
    return true;
}

// LDMIB loads multiple registers from consecutive memory locations using an address from a base register.  The 
// consecutive memory locations start just above this address, and thea ddress of the last of those locations can 
// optinoally be written back to the base register.
bool
EmulateInstructionARM::EmulateLDMIB (ARMEncoding encoding)
{
#if 0
    if ConditionPassed() then
        EncodingSpecificOperations(); 
        address = R[n] + 4;
                  
        for i = 0 to 14 
            if registers<i> == ’1’ then
                  R[i] = MemA[address,4]; address = address + 4; 
        if registers<15> == ’1’ then
            LoadWritePC(MemA[address,4]);
                  
        if wback && registers<n> == ’0’ then R[n] = R[n] + 4*BitCount(registers); 
        if wback && registers<n> == ’1’ then R[n] = bits(32) UNKNOWN;
#endif
                  
    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;
                  
    if (ConditionPassed())
    {
        uint32_t n;
        uint32_t registers = 0;
        bool wback;
        const uint32_t addr_byte_size = GetAddressByteSize();
        switch (encoding)
        {
            case eEncodingA1:
                // n = UInt(Rn); registers = register_list; wback = (W == ’1’);
                n = Bits32 (opcode, 19, 16);
                registers = Bits32 (opcode, 15, 0);
                wback = BitIsSet (opcode, 21);
                  
                // if n == 15 || BitCount(registers) < 1 then UNPREDICTABLE;
                if ((n == 15) || (BitCount (registers) < 1))
                    return false;
                  
                break;
            default:
                return false;
        }
        // address = R[n] + 4;
                  
        int32_t offset = 0;
        addr_t address = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + n, 0, &success);
                  
        if (!success)
            return false;
                  
        address = address + addr_byte_size;
                  
        EmulateInstruction::Context context = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindDWARF,
                                                dwarf_r0 + n,
                                                offset };

        for (int i = 0; i < 14; ++i)
        {
            if (BitIsSet (registers, i))
            {
                // R[i] = MemA[address,4]; address = address + 4;
                
                context.arg2 = offset;
                uint32_t data = ReadMemoryUnsigned (context, address + offset, addr_byte_size, 0, &success);
                if (!success)
                    return false;
                  
                if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + i, data))
                    return false;
                  
                offset += addr_byte_size;
            }
        }
                  
        // if registers<15> == ’1’ then
        //     LoadWritePC(MemA[address,4]);
        if (BitIsSet (registers, 15))
        {
            context.arg2 = offset;
            uint32_t data = ReadMemoryUnsigned (context, address + offset, addr_byte_size, 0, &success);
            if (!success)
                return false;
            // In ARMv5T and above, this is an interworking branch.
            if (!LoadWritePC(context, data))
                return false;
        }
                  
        // if wback && registers<n> == ’0’ then R[n] = R[n] + 4*BitCount(registers);
        if (wback && BitIsClear (registers, n))
        {
            context.arg2 = offset;      
            addr_t addr = ReadRegisterUnsigned (eRegisterKindDWARF, dwarf_r0 + n, 0, &success);
            if (!success)
                return false;
            addr = addr + (addr_byte_size * BitCount (registers));
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + n, addr))
                return false;
        }
                  
        // if wback && registers<n> == ’1’ then R[n] = bits(32) UNKNOWN; // Only possible for encoding A1
        if (wback && BitIsSet (registers, n))
            return WriteBits32Unknown (n);
    }
    return true;
}
                  
// Load Register (immediate) calculates an address from a base register value and
// an immediate offset, loads a word from memory, and writes to a register.
// LDR (immediate, Thumb)
bool
EmulateInstructionARM::EmulateLDRRtRnImm (ARMEncoding encoding)
{
#if 0
    // ARM pseudo code...
    if (ConditionPassed())
    {
        EncodingSpecificOperations(); NullCheckIfThumbEE(15);
        offset_addr = if add then (R[n] + imm32) else (R[n] - imm32);
        address = if index then offset_addr else R[n];
        data = MemU[address,4];
        if wback then R[n] = offset_addr;
        if t == 15 then
            if address<1:0> == '00' then LoadWritePC(data); else UNPREDICTABLE;
        elsif UnalignedSupport() || address<1:0> = '00' then
            R[t] = data;
        else R[t] = bits(32) UNKNOWN; // Can only apply before ARMv7
    }
#endif

    bool success = false;
    const uint32_t opcode = OpcodeAsUnsigned (&success);
    if (!success)
        return false;

    if (ConditionPassed())
    {
        uint32_t Rt; // the destination register
        uint32_t Rn; // the base register
        uint32_t imm32; // the immediate offset used to form the address
        addr_t offset_addr; // the offset address
        addr_t address; // the calculated address
        uint32_t data; // the literal data value from memory load
        bool add, index, wback;
        switch (encoding) {
        case eEncodingT1:
            Rt = Bits32(opcode, 5, 3);
            Rn = Bits32(opcode, 2, 0);
            imm32 = Bits32(opcode, 10, 6) << 2; // imm32 = ZeroExtend(imm5:'00', 32);
            // index = TRUE; add = TRUE; wback = FALSE
            add = true;
            index = true;
            wback = false;
            break;
        default:
            return false;
        }
        uint32_t base = ReadRegisterUnsigned(eRegisterKindDWARF, dwarf_r0 + Rn, 0, &success);
        if (!success)
            return false;
        if (add)
            offset_addr = base + imm32;
        else
            offset_addr = base - imm32;

        address = (index ? offset_addr : base);

        if (wback)
        {
            EmulateInstruction::Context ctx = { EmulateInstruction::eContextRegisterPlusOffset,
                                                eRegisterKindDWARF,
                                                dwarf_r0 + Rn,
                                                (int32_t) (offset_addr - base)};
            if (!WriteRegisterUnsigned (ctx, eRegisterKindDWARF, dwarf_r0 + Rn, offset_addr))
                return false;
        }

        // Prepare to write to the Rt register.
        EmulateInstruction::Context context = {EmulateInstruction::eContextImmediate,
                                               0,
                                               0,
                                               0};

        // Read memory from the address.
        data = ReadMemoryUnsigned(context, address, 4, 0, &success);
        if (!success)
            return false;    
        context.arg0 = data;

        if (Rt == 15)
        {
            if (Bits32(address, 1, 0) == 0)
            {
                if (!LoadWritePC(context, data))
                    return false;
            }
            else
                return false;
        }
        else if (UnalignedSupport() || Bits32(address, 1, 0) == 0)
        {
            if (!WriteRegisterUnsigned (context, eRegisterKindDWARF, dwarf_r0 + Rt, data))
                return false;
        }
        else
            return false;
    }
    return true;
}

EmulateInstructionARM::ARMOpcode*
EmulateInstructionARM::GetARMOpcodeForInstruction (const uint32_t opcode)
{
    static ARMOpcode 
    g_arm_opcodes[] = 
    {
        //----------------------------------------------------------------------
        // Prologue instructions
        //----------------------------------------------------------------------

        // push register(s)
        { 0x0fff0000, 0x092d0000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulatePush, "push <registers>" },
        { 0x0fff0fff, 0x052d0004, ARMvAll,       eEncodingA2, eSize32, &EmulateInstructionARM::EmulatePush, "push <register>" },

        // set r7 to point to a stack offset
        { 0x0ffff000, 0x028d7000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateAddRdSPImmediate, "add r7, sp, #<const>" },
        { 0x0ffff000, 0x024c7000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateSubR7IPImmediate, "sub r7, ip, #<const>"},
        // copy the stack pointer to ip
        { 0x0fffffff, 0x01a0c00d, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateMovRdSP, "mov ip, sp" },
        { 0x0ffff000, 0x028dc000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateAddRdSPImmediate, "add ip, sp, #<const>" },
        { 0x0ffff000, 0x024dc000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateSubIPSPImmediate, "sub ip, sp, #<const>"},

        // adjust the stack pointer
        { 0x0ffff000, 0x024dd000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateSubSPImmdiate, "sub sp, sp, #<const>"},

        // push one register
        // if Rn == '1101' && imm12 == '000000000100' then SEE PUSH;
        { 0x0fff0000, 0x052d0000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateSTRRtSP, "str Rt, [sp, #-imm12]!" },

        // vector push consecutive extension register(s)
        { 0x0fbf0f00, 0x0d2d0b00, ARMV6T2_ABOVE, eEncodingA1, eSize32, &EmulateInstructionARM::EmulateVPUSH, "vpush.64 <list>"},
        { 0x0fbf0f00, 0x0d2d0a00, ARMV6T2_ABOVE, eEncodingA2, eSize32, &EmulateInstructionARM::EmulateVPUSH, "vpush.32 <list>"},

        //----------------------------------------------------------------------
        // Epilogue instructions
        //----------------------------------------------------------------------

        // To resolve ambiguity, "blx <label>" should come before "bl <label>".
        { 0xfe000000, 0xfa000000, ARMV5_ABOVE,   eEncodingA2, eSize32, &EmulateInstructionARM::EmulateBLXImmediate, "blx <label>"},
        { 0x0f000000, 0x0b000000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateBLXImmediate, "bl <label>"},
        { 0x0ffffff0, 0x012fff30, ARMV5_ABOVE,   eEncodingA1, eSize32, &EmulateInstructionARM::EmulateBLXRm, "blx <Rm>"},
        { 0x0fff0000, 0x08bd0000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulatePop, "pop <registers>"},
        { 0x0fff0fff, 0x049d0004, ARMvAll,       eEncodingA2, eSize32, &EmulateInstructionARM::EmulatePop, "pop <register>"},
        { 0x0fbf0f00, 0x0cbd0b00, ARMV6T2_ABOVE, eEncodingA1, eSize32, &EmulateInstructionARM::EmulateVPOP, "vpop.64 <list>"},
        { 0x0fbf0f00, 0x0cbd0a00, ARMV6T2_ABOVE, eEncodingA2, eSize32, &EmulateInstructionARM::EmulateVPOP, "vpop.32 <list>"},

        //----------------------------------------------------------------------
        // Supervisor Call (previously Software Interrupt)
        //----------------------------------------------------------------------
        { 0x0f000000, 0x0f000000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateSVC, "svc #imm24"},

        //----------------------------------------------------------------------
        // Branch instructions
        //----------------------------------------------------------------------
        { 0x0f000000, 0x0a000000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateSVC, "b #imm24"},

        //----------------------------------------------------------------------
        // Load instructions
        //----------------------------------------------------------------------
        { 0x0fd00000, 0x08900000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateLDM, "ldm<c> <Rn>{!} <registers>" },
        { 0x0fd00000, 0x08100000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateLDMDA, "ldmda<c> <Rn>{!} <registers>" },
        { 0x0fd00000, 0x09100000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateLDMDB, "ldmdb<c> <Rn>{!} <registers>" },
        { 0x0fd00000, 0x09900000, ARMvAll,       eEncodingA1, eSize32, &EmulateInstructionARM::EmulateLDMIB, "ldmib<c> <Rn<{!} <registers>" }
        
    };
    static const size_t k_num_arm_opcodes = sizeof(g_arm_opcodes)/sizeof(ARMOpcode);
                  
    for (size_t i=0; i<k_num_arm_opcodes; ++i)
    {
        if ((g_arm_opcodes[i].mask & opcode) == g_arm_opcodes[i].value)
            return &g_arm_opcodes[i];
    }
    return NULL;
}

    
EmulateInstructionARM::ARMOpcode*
EmulateInstructionARM::GetThumbOpcodeForInstruction (const uint32_t opcode)
{

    static ARMOpcode 
    g_thumb_opcodes[] =
    {
        //----------------------------------------------------------------------
        // Prologue instructions
        //----------------------------------------------------------------------

        // push register(s)
        { 0xfffffe00, 0x0000b400, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulatePush, "push <registers>" },
        { 0xffff0000, 0xe92d0000, ARMV6T2_ABOVE, eEncodingT2, eSize32, &EmulateInstructionARM::EmulatePush, "push.w <registers>" },
        { 0xffff0fff, 0xf84d0d04, ARMV6T2_ABOVE, eEncodingT3, eSize32, &EmulateInstructionARM::EmulatePush, "push.w <register>" },

        // set r7 to point to a stack offset
        { 0xffffff00, 0x0000af00, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateAddRdSPImmediate, "add r7, sp, #imm" },
        // copy the stack pointer to r7
        { 0xffffffff, 0x0000466f, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateMovRdSP, "mov r7, sp" },
        // move from high register to low register (comes after "mov r7, sp" to resolve ambiguity)
        { 0xffffffc0, 0x00004640, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateMovLowHigh, "mov r0-r7, r8-r15" },

        // PC-relative load into register (see also EmulateAddSPRm)
        { 0xfffff800, 0x00004800, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateLDRRtPCRelative, "ldr <Rt>, [PC, #imm]"},

        // adjust the stack pointer
        { 0xffffff87, 0x00004485, ARMvAll,       eEncodingT2, eSize16, &EmulateInstructionARM::EmulateAddSPRm, "add sp, <Rm>"},
        { 0xffffff80, 0x0000b080, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateSubSPImmdiate, "add sp, sp, #imm"},
        { 0xfbef8f00, 0xf1ad0d00, ARMV6T2_ABOVE, eEncodingT2, eSize32, &EmulateInstructionARM::EmulateSubSPImmdiate, "sub.w sp, sp, #<const>"},
        { 0xfbff8f00, 0xf2ad0d00, ARMV6T2_ABOVE, eEncodingT3, eSize32, &EmulateInstructionARM::EmulateSubSPImmdiate, "subw sp, sp, #imm12"},

        // vector push consecutive extension register(s)
        { 0xffbf0f00, 0xed2d0b00, ARMV6T2_ABOVE, eEncodingT1, eSize32, &EmulateInstructionARM::EmulateVPUSH, "vpush.64 <list>"},
        { 0xffbf0f00, 0xed2d0a00, ARMV6T2_ABOVE, eEncodingT2, eSize32, &EmulateInstructionARM::EmulateVPUSH, "vpush.32 <list>"},

        //----------------------------------------------------------------------
        // Epilogue instructions
        //----------------------------------------------------------------------

        { 0xffffff80, 0x0000b000, ARMvAll,       eEncodingT2, eSize16, &EmulateInstructionARM::EmulateAddSPImmediate, "add sp, #imm"},
        { 0xffffff87, 0x00004780, ARMV5_ABOVE,   eEncodingT1, eSize16, &EmulateInstructionARM::EmulateBLXRm, "blx <Rm>"},
        // J1 == J2 == 1
        { 0xf800f800, 0xf000f800, ARMV4T_ABOVE,  eEncodingT1, eSize32, &EmulateInstructionARM::EmulateBLXImmediate, "bl <label>"},
        // J1 == J2 == 1
        { 0xf800e800, 0xf000e800, ARMV5_ABOVE,   eEncodingT2, eSize32, &EmulateInstructionARM::EmulateBLXImmediate, "blx <label>"},
        { 0xfffffe00, 0x0000bc00, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulatePop, "pop <registers>"},
        { 0xffff0000, 0xe8bd0000, ARMV6T2_ABOVE, eEncodingT2, eSize32, &EmulateInstructionARM::EmulatePop, "pop.w <registers>" },
        { 0xffff0fff, 0xf85d0d04, ARMV6T2_ABOVE, eEncodingT3, eSize32, &EmulateInstructionARM::EmulatePop, "pop.w <register>" },
        { 0xffbf0f00, 0xecbd0b00, ARMV6T2_ABOVE, eEncodingT1, eSize32, &EmulateInstructionARM::EmulateVPOP, "vpop.64 <list>"},
        { 0xffbf0f00, 0xecbd0a00, ARMV6T2_ABOVE, eEncodingT2, eSize32, &EmulateInstructionARM::EmulateVPOP, "vpop.32 <list>"},

        //----------------------------------------------------------------------
        // Supervisor Call (previously Software Interrupt)
        //----------------------------------------------------------------------
        { 0xffffff00, 0x0000df00, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateSVC, "svc #imm8"},

        //----------------------------------------------------------------------
        // If Then makes up to four following instructions conditional.
        //----------------------------------------------------------------------
        { 0xffffff00, 0x0000bf00, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateIT, "it{<x>{<y>{<z>}}} <firstcond>"},

        //----------------------------------------------------------------------
        // Branch instructions
        //----------------------------------------------------------------------
        // To resolve ambiguity, "b<c> #imm8" should come after "svc #imm8".
        { 0xfffff000, 0x0000d000, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateB, "b<c> #imm8 (outside IT)"},
        { 0xffff8000, 0x0000e000, ARMvAll,       eEncodingT2, eSize16, &EmulateInstructionARM::EmulateB, "b #imm11 (outside or last in IT)"},
        { 0xf800d000, 0xf0008000, ARMV6T2_ABOVE, eEncodingT3, eSize32, &EmulateInstructionARM::EmulateB, "b<c>.w #imm8 (outside IT)"},
        { 0xf800d000, 0xf0009000, ARMV6T2_ABOVE, eEncodingT4, eSize32, &EmulateInstructionARM::EmulateB, "b.w #imm8 (outside or last in IT)"},
        // compare and branch
        { 0xfffff500, 0x0000b100, ARMV6T2_ABOVE, eEncodingT1, eSize16, &EmulateInstructionARM::EmulateCB, "cb{n}z <Rn>, <label>"},

        //----------------------------------------------------------------------
        // Data-processing instructions
        //----------------------------------------------------------------------
        // Make sure "add sp, <Rm>" comes before this instruction, so there's no ambiguity decoding the two.
        { 0xffffff00, 0x00004400, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateAddRdnRm, "add <Rdn>, <Rm>"},
        // move from high register to high register
        { 0xffffff00, 0x00004600, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateMovRdRm, "mov<c> <Rd>, <Rm>"},
        // move from low register to low register
        { 0xffffffc0, 0x00000000, ARMvAll,       eEncodingT2, eSize16, &EmulateInstructionARM::EmulateMovRdRm, "movs <Rd>, <Rm>"},
        // compare a register with immediate
        { 0xfffff800, 0x00002800, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateCmpRnImm, "cmp<c> <Rn>, #imm8"},

        //----------------------------------------------------------------------
        // Load instructions
        //----------------------------------------------------------------------
        { 0xfffff800, 0x0000c800, ARMV4T_ABOVE,  eEncodingT1, eSize16, &EmulateInstructionARM::EmulateLDM, "ldm<c> <Rn>{!} <registers>" },
        { 0xffd02000, 0xe8900000, ARMV6T2_ABOVE, eEncodingT2, eSize32, &EmulateInstructionARM::EmulateLDM, "ldm<c>.w <Rn>{!} <registers>" },
        { 0xffd00000, 0xe9100000, ARMV6T2_ABOVE, eEncodingT1, eSize32, &EmulateInstructionARM::EmulateLDMDB, "ldmdb<c> <Rn>{!} <registers>" },
        { 0xfffff800, 0x00006800, ARMvAll,       eEncodingT1, eSize16, &EmulateInstructionARM::EmulateLDRRtRnImm, "ldr<c> <Rt>, [<Rn>{,#imm}]"},
        // Thumb2 PC-relative load into register
        { 0xff7f0000, 0xf85f0000, ARMV6T2_ABOVE, eEncodingT2, eSize32, &EmulateInstructionARM::EmulateLDRRtPCRelative, "ldr<c>.w <Rt>, [PC, +/-#imm}]"}
        
    };

    const size_t k_num_thumb_opcodes = sizeof(g_thumb_opcodes)/sizeof(ARMOpcode);
    for (size_t i=0; i<k_num_thumb_opcodes; ++i)
    {
        if ((g_thumb_opcodes[i].mask & opcode) == g_thumb_opcodes[i].value)
            return &g_thumb_opcodes[i];
    }
    return NULL;
}

bool
EmulateInstructionARM::SetTargetTriple (const ConstString &triple)
{
    m_arm_isa = 0;
    const char *triple_cstr = triple.GetCString();
    if (triple_cstr)
    {
        const char *dash = ::strchr (triple_cstr, '-');
        if (dash)
        {
            std::string arch (triple_cstr, dash);
            const char *arch_cstr = arch.c_str();
            if (strcasecmp(arch_cstr, "armv4t") == 0)
                m_arm_isa = ARMv4T;
            else if (strcasecmp(arch_cstr, "armv4") == 0)
                m_arm_isa = ARMv4;
            else if (strcasecmp(arch_cstr, "armv5tej") == 0)
                m_arm_isa = ARMv5TEJ;
            else if (strcasecmp(arch_cstr, "armv5te") == 0)
                m_arm_isa = ARMv5TE;
            else if (strcasecmp(arch_cstr, "armv5t") == 0)
                m_arm_isa = ARMv5T;
            else if (strcasecmp(arch_cstr, "armv6k") == 0)
                m_arm_isa = ARMv6K;
            else if (strcasecmp(arch_cstr, "armv6") == 0)
                m_arm_isa = ARMv6;
            else if (strcasecmp(arch_cstr, "armv6t2") == 0)
                m_arm_isa = ARMv6T2;
            else if (strcasecmp(arch_cstr, "armv7") == 0)
                m_arm_isa = ARMv7;
            else if (strcasecmp(arch_cstr, "armv8") == 0)
                m_arm_isa = ARMv8;
        }
    }
    return m_arm_isa != 0;
}


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
EmulateInstructionARM::ArchVersion ()
{
    return m_arm_isa;
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
        // For T1 and T3 encodings of the Branch instruction, it returns the 4-bit
        // 'cond' field of the encoding.
        if (m_inst.opcode_type == eOpcode16 &&
            Bits32(m_inst.opcode.inst16, 15, 12) == 0x0d &&
            Bits32(m_inst.opcode.inst16, 11, 7) != 0x0f)
        {
            return Bits32(m_inst.opcode.inst16, 11, 7);
        }
        else if (m_inst.opcode_type == eOpcode32 &&
                 Bits32(m_inst.opcode.inst32, 31, 27) == 0x1e &&
                 Bits32(m_inst.opcode.inst32, 15, 14) == 0x02 &&
                 Bits32(m_inst.opcode.inst32, 12, 12) == 0x00 &&
                 Bits32(m_inst.opcode.inst32, 25, 22) <= 0x0d)
        {
            return Bits32(m_inst.opcode.inst32, 25, 22);
        }
        
        return m_it_session.GetCond();
    }
    return UINT32_MAX;  // Return invalid value
}

bool
EmulateInstructionARM::BranchWritePC (const Context &context, uint32_t addr)
{
    addr_t target;

    // Check the current instruction set.
    if (CurrentInstrSet() == eModeARM)
        target = addr & 0xfffffffc;
    else
        target = addr & 0xfffffffe;

    if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, target))
        return false;

    return true;
}

// As a side effect, BXWritePC sets context.arg2 to eModeARM or eModeThumb by inspecting addr.
bool
EmulateInstructionARM::BXWritePC (Context &context, uint32_t addr)
{
    addr_t target;
    // If the CPSR is changed due to switching between ARM and Thumb ISETSTATE,
    // we want to record it and issue a WriteRegister callback so the clients
    // can track the mode changes accordingly.
    bool cpsr_changed = false;

    if (BitIsSet(addr, 0))
    {
        if (CurrentInstrSet() != eModeThumb)
        {
            SelectInstrSet(eModeThumb);
            cpsr_changed = true;
        }
        target = addr & 0xfffffffe;
        context.arg2 = eModeThumb;
    }
    else if (BitIsClear(addr, 1))
    {
        if (CurrentInstrSet() != eModeARM)
        {
            SelectInstrSet(eModeARM);
            cpsr_changed = true;
        }
        target = addr & 0xfffffffc;
        context.arg2 = eModeARM;
    }
    else
        return false; // address<1:0> == '10' => UNPREDICTABLE

    if (cpsr_changed)
    {
        if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS, m_new_inst_cpsr))
            return false;
    }
    if (!WriteRegisterUnsigned (context, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, target))
        return false;

    return true;
}

// Dispatches to either BXWritePC or BranchWritePC based on architecture versions.
bool
EmulateInstructionARM::LoadWritePC (Context &context, uint32_t addr)
{
    if (ArchVersion() >= ARMv5T)
        return BXWritePC(context, addr);
    else
        return BranchWritePC((const Context)context, addr);
}

// Dispatches to either BXWritePC or BranchWritePC based on architecture versions and current instruction set.
bool
EmulateInstructionARM::ALUWritePC (Context &context, uint32_t addr)
{
    if (ArchVersion() >= ARMv7 && CurrentInstrSet() == eModeARM)
        return BXWritePC(context, addr);
    else
        return BranchWritePC((const Context)context, addr);
}

EmulateInstructionARM::Mode
EmulateInstructionARM::CurrentInstrSet ()
{
    return m_inst_mode;
}

// Set the 'T' bit of our CPSR.  The m_inst_mode gets updated when the next
// ReadInstruction() is performed.  This function has a side effect of updating
// the m_new_inst_cpsr member variable if necessary.
bool
EmulateInstructionARM::SelectInstrSet (Mode arm_or_thumb)
{
    m_new_inst_cpsr = m_inst_cpsr;
    switch (arm_or_thumb)
    {
    default:
        return false;
    eModeARM:
        // Clear the T bit.
        m_new_inst_cpsr &= ~MASK_CPSR_T;
        break;
    eModeThumb:
        // Set the T bit.
        m_new_inst_cpsr |= MASK_CPSR_T;
        break;
    }
    return true;
}

// This function returns TRUE if the processor currently provides support for
// unaligned memory accesses, or FALSE otherwise. This is always TRUE in ARMv7,
// controllable by the SCTLR.U bit in ARMv6, and always FALSE before ARMv6.
bool
EmulateInstructionARM::UnalignedSupport()
{
    return (ArchVersion() >= ARMv7);
}

// The main addition and subtraction instructions can produce status information
// about both unsigned carry and signed overflow conditions.  This status
// information can be used to synthesize multi-word additions and subtractions.
EmulateInstructionARM::AddWithCarryResult
EmulateInstructionARM::AddWithCarry (uint32_t x, uint32_t y, uint8_t carry_in)
{
    uint32_t result;
    uint8_t carry_out;
    uint8_t overflow;

    uint64_t unsigned_sum = x + y + carry_in;
    int64_t signed_sum = (int32_t)x + (int32_t)y + (int32_t)carry_in;
    
    result = UnsignedBits(unsigned_sum, 31, 0);
    carry_out = (result == unsigned_sum ? 0 : 1);
    overflow = ((int32_t)result == signed_sum ? 0 : 1);
    
    AddWithCarryResult res = { result, carry_out, overflow };
    return res;
}

bool
EmulateInstructionARM::EvaluateInstruction ()
{
    // Advance the ITSTATE bits to their values for the next instruction.
    if (m_inst_mode == eModeThumb && m_it_session.InITBlock())
        m_it_session.ITAdvance();

    return false;
}

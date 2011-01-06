//===-- MachThreadContext_arm.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MachThreadContext_arm.h"

#include <sys/sysctl.h>

#include "ProcessMacOSX.h"
#include "ProcessMacOSXLog.h"
#include "ThreadMacOSX.h"

using namespace lldb_private;

//#define DNB_ARCH_MACH_ARM_DEBUG_SW_STEP 1

static const uint8_t g_arm_breakpoint_opcode[] = { 0xFE, 0xDE, 0xFF, 0xE7 };
static const uint8_t g_thumb_breakpooint_opcode[] = { 0xFE, 0xDE };

// ARM constants used during decoding
#define REG_RD          0
#define LDM_REGLIST     1
#define PC_REG          15
#define PC_REGLIST_BIT  0x8000

// ARM conditions
#define COND_EQ     0x0
#define COND_NE     0x1
#define COND_CS     0x2
#define COND_HS     0x2
#define COND_CC     0x3
#define COND_LO     0x3
#define COND_MI     0x4
#define COND_PL     0x5
#define COND_VS     0x6
#define COND_VC     0x7
#define COND_HI     0x8
#define COND_LS     0x9
#define COND_GE     0xA
#define COND_LT     0xB
#define COND_GT     0xC
#define COND_LE     0xD
#define COND_AL     0xE
#define COND_UNCOND 0xF

#define MASK_CPSR_T (1u << 5)
#define MASK_CPSR_J (1u << 24)

#define MNEMONIC_STRING_SIZE 32
#define OPERAND_STRING_SIZE 128

using namespace lldb;
using namespace lldb_private;

MachThreadContext_arm::MachThreadContext_arm(ThreadMacOSX &thread) :
    MachThreadContext(thread),
    m_hw_single_chained_step_addr(LLDB_INVALID_ADDRESS),
    m_bvr0_reg (LLDB_INVALID_REGNUM),
    m_bcr0_reg (LLDB_INVALID_REGNUM),
    m_bvr0_save (0),
    m_bcr0_save (0)
{
}

MachThreadContext_arm::~MachThreadContext_arm()
{
}

lldb::RegisterContextSP
MachThreadContext_arm::CreateRegisterContext (StackFrame *frame) const
{
    lldb::RegisterContextSP reg_ctx_sp (new RegisterContextMach_arm(m_thread, frame->GetConcreteFrameIndex()));
    return reg_ctx_sp;
}

// Instance init function
void
MachThreadContext_arm::InitializeInstance()
{
    RegisterContext *reg_ctx = m_thread.GetRegisterContext().get();
    assert (reg_ctx != NULL);
    const RegisterInfo * reg_info;
    reg_info = reg_ctx->GetRegisterInfoByName ("bvr0");
    if (reg_info)
        m_bvr0_reg = reg_info->kinds[eRegisterKindLLDB];

    reg_info = reg_ctx->GetRegisterInfoByName ("bcr0");
    if (reg_info)
        m_bcr0_reg = reg_info->kinds[eRegisterKindLLDB];
}


void
MachThreadContext_arm::ThreadWillResume()
{
    // Do we need to step this thread? If so, let the mach thread tell us so.
    if (m_thread.GetState() == eStateStepping)
    {
        bool step_handled = false;
        // This is the primary thread, let the arch do anything it needs
        if (m_thread.GetRegisterContext()->NumSupportedHardwareBreakpoints() > 0)
        {
#if defined (DNB_ARCH_MACH_ARM_DEBUG_SW_STEP)
            bool half_step = m_hw_single_chained_step_addr != LLDB_INVALID_ADDRESS;
#endif
            step_handled = EnableHardwareSingleStep(true) == KERN_SUCCESS;
#if defined (DNB_ARCH_MACH_ARM_DEBUG_SW_STEP)
            if (!half_step)
                step_handled = false;
#endif
        }

#if defined (ENABLE_ARM_SINGLE_STEP)
        if (!step_handled)
        {
            SetSingleStepSoftwareBreakpoints();
        }
#endif
    }
}

bool
MachThreadContext_arm::ShouldStop ()
{
    return true;
}

void
MachThreadContext_arm::RefreshStateAfterStop ()
{
    EnableHardwareSingleStep (false) == KERN_SUCCESS;
}

#if defined (ENABLE_ARM_SINGLE_STEP)

bool
MachThreadContext_arm::ShouldStop ()
{
    return true;
}

bool
MachThreadContext_arm::RefreshStateAfterStop ()
{
    success = EnableHardwareSingleStep(false) == KERN_SUCCESS;

    bool success = true;

    m_state.InvalidateRegisterSet (GPRRegSet);
    m_state.InvalidateRegisterSet (VFPRegSet);
    m_state.InvalidateRegisterSet (EXCRegSet);

    // Are we stepping a single instruction?
    if (ReadGPRRegisters(true) == KERN_SUCCESS)
    {
        // We are single stepping, was this the primary thread?
        if (m_thread.GetState() == eStateStepping)
        {
#if defined (DNB_ARCH_MACH_ARM_DEBUG_SW_STEP)
            success = EnableHardwareSingleStep(false) == KERN_SUCCESS;
            // Hardware single step must work if we are going to test software
            // single step functionality
            assert(success);
            if (m_hw_single_chained_step_addr == LLDB_INVALID_ADDRESS && m_sw_single_step_next_pc != LLDB_INVALID_ADDRESS)
            {
                uint32_t sw_step_next_pc = m_sw_single_step_next_pc & 0xFFFFFFFEu;
                bool sw_step_next_pc_is_thumb = (m_sw_single_step_next_pc & 1) != 0;
                bool actual_next_pc_is_thumb = (m_state.gpr.__cpsr & 0x20) != 0;
                if (m_state.gpr.r[15] != sw_step_next_pc)
                {
                    LogError("curr pc = 0x%8.8x - calculated single step target PC was incorrect: 0x%8.8x != 0x%8.8x", m_state.gpr.r[15], sw_step_next_pc, m_state.gpr.r[15]);
                    exit(1);
                }
                if (actual_next_pc_is_thumb != sw_step_next_pc_is_thumb)
                {
                    LogError("curr pc = 0x%8.8x - calculated single step calculated mode mismatch: sw single mode = %s != %s",
                                m_state.gpr.r[15],
                                actual_next_pc_is_thumb ? "Thumb" : "ARM",
                                sw_step_next_pc_is_thumb ? "Thumb" : "ARM");
                    exit(1);
                }
                m_sw_single_step_next_pc = LLDB_INVALID_ADDRESS;
            }
#else
#if defined (ENABLE_ARM_SINGLE_STEP)
            // Are we software single stepping?
            if (LLDB_BREAK_ID_IS_VALID(m_sw_single_step_break_id) || m_sw_single_step_itblock_break_count)
            {
                // Remove any software single stepping breakpoints that we have set

                // Do we have a normal software single step breakpoint?
                if (LLDB_BREAK_ID_IS_VALID(m_sw_single_step_break_id))
                {
                    ProcessMacOSXLog::LogIf(PD_LOG_STEP, "%s: removing software single step breakpoint (breakID=%d)", __FUNCTION__, m_sw_single_step_break_id);
                    success = m_thread.Process()->DisableBreakpoint(m_sw_single_step_break_id, true);
                    m_sw_single_step_break_id = LLDB_INVALID_BREAK_ID;
                }

                // Do we have any Thumb IT breakpoints?
                if (m_sw_single_step_itblock_break_count > 0)
                {
                    // See if we hit one of our Thumb IT breakpoints?
                    DNBBreakpoint *step_bp = m_thread.Process()->Breakpoints().FindByAddress(m_state.gpr.r[15]);

                    if (step_bp)
                    {
                        // We did hit our breakpoint, tell the breakpoint it was
                        // hit so that it can run its callback routine and fixup
                        // the PC.
                        ProcessMacOSXLog::LogIf(PD_LOG_STEP, "%s: IT software single step breakpoint hit (breakID=%u)", __FUNCTION__, step_bp->GetID());
                        step_bp->BreakpointHit(m_thread.Process()->ProcessID(), m_thread.GetID());
                    }

                    // Remove all Thumb IT breakpoints
                    for (int i = 0; i < m_sw_single_step_itblock_break_count; i++)
                    {
                        if (LLDB_BREAK_ID_IS_VALID(m_sw_single_step_itblock_break_id[i]))
                        {
                            ProcessMacOSXLog::LogIf(PD_LOG_STEP, "%s: removing IT software single step breakpoint (breakID=%d)", __FUNCTION__, m_sw_single_step_itblock_break_id[i]);
                            success = m_thread.Process()->DisableBreakpoint(m_sw_single_step_itblock_break_id[i], true);
                            m_sw_single_step_itblock_break_id[i] = LLDB_INVALID_BREAK_ID;
                        }
                    }
                    m_sw_single_step_itblock_break_count = 0;

                    // Decode instructions up to the current PC to ensure the internal decoder state is valid for the IT block
                    // The decoder has to decode each instruction in the IT block even if it is not executed so that
                    // the fields are correctly updated
                    DecodeITBlockInstructions(m_state.gpr.r[15]);
                }

            }
            else
#endif
                success = EnableHardwareSingleStep(false) == KERN_SUCCESS;
#endif
        }
        else
        {
            // The MachThread will automatically restore the suspend count
            // in ShouldStop (), so we don't need to do anything here if
            // we weren't the primary thread the last time
        }
    }
    return;
}



bool
MachThreadContext_arm::StepNotComplete ()
{
    if (m_hw_single_chained_step_addr != LLDB_INVALID_ADDRESS)
    {
        kern_return_t kret = KERN_INVALID_ARGUMENT;
        kret = ReadGPRRegisters(false);
        if (kret == KERN_SUCCESS)
        {
            if (m_state.gpr.r[15] == m_hw_single_chained_step_addr)
            {
                //ProcessMacOSXLog::LogIf(PD_LOG_STEP, "Need to step some more at 0x%8.8x", m_hw_single_chained_step_addr);
                return true;
            }
        }
    }

    m_hw_single_chained_step_addr = LLDB_INVALID_ADDRESS;
    return false;
}


void
MachThreadContext_arm::DecodeITBlockInstructions(lldb::addr_t curr_pc)

{
    uint16_t opcode16;
    uint32_t opcode32;
    lldb::addr_t next_pc_in_itblock;
    lldb::addr_t pc_in_itblock = m_last_decode_pc;

    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: last_decode_pc=0x%8.8x", __FUNCTION__, m_last_decode_pc);

    // Decode IT block instruction from the instruction following the m_last_decoded_instruction at
    // PC m_last_decode_pc upto and including the instruction at curr_pc
    if (m_thread.Process()->Task().ReadMemory(pc_in_itblock, 2, &opcode16) == 2)
    {
        opcode32 = opcode16;
        pc_in_itblock += 2;
        // Check for 32 bit thumb opcode and read the upper 16 bits if needed
        if (((opcode32 & 0xE000) == 0xE000) && opcode32 & 0x1800)
        {
            // Adjust 'next_pc_in_itblock' to point to the default next Thumb instruction for
            // a 32 bit Thumb opcode
            // Read bits 31:16 of a 32 bit Thumb opcode
            if (m_thread.Process()->Task().ReadMemory(pc_in_itblock, 2, &opcode16) == 2)
            {
                pc_in_itblock += 2;
                // 32 bit thumb opcode
                opcode32 = (opcode32 << 16) | opcode16;
            }
            else
            {
                LogError("%s: Unable to read opcode bits 31:16 for a 32 bit thumb opcode at pc=0x%8.8lx", __FUNCTION__, pc_in_itblock);
            }
        }
    }
    else
    {
        LogError("%s: Error reading 16-bit Thumb instruction at pc=0x%8.8x", __FUNCTION__, pc_in_itblock);
    }

    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: pc_in_itblock=0x%8.8x, curr_pc=0x%8.8x", __FUNCTION__, pc_in_itblock, curr_pc);

    next_pc_in_itblock = pc_in_itblock;
    while (next_pc_in_itblock <= curr_pc)
    {
        arm_error_t decodeError;

        m_last_decode_pc = pc_in_itblock;
        decodeError = DecodeInstructionUsingDisassembler(pc_in_itblock, m_state.gpr.__cpsr, &m_last_decode_arm, &m_last_decode_thumb, &next_pc_in_itblock);

        pc_in_itblock = next_pc_in_itblock;
        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: next_pc_in_itblock=0x%8.8x", __FUNCTION__, next_pc_in_itblock);
    }
}

#endif

// Set the single step bit in the processor status register.
kern_return_t
MachThreadContext_arm::EnableHardwareSingleStep (bool enable)
{
    Error err;
    LogSP log (ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_STEP));

    if (log) log->Printf("%s( enable = %d )", __FUNCTION__, enable);

    if (m_bvr0_reg == LLDB_INVALID_REGNUM || m_bcr0_reg == LLDB_INVALID_REGNUM)
        return KERN_INVALID_ARGUMENT;

    RegisterContext *reg_ctx = m_thread.GetRegisterContext().get();
    uint32_t bvr = 0;
    uint32_t bcr = 0;

    const uint32_t i = 0;
    if (enable)
    {
        m_hw_single_chained_step_addr = LLDB_INVALID_ADDRESS;

        // Save our previous state
        m_bvr0_save = reg_ctx->ReadRegisterAsUnsigned(m_bvr0_reg, 0);
        m_bcr0_save = reg_ctx->ReadRegisterAsUnsigned(m_bcr0_reg, 0);
        lldb::addr_t pc = reg_ctx->GetPC(LLDB_INVALID_ADDRESS);
        lldb::addr_t cpsr = reg_ctx->GetFlags(0);
        if (pc == LLDB_INVALID_ADDRESS)
            return KERN_INVALID_ARGUMENT;

        // Set a breakpoint that will stop when the PC doesn't match the current one!
        bvr = pc & 0xFFFFFFFCu;     // Set the current PC as the breakpoint address
        bcr = BCR_M_IMVA_MISMATCH | // Stop on address mismatch
              S_USER |                  // Stop only in user mode
              BCR_ENABLE;               // Enable this breakpoint
        if (cpsr & 0x20)
        {
            // Thumb breakpoint
            if (pc & 2)
                bcr |= BAS_IMVA_2_3;
            else
                bcr |= BAS_IMVA_0_1;

            uint16_t opcode;
            Error error;
            if (sizeof(opcode) == m_thread.GetProcess().ReadMemory(pc, &opcode, sizeof(opcode), error))
            {
                if (((opcode & 0xE000) == 0xE000) && opcode & 0x1800)
                {
                    // 32 bit thumb opcode...
                    if (pc & 2)
                    {
                        // We can't take care of a 32 bit thumb instruction single step
                        // with just IVA mismatching. We will need to chain an extra
                        // hardware single step in order to complete this single step...
                        m_hw_single_chained_step_addr = pc + 2;
                    }
                    else
                    {
                        // Extend the number of bits to ignore for the mismatch
                        bcr |= BAS_IMVA_ALL;
                    }
                }
            }
        }
        else
        {
            // ARM breakpoint
            bcr |= BAS_IMVA_ALL; // Stop when any address bits change
        }

        ProcessMacOSXLog::LogIf(PD_LOG_STEP, "%s: BVR%u=0x%8.8x  BCR%u=0x%8.8x", __FUNCTION__, i, bvr, i, bcr);

        m_bvr0_save = reg_ctx->ReadRegisterAsUnsigned(m_bvr0_reg, 0);
        m_bcr0_save = reg_ctx->ReadRegisterAsUnsigned(m_bcr0_reg, 0);

//        for (uint32_t j=i+1; j<16; ++j)
//        {
//          // Disable all others
//          m_state.dbg.bvr[j] = 0;
//          m_state.dbg.bcr[j] = 0;
//        }
    }
    else
    {
        // Just restore the state we had before we did single stepping
        bvr = m_bvr0_save;
        bcr = m_bcr0_save;
    }

    if (reg_ctx->WriteRegisterFromUnsigned(m_bvr0_reg, bvr) &&
        reg_ctx->WriteRegisterFromUnsigned(m_bcr0_reg, bcr))
        return KERN_SUCCESS;

    return KERN_INVALID_ARGUMENT;
}

#if defined (ENABLE_ARM_SINGLE_STEP)

// return 1 if bit "BIT" is set in "value"
static inline uint32_t bit(uint32_t value, uint32_t bit)
{
    return (value >> bit) & 1u;
}

// return the bitfield "value[msbit:lsbit]".
static inline uint32_t bits(uint32_t value, uint32_t msbit, uint32_t lsbit)
{
    assert(msbit >= lsbit);
    uint32_t shift_left = sizeof(value) * 8 - 1 - msbit;
    value <<= shift_left;           // shift anything above the msbit off of the unsigned edge
    value >>= shift_left + lsbit;   // shift it back again down to the lsbit (including undoing any shift from above)
    return value;                   // return our result
}

bool
MachThreadContext_arm::ConditionPassed(uint8_t condition, uint32_t cpsr)
{
    uint32_t cpsr_n = bit(cpsr, 31); // Negative condition code flag
    uint32_t cpsr_z = bit(cpsr, 30); // Zero condition code flag
    uint32_t cpsr_c = bit(cpsr, 29); // Carry condition code flag
    uint32_t cpsr_v = bit(cpsr, 28); // Overflow condition code flag

    switch (condition) {
        case COND_EQ: // (0x0)
            if (cpsr_z == 1) return true;
            break;
        case COND_NE: // (0x1)
            if (cpsr_z == 0) return true;
            break;
        case COND_CS: // (0x2)
            if (cpsr_c == 1) return true;
            break;
        case COND_CC: // (0x3)
            if (cpsr_c == 0) return true;
            break;
        case COND_MI: // (0x4)
            if (cpsr_n == 1) return true;
            break;
        case COND_PL: // (0x5)
            if (cpsr_n == 0) return true;
            break;
        case COND_VS: // (0x6)
            if (cpsr_v == 1) return true;
            break;
        case COND_VC: // (0x7)
            if (cpsr_v == 0) return true;
            break;
        case COND_HI: // (0x8)
            if ((cpsr_c == 1) && (cpsr_z == 0)) return true;
            break;
        case COND_LS: // (0x9)
            if ((cpsr_c == 0) || (cpsr_z == 1)) return true;
            break;
        case COND_GE: // (0xA)
            if (cpsr_n == cpsr_v) return true;
            break;
        case COND_LT: // (0xB)
            if (cpsr_n != cpsr_v) return true;
            break;
        case COND_GT: // (0xC)
            if ((cpsr_z == 0) && (cpsr_n == cpsr_v)) return true;
            break;
        case COND_LE: // (0xD)
            if ((cpsr_z == 1) || (cpsr_n != cpsr_v)) return true;
            break;
        default:
            return true;
            break;
    }

    return false;
}

bool
MachThreadContext_arm::ComputeNextPC(lldb::addr_t currentPC, arm_decoded_instruction_t decodedInstruction, bool currentPCIsThumb, lldb::addr_t *targetPC)
{
    lldb::addr_t myTargetPC, addressWherePCLives;
    lldb::pid_t mypid;

    uint32_t cpsr_c = bit(m_state.gpr.cpsr, 29); // Carry condition code flag

    uint32_t firstOperand=0, secondOperand=0, shiftAmount=0, secondOperandAfterShift=0, immediateValue=0;
    uint32_t halfwords=0, baseAddress=0, immediateOffset=0, addressOffsetFromRegister=0, addressOffsetFromRegisterAfterShift;
    uint32_t baseAddressIndex=LLDB_INVALID_INDEX32;
    uint32_t firstOperandIndex=LLDB_INVALID_INDEX32;
    uint32_t secondOperandIndex=LLDB_INVALID_INDEX32;
    uint32_t addressOffsetFromRegisterIndex=LLDB_INVALID_INDEX32;
    uint32_t shiftRegisterIndex=LLDB_INVALID_INDEX32;
    uint16_t registerList16, registerList16NoPC;
    uint8_t registerList8;
    uint32_t numRegistersToLoad=0;

    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: instruction->code=%d", __FUNCTION__, decodedInstruction.instruction->code);

    // Get the following in this switch statement:
    //   - firstOperand, secondOperand, immediateValue, shiftAmount: For arithmetic, logical and move instructions
    //   - baseAddress, immediateOffset, shiftAmount: For LDR
    //   - numRegistersToLoad: For LDM and POP instructions
    switch (decodedInstruction.instruction->code)
    {
            // Arithmetic operations that can change the PC
        case ARM_INST_ADC:
        case ARM_INST_ADCS:
        case ARM_INST_ADD:
        case ARM_INST_ADDS:
        case ARM_INST_AND:
        case ARM_INST_ANDS:
        case ARM_INST_ASR:
        case ARM_INST_ASRS:
        case ARM_INST_BIC:
        case ARM_INST_BICS:
        case ARM_INST_EOR:
        case ARM_INST_EORS:
        case ARM_INST_ORR:
        case ARM_INST_ORRS:
        case ARM_INST_RSB:
        case ARM_INST_RSBS:
        case ARM_INST_RSC:
        case ARM_INST_RSCS:
        case ARM_INST_SBC:
        case ARM_INST_SBCS:
        case ARM_INST_SUB:
        case ARM_INST_SUBS:
            switch (decodedInstruction.addressMode)
            {
                case ARM_ADDR_DATA_IMM:
                    if (decodedInstruction.numOperands != 3)
                    {
                        LogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=1)
                    firstOperandIndex = decodedInstruction.op[1].value; // first operand register index
                    firstOperand = m_state.gpr.r[firstOperandIndex];

                    // Get immediateValue (at index=2)
                    immediateValue = decodedInstruction.op[2].value;

                    break;

                case ARM_ADDR_DATA_REG:
                    if (decodedInstruction.numOperands != 3)
                    {
                        LogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=1)
                    firstOperandIndex = decodedInstruction.op[1].value; // first operand register index
                    firstOperand = m_state.gpr.r[firstOperandIndex];

                    // Get secondOperand register value (at index=2)
                    secondOperandIndex = decodedInstruction.op[2].value; // second operand register index
                    secondOperand = m_state.gpr.r[secondOperandIndex];

                    break;

                case ARM_ADDR_DATA_SCALED_IMM:
                    if (decodedInstruction.numOperands != 4)
                    {
                        LogError("Expected 4 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=1)
                    firstOperandIndex = decodedInstruction.op[1].value; // first operand register index
                    firstOperand = m_state.gpr.r[firstOperandIndex];

                    // Get secondOperand register value (at index=2)
                    secondOperandIndex = decodedInstruction.op[2].value; // second operand register index
                    secondOperand = m_state.gpr.r[secondOperandIndex];

                    // Get shiftAmount as immediate value (at index=3)
                    shiftAmount = decodedInstruction.op[3].value;

                    break;


                case ARM_ADDR_DATA_SCALED_REG:
                    if (decodedInstruction.numOperands != 4)
                    {
                        LogError("Expected 4 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=1)
                    firstOperandIndex = decodedInstruction.op[1].value; // first operand register index
                    firstOperand = m_state.gpr.r[firstOperandIndex];

                    // Get secondOperand register value (at index=2)
                    secondOperandIndex = decodedInstruction.op[2].value; // second operand register index
                    secondOperand = m_state.gpr.r[secondOperandIndex];

                    // Get shiftAmount from register (at index=3)
                    shiftRegisterIndex = decodedInstruction.op[3].value; // second operand register index
                    shiftAmount = m_state.gpr.r[shiftRegisterIndex];

                    break;

                case THUMB_ADDR_HR_HR:
                    if (decodedInstruction.numOperands != 2)
                    {
                        LogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get firstOperand register value (at index=0)
                    firstOperandIndex = decodedInstruction.op[0].value; // first operand register index
                    firstOperand = m_state.gpr.r[firstOperandIndex];

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[1].value; // second operand register index
                    secondOperand = m_state.gpr.r[secondOperandIndex];

                    break;

                default:
                    break;
            }
            break;

            // Logical shifts and move operations that can change the PC
        case ARM_INST_LSL:
        case ARM_INST_LSLS:
        case ARM_INST_LSR:
        case ARM_INST_LSRS:
        case ARM_INST_MOV:
        case ARM_INST_MOVS:
        case ARM_INST_MVN:
        case ARM_INST_MVNS:
        case ARM_INST_ROR:
        case ARM_INST_RORS:
        case ARM_INST_RRX:
        case ARM_INST_RRXS:
            // In these cases, the firstOperand is always 0, as if it does not exist
            switch (decodedInstruction.addressMode)
            {
                case ARM_ADDR_DATA_IMM:
                    if (decodedInstruction.numOperands != 2)
                    {
                        LogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get immediateValue (at index=1)
                    immediateValue = decodedInstruction.op[1].value;

                    break;

                case ARM_ADDR_DATA_REG:
                    if (decodedInstruction.numOperands != 2)
                    {
                        LogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[1].value; // second operand register index
                    secondOperand = m_state.gpr.r[secondOperandIndex];

                    break;

                case ARM_ADDR_DATA_SCALED_IMM:
                    if (decodedInstruction.numOperands != 3)
                    {
                        LogError("Expected 4 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[2].value; // second operand register index
                    secondOperand = m_state.gpr.r[secondOperandIndex];

                    // Get shiftAmount as immediate value (at index=2)
                    shiftAmount = decodedInstruction.op[2].value;

                    break;


                case ARM_ADDR_DATA_SCALED_REG:
                    if (decodedInstruction.numOperands != 3)
                    {
                        LogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[1].value; // second operand register index
                    secondOperand = m_state.gpr.r[secondOperandIndex];

                    // Get shiftAmount from register (at index=2)
                    shiftRegisterIndex = decodedInstruction.op[2].value; // second operand register index
                    shiftAmount = m_state.gpr.r[shiftRegisterIndex];

                    break;

                case THUMB_ADDR_HR_HR:
                    if (decodedInstruction.numOperands != 2)
                    {
                        LogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    if (decodedInstruction.op[0].value != PC_REG)
                    {
                        LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                        return false;
                    }

                    // Get secondOperand register value (at index=1)
                    secondOperandIndex = decodedInstruction.op[1].value; // second operand register index
                    secondOperand = m_state.gpr.r[secondOperandIndex];

                    break;

                default:
                    break;
            }

            break;

            // Simple branches, used to hop around within a routine
        case ARM_INST_B:
            *targetPC = decodedInstruction.targetPC; // Known targetPC
            return true;
            break;

            // Branch-and-link, used to call ARM subroutines
        case ARM_INST_BL:
            *targetPC = decodedInstruction.targetPC; // Known targetPC
            return true;
            break;

            // Branch-and-link with exchange, used to call opposite-mode subroutines
        case ARM_INST_BLX:
            if ((decodedInstruction.addressMode == ARM_ADDR_BRANCH_IMM) ||
                (decodedInstruction.addressMode == THUMB_ADDR_UNCOND))
            {
                *targetPC = decodedInstruction.targetPC; // Known targetPC
                return true;
            }
            else    // addressMode == ARM_ADDR_BRANCH_REG
            {
                // Unknown target unless we're branching to the PC itself,
                //  although this may not work properly with BLX
                if (decodedInstruction.op[REG_RD].value == PC_REG)
                {
                    // this should (almost) never happen
                    *targetPC = decodedInstruction.targetPC; // Known targetPC
                    return true;
                }

                // Get the branch address and return
                if (decodedInstruction.numOperands != 1)
                {
                    LogError("Expected 1 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                    return false;
                }

                // Get branch address in register (at index=0)
                *targetPC = m_state.gpr.r[decodedInstruction.op[0].value];
                return true;
            }
            break;

            // Branch with exchange, used to hop to opposite-mode code
            // Branch to Jazelle code, used to execute Java; included here since it
            //  acts just like BX unless the Jazelle unit is active and JPC is
            //  already loaded into it.
        case ARM_INST_BX:
        case ARM_INST_BXJ:
            // Unknown target unless we're branching to the PC itself,
            //  although this can never switch to Thumb mode and is
            //  therefore pretty much useless
            if (decodedInstruction.op[REG_RD].value == PC_REG)
            {
                // this should (almost) never happen
                *targetPC = decodedInstruction.targetPC; // Known targetPC
                return true;
            }

            // Get the branch address and return
            if (decodedInstruction.numOperands != 1)
            {
                LogError("Expected 1 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                return false;
            }

            // Get branch address in register (at index=0)
            *targetPC = m_state.gpr.r[decodedInstruction.op[0].value];
            return true;
            break;

            // Compare and branch on zero/non-zero (Thumb-16 only)
            // Unusual condition check built into the instruction
        case ARM_INST_CBZ:
        case ARM_INST_CBNZ:
            // Branch address is known at compile time
            // Get the branch address and return
            if (decodedInstruction.numOperands != 2)
            {
                LogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                return false;
            }

            // Get branch address as an immediate value (at index=1)
            *targetPC = decodedInstruction.op[1].value;
            return true;
            break;

            // Load register can be used to load PC, usually with a function pointer
        case ARM_INST_LDR:
            if (decodedInstruction.op[REG_RD].value != PC_REG)
            {
                LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                return false;
            }
            switch (decodedInstruction.addressMode)
            {
                case ARM_ADDR_LSWUB_IMM:
                case ARM_ADDR_LSWUB_IMM_PRE:
                case ARM_ADDR_LSWUB_IMM_POST:
                    if (decodedInstruction.numOperands != 3)
                    {
                        LogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    // Get baseAddress from register (at index=1)
                    baseAddressIndex = decodedInstruction.op[1].value;
                    baseAddress = m_state.gpr.r[baseAddressIndex];

                    // Get immediateOffset (at index=2)
                    immediateOffset = decodedInstruction.op[2].value;
                    break;

                case ARM_ADDR_LSWUB_REG:
                case ARM_ADDR_LSWUB_REG_PRE:
                case ARM_ADDR_LSWUB_REG_POST:
                    if (decodedInstruction.numOperands != 3)
                    {
                        LogError("Expected 3 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    // Get baseAddress from register (at index=1)
                    baseAddressIndex = decodedInstruction.op[1].value;
                    baseAddress = m_state.gpr.r[baseAddressIndex];

                    // Get immediateOffset from register (at index=2)
                    addressOffsetFromRegisterIndex = decodedInstruction.op[2].value;
                    addressOffsetFromRegister = m_state.gpr.r[addressOffsetFromRegisterIndex];

                    break;

                case ARM_ADDR_LSWUB_SCALED:
                case ARM_ADDR_LSWUB_SCALED_PRE:
                case ARM_ADDR_LSWUB_SCALED_POST:
                    if (decodedInstruction.numOperands != 4)
                    {
                        LogError("Expected 4 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                        return false;
                    }

                    // Get baseAddress from register (at index=1)
                    baseAddressIndex = decodedInstruction.op[1].value;
                    baseAddress = m_state.gpr.r[baseAddressIndex];

                    // Get immediateOffset from register (at index=2)
                    addressOffsetFromRegisterIndex = decodedInstruction.op[2].value;
                    addressOffsetFromRegister = m_state.gpr.r[addressOffsetFromRegisterIndex];

                    // Get shiftAmount (at index=3)
                    shiftAmount = decodedInstruction.op[3].value;

                    break;

                default:
                    break;
            }
            break;

            // 32b load multiple operations can load the PC along with everything else,
            //  usually to return from a function call
        case ARM_INST_LDMDA:
        case ARM_INST_LDMDB:
        case ARM_INST_LDMIA:
        case ARM_INST_LDMIB:
            if (decodedInstruction.op[LDM_REGLIST].value & PC_REGLIST_BIT)
            {
                if (decodedInstruction.numOperands != 2)
                {
                    LogError("Expected 2 operands in decoded instruction structure. numOperands is %d!", decodedInstruction.numOperands);
                    return false;
                }

                // Get baseAddress from register (at index=0)
                baseAddressIndex = decodedInstruction.op[0].value;
                baseAddress = m_state.gpr.r[baseAddressIndex];

                // Get registerList from register (at index=1)
                registerList16 = (uint16_t)decodedInstruction.op[1].value;

                // Count number of registers to load in the multiple register list excluding the PC
                registerList16NoPC = registerList16&0x3FFF; // exclude the PC
                numRegistersToLoad=0;
                for (int i = 0; i < 16; i++)
                {
                    if (registerList16NoPC & 0x1) numRegistersToLoad++;
                    registerList16NoPC = registerList16NoPC >> 1;
                }
            }
            else
            {
                LogError("Destination register is not a PC! %s routine should be called on on instructions that modify the PC. Destination register is R%d!", __FUNCTION__, decodedInstruction.op[0].value);
                return false;
            }
            break;

            // Normal 16-bit LD multiple can't touch R15, but POP can
        case ARM_INST_POP:  // Can also get the PC & updates SP
            // Get baseAddress from SP (at index=0)
            baseAddress = m_state.gpr.__sp;

            if (decodedInstruction.thumb16b)
            {
                // Get registerList from register (at index=0)
                registerList8 = (uint8_t)decodedInstruction.op[0].value;

                // Count number of registers to load in the multiple register list
                numRegistersToLoad=0;
                for (int i = 0; i < 8; i++)
                {
                    if (registerList8 & 0x1) numRegistersToLoad++;
                    registerList8 = registerList8 >> 1;
                }
            }
            else
            {
                // Get registerList from register (at index=0)
                registerList16 = (uint16_t)decodedInstruction.op[0].value;

                // Count number of registers to load in the multiple register list excluding the PC
                registerList16NoPC = registerList16&0x3FFF; // exclude the PC
                numRegistersToLoad=0;
                for (int i = 0; i < 16; i++)
                {
                    if (registerList16NoPC & 0x1) numRegistersToLoad++;
                    registerList16NoPC = registerList16NoPC >> 1;
                }
            }
            break;

            // 16b TBB and TBH instructions load a jump address from a table
        case ARM_INST_TBB:
        case ARM_INST_TBH:
            // Get baseAddress from register (at index=0)
            baseAddressIndex = decodedInstruction.op[0].value;
            baseAddress = m_state.gpr.r[baseAddressIndex];

            // Get immediateOffset from register (at index=1)
            addressOffsetFromRegisterIndex = decodedInstruction.op[1].value;
            addressOffsetFromRegister = m_state.gpr.r[addressOffsetFromRegisterIndex];
            break;

            // ThumbEE branch-to-handler instructions: Jump to handlers at some offset
            //  from a special base pointer register (which is unknown at disassembly time)
        case ARM_INST_HB:
        case ARM_INST_HBP:
//          TODO: ARM_INST_HB, ARM_INST_HBP
            break;

        case ARM_INST_HBL:
        case ARM_INST_HBLP:
//          TODO: ARM_INST_HBL, ARM_INST_HBLP
            break;

            // Breakpoint and software interrupt jump to interrupt handler (always ARM)
        case ARM_INST_BKPT:
        case ARM_INST_SMC:
        case ARM_INST_SVC:

            // Return from exception, obviously modifies PC [interrupt only!]
        case ARM_INST_RFEDA:
        case ARM_INST_RFEDB:
        case ARM_INST_RFEIA:
        case ARM_INST_RFEIB:

            // Other instructions either can't change R15 or are "undefined" if you do,
            //  so no sane compiler should ever generate them & we don't care here.
            //  Also, R15 can only legally be used in a read-only manner for the
            //  various ARM addressing mode (to get PC-relative addressing of constants),
            //  but can NOT be used with any of the update modes.
        default:
            LogError("%s should not be called for instruction code %d!", __FUNCTION__, decodedInstruction.instruction->code);
            return false;
            break;
    }

    // Adjust PC if PC is one of the input operands
    if (baseAddressIndex == PC_REG)
    {
        if (currentPCIsThumb)
            baseAddress += 4;
        else
            baseAddress += 8;
    }

    if (firstOperandIndex == PC_REG)
    {
        if (currentPCIsThumb)
            firstOperand += 4;
        else
            firstOperand += 8;
    }

    if (secondOperandIndex == PC_REG)
    {
        if (currentPCIsThumb)
            secondOperand += 4;
        else
            secondOperand += 8;
    }

    if (addressOffsetFromRegisterIndex == PC_REG)
    {
        if (currentPCIsThumb)
            addressOffsetFromRegister += 4;
        else
            addressOffsetFromRegister += 8;
    }

    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE,
        "%s: firstOperand=%8.8x, secondOperand=%8.8x, immediateValue = %d, shiftAmount = %d, baseAddress = %8.8x, addressOffsetFromRegister = %8.8x, immediateOffset = %d, numRegistersToLoad = %d",
        __FUNCTION__,
        firstOperand,
        secondOperand,
        immediateValue,
        shiftAmount,
        baseAddress,
        addressOffsetFromRegister,
        immediateOffset,
        numRegistersToLoad);


    // Calculate following values after applying shiftAmount:
    //   - immediateOffsetAfterShift, secondOperandAfterShift

    switch (decodedInstruction.scaleMode)
    {
        case ARM_SCALE_NONE:
            addressOffsetFromRegisterAfterShift = addressOffsetFromRegister;
            secondOperandAfterShift = secondOperand;
            break;

        case ARM_SCALE_LSL:             // Logical shift left
            addressOffsetFromRegisterAfterShift = addressOffsetFromRegister << shiftAmount;
            secondOperandAfterShift = secondOperand << shiftAmount;
            break;

        case ARM_SCALE_LSR:             // Logical shift right
            addressOffsetFromRegisterAfterShift = addressOffsetFromRegister >> shiftAmount;
            secondOperandAfterShift = secondOperand >> shiftAmount;
            break;

        case ARM_SCALE_ASR:             // Arithmetic shift right
            asm("mov %0, %1, asr %2" : "=r" (addressOffsetFromRegisterAfterShift) : "r" (addressOffsetFromRegister), "r" (shiftAmount));
            asm("mov %0, %1, asr %2" : "=r" (secondOperandAfterShift) : "r" (secondOperand), "r" (shiftAmount));
            break;

        case ARM_SCALE_ROR:             // Rotate right
            asm("mov %0, %1, ror %2" : "=r" (addressOffsetFromRegisterAfterShift) : "r" (addressOffsetFromRegister), "r" (shiftAmount));
            asm("mov %0, %1, ror %2" : "=r" (secondOperandAfterShift) : "r" (secondOperand), "r" (shiftAmount));
            break;

        case ARM_SCALE_RRX:             // Rotate right, pulling in carry (1-bit shift only)
            asm("mov %0, %1, rrx" : "=r" (addressOffsetFromRegisterAfterShift) : "r" (addressOffsetFromRegister));
            asm("mov %0, %1, rrx" : "=r" (secondOperandAfterShift) : "r" (secondOperand));
            break;
    }

    // Emulate instruction to calculate targetPC
    // All branches are already handled in the first switch statement. A branch should not reach this switch
    switch (decodedInstruction.instruction->code)
    {
            // Arithmetic operations that can change the PC
        case ARM_INST_ADC:
        case ARM_INST_ADCS:
            // Add with Carry
            *targetPC = firstOperand + (secondOperandAfterShift + immediateValue) + cpsr_c;
            break;

        case ARM_INST_ADD:
        case ARM_INST_ADDS:
            *targetPC = firstOperand + (secondOperandAfterShift + immediateValue);
            break;

        case ARM_INST_AND:
        case ARM_INST_ANDS:
            *targetPC = firstOperand & (secondOperandAfterShift + immediateValue);
            break;

        case ARM_INST_ASR:
        case ARM_INST_ASRS:
            asm("mov %0, %1, asr %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_BIC:
        case ARM_INST_BICS:
            asm("bic %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_EOR:
        case ARM_INST_EORS:
            asm("eor %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_ORR:
        case ARM_INST_ORRS:
            asm("orr %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_RSB:
        case ARM_INST_RSBS:
            asm("rsb %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_RSC:
        case ARM_INST_RSCS:
            myTargetPC = secondOperandAfterShift - (firstOperand + !cpsr_c);
            *targetPC = myTargetPC;
            break;

        case ARM_INST_SBC:
        case ARM_INST_SBCS:
            asm("sbc %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue  + !cpsr_c));
            *targetPC = myTargetPC;
            break;

        case ARM_INST_SUB:
        case ARM_INST_SUBS:
            asm("sub %0, %1, %2" : "=r" (myTargetPC) : "r" (firstOperand), "r" (secondOperandAfterShift + immediateValue));
            *targetPC = myTargetPC;
            break;

            // Logical shifts and move operations that can change the PC
        case ARM_INST_LSL:
        case ARM_INST_LSLS:
        case ARM_INST_LSR:
        case ARM_INST_LSRS:
        case ARM_INST_MOV:
        case ARM_INST_MOVS:
        case ARM_INST_ROR:
        case ARM_INST_RORS:
        case ARM_INST_RRX:
        case ARM_INST_RRXS:
            myTargetPC = secondOperandAfterShift + immediateValue;
            *targetPC = myTargetPC;
            break;

        case ARM_INST_MVN:
        case ARM_INST_MVNS:
            myTargetPC = !(secondOperandAfterShift + immediateValue);
            *targetPC = myTargetPC;
            break;

            // Load register can be used to load PC, usually with a function pointer
        case ARM_INST_LDR:
            switch (decodedInstruction.addressMode) {
                case ARM_ADDR_LSWUB_IMM_POST:
                case ARM_ADDR_LSWUB_REG_POST:
                case ARM_ADDR_LSWUB_SCALED_POST:
                    addressWherePCLives = baseAddress;
                    break;

                case ARM_ADDR_LSWUB_IMM:
                case ARM_ADDR_LSWUB_REG:
                case ARM_ADDR_LSWUB_SCALED:
                case ARM_ADDR_LSWUB_IMM_PRE:
                case ARM_ADDR_LSWUB_REG_PRE:
                case ARM_ADDR_LSWUB_SCALED_PRE:
                    addressWherePCLives = baseAddress + (addressOffsetFromRegisterAfterShift + immediateOffset);
                    break;

                default:
                    break;
            }

            mypid = m_thread.ProcessID();
            if (PDProcessMemoryRead(mypid, addressWherePCLives, sizeof(lldb::addr_t), &myTargetPC) !=  sizeof(lldb::addr_t))
            {
                LogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

            // 32b load multiple operations can load the PC along with everything else,
            //  usually to return from a function call
        case ARM_INST_LDMDA:
            mypid = m_thread.ProcessID();
            addressWherePCLives = baseAddress;
            if (PDProcessMemoryRead(mypid, addressWherePCLives, sizeof(lldb::addr_t), &myTargetPC) !=  sizeof(lldb::addr_t))
            {
                LogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

        case ARM_INST_LDMDB:
            mypid = m_thread.ProcessID();
            addressWherePCLives = baseAddress - 4;
            if (PDProcessMemoryRead(mypid, addressWherePCLives, sizeof(lldb::addr_t), &myTargetPC) !=  sizeof(lldb::addr_t))
            {
                LogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

        case ARM_INST_LDMIB:
            mypid = m_thread.ProcessID();
            addressWherePCLives = baseAddress + numRegistersToLoad*4 + 4;
            if (PDProcessMemoryRead(mypid, addressWherePCLives, sizeof(lldb::addr_t), &myTargetPC) !=  sizeof(lldb::addr_t))
            {
                LogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

        case ARM_INST_LDMIA: // same as pop
            // Normal 16-bit LD multiple can't touch R15, but POP can
        case ARM_INST_POP:  // Can also get the PC & updates SP
            mypid = m_thread.ProcessID();
            addressWherePCLives = baseAddress + numRegistersToLoad*4;
            if (PDProcessMemoryRead(mypid, addressWherePCLives, sizeof(lldb::addr_t), &myTargetPC) !=  sizeof(lldb::addr_t))
            {
                LogError("Could not read memory at %8.8x to get targetPC when processing the pop instruction!", addressWherePCLives);
                return false;
            }

            *targetPC = myTargetPC;
            break;

            // 16b TBB and TBH instructions load a jump address from a table
        case ARM_INST_TBB:
            mypid = m_thread.ProcessID();
            addressWherePCLives = baseAddress + addressOffsetFromRegisterAfterShift;
            if (PDProcessMemoryRead(mypid, addressWherePCLives, 1, &halfwords) !=  1)
            {
                LogError("Could not read memory at %8.8x to get targetPC when processing the TBB instruction!", addressWherePCLives);
                return false;
            }
            // add 4 to currentPC since we are in Thumb mode and then add 2*halfwords
            *targetPC = (currentPC + 4) + 2*halfwords;
            break;

        case ARM_INST_TBH:
            mypid = m_thread.ProcessID();
            addressWherePCLives = ((baseAddress + (addressOffsetFromRegisterAfterShift << 1)) & ~0x1);
            if (PDProcessMemoryRead(mypid, addressWherePCLives, 2, &halfwords) !=  2)
            {
                LogError("Could not read memory at %8.8x to get targetPC when processing the TBH instruction!", addressWherePCLives);
                return false;
            }
            // add 4 to currentPC since we are in Thumb mode and then add 2*halfwords
            *targetPC = (currentPC + 4) + 2*halfwords;
            break;

            // ThumbEE branch-to-handler instructions: Jump to handlers at some offset
            //  from a special base pointer register (which is unknown at disassembly time)
        case ARM_INST_HB:
        case ARM_INST_HBP:
            //          TODO: ARM_INST_HB, ARM_INST_HBP
            break;

        case ARM_INST_HBL:
        case ARM_INST_HBLP:
            //          TODO: ARM_INST_HBL, ARM_INST_HBLP
            break;

            // Breakpoint and software interrupt jump to interrupt handler (always ARM)
        case ARM_INST_BKPT:
        case ARM_INST_SMC:
        case ARM_INST_SVC:
            //          TODO: ARM_INST_BKPT, ARM_INST_SMC, ARM_INST_SVC
            break;

            // Return from exception, obviously modifies PC [interrupt only!]
        case ARM_INST_RFEDA:
        case ARM_INST_RFEDB:
        case ARM_INST_RFEIA:
        case ARM_INST_RFEIB:
            //          TODO: ARM_INST_RFEDA, ARM_INST_RFEDB, ARM_INST_RFEIA, ARM_INST_RFEIB
            break;

            // Other instructions either can't change R15 or are "undefined" if you do,
            //  so no sane compiler should ever generate them & we don't care here.
            //  Also, R15 can only legally be used in a read-only manner for the
            //  various ARM addressing mode (to get PC-relative addressing of constants),
            //  but can NOT be used with any of the update modes.
        default:
            LogError("%s should not be called for instruction code %d!", __FUNCTION__, decodedInstruction.instruction->code);
            return false;
            break;
    }

    return true;
}

void
MachThreadContext_arm::EvaluateNextInstructionForSoftwareBreakpointSetup(lldb::addr_t currentPC, uint32_t cpsr, bool currentPCIsThumb, lldb::addr_t *nextPC, bool *nextPCIsThumb)
{
    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "MachThreadContext_arm::EvaluateNextInstructionForSoftwareBreakpointSetup() called");

    lldb::addr_t targetPC = LLDB_INVALID_ADDRESS;
    uint32_t registerValue;
    arm_error_t decodeError;
    lldb::addr_t currentPCInITBlock, nextPCInITBlock;
    int i;
    bool last_decoded_instruction_executes = true;

    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: default nextPC=0x%8.8x (%s)", __FUNCTION__, *nextPC, *nextPCIsThumb ? "Thumb" : "ARM");

    // Update *nextPC and *nextPCIsThumb for special cases
    if (m_last_decode_thumb.itBlockRemaining) // we are in an IT block
    {
        // Set the nextPC to the PC of the instruction which will execute in the IT block
        // If none of the instruction execute in the IT block based on the condition flags,
        // then point to the instruction immediately following the IT block
        const int itBlockRemaining = m_last_decode_thumb.itBlockRemaining;
        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: itBlockRemaining=%8.8x", __FUNCTION__, itBlockRemaining);

        // Determine the PC at which the next instruction resides
        if (m_last_decode_arm.thumb16b)
            currentPCInITBlock = currentPC + 2;
        else
            currentPCInITBlock = currentPC + 4;

        for (i = 0; i < itBlockRemaining; i++)
        {
            ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: currentPCInITBlock=%8.8x", __FUNCTION__, currentPCInITBlock);
            decodeError = DecodeInstructionUsingDisassembler(currentPCInITBlock, cpsr, &m_last_decode_arm, &m_last_decode_thumb, &nextPCInITBlock);

            if (decodeError != ARM_SUCCESS)
                LogError("unable to disassemble instruction at 0x%8.8lx", currentPCInITBlock);

            ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: condition=%d", __FUNCTION__, m_last_decode_arm.condition);
            if (ConditionPassed(m_last_decode_arm.condition, cpsr))
            {
                ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: Condition codes matched for instruction %d", __FUNCTION__, i);
                break; // break from the for loop
            }
            else
            {
                ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: Condition codes DID NOT matched for instruction %d", __FUNCTION__, i);
            }

            // update currentPC and nextPCInITBlock
            currentPCInITBlock = nextPCInITBlock;
        }

        if (i == itBlockRemaining) // We came out of the IT block without executing any instructions
            last_decoded_instruction_executes = false;

        *nextPC = currentPCInITBlock;
        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: After IT block step-through: *nextPC=%8.8x", __FUNCTION__, *nextPC);
    }

    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE,
                    "%s: cpsr = %8.8x, thumb16b = %d, thumb = %d, branch = %d, conditional = %d, knownTarget = %d, links = %d, canSwitchMode = %d, doesSwitchMode = %d",
                    __FUNCTION__,
                    cpsr,
                    m_last_decode_arm.thumb16b,
                    m_last_decode_arm.thumb,
                    m_last_decode_arm.branch,
                    m_last_decode_arm.conditional,
                    m_last_decode_arm.knownTarget,
                    m_last_decode_arm.links,
                    m_last_decode_arm.canSwitchMode,
                    m_last_decode_arm.doesSwitchMode);


    if (last_decoded_instruction_executes &&                    // Was this a conditional instruction that did execute?
        m_last_decode_arm.branch &&                             // Can this instruction change the PC?
        (m_last_decode_arm.instruction->code != ARM_INST_SVC))  // If this instruction is not an SVC instruction
    {
        // Set targetPC. Compute if needed.
        if (m_last_decode_arm.knownTarget)
        {
            // Fixed, known PC-relative
            targetPC = m_last_decode_arm.targetPC;
        }
        else
        {
            // if targetPC is not known at compile time (PC-relative target), compute targetPC
            if (!ComputeNextPC(currentPC, m_last_decode_arm, currentPCIsThumb, &targetPC))
            {
                LogError("%s: Unable to compute targetPC for instruction at 0x%8.8lx", __FUNCTION__, currentPC);
                targetPC = LLDB_INVALID_ADDRESS;
            }
        }

        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: targetPC=0x%8.8x, cpsr=0x%8.8x, condition=0x%hhx", __FUNCTION__, targetPC, cpsr, m_last_decode_arm.condition);

        // Refine nextPC computation
        if ((m_last_decode_arm.instruction->code == ARM_INST_CBZ) ||
            (m_last_decode_arm.instruction->code == ARM_INST_CBNZ))
        {
            // Compare and branch on zero/non-zero (Thumb-16 only)
            // Unusual condition check built into the instruction
            registerValue = m_state.gpr.r[m_last_decode_arm.op[REG_RD].value];

            if (m_last_decode_arm.instruction->code == ARM_INST_CBZ)
            {
                if (registerValue == 0)
                    *nextPC = targetPC;
            }
            else
            {
                if (registerValue != 0)
                    *nextPC = targetPC;
            }
        }
        else if (m_last_decode_arm.conditional) // Is the change conditional on flag results?
        {
            if (ConditionPassed(m_last_decode_arm.condition, cpsr)) // conditions match
            {
                ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: Condition matched!", __FUNCTION__);
                *nextPC = targetPC;
            }
            else
            {
                ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: Condition did not match!", __FUNCTION__);
            }
        }
        else
        {
            *nextPC = targetPC;
        }

        // Refine nextPCIsThumb computation
        if (m_last_decode_arm.doesSwitchMode)
        {
            *nextPCIsThumb = !currentPCIsThumb;
        }
        else if (m_last_decode_arm.canSwitchMode)
        {
            // Legal to switch ARM <--> Thumb mode with this branch
            // dependent on bit[0] of targetPC
            *nextPCIsThumb = (*nextPC & 1u) != 0;
        }
        else
        {
            *nextPCIsThumb = currentPCIsThumb;
        }
    }

    ProcessMacOSXLog::LogIf(PD_LOG_STEP, "%s: calculated nextPC=0x%8.8x (%s)", __FUNCTION__, *nextPC, *nextPCIsThumb ? "Thumb" : "ARM");
}


arm_error_t
MachThreadContext_arm::DecodeInstructionUsingDisassembler(lldb::addr_t curr_pc, uint32_t curr_cpsr, arm_decoded_instruction_t *decodedInstruction, thumb_static_data_t *thumbStaticData, lldb::addr_t *next_pc)
{

    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: pc=0x%8.8x, cpsr=0x%8.8x", __FUNCTION__, curr_pc, curr_cpsr);

    const uint32_t isetstate_mask = MASK_CPSR_T | MASK_CPSR_J;
    const uint32_t curr_isetstate = curr_cpsr & isetstate_mask;
    uint32_t opcode32;
    lldb::addr_t nextPC = curr_pc;
    arm_error_t decodeReturnCode = ARM_SUCCESS;

    m_last_decode_pc = curr_pc;
    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: last_decode_pc=0x%8.8x", __FUNCTION__, m_last_decode_pc);

    switch (curr_isetstate) {
        case 0x0: // ARM Instruction
            // Read the ARM opcode
            if (m_thread.Process()->Task().ReadMemory(curr_pc, 4, &opcode32) != 4)
            {
                LogError("unable to read opcode bits 31:0 for an ARM opcode at 0x%8.8lx", curr_pc);
                decodeReturnCode = ARM_ERROR;
            }
            else
            {
                nextPC += 4;
                decodeReturnCode = ArmDisassembler((uint64_t)curr_pc, opcode32, false, decodedInstruction, NULL, 0, NULL, 0);

                if (decodeReturnCode != ARM_SUCCESS)
                    LogError("Unable to decode ARM instruction 0x%8.8x at 0x%8.8lx", opcode32, curr_pc);
            }
            break;

        case 0x20: // Thumb Instruction
            uint16_t opcode16;
            // Read the a 16 bit Thumb opcode
            if (m_thread.Process()->Task().ReadMemory(curr_pc, 2, &opcode16) != 2)
            {
                LogError("unable to read opcode bits 15:0 for a thumb opcode at 0x%8.8lx", curr_pc);
                decodeReturnCode = ARM_ERROR;
            }
            else
            {
                nextPC += 2;
                opcode32 = opcode16;

                decodeReturnCode = ThumbDisassembler((uint64_t)curr_pc, opcode16, false, false, thumbStaticData, decodedInstruction, NULL, 0, NULL, 0);

                switch (decodeReturnCode) {
                    case ARM_SKIP:
                        // 32 bit thumb opcode
                        nextPC += 2;
                        if (m_thread.Process()->Task().ReadMemory(curr_pc+2, 2, &opcode16) != 2)
                        {
                            LogError("unable to read opcode bits 15:0 for a thumb opcode at 0x%8.8lx", curr_pc+2);
                        }
                        else
                        {
                            opcode32 = (opcode32 << 16) | opcode16;

                            decodeReturnCode = ThumbDisassembler((uint64_t)(curr_pc+2), opcode16, false, false, thumbStaticData, decodedInstruction, NULL, 0, NULL, 0);

                            if (decodeReturnCode != ARM_SUCCESS)
                                LogError("Unable to decode 2nd half of Thumb instruction 0x%8.4hx at 0x%8.8lx", opcode16, curr_pc+2);
                            break;
                        }
                        break;

                    case ARM_SUCCESS:
                        // 16 bit thumb opcode; at this point we are done decoding the opcode
                        break;

                    default:
                        LogError("Unable to decode Thumb instruction 0x%8.4hx at 0x%8.8lx", opcode16, curr_pc);
                        decodeReturnCode = ARM_ERROR;
                        break;
                }
            }
            break;

        default:
            break;
    }

    if (next_pc)
        *next_pc = nextPC;

    return decodeReturnCode;
}

bool
MachThreadContext_arm::BreakpointHit(lldb::pid_t pid, lldb::tid_t tid, lldb::user_id_t breakID, void *baton)
{
    lldb::addr_t bkpt_pc = (lldb::addr_t)baton;
    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s(pid = %i, tid = %4.4x, breakID = %u, baton = %p): Setting PC to 0x%8.8x", __FUNCTION__, pid, tid, breakID, baton, bkpt_pc);
    return PDThreadSetRegisterValueByID(pid, tid, eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC, bkpt_pc);
}

// Set the single step bit in the processor status register.
kern_return_t
MachThreadContext_arm::SetSingleStepSoftwareBreakpoints()
{
    Error err;
    err = ReadGPRRegisters(false);

    if (err.Fail())
    {
        err.Log("%s: failed to read the GPR registers", __FUNCTION__);
        return err.GetError();
    }

    lldb::addr_t curr_pc = m_state.gpr.r[15];
    uint32_t curr_cpsr = m_state.gpr.__cpsr;
    lldb::addr_t next_pc = curr_pc;

    bool curr_pc_is_thumb = (m_state.gpr.__cpsr & 0x20) != 0;
    bool next_pc_is_thumb = curr_pc_is_thumb;

    uint32_t curr_itstate = ((curr_cpsr & 0x6000000) >> 25) | ((curr_cpsr & 0xFC00) >> 8);
    bool inITBlock = (curr_itstate & 0xF) ? 1 : 0;
    bool lastInITBlock = ((curr_itstate & 0xF) == 0x8) ? 1 : 0;

    ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: curr_pc=0x%8.8x (%s), curr_itstate=0x%x, inITBlock=%d, lastInITBlock=%d", __FUNCTION__, curr_pc, curr_pc_is_thumb ? "Thumb" : "ARM", curr_itstate, inITBlock, lastInITBlock);

    // If the instruction is not in the IT block, then decode using the Disassembler and compute next_pc
    if (!inITBlock)
    {
        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: Decoding an instruction NOT in the IT block", __FUNCTION__);

        arm_error_t decodeReturnCode =  DecodeInstructionUsingDisassembler(curr_pc, curr_cpsr, &m_last_decode_arm, &m_last_decode_thumb, &next_pc);

        if (decodeReturnCode != ARM_SUCCESS)
        {
            err = KERN_INVALID_ARGUMENT;
            LogError("MachThreadContext_arm::SetSingleStepSoftwareBreakpoints: Unable to disassemble instruction at 0x%8.8lx", curr_pc);
        }
    }
    else
    {
        next_pc = curr_pc + ((m_last_decode_arm.thumb16b) ? 2 : 4);
    }

    // Instruction is NOT in the IT block OR
    if (!inITBlock)
    {
        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: normal instruction", __FUNCTION__);
        EvaluateNextInstructionForSoftwareBreakpointSetup(curr_pc, m_state.gpr.__cpsr, curr_pc_is_thumb, &next_pc, &next_pc_is_thumb);
    }
    else if (inITBlock && !m_last_decode_arm.setsFlags)
    {
        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: IT instruction that doesn't set flags", __FUNCTION__);
        EvaluateNextInstructionForSoftwareBreakpointSetup(curr_pc, m_state.gpr.__cpsr, curr_pc_is_thumb, &next_pc, &next_pc_is_thumb);
    }
    else if (lastInITBlock && m_last_decode_arm.branch)
    {
        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: IT instruction which last in the IT block and is a branch", __FUNCTION__);
        EvaluateNextInstructionForSoftwareBreakpointSetup(curr_pc, m_state.gpr.__cpsr, curr_pc_is_thumb, &next_pc, &next_pc_is_thumb);
    }
    else
    {
        // Instruction is in IT block and can modify the CPSR flags
        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: IT instruction that sets flags", __FUNCTION__);

        // NOTE: When this point of code is reached, the instruction at curr_pc has already been decoded
        // inside the function ShouldStop (). Therefore m_last_decode_arm, m_last_decode_thumb
        // reflect the decoded instruction at curr_pc

        // If we find an instruction inside the IT block which will set/modify the condition flags (NZCV bits in CPSR),
        // we set breakpoints at all remaining instructions inside the IT block starting from the instruction immediately
        // following this one AND a breakpoint at the instruction immediately following the IT block. We do this because
        // we cannot determine the next_pc until the instruction at which we are currently stopped executes. Hence we
        // insert (m_last_decode_thumb.itBlockRemaining+1) 16-bit Thumb breakpoints at consecutive memory locations
        // starting at addrOfNextInstructionInITBlock. We record these breakpoints in class variable m_sw_single_step_itblock_break_id[],
        // and also record the total number of IT breakpoints set in the variable 'm_sw_single_step_itblock_break_count'.

        // The instructions inside the IT block, which are replaced by the 16-bit Thumb breakpoints (opcode=0xDEFE)
        // instructions, can be either Thumb-16 or Thumb-32. When a Thumb-32 instruction (say, inst#1) is replaced  Thumb
        // by a 16-bit breakpoint (OS only supports 16-bit breakpoints in Thumb mode and 32-bit breakpoints in ARM mode), the
        // breakpoint for the next instruction (say instr#2) is saved in the upper half of this Thumb-32 (instr#1)
        // instruction. Hence if the execution stops at Breakpoint2 corresponding to instr#2, the PC is offset by 16-bits.
        // We therefore have to keep track of PC of each instruction in the IT block that is being replaced with the 16-bit
        // Thumb breakpoint, to ensure that when the breakpoint is hit, the PC is adjusted to the correct value. We save
        // the actual PC corresponding to each instruction in the IT block by associating a call back with each breakpoint
        // we set and passing it as a baton. When the breakpoint hits and the callback routine is called, the routine
        // adjusts the PC based on the baton that is passed to it.

        lldb::addr_t addrOfNextInstructionInITBlock, pcInITBlock, nextPCInITBlock, bpAddressInITBlock;
        uint16_t opcode16;
        uint32_t opcode32;

        addrOfNextInstructionInITBlock = (m_last_decode_arm.thumb16b) ? curr_pc + 2 : curr_pc + 4;

        pcInITBlock = addrOfNextInstructionInITBlock;

        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: itBlockRemaining=%d", __FUNCTION__, m_last_decode_thumb.itBlockRemaining);

        m_sw_single_step_itblock_break_count = 0;
        for (int i = 0; i <= m_last_decode_thumb.itBlockRemaining; i++)
        {
            if (LLDB_BREAK_ID_IS_VALID(m_sw_single_step_itblock_break_id[i]))
            {
                LogError("FunctionProfiler::SetSingleStepSoftwareBreakpoints(): Array m_sw_single_step_itblock_break_id should not contain any valid breakpoint IDs at this point. But found a valid breakID=%d at index=%d", m_sw_single_step_itblock_break_id[i], i);
            }
            else
            {
                nextPCInITBlock = pcInITBlock;
                // Compute nextPCInITBlock based on opcode present at pcInITBlock
                if (m_thread.Process()->Task().ReadMemory(pcInITBlock, 2, &opcode16) == 2)
                {
                    opcode32 = opcode16;
                    nextPCInITBlock += 2;

                    // Check for 32 bit thumb opcode and read the upper 16 bits if needed
                    if (((opcode32 & 0xE000) == 0xE000) && (opcode32 & 0x1800))
                    {
                        // Adjust 'next_pc_in_itblock' to point to the default next Thumb instruction for
                        // a 32 bit Thumb opcode
                        // Read bits 31:16 of a 32 bit Thumb opcode
                        if (m_thread.Process()->Task().ReadMemory(pcInITBlock+2, 2, &opcode16) == 2)
                        {
                            // 32 bit thumb opcode
                            opcode32 = (opcode32 << 16) | opcode16;
                            nextPCInITBlock += 2;
                        }
                        else
                        {
                            LogError("FunctionProfiler::SetSingleStepSoftwareBreakpoints(): Unable to read opcode bits 31:16 for a 32 bit thumb opcode at pc=0x%8.8lx", nextPCInITBlock);
                        }
                    }
                }
                else
                {
                    LogError("FunctionProfiler::SetSingleStepSoftwareBreakpoints(): Error reading 16-bit Thumb instruction at pc=0x%8.8x", nextPCInITBlock);
                }


                // Set breakpoint and associate a callback function with it
                bpAddressInITBlock = addrOfNextInstructionInITBlock + 2*i;
                ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: Setting IT breakpoint[%d] at address: 0x%8.8x", __FUNCTION__, i, bpAddressInITBlock);

                m_sw_single_step_itblock_break_id[i] = m_thread.Process()->CreateBreakpoint(bpAddressInITBlock, 2, false, m_thread.GetID());
                if (!LLDB_BREAK_ID_IS_VALID(m_sw_single_step_itblock_break_id[i]))
                    err = KERN_INVALID_ARGUMENT;
                else
                {
                    ProcessMacOSXLog::LogIf(PD_LOG_STEP, "%s: Set IT breakpoint[%i]=%d set at 0x%8.8x for instruction at 0x%8.8x", __FUNCTION__, i, m_sw_single_step_itblock_break_id[i], bpAddressInITBlock, pcInITBlock);

                    // Set the breakpoint callback for these special IT breakpoints
                    // so that if one of these breakpoints gets hit, it knows to
                    // update the PC to the original address of the conditional
                    // IT instruction.
                    PDBreakpointSetCallback(m_thread.ProcessID(), m_sw_single_step_itblock_break_id[i], MachThreadContext_arm::BreakpointHit, (void*)pcInITBlock);
                    m_sw_single_step_itblock_break_count++;
                }
            }

            pcInITBlock = nextPCInITBlock;
        }

        ProcessMacOSXLog::LogIf(PD_LOG_STEP | PD_LOG_VERBOSE, "%s: Set %u IT software single breakpoints.", __FUNCTION__, m_sw_single_step_itblock_break_count);

    }

    ProcessMacOSXLog::LogIf(PD_LOG_STEP, "%s: next_pc=0x%8.8x (%s)", __FUNCTION__, next_pc, next_pc_is_thumb ? "Thumb" : "ARM");

    if (next_pc & 0x1)
    {
        assert(next_pc_is_thumb);
    }

    if (next_pc_is_thumb)
    {
        next_pc &= ~0x1;
    }
    else
    {
        assert((next_pc & 0x3) == 0);
    }

    if (!inITBlock || (inITBlock && !m_last_decode_arm.setsFlags) || (lastInITBlock && m_last_decode_arm.branch))
    {
        err = KERN_SUCCESS;

#if defined DNB_ARCH_MACH_ARM_DEBUG_SW_STEP
        m_sw_single_step_next_pc = next_pc;
        if (next_pc_is_thumb)
            m_sw_single_step_next_pc |= 1;  // Set bit zero if the next PC is expected to be Thumb
#else
        const DNBBreakpoint *bp = m_thread.Process()->Breakpoints().FindByAddress(next_pc);

        if (bp == NULL)
        {
            m_sw_single_step_break_id = m_thread.Process()->CreateBreakpoint(next_pc, next_pc_is_thumb ? 2 : 4, false, m_thread.GetID());
            if (!LLDB_BREAK_ID_IS_VALID(m_sw_single_step_break_id))
                err = KERN_INVALID_ARGUMENT;
            ProcessMacOSXLog::LogIf(PD_LOG_STEP, "%s: software single step breakpoint with breakID=%d set at 0x%8.8x", __FUNCTION__, m_sw_single_step_break_id, next_pc);
        }
#endif
    }

    return err.GetError();
}

#endif

MachThreadContext*
MachThreadContext_arm::Create (const ArchSpec &arch_spec, ThreadMacOSX &thread)
{
    return new MachThreadContext_arm(thread);
}

void
MachThreadContext_arm::Initialize()
{
    ArchSpec arch_spec(eArchTypeMachO, 12, UINT32_MAX);
    ProcessMacOSX::AddArchCreateCallback(arch_spec, MachThreadContext_arm::Create);
}

//===-- UnwindAssemblyInstEmulation.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnwindAssemblyInstEmulation.h"

#include "llvm-c/EnhancedDisassembly.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;



//-----------------------------------------------------------------------------------------------
//  UnwindAssemblyParser_x86 method definitions 
//-----------------------------------------------------------------------------------------------

bool
UnwindAssemblyInstEmulation::GetNonCallSiteUnwindPlanFromAssembly (AddressRange& range, 
                                                                   Thread& thread, 
                                                                   UnwindPlan& unwind_plan)
{
#if 0
    UnwindPlan::Row row;
    UnwindPlan::Row::RegisterLocation regloc;
    
    m_unwind_plan_sp->SetRegisterKind (eRegisterKindGeneric);
    row.SetCFARegister (LLDB_REGNUM_GENERIC_FP);
    row.SetCFAOffset (2 * 8);
    row.SetOffset (0);
    
    regloc.SetAtCFAPlusOffset (2 * -8);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_FP, regloc);
    regloc.SetAtCFAPlusOffset (1 * -8);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_PC, regloc);
    regloc.SetIsCFAPlusOffset (0);
    row.SetRegisterInfo (LLDB_REGNUM_GENERIC_SP, regloc);
    
    m_unwind_plan_sp->AppendRow (row);
    m_unwind_plan_sp->SetSourceName ("x86_64 architectural default");
#endif

    if (range.GetByteSize() > 0 && 
        range.GetBaseAddress().IsValid() &&
        m_inst_emulator_ap.get())
    {
#if  0
        Target &target = thread.GetProcess().GetTarget();
        const ArchSpec &target_arch = target.GetArchitecture();
        bool prefer_file_cache = true;
        Error error;
        DataBufferHeap data_buffer (range.GetByteSize(), 0);
        if (target.ReadMemory (range.GetBaseAddress(), 
                               prefer_file_cache, 
                               data_buffer.GetBytes(),
                               data_buffer.GetByteSize(), 
                               error) == data_buffer.GetByteSize())
        {
            DataExtractor data (data_buffer.GetBytes(), 
                                data_buffer.GetByteSize(), 
                                target_arch.GetByteOrder(), 
                                target_arch.GetAddressByteSize());
        }
#endif
        StreamFile strm (stdout, false);
        
        ExecutionContext exe_ctx;
        thread.CalculateExecutionContext(exe_ctx);
        DisassemblerSP disasm_sp (Disassembler::DisassembleRange (m_arch,
                                                                  NULL,
                                                                  exe_ctx,
                                                                  range));
        if (disasm_sp)
        {
            
            m_range_ptr = &range;
            m_thread_ptr = &thread;
            m_unwind_plan_ptr = &unwind_plan;

            const uint32_t addr_byte_size = m_arch.GetAddressByteSize();
            const bool show_address = true;
            const bool show_bytes = true;
            const bool raw = false;
            // Initialize the stack pointer with a known value. In the 32 bit case
            // it will be 0x80000000, and in the 64 bit case 0x8000000000000000.
            // We use the address byte size to be safe for any future addresss sizes
            SetRegisterValue (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP, (1ull << ((addr_byte_size * 8) - 1)));
                
            const InstructionList &inst_list = disasm_sp->GetInstructionList ();
            const size_t num_instructions = inst_list.GetSize();
            for (size_t idx=0; idx<num_instructions; ++idx)
            {
                Instruction *inst = inst_list.GetInstructionAtIndex (idx).get();
                if (inst)
                {
                    inst->Dump(&strm, inst_list.GetMaxOpcocdeByteSize (), show_address, show_bytes, &exe_ctx, raw);
                    strm.EOL();

                    m_inst_emulator_ap->SetInstruction (inst->GetOpcode(), inst->GetAddress(), exe_ctx.target);
                    m_inst_emulator_ap->EvaluateInstruction (eEmulateInstructionOptionIgnoreConditions);
                }
            }
        }
    }
    return false;
}

bool
UnwindAssemblyInstEmulation::GetFastUnwindPlan (AddressRange& func, 
                                                Thread& thread, 
                                                UnwindPlan &unwind_plan)
{
    return false;
}

bool
UnwindAssemblyInstEmulation::FirstNonPrologueInsn (AddressRange& func, 
                                                   Target& target, 
                                                   Thread* thread, 
                                                   Address& first_non_prologue_insn)
{
    return false;
}

UnwindAssembly *
UnwindAssemblyInstEmulation::CreateInstance (const ArchSpec &arch)
{
    std::auto_ptr<lldb_private::EmulateInstruction> inst_emulator_ap (EmulateInstruction::FindPlugin (arch, eInstructionTypePrologueEpilogue, NULL));
    // Make sure that all prologue instructions are handled
    if (inst_emulator_ap.get())
        return new UnwindAssemblyInstEmulation (arch, inst_emulator_ap.release());
    return NULL;
}


//------------------------------------------------------------------
// PluginInterface protocol in UnwindAssemblyParser_x86
//------------------------------------------------------------------

const char *
UnwindAssemblyInstEmulation::GetPluginName()
{
    return "UnwindAssemblyInstEmulation";
}

const char *
UnwindAssemblyInstEmulation::GetShortPluginName()
{
    return "unwindassembly.inst-emulation";
}


uint32_t
UnwindAssemblyInstEmulation::GetPluginVersion()
{
    return 1;
}

void
UnwindAssemblyInstEmulation::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
UnwindAssemblyInstEmulation::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
UnwindAssemblyInstEmulation::GetPluginNameStatic()
{
    return "UnwindAssemblyInstEmulation";
}

const char *
UnwindAssemblyInstEmulation::GetPluginDescriptionStatic()
{
    return "Instruction emulation based unwind information.";
}



size_t
UnwindAssemblyInstEmulation::ReadMemory (EmulateInstruction *instruction,
                                         void *baton,
                                         const EmulateInstruction::Context &context, 
                                         lldb::addr_t addr, 
                                         void *dst,
                                         size_t dst_len)
{
    //UnwindAssemblyInstEmulation *inst_emulator = (UnwindAssemblyInstEmulation *)baton;
    printf ("UnwindAssemblyInstEmulation::ReadMemory    (context.type = %i, context.info_type = %i, addr = 0x%16.16llx, dst = %p, dst_len = %zu)\n", 
            context.type,
            context.info_type,
            addr,
            dst,
            dst_len);
    return dst_len;
}

size_t
UnwindAssemblyInstEmulation::WriteMemory (EmulateInstruction *instruction,
                                          void *baton,
                                          const EmulateInstruction::Context &context, 
                                          lldb::addr_t addr, 
                                          const void *dst,
                                          size_t dst_len)
{
    // UnwindAssemblyInstEmulation *inst_emulator = (UnwindAssemblyInstEmulation *)baton;
    
    DataExtractor data (dst, 
                        dst_len, 
                        instruction->GetArchitecture ().GetByteOrder(), 
                        instruction->GetArchitecture ().GetAddressByteSize());
    StreamFile strm(stdout, false);

    strm.Printf ("UnwindAssemblyInstEmulation::WriteMemory   (context.type = %i, context.info_type = %i, ",
                 context.type,
                 context.info_type);
    data.Dump(&strm, 0, eFormatBytes, 1, dst_len, UINT32_MAX, addr, 0, 0);
    strm.EOL();
    return dst_len;
}

bool
UnwindAssemblyInstEmulation::ReadRegister (EmulateInstruction *instruction,
                                           void *baton,
                                           uint32_t reg_kind, 
                                           uint32_t reg_num,
                                           uint64_t &reg_value)
{
    UnwindAssemblyInstEmulation *inst_emulator = (UnwindAssemblyInstEmulation *)baton;
    const char *reg_name = instruction->GetRegisterName (reg_kind, reg_num);

    reg_value = inst_emulator->GetRegisterValue (reg_kind, reg_num);

    printf ("UnwindAssemblyInstEmulation::ReadRegister  (name = \"%s\") => value = 0x%16.16llx\n", reg_name, reg_value);

    return true;
}

bool
UnwindAssemblyInstEmulation::WriteRegister (EmulateInstruction *instruction,
                                            void *baton,
                                            const EmulateInstruction::Context &context, 
                                            uint32_t reg_kind, 
                                            uint32_t reg_num,
                                            uint64_t reg_value)
{
    UnwindAssemblyInstEmulation *inst_emulator = (UnwindAssemblyInstEmulation *)baton;
    const char *reg_name = instruction->GetRegisterName (reg_kind, reg_num);
    
    printf ("UnwindAssemblyInstEmulation::WriteRegister (name = \"%s\", value = 0x%16.16llx, context.type = %i, context.info_type = %i)\n", 
            reg_name,
            reg_value,
            context.type,
            context.info_type);

    inst_emulator->SetRegisterValue (reg_kind, reg_num, reg_value);

    switch (context.type)
    {
        case EmulateInstruction::eContextInvalid:
        case EmulateInstruction::eContextReadOpcode:
        case EmulateInstruction::eContextImmediate:
        case EmulateInstruction::eContextAdjustBaseRegister:
        case EmulateInstruction::eContextRegisterPlusOffset:
        case EmulateInstruction::eContextAdjustPC:
        case EmulateInstruction::eContextRegisterStore:
        case EmulateInstruction::eContextRegisterLoad:  
        case EmulateInstruction::eContextRelativeBranchImmediate:
        case EmulateInstruction::eContextAbsoluteBranchRegister:
        case EmulateInstruction::eContextSupervisorCall:
        case EmulateInstruction::eContextTableBranchReadMemory:
        case EmulateInstruction::eContextWriteRegisterRandomBits:
        case EmulateInstruction::eContextWriteMemoryRandomBits:
        case EmulateInstruction::eContextMultiplication:
        case EmulateInstruction::eContextAddition:
        case EmulateInstruction::eContextSubtraction:
        case EmulateInstruction::eContextAdvancePC:    
        case EmulateInstruction::eContextReturnFromException:
            break;

        case EmulateInstruction::eContextPushRegisterOnStack:
            break;
            
        case EmulateInstruction::eContextPopRegisterOffStack:
            break;

        case EmulateInstruction::eContextAdjustStackPointer:
            break;
    }
    return true;
}



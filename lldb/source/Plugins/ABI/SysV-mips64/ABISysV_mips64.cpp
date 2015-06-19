//===-- ABISysV_mips64.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ABISysV_mips64.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"

using namespace lldb;
using namespace lldb_private;

enum gcc_dwarf_regnums
{
    gcc_dwarf_r0 = 0,
    gcc_dwarf_r1,
    gcc_dwarf_r2,
    gcc_dwarf_r3,
    gcc_dwarf_r4,
    gcc_dwarf_r5,
    gcc_dwarf_r6,
    gcc_dwarf_r7,
    gcc_dwarf_r8,
    gcc_dwarf_r9,
    gcc_dwarf_r10,
    gcc_dwarf_r11,
    gcc_dwarf_r12,
    gcc_dwarf_r13,
    gcc_dwarf_r14,
    gcc_dwarf_r15,
    gcc_dwarf_r16,
    gcc_dwarf_r17,
    gcc_dwarf_r18,
    gcc_dwarf_r19,
    gcc_dwarf_r20,
    gcc_dwarf_r21,
    gcc_dwarf_r22,
    gcc_dwarf_r23,
    gcc_dwarf_r24,
    gcc_dwarf_r25,
    gcc_dwarf_r26,
    gcc_dwarf_r27,
    gcc_dwarf_r28,
    gcc_dwarf_r29,
    gcc_dwarf_r30,
    gcc_dwarf_r31,
    gcc_dwarf_sr,
    gcc_dwarf_lo,
    gcc_dwarf_hi,
    gcc_dwarf_bad,
    gcc_dwarf_cause,
    gcc_dwarf_pc
};

enum gdb_regnums
{
    gdb_r0 = 0,
    gdb_r1,
    gdb_r2,
    gdb_r3,
    gdb_r4,
    gdb_r5,
    gdb_r6,
    gdb_r7,
    gdb_r8,
    gdb_r9,
    gdb_r10,
    gdb_r11,
    gdb_r12,
    gdb_r13,
    gdb_r14,
    gdb_r15,
    gdb_r16,
    gdb_r17,
    gdb_r18,
    gdb_r19,
    gdb_r20,
    gdb_r21,
    gdb_r22,
    gdb_r23,
    gdb_r24,
    gdb_r25,
    gdb_r26,
    gdb_r27,
    gdb_r28,
    gdb_r29,
    gdb_r30,
    gdb_r31,
    gdb_sr,
    gdb_lo,
    gdb_hi,
    gdb_bad,
    gdb_cause,
    gdb_pc
};

static const RegisterInfo
g_register_infos_mips64[] =
{
   //  NAME      ALT    SZ OFF ENCODING        FORMAT        COMPILER                DWARF                 GENERIC                   GDB           LLDB NATIVE       VALUE REGS  INVALIDATE REGS
  //  ========  ======  == === =============  =================== ============ ===================== ==================== =================     ====================== ========== ===============
    { "r0"    , "zero", 8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r0,      gcc_dwarf_r0,           LLDB_INVALID_REGNUM,        gdb_r0,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r1"    , "AT",   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r1,      gcc_dwarf_r1,           LLDB_INVALID_REGNUM,        gdb_r1,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r2"    , "v0",   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r2,      gcc_dwarf_r2,           LLDB_INVALID_REGNUM,        gdb_r2,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r3"    , "v1",   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r3,      gcc_dwarf_r3,           LLDB_INVALID_REGNUM,        gdb_r3,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r4"    , "arg1", 8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r4,      gcc_dwarf_r4,           LLDB_REGNUM_GENERIC_ARG1,   gdb_r4,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r5"    , "arg2", 8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r5,      gcc_dwarf_r5,           LLDB_REGNUM_GENERIC_ARG2,   gdb_r5,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r6"    , "arg3", 8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r6,      gcc_dwarf_r6,           LLDB_REGNUM_GENERIC_ARG3,   gdb_r6,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r7"    , "arg4", 8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r7,      gcc_dwarf_r7,           LLDB_REGNUM_GENERIC_ARG4,   gdb_r7,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r8"    , "arg5", 8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r8,      gcc_dwarf_r8,           LLDB_REGNUM_GENERIC_ARG5,   gdb_r8,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r9"    , "arg6", 8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r9,      gcc_dwarf_r9,           LLDB_REGNUM_GENERIC_ARG6,   gdb_r9,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r10"   , "arg7", 8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r10,     gcc_dwarf_r10,          LLDB_REGNUM_GENERIC_ARG7,   gdb_r10,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r11"   , "arg8", 8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r11,     gcc_dwarf_r11,          LLDB_REGNUM_GENERIC_ARG8,   gdb_r11,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r12"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r12,     gcc_dwarf_r12,          LLDB_INVALID_REGNUM,        gdb_r12,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r13"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r13,     gcc_dwarf_r13,          LLDB_INVALID_REGNUM,        gdb_r13,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r14"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r14,     gcc_dwarf_r14,          LLDB_INVALID_REGNUM,        gdb_r14,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r15"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r15,     gcc_dwarf_r15,          LLDB_INVALID_REGNUM,        gdb_r15,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r16"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r16,     gcc_dwarf_r16,          LLDB_INVALID_REGNUM,        gdb_r16,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r17"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r17,     gcc_dwarf_r17,          LLDB_INVALID_REGNUM,        gdb_r17,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r18"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r18,     gcc_dwarf_r18,          LLDB_INVALID_REGNUM,        gdb_r18,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r19"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r19,     gcc_dwarf_r19,          LLDB_INVALID_REGNUM,        gdb_r19,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r20"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r20,     gcc_dwarf_r20,          LLDB_INVALID_REGNUM,        gdb_r20,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r21"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r21,     gcc_dwarf_r21,          LLDB_INVALID_REGNUM,        gdb_r21,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r22"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r22,     gcc_dwarf_r22,          LLDB_INVALID_REGNUM,        gdb_r22,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r23"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r23,     gcc_dwarf_r23,          LLDB_INVALID_REGNUM,        gdb_r23,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r24"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r24,     gcc_dwarf_r24,          LLDB_INVALID_REGNUM,        gdb_r24,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r25"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r25,     gcc_dwarf_r25,          LLDB_INVALID_REGNUM,        gdb_r25,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r26"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r26,     gcc_dwarf_r26,          LLDB_INVALID_REGNUM,        gdb_r26,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r27"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r27,     gcc_dwarf_r27,          LLDB_INVALID_REGNUM,        gdb_r27,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r28"   , "gp",   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r28,     gcc_dwarf_r28,          LLDB_INVALID_REGNUM,        gdb_r28,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r29"   , "sp",   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r29,     gcc_dwarf_r29,          LLDB_REGNUM_GENERIC_SP,     gdb_r29,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r30"   , "fp",   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r30,     gcc_dwarf_r30,          LLDB_REGNUM_GENERIC_FP,     gdb_r30,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r31"   , "ra",   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_r31,     gcc_dwarf_r31,          LLDB_REGNUM_GENERIC_RA,     gdb_r31,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "sr"    , NULL,   4,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_sr,      gcc_dwarf_sr,           LLDB_REGNUM_GENERIC_FLAGS,  gdb_sr,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "lo"    , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_lo,      gcc_dwarf_lo,           LLDB_INVALID_REGNUM,        gdb_lo,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "hi"    , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_hi,      gcc_dwarf_hi,           LLDB_INVALID_REGNUM,        gdb_hi,     LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "bad"   , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_bad,     gcc_dwarf_bad,          LLDB_INVALID_REGNUM,        gdb_bad,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "cause" , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_cause,   gcc_dwarf_cause,        LLDB_INVALID_REGNUM,        gdb_cause,  LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "pc"    , NULL,   8,  0, eEncodingUint, eFormatHex,  { gcc_dwarf_pc,      gcc_dwarf_pc,           LLDB_REGNUM_GENERIC_PC,     gdb_pc,     LLDB_INVALID_REGNUM },  NULL,      NULL},
};

static const uint32_t k_num_register_infos = llvm::array_lengthof(g_register_infos_mips64);

const lldb_private::RegisterInfo *
ABISysV_mips64::GetRegisterInfoArray (uint32_t &count)
{
    count = k_num_register_infos;
    return g_register_infos_mips64;
}

size_t
ABISysV_mips64::GetRedZoneSize () const
{
    return 0;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
ABISP
ABISysV_mips64::CreateInstance (const ArchSpec &arch)
{
    static ABISP g_abi_sp;
    const llvm::Triple::ArchType arch_type = arch.GetTriple().getArch();
    if ((arch_type == llvm::Triple::mips64) ||
        (arch_type == llvm::Triple::mips64el))
    {
        if (!g_abi_sp)
            g_abi_sp.reset (new ABISysV_mips64);
        return g_abi_sp;
    }
    return ABISP();
}

bool
ABISysV_mips64::PrepareTrivialCall (Thread &thread,
                                  addr_t sp,
                                  addr_t func_addr,
                                  addr_t return_addr,
                                  llvm::ArrayRef<addr_t> args) const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (log)
    {
        StreamString s;
        s.Printf("ABISysV_mips64::PrepareTrivialCall (tid = 0x%" PRIx64 ", sp = 0x%" PRIx64 ", func_addr = 0x%" PRIx64 ", return_addr = 0x%" PRIx64,
                    thread.GetID(),
                    (uint64_t)sp,
                    (uint64_t)func_addr,
                    (uint64_t)return_addr);

        for (size_t i = 0; i < args.size(); ++i)
            s.Printf (", arg%zd = 0x%" PRIx64, i + 1, args[i]);
        s.PutCString (")");
        log->PutCString(s.GetString().c_str());
    }

    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;

    const RegisterInfo *reg_info = NULL;

    if (args.size() > 8) // TODO handle more than 8 arguments
        return false;

    for (size_t i = 0; i < args.size(); ++i)
    {
        reg_info = reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1 + i);
        if (log)
            log->Printf("About to write arg%zd (0x%" PRIx64 ") into %s", i + 1, args[i], reg_info->name);
        if (!reg_ctx->WriteRegisterFromUnsigned(reg_info, args[i]))
            return false;
    }

    // First, align the SP

    if (log)
        log->Printf("16-byte aligning SP: 0x%" PRIx64 " to 0x%" PRIx64, (uint64_t)sp, (uint64_t)(sp & ~0xfull));

    sp &= ~(0xfull); // 16-byte alignment

    Error error;
    const RegisterInfo *pc_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    const RegisterInfo *sp_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    const RegisterInfo *ra_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA);

    if (log)
    log->Printf("Writing SP: 0x%" PRIx64, (uint64_t)sp);

    // Set "sp" to the requested value
    if (!reg_ctx->WriteRegisterFromUnsigned (sp_reg_info, sp))
        return false;

    if (log)
    log->Printf("Writing RA: 0x%" PRIx64, (uint64_t)return_addr);

    // Set "ra" to the return address
    if (!reg_ctx->WriteRegisterFromUnsigned (ra_reg_info, return_addr))
        return false;

    if (log)
        log->Printf("Writing PC: 0x%" PRIx64, (uint64_t)func_addr);

    // Set pc to the address of the called function.
    if (!reg_ctx->WriteRegisterFromUnsigned (pc_reg_info, func_addr))
        return false;

    return true;
}

bool
ABISysV_mips64::GetArgumentValues (Thread &thread, ValueList &values) const
{
    return false;
}

Error
ABISysV_mips64::SetReturnValueObject(lldb::StackFrameSP &frame_sp, lldb::ValueObjectSP &new_value_sp)
{
    Error error;
    if (!new_value_sp)
    {
        error.SetErrorString("Empty value object for return value.");
        return error;
    }

    ClangASTType clang_type = new_value_sp->GetClangType();
    if (!clang_type)
    {
        error.SetErrorString ("Null clang type for return value.");
        return error;
    }

    Thread *thread = frame_sp->GetThread().get();

    RegisterContext *reg_ctx = thread->GetRegisterContext().get();

    if (!reg_ctx)
        error.SetErrorString("no registers are available");
        
    DataExtractor data;
    Error data_error;
    size_t num_bytes = new_value_sp->GetData(data, data_error);
    if (data_error.Fail())
    {
        error.SetErrorStringWithFormat("Couldn't convert return value to raw data: %s", data_error.AsCString());
        return error;
    }

    const uint32_t type_flags = clang_type.GetTypeInfo (NULL);
    
    if (type_flags & eTypeIsScalar ||
        type_flags & eTypeIsPointer)
    {
        if (type_flags & eTypeIsInteger ||
            type_flags & eTypeIsPointer )
        {
            lldb::offset_t offset = 0;
            
            if (num_bytes <= 16)
            {
                const RegisterInfo *r2_info = reg_ctx->GetRegisterInfoByName("r2", 0);
                if (num_bytes <= 8)
                {
                    uint64_t raw_value = data.GetMaxU64(&offset, num_bytes);
                        
                    if (!reg_ctx->WriteRegisterFromUnsigned (r2_info, raw_value))
                        error.SetErrorString ("failed to write register r2");
                }
                else
                {
                    uint64_t raw_value = data.GetMaxU64(&offset, 8);
                    if (reg_ctx->WriteRegisterFromUnsigned (r2_info, raw_value))
                    {
                        const RegisterInfo *r3_info = reg_ctx->GetRegisterInfoByName("r3", 0);
                        raw_value = data.GetMaxU64(&offset, num_bytes - offset);
                        
                        if (!reg_ctx->WriteRegisterFromUnsigned (r3_info, raw_value))
                            error.SetErrorString ("failed to write register r3");
                    }
                    else
                        error.SetErrorString ("failed to write register r2");
                }
            }
            else
            {
                error.SetErrorString("We don't support returning longer than 128 bit integer values at present.");
            }
        }
        else if (type_flags & eTypeIsFloat)
        {
            error.SetErrorString("TODO: Handle Float Types.");
        }
    }
    else if (type_flags & eTypeIsVector)
    {
        error.SetErrorString("returning vector values are not supported");
    }

    return error;
}


ValueObjectSP
ABISysV_mips64::GetReturnValueObjectSimple (Thread &thread, ClangASTType &return_clang_type) const
{
    ValueObjectSP return_valobj_sp;
    return return_valobj_sp;
}

ValueObjectSP
ABISysV_mips64::GetReturnValueObjectImpl (Thread &thread, ClangASTType &return_clang_type) const
{
    ValueObjectSP return_valobj_sp;
    Value value;

    ExecutionContext exe_ctx (thread.shared_from_this());
    if (exe_ctx.GetTargetPtr() == NULL || exe_ctx.GetProcessPtr() == NULL)
        return return_valobj_sp;

    value.SetClangType(return_clang_type);

    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return return_valobj_sp;

    const size_t byte_size = return_clang_type.GetByteSize(nullptr);
    const uint32_t type_flags = return_clang_type.GetTypeInfo (NULL);

    if (type_flags & eTypeIsScalar)
    {
        value.SetValueType(Value::eValueTypeScalar);

        bool success = false;
        if (type_flags & eTypeIsInteger)
        {
            // Extract the register context so we can read arguments from registers
            // In MIPS register "r2" (v0) holds the integer function return values

            uint64_t raw_value = reg_ctx->ReadRegisterAsUnsigned(reg_ctx->GetRegisterInfoByName("r2", 0), 0);

            const bool is_signed = (type_flags & eTypeIsSigned) != 0;
            switch (byte_size)
            {
                default:
                    break;

                case sizeof(uint64_t):
                    if (is_signed)
                        value.GetScalar() = (int64_t)(raw_value);
                    else
                        value.GetScalar() = (uint64_t)(raw_value);
                    success = true;
                    break;

                case sizeof(uint32_t):
                    if (is_signed)
                        value.GetScalar() = (int32_t)(raw_value & UINT32_MAX);
                    else
                        value.GetScalar() = (uint32_t)(raw_value & UINT32_MAX);
                    success = true;
                    break;

                case sizeof(uint16_t):
                    if (is_signed)
                        value.GetScalar() = (int16_t)(raw_value & UINT16_MAX);
                    else
                        value.GetScalar() = (uint16_t)(raw_value & UINT16_MAX);
                    success = true;
                    break;

                case sizeof(uint8_t):
                    if (is_signed)
                        value.GetScalar() = (int8_t)(raw_value & UINT8_MAX);
                    else
                        value.GetScalar() = (uint8_t)(raw_value & UINT8_MAX);
                    success = true;
                    break;
            }
        }

        if (success)
        return_valobj_sp = ValueObjectConstResult::Create (thread.GetStackFrameAtIndex(0).get(),
                                                           value,
                                                           ConstString(""));
    }
    else if (type_flags & eTypeIsPointer)
    {
        value.SetValueType(Value::eValueTypeScalar);
        uint64_t raw_value = reg_ctx->ReadRegisterAsUnsigned(reg_ctx->GetRegisterInfoByName("r2", 0), 0);
        value.GetScalar() = (uint64_t)(raw_value);

        return_valobj_sp = ValueObjectConstResult::Create (thread.GetStackFrameAtIndex(0).get(),
                                                           value,
                                                           ConstString(""));
    }
    else if (type_flags & eTypeIsVector)
    {
        // TODO: Handle vector types
    }
    return return_valobj_sp;
}

bool
ABISysV_mips64::CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    UnwindPlan::RowSP row(new UnwindPlan::Row);

    // Our Call Frame Address is the stack pointer value
    row->GetCFAValue().SetIsRegisterPlusOffset(gcc_dwarf_r29, 0);

    // The previous PC is in the RA
    row->SetRegisterLocationToRegister(gcc_dwarf_pc, gcc_dwarf_r31, true);
    unwind_plan.AppendRow (row);

    // All other registers are the same.

    unwind_plan.SetSourceName ("mips64 at-func-entry default");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetReturnAddressRegister(gcc_dwarf_r31);
    return true;
}

bool
ABISysV_mips64::CreateDefaultUnwindPlan (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    UnwindPlan::RowSP row(new UnwindPlan::Row);

    row->GetCFAValue().SetIsRegisterPlusOffset(gcc_dwarf_r29, 0);

    row->SetRegisterLocationToRegister(gcc_dwarf_pc, gcc_dwarf_r31, true);

    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("mips64 default unwind plan");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    return true;
}

bool
ABISysV_mips64::RegisterIsVolatile (const RegisterInfo *reg_info)
{
    return !RegisterIsCalleeSaved (reg_info);
}

bool
ABISysV_mips64::RegisterIsCalleeSaved (const RegisterInfo *reg_info)
{
    if (reg_info)
    {
        // Preserved registers are :
        // r16-r23, r28, r29, r30, r31

        int reg = ((reg_info->byte_offset) / 8);

        bool save  = (reg >= 16) && (reg <= 23);
             save |= (reg >= 28) && (reg <= 31);

        return save;
    }
    return false;
}

void
ABISysV_mips64::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "System V ABI for mips64 targets",
                                   CreateInstance);
}

void
ABISysV_mips64::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

lldb_private::ConstString
ABISysV_mips64::GetPluginNameStatic()
{
    static ConstString g_name("sysv-mips64");
    return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
ABISysV_mips64::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ABISysV_mips64::GetPluginVersion()
{
    return 1;
}

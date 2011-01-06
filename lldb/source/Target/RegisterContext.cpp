//===-- RegisterContext.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/RegisterContext.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

RegisterContext::RegisterContext (Thread &thread, uint32_t concrete_frame_idx) :
    m_thread (thread),
    m_concrete_frame_idx (concrete_frame_idx)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
RegisterContext::~RegisterContext()
{
}

const RegisterInfo *
RegisterContext::GetRegisterInfoByName (const char *reg_name, uint32_t start_idx)
{
    if (reg_name && reg_name[0])
    {
        const uint32_t num_registers = GetRegisterCount();
        for (uint32_t reg = start_idx; reg < num_registers; ++reg)
        {
            const RegisterInfo * reg_info = GetRegisterInfoAtIndex(reg);

            if ((reg_info->name != NULL && ::strcasecmp (reg_info->name, reg_name) == 0) ||
                (reg_info->alt_name != NULL && ::strcasecmp (reg_info->alt_name, reg_name) == 0))
            {
                return reg_info;
            }
        }
    }
    return NULL;
}

const char *
RegisterContext::GetRegisterName (uint32_t reg)
{
    const RegisterInfo * reg_info = GetRegisterInfoAtIndex(reg);
    if (reg_info)
        return reg_info->name;
    return NULL;
}

uint64_t
RegisterContext::GetPC(uint64_t fail_value)
{
    uint32_t reg = ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    return ReadRegisterAsUnsigned (reg, fail_value);
}

bool
RegisterContext::SetPC(uint64_t pc)
{
    uint32_t reg = ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    bool success = WriteRegisterFromUnsigned (reg, pc);
    if (success)
    {
        StackFrameSP frame_sp(m_thread.GetFrameWithConcreteFrameIndex (m_concrete_frame_idx));
        if (frame_sp)
            frame_sp->ChangePC(pc);
        else
            m_thread.ClearStackFrames ();
    }
    return success;
}

uint64_t
RegisterContext::GetSP(uint64_t fail_value)
{
    uint32_t reg = ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    return ReadRegisterAsUnsigned (reg, fail_value);
}

bool
RegisterContext::SetSP(uint64_t sp)
{
    uint32_t reg = ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    return WriteRegisterFromUnsigned (reg, sp);
}

uint64_t
RegisterContext::GetFP(uint64_t fail_value)
{
    uint32_t reg = ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP);
    return ReadRegisterAsUnsigned (reg, fail_value);
}

bool
RegisterContext::SetFP(uint64_t fp)
{
    uint32_t reg = ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP);
    return WriteRegisterFromUnsigned (reg, fp);
}

uint64_t
RegisterContext::GetReturnAddress (uint64_t fail_value)
{
    uint32_t reg = ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA);
    return ReadRegisterAsUnsigned (reg, fail_value);
}

uint64_t
RegisterContext::GetFlags (uint64_t fail_value)
{
    uint32_t reg = ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS);
    return ReadRegisterAsUnsigned (reg, fail_value);
}


uint64_t
RegisterContext::ReadRegisterAsUnsigned (uint32_t reg, uint64_t fail_value)
{
    if (reg != LLDB_INVALID_REGNUM)
    {
        Scalar value;
        if (ReadRegisterValue (reg, value))
            return value.GetRawBits64(fail_value);
    }
    return fail_value;
}

bool
RegisterContext::WriteRegisterFromUnsigned (uint32_t reg, uint64_t uval)
{
    if (reg == LLDB_INVALID_REGNUM)
        return false;
    Scalar value(uval);
    return WriteRegisterValue (reg, value);
}

lldb::tid_t
RegisterContext::GetThreadID() const
{
    return m_thread.GetID();
}

uint32_t
RegisterContext::NumSupportedHardwareBreakpoints ()
{
    return 0;
}

uint32_t
RegisterContext::SetHardwareBreakpoint (lldb::addr_t addr, size_t size)
{
    return LLDB_INVALID_INDEX32;
}

bool
RegisterContext::ClearHardwareBreakpoint (uint32_t hw_idx)
{
    return false;
}


uint32_t
RegisterContext::NumSupportedHardwareWatchpoints ()
{
    return 0;
}

uint32_t
RegisterContext::SetHardwareWatchpoint (lldb::addr_t addr, size_t size, bool read, bool write)
{
    return LLDB_INVALID_INDEX32;
}

bool
RegisterContext::ClearHardwareWatchpoint (uint32_t hw_index)
{
    return false;
}

bool
RegisterContext::HardwareSingleStep (bool enable)
{
    return false;
}

Target *
RegisterContext::CalculateTarget ()
{
    return m_thread.CalculateTarget();
}


Process *
RegisterContext::CalculateProcess ()
{
    return m_thread.CalculateProcess ();
}

Thread *
RegisterContext::CalculateThread ()
{
    return &m_thread;
}

StackFrame *
RegisterContext::CalculateStackFrame ()
{
    // Register contexts might belong to many frames if we have inlined 
    // functions inside a frame since all inlined functions share the
    // same registers, so we can't definitively say which frame we come from...
    return NULL;
}

void
RegisterContext::CalculateExecutionContext (ExecutionContext &exe_ctx)
{
    m_thread.CalculateExecutionContext (exe_ctx);
}


bool
RegisterContext::ConvertBetweenRegisterKinds (int source_rk, uint32_t source_regnum, int target_rk, uint32_t& target_regnum)
{
    const uint32_t num_registers = GetRegisterCount();
    for (uint32_t reg = 0; reg < num_registers; ++reg)
    {
        const RegisterInfo * reg_info = GetRegisterInfoAtIndex (reg);

        if (reg_info->kinds[source_rk] == source_regnum)
        {
            target_regnum = reg_info->kinds[target_rk];
            if (target_regnum == LLDB_INVALID_REGNUM)
            {
                return false;
            }
            else
            {
                return true;
            }
        } 
    }
    return false;
}

bool
RegisterContext::ReadRegisterValue (uint32_t reg, Scalar &value)
{
    DataExtractor data;
    if (!ReadRegisterBytes (reg, data))
        return false;

    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    uint32_t offset = 0;
    switch (reg_info->encoding)
    {
    case eEncodingUint:
        switch (reg_info->byte_size)
        {
        case 1:
            {
                value = data.GetU8 (&offset);
                return true;
            }
        case 2:
            {
                value = data.GetU16 (&offset);
                return true;
            }
        case 4:
            {
                value = data.GetU32 (&offset);
                return true;
            }
        case 8:
            {
                value = data.GetU64 (&offset);
                return true;
            }
        }
        break;
    case eEncodingSint:
        switch (reg_info->byte_size)
        {
        case 1:
            {
                int8_t v;
                if (data.ExtractBytes (0, sizeof (int8_t), eByteOrderHost, &v) != sizeof (int8_t))
                    return false;
                value = v;
                return true;
            }
        case 2:
            {
                int16_t v;
                if (data.ExtractBytes (0, sizeof (int16_t), eByteOrderHost, &v) != sizeof (int16_t))
                    return false;
                value = v;
                return true;
            }
        case 4:
            {
                int32_t v;
                if (data.ExtractBytes (0, sizeof (int32_t), eByteOrderHost, &v) != sizeof (int32_t))
                    return false;
                value = v;
                return true;
            }
        case 8:
            {
                int64_t v;
                if (data.ExtractBytes (0, sizeof (int64_t), eByteOrderHost, &v) != sizeof (int64_t))
                    return false;
                value = v;
                return true;
            }
        }
        break;
    case eEncodingIEEE754:
        switch (reg_info->byte_size)
        {
        case sizeof (float):
            {
                float v;
                if (data.ExtractBytes (0, sizeof (float), eByteOrderHost, &v) != sizeof (float))
                    return false;
                value = v;
                return true;
            }
        case sizeof (double):
            {
                double v;
                if (data.ExtractBytes (0, sizeof (double), eByteOrderHost, &v) != sizeof (double))
                    return false;
                value = v;
                return true;
            }
        case sizeof (long double):
            {
                double v;
                if (data.ExtractBytes (0, sizeof (long double), eByteOrderHost, &v) != sizeof (long double))
                    return false;
                value = v;
                return true;
            }
        }
        break;
    }
    return false;
}

bool 
RegisterContext::WriteRegisterValue (uint32_t reg, const Scalar &value)
{
    DataExtractor data;
    if (!value.IsValid())
        return false;
    if (!value.GetData (data))
        return false;

    return WriteRegisterBytes (reg, data);
}

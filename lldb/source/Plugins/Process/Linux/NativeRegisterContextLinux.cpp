//===-- NativeRegisterContextLinux.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextLinux.h"

#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Host/linux/Ptrace.h"

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"

using namespace lldb_private;
using namespace lldb_private::process_linux;

namespace
{

class ReadRegOperation : public NativeProcessLinux::Operation
{
public:
    ReadRegOperation(lldb::tid_t tid, uint32_t offset, const char *reg_name, RegisterValue &value) :
    	m_tid(tid),
        m_offset(static_cast<uintptr_t>(offset)),
        m_reg_name(reg_name),
        m_value(value)
    { }

    void
    Execute(NativeProcessLinux *monitor) override;

private:
    lldb::tid_t m_tid;
    uintptr_t m_offset;
    const char *m_reg_name;
    RegisterValue &m_value;
};

class WriteRegOperation : public NativeProcessLinux::Operation
{
public:
    WriteRegOperation(lldb::tid_t tid, unsigned offset, const char *reg_name, const RegisterValue &value) :
    	m_tid(tid),
        m_offset(offset),
        m_reg_name(reg_name),
        m_value(value)
    { }

    void
    Execute(NativeProcessLinux *monitor) override;

private:
    lldb::tid_t m_tid;
    uintptr_t m_offset;
    const char *m_reg_name;
    const RegisterValue &m_value;
};

class ReadGPROperation : public NativeProcessLinux::Operation
{
public:
    ReadGPROperation(lldb::tid_t tid, void *buf, size_t buf_size) :
    	m_tid(tid), m_buf(buf), m_buf_size(buf_size)
    { }

    void Execute(NativeProcessLinux *monitor) override;

private:
    lldb::tid_t m_tid;
    void *m_buf;
    size_t m_buf_size;
};

class WriteGPROperation : public NativeProcessLinux::Operation
{
public:
    WriteGPROperation(lldb::tid_t tid, void *buf, size_t buf_size) :
    	m_tid(tid), m_buf(buf), m_buf_size(buf_size)
    { }

    void Execute(NativeProcessLinux *monitor) override;

private:
    lldb::tid_t m_tid;
    void *m_buf;
    size_t m_buf_size;
};

class ReadFPROperation : public NativeProcessLinux::Operation
{
public:
    ReadFPROperation(lldb::tid_t tid, void *buf, size_t buf_size) :
        m_tid(tid), m_buf(buf), m_buf_size(buf_size)
    { }

    void Execute(NativeProcessLinux *monitor) override;

private:
    lldb::tid_t m_tid;
    void *m_buf;
    size_t m_buf_size;
};

class WriteFPROperation : public NativeProcessLinux::Operation
{
public:
    WriteFPROperation(lldb::tid_t tid, void *buf, size_t buf_size) :
        m_tid(tid), m_buf(buf), m_buf_size(buf_size)
    { }

    void Execute(NativeProcessLinux *monitor) override;

private:
    lldb::tid_t m_tid;
    void *m_buf;
    size_t m_buf_size;
};

class ReadRegisterSetOperation : public NativeProcessLinux::Operation
{
public:
    ReadRegisterSetOperation(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset) :
        m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_regset(regset)
    { }

    void Execute(NativeProcessLinux *monitor) override;

private:
    lldb::tid_t m_tid;
    void *m_buf;
    size_t m_buf_size;
    const unsigned int m_regset;
};

class WriteRegisterSetOperation : public NativeProcessLinux::Operation
{
public:
    WriteRegisterSetOperation(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset) :
        m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_regset(regset)
    { }

    void Execute(NativeProcessLinux *monitor) override;

private:
    lldb::tid_t m_tid;
    void *m_buf;
    size_t m_buf_size;
    const unsigned int m_regset;
};

} // end of anonymous namespace

void
ReadRegOperation::Execute(NativeProcessLinux *monitor)
{
	Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_REGISTERS));

    lldb::addr_t data = static_cast<unsigned long>(NativeProcessLinux::PtraceWrapper(PTRACE_PEEKUSER, m_tid, (void*)m_offset, nullptr, 0, m_error));
    if (m_error.Success())
        m_value = data;

    if (log)
        log->Printf ("NativeProcessLinux::%s() reg %s: 0x%" PRIx64, __FUNCTION__, m_reg_name, data);
}

void
WriteRegOperation::Execute(NativeProcessLinux *monitor)
{
    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_REGISTERS));

    void* buf = (void*)m_value.GetAsUInt64();

    if (log)
        log->Printf ("NativeProcessLinux::%s() reg %s: %p", __FUNCTION__, m_reg_name, buf);
    NativeProcessLinux::PtraceWrapper(PTRACE_POKEUSER, m_tid, (void*)m_offset, buf, 0, m_error);
}

void
ReadGPROperation::Execute(NativeProcessLinux *monitor)
{
    NativeProcessLinux::PtraceWrapper(PTRACE_GETREGS, m_tid, nullptr, m_buf, m_buf_size, m_error);
}

void
WriteGPROperation::Execute(NativeProcessLinux *monitor)
{
    NativeProcessLinux::PtraceWrapper(PTRACE_SETREGS, m_tid, nullptr, m_buf, m_buf_size, m_error);
}

void
ReadFPROperation::Execute(NativeProcessLinux *monitor)
{
    NativeProcessLinux::PtraceWrapper(PTRACE_GETFPREGS, m_tid, nullptr, m_buf, m_buf_size, m_error);
}

void
WriteFPROperation::Execute(NativeProcessLinux *monitor)
{
    NativeProcessLinux::PtraceWrapper(PTRACE_SETFPREGS, m_tid, nullptr, m_buf, m_buf_size, m_error);
}

void
ReadRegisterSetOperation::Execute(NativeProcessLinux *monitor)
{
    NativeProcessLinux::PtraceWrapper(PTRACE_GETREGSET, m_tid, (void *)&m_regset, m_buf, m_buf_size, m_error);
}

void
WriteRegisterSetOperation::Execute(NativeProcessLinux *monitor)
{
    NativeProcessLinux::PtraceWrapper(PTRACE_SETREGSET, m_tid, (void *)&m_regset, m_buf, m_buf_size, m_error);
}

NativeRegisterContextLinux::NativeRegisterContextLinux(NativeThreadProtocol &native_thread,
                                                       uint32_t concrete_frame_idx,
                                                       RegisterInfoInterface *reg_info_interface_p) :
	NativeRegisterContextRegisterInfo(native_thread, concrete_frame_idx, reg_info_interface_p)
{}

lldb::ByteOrder
NativeRegisterContextLinux::GetByteOrder() const
{
    // Get the target process whose privileged thread was used for the register read.
    lldb::ByteOrder byte_order = lldb::eByteOrderInvalid;

    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return byte_order;

    if (!process_sp->GetByteOrder (byte_order))
    {
        // FIXME log here
    }

    return byte_order;
}

Error
NativeRegisterContextLinux::ReadRegisterRaw(uint32_t reg_index, RegisterValue &reg_value)
{
    const RegisterInfo *const reg_info = GetRegisterInfoAtIndex(reg_index);
    if (!reg_info)
    	return Error("register %" PRIu32 " not found", reg_index);

    NativeProcessProtocolSP process_sp(m_thread.GetProcess());
    if (!process_sp)
        return Error("NativeProcessProtocol is NULL");

    NativeProcessLinux* process_p = static_cast<NativeProcessLinux*>(process_sp.get());
    return process_p->DoOperation(GetReadRegisterValueOperation(reg_info->byte_offset,
                                                                reg_info->name,
                                                                reg_info->byte_size,
                                                                reg_value));
}

Error
NativeRegisterContextLinux::WriteRegisterRaw(uint32_t reg_index, const RegisterValue &reg_value)
{
    uint32_t reg_to_write = reg_index;
    RegisterValue value_to_write = reg_value;

    // Check if this is a subregister of a full register.
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex(reg_index);
    if (reg_info->invalidate_regs && (reg_info->invalidate_regs[0] != LLDB_INVALID_REGNUM))
    {
		Error error;

        RegisterValue full_value;
        uint32_t full_reg = reg_info->invalidate_regs[0];
        const RegisterInfo *full_reg_info = GetRegisterInfoAtIndex(full_reg);

        // Read the full register.
        error = ReadRegister(full_reg_info, full_value);
        if (error.Fail ())
            return error;

        lldb::ByteOrder byte_order = GetByteOrder();
        uint8_t dst[RegisterValue::kMaxRegisterByteSize];

        // Get the bytes for the full register.
        const uint32_t dest_size = full_value.GetAsMemoryData (full_reg_info,
                                                               dst,
                                                               sizeof(dst),
                                                               byte_order,
                                                               error);
        if (error.Success() && dest_size)
        {
            uint8_t src[RegisterValue::kMaxRegisterByteSize];

            // Get the bytes for the source data.
            const uint32_t src_size = reg_value.GetAsMemoryData (reg_info, src, sizeof(src), byte_order, error);
            if (error.Success() && src_size && (src_size < dest_size))
            {
                // Copy the src bytes to the destination.
                memcpy (dst + (reg_info->byte_offset & 0x1), src, src_size);
                // Set this full register as the value to write.
                value_to_write.SetBytes(dst, full_value.GetByteSize(), byte_order);
                value_to_write.SetType(full_reg_info);
                reg_to_write = full_reg;
            }
        }
    }

    NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
	    return Error("NativeProcessProtocol is NULL");

    const RegisterInfo *const register_to_write_info_p = GetRegisterInfoAtIndex (reg_to_write);
    assert (register_to_write_info_p && "register to write does not have valid RegisterInfo");
    if (!register_to_write_info_p)
        return Error("NativeRegisterContextLinux::%s failed to get RegisterInfo for write register index %" PRIu32, __FUNCTION__, reg_to_write);

    NativeProcessLinux* process_p = static_cast<NativeProcessLinux*> (process_sp.get ());
    return process_p->DoOperation(GetWriteRegisterValueOperation(reg_info->byte_offset,
                                                                 reg_info->name,
                                                                 reg_value));
}

Error
NativeRegisterContextLinux::ReadGPR()
{
	NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return Error("NativeProcessProtocol is NULL");

    void* buf = GetGPRBuffer();
    if (!buf)
    	return Error("GPR buffer is NULL");
    size_t buf_size = GetGPRSize();

    NativeProcessLinux* process_p = static_cast<NativeProcessLinux*>(process_sp.get());
    return process_p->DoOperation(GetReadGPROperation(buf, buf_size));
}

Error
NativeRegisterContextLinux::WriteGPR()
{
	NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return Error("NativeProcessProtocol is NULL");

    void* buf = GetGPRBuffer();
    if (!buf)
    	return Error("GPR buffer is NULL");
    size_t buf_size = GetGPRSize();

    NativeProcessLinux* process_p = static_cast<NativeProcessLinux*>(process_sp.get());
    return process_p->DoOperation(GetWriteGPROperation(buf, buf_size));
}

Error
NativeRegisterContextLinux::ReadFPR()
{
	NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return Error("NativeProcessProtocol is NULL");

    void* buf = GetFPRBuffer();
    if (!buf)
    	return Error("GPR buffer is NULL");
    size_t buf_size = GetFPRSize();

    NativeProcessLinux* process_p = static_cast<NativeProcessLinux*>(process_sp.get());
    return process_p->DoOperation(GetReadFPROperation(buf, buf_size));
}

Error
NativeRegisterContextLinux::WriteFPR()
{
	NativeProcessProtocolSP process_sp (m_thread.GetProcess ());
    if (!process_sp)
        return Error("NativeProcessProtocol is NULL");

    void* buf = GetFPRBuffer();
    if (!buf)
    	return Error("GPR buffer is NULL");
    size_t buf_size = GetFPRSize();

    NativeProcessLinux* process_p = static_cast<NativeProcessLinux*>(process_sp.get());
    return process_p->DoOperation(GetWriteFPROperation(buf, buf_size));
}

Error
NativeRegisterContextLinux::ReadRegisterSet(void *buf, size_t buf_size, unsigned int regset)
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess());
    if (!process_sp)
        return Error("NativeProcessProtocol is NULL");
    NativeProcessLinux* process_p = static_cast<NativeProcessLinux*>(process_sp.get());

    ReadRegisterSetOperation op(m_thread.GetID(), buf, buf_size, regset);
    return process_p->DoOperation(&op);
}

Error
NativeRegisterContextLinux::WriteRegisterSet(void *buf, size_t buf_size, unsigned int regset)
{
    NativeProcessProtocolSP process_sp (m_thread.GetProcess());
    if (!process_sp)
        return Error("NativeProcessProtocol is NULL");
    NativeProcessLinux* process_p = static_cast<NativeProcessLinux*>(process_sp.get());

    WriteRegisterSetOperation op(m_thread.GetID(), buf, buf_size, regset);
    return process_p->DoOperation(&op);
}

NativeProcessLinux::OperationUP
NativeRegisterContextLinux::GetReadRegisterValueOperation(uint32_t offset,
                                                          const char* reg_name,
	                                                      uint32_t size,
	                                                      RegisterValue &value)
{
	return NativeProcessLinux::OperationUP(new ReadRegOperation(m_thread.GetID(), offset, reg_name, value));
}

NativeProcessLinux::OperationUP
NativeRegisterContextLinux::GetWriteRegisterValueOperation(uint32_t offset,
                                                           const char* reg_name,
	                                                       const RegisterValue &value)
{
	return NativeProcessLinux::OperationUP(new WriteRegOperation(m_thread.GetID(), offset, reg_name, value));
}

NativeProcessLinux::OperationUP
NativeRegisterContextLinux::GetReadGPROperation(void *buf, size_t buf_size)
{
	return NativeProcessLinux::OperationUP(new ReadGPROperation(m_thread.GetID(), buf, buf_size));
}

NativeProcessLinux::OperationUP
NativeRegisterContextLinux::GetWriteGPROperation(void *buf, size_t buf_size)
{
	return NativeProcessLinux::OperationUP(new WriteGPROperation(m_thread.GetID(), buf, buf_size));
}

NativeProcessLinux::OperationUP
NativeRegisterContextLinux::GetReadFPROperation(void *buf, size_t buf_size)
{
	return NativeProcessLinux::OperationUP(new ReadFPROperation(m_thread.GetID(), buf, buf_size));
}

NativeProcessLinux::OperationUP
NativeRegisterContextLinux::GetWriteFPROperation(void *buf, size_t buf_size)
{
	return NativeProcessLinux::OperationUP(new WriteFPROperation(m_thread.GetID(), buf, buf_size));
}

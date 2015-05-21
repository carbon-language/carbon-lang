//===-- NativeProcessLinux.cpp -------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "NativeProcessLinux.h"

// C Includes
#include <errno.h>
#include <poll.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

// C++ Includes
#include <fstream>
#include <sstream>
#include <string>

// Other libraries and framework includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/State.h"
#include "lldb/Host/common/NativeBreakpoint.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostNativeThread.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/PseudoTerminal.h"

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "Plugins/Process/Utility/LinuxSignals.h"
#include "Utility/StringExtractor.h"
#include "NativeThreadLinux.h"
#include "ProcFileReader.h"
#include "Procfs.h"

// System includes - They have to be included after framework includes because they define some
// macros which collide with variable names in other modules
#include <linux/unistd.h>
#include <sys/socket.h>

#include <sys/types.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>

#if defined (__arm64__) || defined (__aarch64__)
// NT_PRSTATUS and NT_FPREGSET definition
#include <elf.h>
#endif

#include "lldb/Host/linux/Personality.h"
#include "lldb/Host/linux/Ptrace.h"
#include "lldb/Host/linux/Signalfd.h"
#include "lldb/Host/android/Android.h"

#define LLDB_PERSONALITY_GET_CURRENT_SETTINGS  0xffffffff

// Support hardware breakpoints in case it has not been defined
#ifndef TRAP_HWBKPT
  #define TRAP_HWBKPT 4
#endif

// We disable the tracing of ptrace calls for integration builds to
// avoid the additional indirection and checks.
#ifndef LLDB_CONFIGURATION_BUILDANDINTEGRATION
#define PTRACE(req, pid, addr, data, data_size, error) \
    PtraceWrapper((req), (pid), (addr), (data), (data_size), (error), #req, __FILE__, __LINE__)
#else
#define PTRACE(req, pid, addr, data, data_size, error) \
    PtraceWrapper((req), (pid), (addr), (data), (data_size), (error))
#endif

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;
using namespace llvm;

// Private bits we only need internally.
namespace
{
    const UnixSignals&
    GetUnixSignals ()
    {
        static process_linux::LinuxSignals signals;
        return signals;
    }

    Error
    ResolveProcessArchitecture (lldb::pid_t pid, Platform &platform, ArchSpec &arch)
    {
        // Grab process info for the running process.
        ProcessInstanceInfo process_info;
        if (!platform.GetProcessInfo (pid, process_info))
            return Error("failed to get process info");

        // Resolve the executable module.
        ModuleSP exe_module_sp;
        ModuleSpec exe_module_spec(process_info.GetExecutableFile(), process_info.GetArchitecture());
        FileSpecList executable_search_paths (Target::GetDefaultExecutableSearchPaths ());
        Error error = platform.ResolveExecutable(
            exe_module_spec,
            exe_module_sp,
            executable_search_paths.GetSize () ? &executable_search_paths : NULL);

        if (!error.Success ())
            return error;

        // Check if we've got our architecture from the exe_module.
        arch = exe_module_sp->GetArchitecture ();
        if (arch.IsValid ())
            return Error();
        else
            return Error("failed to retrieve a valid architecture from the exe module");
    }

    void
    DisplayBytes (StreamString &s, void *bytes, uint32_t count)
    {
        uint8_t *ptr = (uint8_t *)bytes;
        const uint32_t loop_count = std::min<uint32_t>(DEBUG_PTRACE_MAXBYTES, count);
        for(uint32_t i=0; i<loop_count; i++)
        {
            s.Printf ("[%x]", *ptr);
            ptr++;
        }
    }

    void
    PtraceDisplayBytes(int &req, void *data, size_t data_size)
    {
        StreamString buf;
        Log *verbose_log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (
                    POSIX_LOG_PTRACE | POSIX_LOG_VERBOSE));

        if (verbose_log)
        {
            switch(req)
            {
            case PTRACE_POKETEXT:
            {
                DisplayBytes(buf, &data, 8);
                verbose_log->Printf("PTRACE_POKETEXT %s", buf.GetData());
                break;
            }
            case PTRACE_POKEDATA:
            {
                DisplayBytes(buf, &data, 8);
                verbose_log->Printf("PTRACE_POKEDATA %s", buf.GetData());
                break;
            }
            case PTRACE_POKEUSER:
            {
                DisplayBytes(buf, &data, 8);
                verbose_log->Printf("PTRACE_POKEUSER %s", buf.GetData());
                break;
            }
            case PTRACE_SETREGS:
            {
                DisplayBytes(buf, data, data_size);
                verbose_log->Printf("PTRACE_SETREGS %s", buf.GetData());
                break;
            }
            case PTRACE_SETFPREGS:
            {
                DisplayBytes(buf, data, data_size);
                verbose_log->Printf("PTRACE_SETFPREGS %s", buf.GetData());
                break;
            }
            case PTRACE_SETSIGINFO:
            {
                DisplayBytes(buf, data, sizeof(siginfo_t));
                verbose_log->Printf("PTRACE_SETSIGINFO %s", buf.GetData());
                break;
            }
            case PTRACE_SETREGSET:
            {
                // Extract iov_base from data, which is a pointer to the struct IOVEC
                DisplayBytes(buf, *(void **)data, data_size);
                verbose_log->Printf("PTRACE_SETREGSET %s", buf.GetData());
                break;
            }
            default:
            {
            }
            }
        }
    }

    // Wrapper for ptrace to catch errors and log calls.
    // Note that ptrace sets errno on error because -1 can be a valid result (i.e. for PTRACE_PEEK*)
    long
    PtraceWrapper(int req, lldb::pid_t pid, void *addr, void *data, size_t data_size, Error& error,
                  const char* reqName, const char* file, int line)
    {
        long int result;

        Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PTRACE));

        PtraceDisplayBytes(req, data, data_size);

        error.Clear();
        errno = 0;
        if (req == PTRACE_GETREGSET || req == PTRACE_SETREGSET)
            result = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), *(unsigned int *)addr, data);
        else
            result = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), addr, data);

        if (result == -1)
            error.SetErrorToErrno();

        if (log)
            log->Printf("ptrace(%s, %" PRIu64 ", %p, %p, %zu)=%lX called from file %s line %d",
                    reqName, pid, addr, data, data_size, result, file, line);

        PtraceDisplayBytes(req, data, data_size);

        if (log && error.GetError() != 0)
        {
            const char* str;
            switch (error.GetError())
            {
            case ESRCH:  str = "ESRCH"; break;
            case EINVAL: str = "EINVAL"; break;
            case EBUSY:  str = "EBUSY"; break;
            case EPERM:  str = "EPERM"; break;
            default:     str = error.AsCString();
            }
            log->Printf("ptrace() failed; errno=%d (%s)", error.GetError(), str);
        }

        return result;
    }

#ifdef LLDB_CONFIGURATION_BUILDANDINTEGRATION
    // Wrapper for ptrace when logging is not required.
    // Sets errno to 0 prior to calling ptrace.
    long
    PtraceWrapper(int req, lldb::pid_t pid, void *addr, void *data, size_t data_size, Error& error)
    {
        long result = 0;

        error.Clear();
        errno = 0;
        if (req == PTRACE_GETREGSET || req == PTRACE_SETREGSET)
            result = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), *(unsigned int *)addr, data);
        else
            result = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), addr, data);

        if (result == -1)
            error.SetErrorToErrno();
        return result;
    }
#endif

    //------------------------------------------------------------------------------
    // Static implementations of NativeProcessLinux::ReadMemory and
    // NativeProcessLinux::WriteMemory.  This enables mutual recursion between these
    // functions without needed to go thru the thread funnel.

    size_t
    DoReadMemory(
        lldb::pid_t pid,
        lldb::addr_t vm_addr,
        void *buf,
        size_t size,
        Error &error)
    {
        // ptrace word size is determined by the host, not the child
        static const unsigned word_size = sizeof(void*);
        unsigned char *dst = static_cast<unsigned char*>(buf);
        size_t bytes_read;
        size_t remainder;
        long data;

        Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_ALL));
        if (log)
            ProcessPOSIXLog::IncNestLevel();
        if (log && ProcessPOSIXLog::AtTopNestLevel() && log->GetMask().Test(POSIX_LOG_MEMORY))
            log->Printf ("NativeProcessLinux::%s(%" PRIu64 ", %d, %p, %p, %zd, _)", __FUNCTION__,
                    pid, word_size, (void*)vm_addr, buf, size);

        assert(sizeof(data) >= word_size);
        for (bytes_read = 0; bytes_read < size; bytes_read += remainder)
        {
            data = PTRACE(PTRACE_PEEKDATA, pid, (void*)vm_addr, nullptr, 0, error);
            if (error.Fail())
            {
                if (log)
                    ProcessPOSIXLog::DecNestLevel();
                return bytes_read;
            }

            remainder = size - bytes_read;
            remainder = remainder > word_size ? word_size : remainder;

            // Copy the data into our buffer
            for (unsigned i = 0; i < remainder; ++i)
                dst[i] = ((data >> i*8) & 0xFF);

            if (log && ProcessPOSIXLog::AtTopNestLevel() &&
                    (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_LONG) ||
                            (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_SHORT) &&
                                    size <= POSIX_LOG_MEMORY_SHORT_BYTES)))
            {
                uintptr_t print_dst = 0;
                // Format bytes from data by moving into print_dst for log output
                for (unsigned i = 0; i < remainder; ++i)
                    print_dst |= (((data >> i*8) & 0xFF) << i*8);
                log->Printf ("NativeProcessLinux::%s() [%p]:0x%lx (0x%lx)", __FUNCTION__,
                        (void*)vm_addr, print_dst, (unsigned long)data);
            }

            vm_addr += word_size;
            dst += word_size;
        }

        if (log)
            ProcessPOSIXLog::DecNestLevel();
        return bytes_read;
    }

    size_t
    DoWriteMemory(
        lldb::pid_t pid,
        lldb::addr_t vm_addr,
        const void *buf,
        size_t size,
        Error &error)
    {
        // ptrace word size is determined by the host, not the child
        static const unsigned word_size = sizeof(void*);
        const unsigned char *src = static_cast<const unsigned char*>(buf);
        size_t bytes_written = 0;
        size_t remainder;

        Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_ALL));
        if (log)
            ProcessPOSIXLog::IncNestLevel();
        if (log && ProcessPOSIXLog::AtTopNestLevel() && log->GetMask().Test(POSIX_LOG_MEMORY))
            log->Printf ("NativeProcessLinux::%s(%" PRIu64 ", %u, %p, %p, %" PRIu64 ")", __FUNCTION__,
                    pid, word_size, (void*)vm_addr, buf, size);

        for (bytes_written = 0; bytes_written < size; bytes_written += remainder)
        {
            remainder = size - bytes_written;
            remainder = remainder > word_size ? word_size : remainder;

            if (remainder == word_size)
            {
                unsigned long data = 0;
                assert(sizeof(data) >= word_size);
                for (unsigned i = 0; i < word_size; ++i)
                    data |= (unsigned long)src[i] << i*8;

                if (log && ProcessPOSIXLog::AtTopNestLevel() &&
                        (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_LONG) ||
                                (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_SHORT) &&
                                        size <= POSIX_LOG_MEMORY_SHORT_BYTES)))
                    log->Printf ("NativeProcessLinux::%s() [%p]:0x%lx (0x%lx)", __FUNCTION__,
                            (void*)vm_addr, *(const unsigned long*)src, data);

                if (PTRACE(PTRACE_POKEDATA, pid, (void*)vm_addr, (void*)data, 0, error))
                {
                    if (log)
                        ProcessPOSIXLog::DecNestLevel();
                    return bytes_written;
                }
            }
            else
            {
                unsigned char buff[8];
                if (DoReadMemory(pid, vm_addr,
                                buff, word_size, error) != word_size)
                {
                    if (log)
                        ProcessPOSIXLog::DecNestLevel();
                    return bytes_written;
                }

                memcpy(buff, src, remainder);

                if (DoWriteMemory(pid, vm_addr,
                                buff, word_size, error) != word_size)
                {
                    if (log)
                        ProcessPOSIXLog::DecNestLevel();
                    return bytes_written;
                }

                if (log && ProcessPOSIXLog::AtTopNestLevel() &&
                        (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_LONG) ||
                                (log->GetMask().Test(POSIX_LOG_MEMORY_DATA_SHORT) &&
                                        size <= POSIX_LOG_MEMORY_SHORT_BYTES)))
                    log->Printf ("NativeProcessLinux::%s() [%p]:0x%lx (0x%lx)", __FUNCTION__,
                            (void*)vm_addr, *(const unsigned long*)src, *(unsigned long*)buff);
            }

            vm_addr += word_size;
            src += word_size;
        }
        if (log)
            ProcessPOSIXLog::DecNestLevel();
        return bytes_written;
    }

    //------------------------------------------------------------------------------
    /// @class Operation
    /// @brief Represents a NativeProcessLinux operation.
    ///
    /// Under Linux, it is not possible to ptrace() from any other thread but the
    /// one that spawned or attached to the process from the start.  Therefore, when
    /// a NativeProcessLinux is asked to deliver or change the state of an inferior
    /// process the operation must be "funneled" to a specific thread to perform the
    /// task.  The Operation class provides an abstract base for all services the
    /// NativeProcessLinux must perform via the single virtual function Execute, thus
    /// encapsulating the code that needs to run in the privileged context.
    class Operation
    {
    public:
        Operation () : m_error() { }

        virtual
        ~Operation() {}

        virtual void
        Execute (NativeProcessLinux *process) = 0;

        const Error &
        GetError () const { return m_error; }

    protected:
        Error m_error;
    };

    //------------------------------------------------------------------------------
    /// @class ReadOperation
    /// @brief Implements NativeProcessLinux::ReadMemory.
    class ReadOperation : public Operation
    {
    public:
        ReadOperation(
            lldb::addr_t addr,
            void *buff,
            size_t size,
            size_t &result) :
            Operation (),
            m_addr (addr),
            m_buff (buff),
            m_size (size),
            m_result (result)
            {
            }

        void Execute (NativeProcessLinux *process) override;

    private:
        lldb::addr_t m_addr;
        void *m_buff;
        size_t m_size;
        size_t &m_result;
    };

    void
    ReadOperation::Execute (NativeProcessLinux *process)
    {
        m_result = DoReadMemory (process->GetID (), m_addr, m_buff, m_size, m_error);
    }

    //------------------------------------------------------------------------------
    /// @class WriteOperation
    /// @brief Implements NativeProcessLinux::WriteMemory.
    class WriteOperation : public Operation
    {
    public:
        WriteOperation(
            lldb::addr_t addr,
            const void *buff,
            size_t size,
            size_t &result) :
            Operation (),
            m_addr (addr),
            m_buff (buff),
            m_size (size),
            m_result (result)
            {
            }

        void Execute (NativeProcessLinux *process) override;

    private:
        lldb::addr_t m_addr;
        const void *m_buff;
        size_t m_size;
        size_t &m_result;
    };

    void
    WriteOperation::Execute(NativeProcessLinux *process)
    {
        m_result = DoWriteMemory (process->GetID (), m_addr, m_buff, m_size, m_error);
    }

    //------------------------------------------------------------------------------
    /// @class ReadRegOperation
    /// @brief Implements NativeProcessLinux::ReadRegisterValue.
    class ReadRegOperation : public Operation
    {
    public:
        ReadRegOperation(lldb::tid_t tid, uint32_t offset, const char *reg_name,
                RegisterValue &value)
            : m_tid(tid),
              m_offset(static_cast<uintptr_t> (offset)),
              m_reg_name(reg_name),
              m_value(value)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        uintptr_t m_offset;
        const char *m_reg_name;
        RegisterValue &m_value;
    };

    void
    ReadRegOperation::Execute(NativeProcessLinux *monitor)
    {
#if defined (__arm64__) || defined (__aarch64__)
        if (m_offset > sizeof(struct user_pt_regs))
        {
            uintptr_t offset = m_offset - sizeof(struct user_pt_regs);
            if (offset > sizeof(struct user_fpsimd_state))
            {
                m_error.SetErrorString("invalid offset value");
                return;
            }
            elf_fpregset_t regs;
            int regset = NT_FPREGSET;
            struct iovec ioVec;

            ioVec.iov_base = &regs;
            ioVec.iov_len = sizeof regs;
            PTRACE(PTRACE_GETREGSET, m_tid, &regset, &ioVec, sizeof regs, m_error);
            if (m_error.Success())
            {
                ArchSpec arch;
                if (monitor->GetArchitecture(arch))
                    m_value.SetBytes((void *)(((unsigned char *)(&regs)) + offset), 16, arch.GetByteOrder());
                else
                    m_error.SetErrorString("failed to get architecture");
            }
        }
        else
        {
            elf_gregset_t regs;
            int regset = NT_PRSTATUS;
            struct iovec ioVec;

            ioVec.iov_base = &regs;
            ioVec.iov_len = sizeof regs;
            PTRACE(PTRACE_GETREGSET, m_tid, &regset, &ioVec, sizeof regs, m_error);
            if (m_error.Success())
            {
                ArchSpec arch;
                if (monitor->GetArchitecture(arch))
                    m_value.SetBytes((void *)(((unsigned char *)(regs)) + m_offset), 8, arch.GetByteOrder());
                else
                    m_error.SetErrorString("failed to get architecture");
            }
        }
#elif defined (__mips__)
        elf_gregset_t regs;
        PTRACE(PTRACE_GETREGS, m_tid, NULL, &regs, sizeof regs, m_error);
        if (m_error.Success())
        {
            lldb_private::ArchSpec arch;
            if (monitor->GetArchitecture(arch))
                m_value.SetBytes((void *)(((unsigned char *)(regs)) + m_offset), 8, arch.GetByteOrder());
            else
                m_error.SetErrorString("failed to get architecture");
        }
#else
        Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_REGISTERS));

        lldb::addr_t data = static_cast<unsigned long>(PTRACE(PTRACE_PEEKUSER, m_tid, (void*)m_offset, nullptr, 0, m_error));
        if (m_error.Success())
            m_value = data;

        if (log)
            log->Printf ("NativeProcessLinux::%s() reg %s: 0x%" PRIx64, __FUNCTION__,
                    m_reg_name, data);
#endif
    }

    //------------------------------------------------------------------------------
    /// @class WriteRegOperation
    /// @brief Implements NativeProcessLinux::WriteRegisterValue.
    class WriteRegOperation : public Operation
    {
    public:
        WriteRegOperation(lldb::tid_t tid, unsigned offset, const char *reg_name,
                const RegisterValue &value)
            : m_tid(tid),
              m_offset(offset),
              m_reg_name(reg_name),
              m_value(value)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        uintptr_t m_offset;
        const char *m_reg_name;
        const RegisterValue &m_value;
    };

    void
    WriteRegOperation::Execute(NativeProcessLinux *monitor)
    {
#if defined (__arm64__) || defined (__aarch64__)
        if (m_offset > sizeof(struct user_pt_regs))
        {
            uintptr_t offset = m_offset - sizeof(struct user_pt_regs);
            if (offset > sizeof(struct user_fpsimd_state))
            {
                m_error.SetErrorString("invalid offset value");
                return;
            }
            elf_fpregset_t regs;
            int regset = NT_FPREGSET;
            struct iovec ioVec;

            ioVec.iov_base = &regs;
            ioVec.iov_len = sizeof regs;
            PTRACE(PTRACE_GETREGSET, m_tid, &regset, &ioVec, sizeof regs, m_error);
            if (m_error.Success())
            {
                ::memcpy((void *)(((unsigned char *)(&regs)) + offset), m_value.GetBytes(), 16);
                PTRACE(PTRACE_SETREGSET, m_tid, &regset, &ioVec, sizeof regs, m_error);
            }
        }
        else
        {
            elf_gregset_t regs;
            int regset = NT_PRSTATUS;
            struct iovec ioVec;

            ioVec.iov_base = &regs;
            ioVec.iov_len = sizeof regs;
            PTRACE(PTRACE_GETREGSET, m_tid, &regset, &ioVec, sizeof regs, m_error);
            if (m_error.Success())
            {
                ::memcpy((void *)(((unsigned char *)(&regs)) + m_offset), m_value.GetBytes(), 8);
                PTRACE(PTRACE_SETREGSET, m_tid, &regset, &ioVec, sizeof regs, m_error);
            }
        }
#elif defined (__mips__)
        elf_gregset_t regs;
        PTRACE(PTRACE_GETREGS, m_tid, NULL, &regs, sizeof regs, m_error);
        if (m_error.Success())
        {
            ::memcpy((void *)(((unsigned char *)(&regs)) + m_offset), m_value.GetBytes(), 8);
            PTRACE(PTRACE_SETREGS, m_tid, NULL, &regs, sizeof regs, m_error);
        }
#else
        void* buf;
        Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_REGISTERS));

        buf = (void*) m_value.GetAsUInt64();

        if (log)
            log->Printf ("NativeProcessLinux::%s() reg %s: %p", __FUNCTION__, m_reg_name, buf);
        PTRACE(PTRACE_POKEUSER, m_tid, (void*)m_offset, buf, 0, m_error);
#endif
    }

    //------------------------------------------------------------------------------
    /// @class ReadGPROperation
    /// @brief Implements NativeProcessLinux::ReadGPR.
    class ReadGPROperation : public Operation
    {
    public:
        ReadGPROperation(lldb::tid_t tid, void *buf, size_t buf_size)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
    };

    void
    ReadGPROperation::Execute(NativeProcessLinux *monitor)
    {
#if defined (__arm64__) || defined (__aarch64__)
        int regset = NT_PRSTATUS;
        struct iovec ioVec;

        ioVec.iov_base = m_buf;
        ioVec.iov_len = m_buf_size;
        PTRACE(PTRACE_GETREGSET, m_tid, &regset, &ioVec, m_buf_size, m_error);
#else
        PTRACE(PTRACE_GETREGS, m_tid, nullptr, m_buf, m_buf_size, m_error);
#endif
    }

    //------------------------------------------------------------------------------
    /// @class ReadFPROperation
    /// @brief Implements NativeProcessLinux::ReadFPR.
    class ReadFPROperation : public Operation
    {
    public:
        ReadFPROperation(lldb::tid_t tid, void *buf, size_t buf_size)
            : m_tid(tid),
              m_buf(buf),
              m_buf_size(buf_size)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
    };

    void
    ReadFPROperation::Execute(NativeProcessLinux *monitor)
    {
#if defined (__arm64__) || defined (__aarch64__)
        int regset = NT_FPREGSET;
        struct iovec ioVec;

        ioVec.iov_base = m_buf;
        ioVec.iov_len = m_buf_size;
        PTRACE(PTRACE_GETREGSET, m_tid, &regset, &ioVec, m_buf_size, m_error);
#else
        PTRACE(PTRACE_GETFPREGS, m_tid, nullptr, m_buf, m_buf_size, m_error);
#endif
    }

    //------------------------------------------------------------------------------
    /// @class ReadDBGROperation
    /// @brief Implements NativeProcessLinux::ReadHardwareDebugInfo.
    class ReadDBGROperation : public Operation
    {
    public:
        ReadDBGROperation(lldb::tid_t tid, unsigned int &count_wp, unsigned int &count_bp)
            : m_tid(tid),
              m_count_wp(count_wp),
              m_count_bp(count_bp)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        unsigned int &m_count_wp;
        unsigned int &m_count_bp;
    };

    void
    ReadDBGROperation::Execute(NativeProcessLinux *monitor)
    {
#if defined (__arm64__) || defined (__aarch64__)
       int regset = NT_ARM_HW_WATCH;
       struct iovec ioVec;
       struct user_hwdebug_state dreg_state;

       ioVec.iov_base = &dreg_state;
       ioVec.iov_len = sizeof (dreg_state);

       PTRACE(PTRACE_GETREGSET, m_tid, &regset, &ioVec, ioVec.iov_len, m_error);

       m_count_wp = dreg_state.dbg_info & 0xff;
       regset = NT_ARM_HW_BREAK;

       PTRACE(PTRACE_GETREGSET, m_tid, &regset, &ioVec, ioVec.iov_len, m_error);
       m_count_bp = dreg_state.dbg_info & 0xff;
#endif
    }


    //------------------------------------------------------------------------------
    /// @class ReadRegisterSetOperation
    /// @brief Implements NativeProcessLinux::ReadRegisterSet.
    class ReadRegisterSetOperation : public Operation
    {
    public:
        ReadRegisterSetOperation(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_regset(regset)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
        const unsigned int m_regset;
    };

    void
    ReadRegisterSetOperation::Execute(NativeProcessLinux *monitor)
    {
        PTRACE(PTRACE_GETREGSET, m_tid, (void *)&m_regset, m_buf, m_buf_size, m_error);
    }

    //------------------------------------------------------------------------------
    /// @class WriteGPROperation
    /// @brief Implements NativeProcessLinux::WriteGPR.
    class WriteGPROperation : public Operation
    {
    public:
        WriteGPROperation(lldb::tid_t tid, void *buf, size_t buf_size)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
    };

    void
    WriteGPROperation::Execute(NativeProcessLinux *monitor)
    {
#if defined (__arm64__) || defined (__aarch64__)
        int regset = NT_PRSTATUS;
        struct iovec ioVec;

        ioVec.iov_base = m_buf;
        ioVec.iov_len = m_buf_size;
        PTRACE(PTRACE_SETREGSET, m_tid, &regset, &ioVec, m_buf_size, m_error);
#else
        PTRACE(PTRACE_SETREGS, m_tid, NULL, m_buf, m_buf_size, m_error);
#endif
    }

    //------------------------------------------------------------------------------
    /// @class WriteFPROperation
    /// @brief Implements NativeProcessLinux::WriteFPR.
    class WriteFPROperation : public Operation
    {
    public:
        WriteFPROperation(lldb::tid_t tid, void *buf, size_t buf_size)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
    };

    void
    WriteFPROperation::Execute(NativeProcessLinux *monitor)
    {
#if defined (__arm64__) || defined (__aarch64__)
        int regset = NT_FPREGSET;
        struct iovec ioVec;

        ioVec.iov_base = m_buf;
        ioVec.iov_len = m_buf_size;
        PTRACE(PTRACE_SETREGSET, m_tid, &regset, &ioVec, m_buf_size, m_error);
#else
        PTRACE(PTRACE_SETFPREGS, m_tid, NULL, m_buf, m_buf_size, m_error);
#endif
    }

    //------------------------------------------------------------------------------
    /// @class WriteDBGROperation
    /// @brief Implements NativeProcessLinux::WriteHardwareDebugRegs.
    class WriteDBGROperation : public Operation
    {
    public:
        WriteDBGROperation(lldb::tid_t tid, lldb::addr_t *addr_buf,
                           uint32_t *cntrl_buf, int type, int count)
            : m_tid(tid),
              m_address(addr_buf),
              m_control(cntrl_buf),
              m_type(type),
              m_count(count)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        lldb::addr_t * m_address;
        uint32_t * m_control;
        int m_type;
        int m_count;
    };

    void
    WriteDBGROperation::Execute(NativeProcessLinux *monitor)
    {
#if defined (__arm64__) || defined (__aarch64__)
        struct iovec ioVec;
        struct user_hwdebug_state dreg_state;

        memset (&dreg_state, 0, sizeof (dreg_state));
        ioVec.iov_base = &dreg_state;
        ioVec.iov_len = sizeof (dreg_state);

        if (m_type == 0)
            m_type = NT_ARM_HW_WATCH;
        else
            m_type = NT_ARM_HW_BREAK;

        for (int i = 0; i < m_count; i++)
        {
            dreg_state.dbg_regs[i].addr = m_address[i];
            dreg_state.dbg_regs[i].ctrl = m_control[i];
        }

        PTRACE(PTRACE_SETREGSET, m_tid, &m_type, &ioVec, ioVec.iov_len, m_error);
#endif
    }


    //------------------------------------------------------------------------------
    /// @class WriteRegisterSetOperation
    /// @brief Implements NativeProcessLinux::WriteRegisterSet.
    class WriteRegisterSetOperation : public Operation
    {
    public:
        WriteRegisterSetOperation(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_regset(regset)
            { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
        const unsigned int m_regset;
    };

    void
    WriteRegisterSetOperation::Execute(NativeProcessLinux *monitor)
    {
        PTRACE(PTRACE_SETREGSET, m_tid, (void *)&m_regset, m_buf, m_buf_size, m_error);
    }

    //------------------------------------------------------------------------------
    /// @class ResumeOperation
    /// @brief Implements NativeProcessLinux::Resume.
    class ResumeOperation : public Operation
    {
    public:
        ResumeOperation(lldb::tid_t tid, uint32_t signo) :
            m_tid(tid), m_signo(signo) { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        uint32_t m_signo;
    };

    void
    ResumeOperation::Execute(NativeProcessLinux *monitor)
    {
        intptr_t data = 0;

        if (m_signo != LLDB_INVALID_SIGNAL_NUMBER)
            data = m_signo;

        PTRACE(PTRACE_CONT, m_tid, nullptr, (void*)data, 0, m_error);
        if (m_error.Fail())
        {
            Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

            if (log)
                log->Printf ("ResumeOperation (%"  PRIu64 ") failed: %s", m_tid, m_error.AsCString());
        }
    }

    //------------------------------------------------------------------------------
    /// @class SingleStepOperation
    /// @brief Implements NativeProcessLinux::SingleStep.
    class SingleStepOperation : public Operation
    {
    public:
        SingleStepOperation(lldb::tid_t tid, uint32_t signo)
            : m_tid(tid), m_signo(signo) { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        uint32_t m_signo;
    };

    void
    SingleStepOperation::Execute(NativeProcessLinux *monitor)
    {
        intptr_t data = 0;

        if (m_signo != LLDB_INVALID_SIGNAL_NUMBER)
            data = m_signo;

        PTRACE(PTRACE_SINGLESTEP, m_tid, nullptr, (void*)data, 0, m_error);
    }

    //------------------------------------------------------------------------------
    /// @class SiginfoOperation
    /// @brief Implements NativeProcessLinux::GetSignalInfo.
    class SiginfoOperation : public Operation
    {
    public:
        SiginfoOperation(lldb::tid_t tid, void *info)
            : m_tid(tid), m_info(info) { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        void *m_info;
    };

    void
    SiginfoOperation::Execute(NativeProcessLinux *monitor)
    {
        PTRACE(PTRACE_GETSIGINFO, m_tid, nullptr, m_info, 0, m_error);
    }

    //------------------------------------------------------------------------------
    /// @class EventMessageOperation
    /// @brief Implements NativeProcessLinux::GetEventMessage.
    class EventMessageOperation : public Operation
    {
    public:
        EventMessageOperation(lldb::tid_t tid, unsigned long *message)
            : m_tid(tid), m_message(message) { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
        unsigned long *m_message;
    };

    void
    EventMessageOperation::Execute(NativeProcessLinux *monitor)
    {
        PTRACE(PTRACE_GETEVENTMSG, m_tid, nullptr, m_message, 0, m_error);
    }

    class DetachOperation : public Operation
    {
    public:
        DetachOperation(lldb::tid_t tid) : m_tid(tid) { }

        void Execute(NativeProcessLinux *monitor) override;

    private:
        lldb::tid_t m_tid;
    };

    void
    DetachOperation::Execute(NativeProcessLinux *monitor)
    {
        PTRACE(PTRACE_DETACH, m_tid, nullptr, 0, 0, m_error);
    }
} // end of anonymous namespace

// Simple helper function to ensure flags are enabled on the given file
// descriptor.
static Error
EnsureFDFlags(int fd, int flags)
{
    Error error;

    int status = fcntl(fd, F_GETFL);
    if (status == -1)
    {
        error.SetErrorToErrno();
        return error;
    }

    if (fcntl(fd, F_SETFL, status | flags) == -1)
    {
        error.SetErrorToErrno();
        return error;
    }

    return error;
}

// This class encapsulates the privileged thread which performs all ptrace and wait operations on
// the inferior. The thread consists of a main loop which waits for events and processes them
//   - SIGCHLD (delivered over a signalfd file descriptor): These signals notify us of events in
//     the inferior process. Upon receiving this signal we do a waitpid to get more information
//     and dispatch to NativeProcessLinux::MonitorCallback.
//   - requests for ptrace operations: These initiated via the DoOperation method, which funnels
//     them to the Monitor thread via m_operation member. The Monitor thread is signaled over a
//     pipe, and the completion of the operation is signalled over the semaphore.
//   - thread exit event: this is signaled from the Monitor destructor by closing the write end
//     of the command pipe.
class NativeProcessLinux::Monitor
{
private:
    // The initial monitor operation (launch or attach). It returns a inferior process id.
    std::unique_ptr<InitialOperation> m_initial_operation_up;

    ::pid_t                           m_child_pid = -1;
    NativeProcessLinux              * m_native_process;

    enum { READ, WRITE };
    int        m_pipefd[2] = {-1, -1};
    int        m_signal_fd = -1;
    HostThread m_thread;

    // current operation which must be executed on the priviliged thread
    Mutex      m_operation_mutex;
    Operation *m_operation = nullptr;
    sem_t      m_operation_sem;
    Error      m_operation_error;

    unsigned   m_operation_nesting_level = 0;

    static constexpr char operation_command   = 'o';
    static constexpr char begin_block_command = '{';
    static constexpr char end_block_command   = '}';

    void
    HandleSignals();

    void
    HandleWait();

    // Returns true if the thread should exit.
    bool
    HandleCommands();

    void
    MainLoop();

    static void *
    RunMonitor(void *arg);

    Error
    WaitForAck();

    void
    BeginOperationBlock()
    {
        write(m_pipefd[WRITE], &begin_block_command, sizeof operation_command);
        WaitForAck();
    }

    void
    EndOperationBlock()
    {
        write(m_pipefd[WRITE], &end_block_command, sizeof operation_command);
        WaitForAck();
    }

public:
    Monitor(const InitialOperation &initial_operation,
            NativeProcessLinux *native_process)
        : m_initial_operation_up(new InitialOperation(initial_operation)),
          m_native_process(native_process)
    {
        sem_init(&m_operation_sem, 0, 0);
    }

    ~Monitor();

    Error
    Initialize();

    void
    Terminate();

    void
    DoOperation(Operation *op);

    class ScopedOperationLock {
        Monitor &m_monitor;

    public:
        ScopedOperationLock(Monitor &monitor)
            : m_monitor(monitor)
        { m_monitor.BeginOperationBlock(); }

        ~ScopedOperationLock()
        { m_monitor.EndOperationBlock(); }
    };
};
constexpr char NativeProcessLinux::Monitor::operation_command;
constexpr char NativeProcessLinux::Monitor::begin_block_command;
constexpr char NativeProcessLinux::Monitor::end_block_command;

Error
NativeProcessLinux::Monitor::Initialize()
{
    Error error;

    // We get a SIGCHLD every time something interesting happens with the inferior. We shall be
    // listening for these signals over a signalfd file descriptors. This allows us to wait for
    // multiple kinds of events with select.
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGCHLD);
    m_signal_fd = signalfd(-1, &signals, SFD_NONBLOCK | SFD_CLOEXEC);
    if (m_signal_fd < 0)
    {
        return Error("NativeProcessLinux::Monitor::%s failed due to signalfd failure. Monitoring the inferior will be impossible: %s",
                    __FUNCTION__, strerror(errno));

    }

    if (pipe2(m_pipefd, O_CLOEXEC) == -1)
    {
        error.SetErrorToErrno();
        return error;
    }

    if ((error = EnsureFDFlags(m_pipefd[READ], O_NONBLOCK)).Fail()) {
        return error;
    }

    static const char g_thread_name[] = "lldb.process.nativelinux.monitor";
    m_thread = ThreadLauncher::LaunchThread(g_thread_name, Monitor::RunMonitor, this, nullptr);
    if (!m_thread.IsJoinable())
        return Error("Failed to create monitor thread for NativeProcessLinux.");

    // Wait for initial operation to complete.
    return WaitForAck();
}

void
NativeProcessLinux::Monitor::DoOperation(Operation *op)
{
    if (m_thread.EqualsThread(pthread_self())) {
        // If we're on the Monitor thread, we can simply execute the operation.
        op->Execute(m_native_process);
        return;
    }

    // Otherwise we need to pass the operation to the Monitor thread so it can handle it.
    Mutex::Locker lock(m_operation_mutex);

    m_operation = op;

    // notify the thread that an operation is ready to be processed
    write(m_pipefd[WRITE], &operation_command, sizeof operation_command);

    WaitForAck();
}

void
NativeProcessLinux::Monitor::Terminate()
{
    if (m_pipefd[WRITE] >= 0)
    {
        close(m_pipefd[WRITE]);
        m_pipefd[WRITE] = -1;
    }
    if (m_thread.IsJoinable())
        m_thread.Join(nullptr);
}

NativeProcessLinux::Monitor::~Monitor()
{
    Terminate();
    if (m_pipefd[READ] >= 0)
        close(m_pipefd[READ]);
    if (m_signal_fd >= 0)
        close(m_signal_fd);
    sem_destroy(&m_operation_sem);
}

void
NativeProcessLinux::Monitor::HandleSignals()
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    // We don't really care about the content of the SIGCHLD siginfo structure, as we will get
    // all the information from waitpid(). We just need to read all the signals so that we can
    // sleep next time we reach select().
    while (true)
    {
        signalfd_siginfo info;
        ssize_t size = read(m_signal_fd, &info, sizeof info);
        if (size == -1)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                break; // We are done.
            if (errno == EINTR)
                continue;
            if (log)
                log->Printf("NativeProcessLinux::Monitor::%s reading from signalfd file descriptor failed: %s",
                        __FUNCTION__, strerror(errno));
            break;
        }
        if (size != sizeof info)
        {
            // We got incomplete information structure. This should not happen, let's just log
            // that.
            if (log)
                log->Printf("NativeProcessLinux::Monitor::%s reading from signalfd file descriptor returned incomplete data: "
                        "structure size is %zd, read returned %zd bytes",
                        __FUNCTION__, sizeof info, size);
            break;
        }
        if (log)
            log->Printf("NativeProcessLinux::Monitor::%s received signal %s(%d).", __FUNCTION__,
                Host::GetSignalAsCString(info.ssi_signo), info.ssi_signo);
    }
}

void
NativeProcessLinux::Monitor::HandleWait()
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    // Process all pending waitpid notifications.
    while (true)
    {
        int status = -1;
        ::pid_t wait_pid = waitpid(m_child_pid, &status, __WALL | WNOHANG);

        if (wait_pid == 0)
            break; // We are done.

        if (wait_pid == -1)
        {
            if (errno == EINTR)
                continue;

            if (log)
              log->Printf("NativeProcessLinux::Monitor::%s waitpid (pid = %" PRIi32 ", &status, __WALL | WNOHANG) failed: %s",
                      __FUNCTION__, m_child_pid, strerror(errno));
            break;
        }

        bool exited = false;
        int signal = 0;
        int exit_status = 0;
        const char *status_cstr = NULL;
        if (WIFSTOPPED(status))
        {
            signal = WSTOPSIG(status);
            status_cstr = "STOPPED";
        }
        else if (WIFEXITED(status))
        {
            exit_status = WEXITSTATUS(status);
            status_cstr = "EXITED";
            exited = true;
        }
        else if (WIFSIGNALED(status))
        {
            signal = WTERMSIG(status);
            status_cstr = "SIGNALED";
            if (wait_pid == abs(m_child_pid)) {
                exited = true;
                exit_status = -1;
            }
        }
        else
            status_cstr = "(\?\?\?)";

        if (log)
            log->Printf("NativeProcessLinux::Monitor::%s: waitpid (pid = %" PRIi32 ", &status, __WALL | WNOHANG)"
                "=> pid = %" PRIi32 ", status = 0x%8.8x (%s), signal = %i, exit_state = %i",
                __FUNCTION__, m_child_pid, wait_pid, status, status_cstr, signal, exit_status);

        m_native_process->MonitorCallback (wait_pid, exited, signal, exit_status);
    }
}

bool
NativeProcessLinux::Monitor::HandleCommands()
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

    while (true)
    {
        char command = 0;
        ssize_t size = read(m_pipefd[READ], &command, sizeof command);
        if (size == -1)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return false;
            if (errno == EINTR)
                continue;
            if (log)
                log->Printf("NativeProcessLinux::Monitor::%s exiting because read from command file descriptor failed: %s", __FUNCTION__, strerror(errno));
            return true;
        }
        if (size == 0) // end of file - write end closed
        {
            if (log)
                log->Printf("NativeProcessLinux::Monitor::%s exit command received, exiting...", __FUNCTION__);
            assert(m_operation_nesting_level == 0 && "Unbalanced begin/end block commands detected");
            return true; // We are done.
        }

        switch (command)
        {
        case operation_command:
            m_operation->Execute(m_native_process);
            break;
        case begin_block_command:
            ++m_operation_nesting_level;
            break;
        case end_block_command:
            assert(m_operation_nesting_level > 0);
            --m_operation_nesting_level;
            break;
        default:
            if (log)
                log->Printf("NativeProcessLinux::Monitor::%s received unknown command '%c'",
                        __FUNCTION__, command);
        }

        // notify calling thread that the command has been processed
        sem_post(&m_operation_sem);
    }
}

void
NativeProcessLinux::Monitor::MainLoop()
{
    ::pid_t child_pid = (*m_initial_operation_up)(m_operation_error);
    m_initial_operation_up.reset();
    m_child_pid = -getpgid(child_pid),
    sem_post(&m_operation_sem);

    while (true)
    {
        fd_set fds;
        FD_ZERO(&fds);
        // Only process waitpid events if we are outside of an operation block. Any pending
        // events will be processed after we leave the block.
        if (m_operation_nesting_level == 0)
            FD_SET(m_signal_fd, &fds);
        FD_SET(m_pipefd[READ], &fds);

        int max_fd = std::max(m_signal_fd, m_pipefd[READ]) + 1;
        int r = select(max_fd, &fds, nullptr, nullptr, nullptr);
        if (r < 0)
        {
            Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
            if (log)
                log->Printf("NativeProcessLinux::Monitor::%s exiting because select failed: %s",
                        __FUNCTION__, strerror(errno));
            return;
        }

        if (FD_ISSET(m_pipefd[READ], &fds))
        {
            if (HandleCommands())
                return;
        }

        if (FD_ISSET(m_signal_fd, &fds))
        {
            HandleSignals();
            HandleWait();
        }
    }
}

Error
NativeProcessLinux::Monitor::WaitForAck()
{
    Error error;
    while (sem_wait(&m_operation_sem) != 0)
    {
        if (errno == EINTR)
            continue;

        error.SetErrorToErrno();
        return error;
    }

    return m_operation_error;
}

void *
NativeProcessLinux::Monitor::RunMonitor(void *arg)
{
    static_cast<Monitor *>(arg)->MainLoop();
    return nullptr;
}


NativeProcessLinux::LaunchArgs::LaunchArgs(Module *module,
                                       char const **argv,
                                       char const **envp,
                                       const std::string &stdin_path,
                                       const std::string &stdout_path,
                                       const std::string &stderr_path,
                                       const char *working_dir,
                                       const ProcessLaunchInfo &launch_info)
    : m_module(module),
      m_argv(argv),
      m_envp(envp),
      m_stdin_path(stdin_path),
      m_stdout_path(stdout_path),
      m_stderr_path(stderr_path),
      m_working_dir(working_dir),
      m_launch_info(launch_info)
{
}

NativeProcessLinux::LaunchArgs::~LaunchArgs()
{ }

// -----------------------------------------------------------------------------
// Public Static Methods
// -----------------------------------------------------------------------------

Error
NativeProcessLinux::LaunchProcess (
    Module *exe_module,
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate,
    NativeProcessProtocolSP &native_process_sp)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    Error error;

    // Verify the working directory is valid if one was specified.
    const char* working_dir = launch_info.GetWorkingDirectory ();
    if (working_dir)
    {
      FileSpec working_dir_fs (working_dir, true);
      if (!working_dir_fs || working_dir_fs.GetFileType () != FileSpec::eFileTypeDirectory)
      {
          error.SetErrorStringWithFormat ("No such file or directory: %s", working_dir);
          return error;
      }
    }

    const FileAction *file_action;

    // Default of NULL will mean to use existing open file descriptors.
    std::string stdin_path;
    std::string stdout_path;
    std::string stderr_path;

    file_action = launch_info.GetFileActionForFD (STDIN_FILENO);
    if (file_action)
        stdin_path = file_action->GetPath ();

    file_action = launch_info.GetFileActionForFD (STDOUT_FILENO);
    if (file_action)
        stdout_path = file_action->GetPath ();

    file_action = launch_info.GetFileActionForFD (STDERR_FILENO);
    if (file_action)
        stderr_path = file_action->GetPath ();

    if (log)
    {
        if (!stdin_path.empty ())
            log->Printf ("NativeProcessLinux::%s setting STDIN to '%s'", __FUNCTION__, stdin_path.c_str ());
        else
            log->Printf ("NativeProcessLinux::%s leaving STDIN as is", __FUNCTION__);

        if (!stdout_path.empty ())
            log->Printf ("NativeProcessLinux::%s setting STDOUT to '%s'", __FUNCTION__, stdout_path.c_str ());
        else
            log->Printf ("NativeProcessLinux::%s leaving STDOUT as is", __FUNCTION__);

        if (!stderr_path.empty ())
            log->Printf ("NativeProcessLinux::%s setting STDERR to '%s'", __FUNCTION__, stderr_path.c_str ());
        else
            log->Printf ("NativeProcessLinux::%s leaving STDERR as is", __FUNCTION__);
    }

    // Create the NativeProcessLinux in launch mode.
    native_process_sp.reset (new NativeProcessLinux ());

    if (log)
    {
        int i = 0;
        for (const char **args = launch_info.GetArguments ().GetConstArgumentVector (); *args; ++args, ++i)
        {
            log->Printf ("NativeProcessLinux::%s arg %d: \"%s\"", __FUNCTION__, i, *args ? *args : "nullptr");
            ++i;
        }
    }

    if (!native_process_sp->RegisterNativeDelegate (native_delegate))
    {
        native_process_sp.reset ();
        error.SetErrorStringWithFormat ("failed to register the native delegate");
        return error;
    }

    std::static_pointer_cast<NativeProcessLinux> (native_process_sp)->LaunchInferior (
            exe_module,
            launch_info.GetArguments ().GetConstArgumentVector (),
            launch_info.GetEnvironmentEntries ().GetConstArgumentVector (),
            stdin_path,
            stdout_path,
            stderr_path,
            working_dir,
            launch_info,
            error);

    if (error.Fail ())
    {
        native_process_sp.reset ();
        if (log)
            log->Printf ("NativeProcessLinux::%s failed to launch process: %s", __FUNCTION__, error.AsCString ());
        return error;
    }

    launch_info.SetProcessID (native_process_sp->GetID ());

    return error;
}

Error
NativeProcessLinux::AttachToProcess (
    lldb::pid_t pid,
    NativeProcessProtocol::NativeDelegate &native_delegate,
    NativeProcessProtocolSP &native_process_sp)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log && log->GetMask ().Test (POSIX_LOG_VERBOSE))
        log->Printf ("NativeProcessLinux::%s(pid = %" PRIi64 ")", __FUNCTION__, pid);

    // Grab the current platform architecture.  This should be Linux,
    // since this code is only intended to run on a Linux host.
    PlatformSP platform_sp (Platform::GetHostPlatform ());
    if (!platform_sp)
        return Error("failed to get a valid default platform");

    // Retrieve the architecture for the running process.
    ArchSpec process_arch;
    Error error = ResolveProcessArchitecture (pid, *platform_sp.get (), process_arch);
    if (!error.Success ())
        return error;

    std::shared_ptr<NativeProcessLinux> native_process_linux_sp (new NativeProcessLinux ());

    if (!native_process_linux_sp->RegisterNativeDelegate (native_delegate))
    {
        error.SetErrorStringWithFormat ("failed to register the native delegate");
        return error;
    }

    native_process_linux_sp->AttachToInferior (pid, error);
    if (!error.Success ())
        return error;

    native_process_sp = native_process_linux_sp;
    return error;
}

// -----------------------------------------------------------------------------
// Public Instance Methods
// -----------------------------------------------------------------------------

NativeProcessLinux::NativeProcessLinux () :
    NativeProcessProtocol (LLDB_INVALID_PROCESS_ID),
    m_arch (),
    m_supports_mem_region (eLazyBoolCalculate),
    m_mem_region_cache (),
    m_mem_region_cache_mutex ()
{
}

//------------------------------------------------------------------------------
// NativeProcessLinux spawns a new thread which performs all operations on the inferior process.
// Refer to Monitor and Operation classes to see why this is necessary.
//------------------------------------------------------------------------------
void
NativeProcessLinux::LaunchInferior (
    Module *module,
    const char *argv[],
    const char *envp[],
    const std::string &stdin_path,
    const std::string &stdout_path,
    const std::string &stderr_path,
    const char *working_dir,
    const ProcessLaunchInfo &launch_info,
    Error &error)
{
    if (module)
        m_arch = module->GetArchitecture ();

    SetState (eStateLaunching);

    std::unique_ptr<LaunchArgs> args(
        new LaunchArgs(
            module, argv, envp,
            stdin_path, stdout_path, stderr_path,
            working_dir, launch_info));

    StartMonitorThread ([&] (Error &e) { return Launch(args.get(), e); }, error);
    if (!error.Success ())
        return;
}

void
NativeProcessLinux::AttachToInferior (lldb::pid_t pid, Error &error)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessLinux::%s (pid = %" PRIi64 ")", __FUNCTION__, pid);

    // We can use the Host for everything except the ResolveExecutable portion.
    PlatformSP platform_sp = Platform::GetHostPlatform ();
    if (!platform_sp)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s (pid = %" PRIi64 "): no default platform set", __FUNCTION__, pid);
        error.SetErrorString ("no default platform available");
        return;
    }

    // Gather info about the process.
    ProcessInstanceInfo process_info;
    if (!platform_sp->GetProcessInfo (pid, process_info))
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s (pid = %" PRIi64 "): failed to get process info", __FUNCTION__, pid);
        error.SetErrorString ("failed to get process info");
        return;
    }

    // Resolve the executable module
    ModuleSP exe_module_sp;
    FileSpecList executable_search_paths (Target::GetDefaultExecutableSearchPaths());
    ModuleSpec exe_module_spec(process_info.GetExecutableFile(), process_info.GetArchitecture());
    error = platform_sp->ResolveExecutable(exe_module_spec, exe_module_sp,
                                           executable_search_paths.GetSize() ? &executable_search_paths : NULL);
    if (!error.Success())
        return;

    // Set the architecture to the exe architecture.
    m_arch = exe_module_sp->GetArchitecture();
    if (log)
        log->Printf ("NativeProcessLinux::%s (pid = %" PRIi64 ") detected architecture %s", __FUNCTION__, pid, m_arch.GetArchitectureName ());

    m_pid = pid;
    SetState(eStateAttaching);

    StartMonitorThread ([=] (Error &e) { return Attach(pid, e); }, error);
    if (!error.Success ())
        return;
}

void
NativeProcessLinux::Terminate ()
{
    m_monitor_up->Terminate();
}

::pid_t
NativeProcessLinux::Launch(LaunchArgs *args, Error &error)
{
    assert (args && "null args");

    const char **argv = args->m_argv;
    const char **envp = args->m_envp;
    const char *working_dir = args->m_working_dir;

    lldb_utility::PseudoTerminal terminal;
    const size_t err_len = 1024;
    char err_str[err_len];
    lldb::pid_t pid;
    NativeThreadProtocolSP thread_sp;

    lldb::ThreadSP inferior;

    // Propagate the environment if one is not supplied.
    if (envp == NULL || envp[0] == NULL)
        envp = const_cast<const char **>(environ);

    if ((pid = terminal.Fork(err_str, err_len)) == static_cast<lldb::pid_t> (-1))
    {
        error.SetErrorToGenericError();
        error.SetErrorStringWithFormat("Process fork failed: %s", err_str);
        return -1;
    }

    // Recognized child exit status codes.
    enum {
        ePtraceFailed = 1,
        eDupStdinFailed,
        eDupStdoutFailed,
        eDupStderrFailed,
        eChdirFailed,
        eExecFailed,
        eSetGidFailed
    };

    // Child process.
    if (pid == 0)
    {
        // FIXME consider opening a pipe between parent/child and have this forked child
        // send log info to parent re: launch status, in place of the log lines removed here.

        // Start tracing this child that is about to exec.
        PTRACE(PTRACE_TRACEME, 0, nullptr, nullptr, 0, error);
        if (error.Fail())
            exit(ePtraceFailed);

        // terminal has already dupped the tty descriptors to stdin/out/err.
        // This closes original fd from which they were copied (and avoids
        // leaking descriptors to the debugged process.
        terminal.CloseSlaveFileDescriptor();

        // Do not inherit setgid powers.
        if (setgid(getgid()) != 0)
            exit(eSetGidFailed);

        // Attempt to have our own process group.
        if (setpgid(0, 0) != 0)
        {
            // FIXME log that this failed. This is common.
            // Don't allow this to prevent an inferior exec.
        }

        // Dup file descriptors if needed.
        if (!args->m_stdin_path.empty ())
            if (!DupDescriptor(args->m_stdin_path.c_str (), STDIN_FILENO, O_RDONLY))
                exit(eDupStdinFailed);

        if (!args->m_stdout_path.empty ())
            if (!DupDescriptor(args->m_stdout_path.c_str (), STDOUT_FILENO, O_WRONLY | O_CREAT | O_TRUNC))
                exit(eDupStdoutFailed);

        if (!args->m_stderr_path.empty ())
            if (!DupDescriptor(args->m_stderr_path.c_str (), STDERR_FILENO, O_WRONLY | O_CREAT | O_TRUNC))
                exit(eDupStderrFailed);

        // Close everything besides stdin, stdout, and stderr that has no file
        // action to avoid leaking
        for (int fd = 3; fd < sysconf(_SC_OPEN_MAX); ++fd)
            if (!args->m_launch_info.GetFileActionForFD(fd))
                close(fd);

        // Change working directory
        if (working_dir != NULL && working_dir[0])
          if (0 != ::chdir(working_dir))
              exit(eChdirFailed);

        // Disable ASLR if requested.
        if (args->m_launch_info.GetFlags ().Test (lldb::eLaunchFlagDisableASLR))
        {
            const int old_personality = personality (LLDB_PERSONALITY_GET_CURRENT_SETTINGS);
            if (old_personality == -1)
            {
                // Can't retrieve Linux personality.  Cannot disable ASLR.
            }
            else
            {
                const int new_personality = personality (ADDR_NO_RANDOMIZE | old_personality);
                if (new_personality == -1)
                {
                    // Disabling ASLR failed.
                }
                else
                {
                    // Disabling ASLR succeeded.
                }
            }
        }

        // Execute.  We should never return...
        execve(argv[0],
               const_cast<char *const *>(argv),
               const_cast<char *const *>(envp));

        // ...unless exec fails.  In which case we definitely need to end the child here.
        exit(eExecFailed);
    }

    //
    // This is the parent code here.
    //
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // Wait for the child process to trap on its call to execve.
    ::pid_t wpid;
    int status;
    if ((wpid = waitpid(pid, &status, 0)) < 0)
    {
        error.SetErrorToErrno();
        if (log)
            log->Printf ("NativeProcessLinux::%s waitpid for inferior failed with %s",
                    __FUNCTION__, error.AsCString ());

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        SetState (StateType::eStateInvalid);

        return -1;
    }
    else if (WIFEXITED(status))
    {
        // open, dup or execve likely failed for some reason.
        error.SetErrorToGenericError();
        switch (WEXITSTATUS(status))
        {
            case ePtraceFailed:
                error.SetErrorString("Child ptrace failed.");
                break;
            case eDupStdinFailed:
                error.SetErrorString("Child open stdin failed.");
                break;
            case eDupStdoutFailed:
                error.SetErrorString("Child open stdout failed.");
                break;
            case eDupStderrFailed:
                error.SetErrorString("Child open stderr failed.");
                break;
            case eChdirFailed:
                error.SetErrorString("Child failed to set working directory.");
                break;
            case eExecFailed:
                error.SetErrorString("Child exec failed.");
                break;
            case eSetGidFailed:
                error.SetErrorString("Child setgid failed.");
                break;
            default:
                error.SetErrorString("Child returned unknown exit status.");
                break;
        }

        if (log)
        {
            log->Printf ("NativeProcessLinux::%s inferior exited with status %d before issuing a STOP",
                    __FUNCTION__,
                    WEXITSTATUS(status));
        }

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        SetState (StateType::eStateInvalid);

        return -1;
    }
    assert(WIFSTOPPED(status) && (wpid == static_cast< ::pid_t> (pid)) &&
           "Could not sync with inferior process.");

    if (log)
        log->Printf ("NativeProcessLinux::%s inferior started, now in stopped state", __FUNCTION__);

    error = SetDefaultPtraceOpts(pid);
    if (error.Fail())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior failed to set default ptrace options: %s",
                    __FUNCTION__, error.AsCString ());

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        SetState (StateType::eStateInvalid);

        return -1;
    }

    // Release the master terminal descriptor and pass it off to the
    // NativeProcessLinux instance.  Similarly stash the inferior pid.
    m_terminal_fd = terminal.ReleaseMasterFileDescriptor();
    m_pid = pid;

    // Set the terminal fd to be in non blocking mode (it simplifies the
    // implementation of ProcessLinux::GetSTDOUT to have a non-blocking
    // descriptor to read from).
    error = EnsureFDFlags(m_terminal_fd, O_NONBLOCK);
    if (error.Fail())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior EnsureFDFlags failed for ensuring terminal O_NONBLOCK setting: %s",
                    __FUNCTION__, error.AsCString ());

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        SetState (StateType::eStateInvalid);

        return -1;
    }

    if (log)
        log->Printf ("NativeProcessLinux::%s() adding pid = %" PRIu64, __FUNCTION__, pid);

    thread_sp = AddThread (pid);
    assert (thread_sp && "AddThread() returned a nullptr thread");
    std::static_pointer_cast<NativeThreadLinux> (thread_sp)->SetStoppedBySignal (SIGSTOP);
    ThreadWasCreated(pid);

    // Let our process instance know the thread has stopped.
    SetCurrentThreadID (thread_sp->GetID ());
    SetState (StateType::eStateStopped);

    if (log)
    {
        if (error.Success ())
        {
            log->Printf ("NativeProcessLinux::%s inferior launching succeeded", __FUNCTION__);
        }
        else
        {
            log->Printf ("NativeProcessLinux::%s inferior launching failed: %s",
                __FUNCTION__, error.AsCString ());
            return -1;
        }
    }
    return pid;
}

::pid_t
NativeProcessLinux::Attach(lldb::pid_t pid, Error &error)
{
    lldb::ThreadSP inferior;
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // Use a map to keep track of the threads which we have attached/need to attach.
    Host::TidMap tids_to_attach;
    if (pid <= 1)
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Attaching to process 1 is not allowed.");
        return -1;
    }

    while (Host::FindProcessThreads(pid, tids_to_attach))
    {
        for (Host::TidMap::iterator it = tids_to_attach.begin();
             it != tids_to_attach.end();)
        {
            if (it->second == false)
            {
                lldb::tid_t tid = it->first;

                // Attach to the requested process.
                // An attach will cause the thread to stop with a SIGSTOP.
                PTRACE(PTRACE_ATTACH, tid, nullptr, nullptr, 0, error);
                if (error.Fail())
                {
                    // No such thread. The thread may have exited.
                    // More error handling may be needed.
                    if (error.GetError() == ESRCH)
                    {
                        it = tids_to_attach.erase(it);
                        continue;
                    }
                    else
                        return -1;
                }

                int status;
                // Need to use __WALL otherwise we receive an error with errno=ECHLD
                // At this point we should have a thread stopped if waitpid succeeds.
                if ((status = waitpid(tid, NULL, __WALL)) < 0)
                {
                    // No such thread. The thread may have exited.
                    // More error handling may be needed.
                    if (errno == ESRCH)
                    {
                        it = tids_to_attach.erase(it);
                        continue;
                    }
                    else
                    {
                        error.SetErrorToErrno();
                        return -1;
                    }
                }

                error = SetDefaultPtraceOpts(tid);
                if (error.Fail())
                    return -1;

                if (log)
                    log->Printf ("NativeProcessLinux::%s() adding tid = %" PRIu64, __FUNCTION__, tid);

                it->second = true;

                // Create the thread, mark it as stopped.
                NativeThreadProtocolSP thread_sp (AddThread (static_cast<lldb::tid_t> (tid)));
                assert (thread_sp && "AddThread() returned a nullptr");

                // This will notify this is a new thread and tell the system it is stopped.
                std::static_pointer_cast<NativeThreadLinux> (thread_sp)->SetStoppedBySignal (SIGSTOP);
                ThreadWasCreated(tid);
                SetCurrentThreadID (thread_sp->GetID ());
            }

            // move the loop forward
            ++it;
        }
    }

    if (tids_to_attach.size() > 0)
    {
        m_pid = pid;
        // Let our process instance know the thread has stopped.
        SetState (StateType::eStateStopped);
    }
    else
    {
        error.SetErrorToGenericError();
        error.SetErrorString("No such process.");
        return -1;
    }

    return pid;
}

Error
NativeProcessLinux::SetDefaultPtraceOpts(lldb::pid_t pid)
{
    long ptrace_opts = 0;

    // Have the child raise an event on exit.  This is used to keep the child in
    // limbo until it is destroyed.
    ptrace_opts |= PTRACE_O_TRACEEXIT;

    // Have the tracer trace threads which spawn in the inferior process.
    // TODO: if we want to support tracing the inferiors' child, add the
    // appropriate ptrace flags here (PTRACE_O_TRACEFORK, PTRACE_O_TRACEVFORK)
    ptrace_opts |= PTRACE_O_TRACECLONE;

    // Have the tracer notify us before execve returns
    // (needed to disable legacy SIGTRAP generation)
    ptrace_opts |= PTRACE_O_TRACEEXEC;

    Error error;
    PTRACE(PTRACE_SETOPTIONS, pid, nullptr, (void*)ptrace_opts, 0, error);
    return error;
}

static ExitType convert_pid_status_to_exit_type (int status)
{
    if (WIFEXITED (status))
        return ExitType::eExitTypeExit;
    else if (WIFSIGNALED (status))
        return ExitType::eExitTypeSignal;
    else if (WIFSTOPPED (status))
        return ExitType::eExitTypeStop;
    else
    {
        // We don't know what this is.
        return ExitType::eExitTypeInvalid;
    }
}

static int convert_pid_status_to_return_code (int status)
{
    if (WIFEXITED (status))
        return WEXITSTATUS (status);
    else if (WIFSIGNALED (status))
        return WTERMSIG (status);
    else if (WIFSTOPPED (status))
        return WSTOPSIG (status);
    else
    {
        // We don't know what this is.
        return ExitType::eExitTypeInvalid;
    }
}

// Handles all waitpid events from the inferior process.
void
NativeProcessLinux::MonitorCallback(lldb::pid_t pid,
                                    bool exited,
                                    int signal,
                                    int status)
{
    Log *log (GetLogIfAnyCategoriesSet (LIBLLDB_LOG_PROCESS));

    // Certain activities differ based on whether the pid is the tid of the main thread.
    const bool is_main_thread = (pid == GetID ());

    // Handle when the thread exits.
    if (exited)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() got exit signal(%d) , tid = %"  PRIu64 " (%s main thread)", __FUNCTION__, signal, pid, is_main_thread ? "is" : "is not");

        // This is a thread that exited.  Ensure we're not tracking it anymore.
        const bool thread_found = StopTrackingThread (pid);

        if (is_main_thread)
        {
            // We only set the exit status and notify the delegate if we haven't already set the process
            // state to an exited state.  We normally should have received a SIGTRAP | (PTRACE_EVENT_EXIT << 8)
            // for the main thread.
            const bool already_notified = (GetState() == StateType::eStateExited) || (GetState () == StateType::eStateCrashed);
            if (!already_notified)
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s() tid = %"  PRIu64 " handling main thread exit (%s), expected exit state already set but state was %s instead, setting exit state now", __FUNCTION__, pid, thread_found ? "stopped tracking thread metadata" : "thread metadata not found", StateAsCString (GetState ()));
                // The main thread exited.  We're done monitoring.  Report to delegate.
                SetExitStatus (convert_pid_status_to_exit_type (status), convert_pid_status_to_return_code (status), nullptr, true);

                // Notify delegate that our process has exited.
                SetState (StateType::eStateExited, true);
            }
            else
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s() tid = %"  PRIu64 " main thread now exited (%s)", __FUNCTION__, pid, thread_found ? "stopped tracking thread metadata" : "thread metadata not found");
            }
        }
        else
        {
            // Do we want to report to the delegate in this case?  I think not.  If this was an orderly
            // thread exit, we would already have received the SIGTRAP | (PTRACE_EVENT_EXIT << 8) signal,
            // and we would have done an all-stop then.
            if (log)
                log->Printf ("NativeProcessLinux::%s() tid = %"  PRIu64 " handling non-main thread exit (%s)", __FUNCTION__, pid, thread_found ? "stopped tracking thread metadata" : "thread metadata not found");
        }
        return;
    }

    // Get details on the signal raised.
    siginfo_t info;
    const auto err = GetSignalInfo(pid, &info);
    if (err.Success())
    {
        // We have retrieved the signal info.  Dispatch appropriately.
        if (info.si_signo == SIGTRAP)
            MonitorSIGTRAP(&info, pid);
        else
            MonitorSignal(&info, pid, exited);
    }
    else
    {
        if (err.GetError() == EINVAL)
        {
            // This is a group stop reception for this tid.
            // We can reach here if we reinject SIGSTOP, SIGSTP, SIGTTIN or SIGTTOU into the
            // tracee, triggering the group-stop mechanism. Normally receiving these would stop
            // the process, pending a SIGCONT. Simulating this state in a debugger is hard and is
            // generally not needed (one use case is debugging background task being managed by a
            // shell). For general use, it is sufficient to stop the process in a signal-delivery
            // stop which happens before the group stop. This done by MonitorSignal and works
            // correctly for all signals.
            if (log)
                log->Printf("NativeProcessLinux::%s received a group stop for pid %" PRIu64 " tid %" PRIu64 ". Transparent handling of group stops not supported, resuming the thread.", __FUNCTION__, GetID (), pid);
            Resume(pid, signal);
        }
        else
        {
            // ptrace(GETSIGINFO) failed (but not due to group-stop).

            // A return value of ESRCH means the thread/process is no longer on the system,
            // so it was killed somehow outside of our control.  Either way, we can't do anything
            // with it anymore.

            // Stop tracking the metadata for the thread since it's entirely off the system now.
            const bool thread_found = StopTrackingThread (pid);

            if (log)
                log->Printf ("NativeProcessLinux::%s GetSignalInfo failed: %s, tid = %" PRIu64 ", signal = %d, status = %d (%s, %s, %s)",
                             __FUNCTION__, err.AsCString(), pid, signal, status, err.GetError() == ESRCH ? "thread/process killed" : "unknown reason", is_main_thread ? "is main thread" : "is not main thread", thread_found ? "thread metadata removed" : "thread metadata not found");

            if (is_main_thread)
            {
                // Notify the delegate - our process is not available but appears to have been killed outside
                // our control.  Is eStateExited the right exit state in this case?
                SetExitStatus (convert_pid_status_to_exit_type (status), convert_pid_status_to_return_code (status), nullptr, true);
                SetState (StateType::eStateExited, true);
            }
            else
            {
                // This thread was pulled out from underneath us.  Anything to do here? Do we want to do an all stop?
                if (log)
                    log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 " non-main thread exit occurred, didn't tell delegate anything since thread disappeared out from underneath us", __FUNCTION__, GetID (), pid);
            }
        }
    }
}

void
NativeProcessLinux::WaitForNewThread(::pid_t tid)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    NativeThreadProtocolSP new_thread_sp = GetThreadByID(tid);

    if (new_thread_sp)
    {
        // We are already tracking the thread - we got the event on the new thread (see
        // MonitorSignal) before this one. We are done.
        return;
    }

    // The thread is not tracked yet, let's wait for it to appear.
    int status = -1;
    ::pid_t wait_pid;
    do
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() received thread creation event for tid %" PRIu32 ". tid not tracked yet, waiting for thread to appear...", __FUNCTION__, tid);
        wait_pid = waitpid(tid, &status, __WALL);
    }
    while (wait_pid == -1 && errno == EINTR);
    // Since we are waiting on a specific tid, this must be the creation event. But let's do
    // some checks just in case.
    if (wait_pid != tid) {
        if (log)
            log->Printf ("NativeProcessLinux::%s() waiting for tid %" PRIu32 " failed. Assuming the thread has disappeared in the meantime", __FUNCTION__, tid);
        // The only way I know of this could happen is if the whole process was
        // SIGKILLed in the mean time. In any case, we can't do anything about that now.
        return;
    }
    if (WIFEXITED(status))
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() waiting for tid %" PRIu32 " returned an 'exited' event. Not tracking the thread.", __FUNCTION__, tid);
        // Also a very improbable event.
        return;
    }

    siginfo_t info;
    Error error = GetSignalInfo(tid, &info);
    if (error.Fail())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() GetSignalInfo for tid %" PRIu32 " failed. Assuming the thread has disappeared in the meantime.", __FUNCTION__, tid);
        return;
    }

    if (((info.si_pid != 0) || (info.si_code != SI_USER)) && log)
    {
        // We should be getting a thread creation signal here, but we received something
        // else. There isn't much we can do about it now, so we will just log that. Since the
        // thread is alive and we are receiving events from it, we shall pretend that it was
        // created properly.
        log->Printf ("NativeProcessLinux::%s() GetSignalInfo for tid %" PRIu32 " received unexpected signal with code %d from pid %d.", __FUNCTION__, tid, info.si_code, info.si_pid);
    }

    if (log)
        log->Printf ("NativeProcessLinux::%s() pid = %" PRIu64 ": tracking new thread tid %" PRIu32,
                 __FUNCTION__, GetID (), tid);

    new_thread_sp = AddThread(tid);
    std::static_pointer_cast<NativeThreadLinux> (new_thread_sp)->SetRunning ();
    Resume (tid, LLDB_INVALID_SIGNAL_NUMBER);
    ThreadWasCreated(tid);
}

void
NativeProcessLinux::MonitorSIGTRAP(const siginfo_t *info, lldb::pid_t pid)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    const bool is_main_thread = (pid == GetID ());

    assert(info && info->si_signo == SIGTRAP && "Unexpected child signal!");
    if (!info)
        return;

    Mutex::Locker locker (m_threads_mutex);

    // See if we can find a thread for this signal.
    NativeThreadProtocolSP thread_sp = GetThreadByID (pid);
    if (!thread_sp)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " no thread found for tid %" PRIu64, __FUNCTION__, GetID (), pid);
    }

    switch (info->si_code)
    {
    // TODO: these two cases are required if we want to support tracing of the inferiors' children.  We'd need this to debug a monitor.
    // case (SIGTRAP | (PTRACE_EVENT_FORK << 8)):
    // case (SIGTRAP | (PTRACE_EVENT_VFORK << 8)):

    case (SIGTRAP | (PTRACE_EVENT_CLONE << 8)):
    {
        // This is the notification on the parent thread which informs us of new thread
        // creation.
        // We don't want to do anything with the parent thread so we just resume it. In case we
        // want to implement "break on thread creation" functionality, we would need to stop
        // here.

        unsigned long event_message = 0;
        if (GetEventMessage (pid, &event_message).Fail())
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " received thread creation event but GetEventMessage failed so we don't know the new tid", __FUNCTION__, pid);
        } else 
            WaitForNewThread(event_message);

        Resume (pid, LLDB_INVALID_SIGNAL_NUMBER);
        break;
    }

    case (SIGTRAP | (PTRACE_EVENT_EXEC << 8)):
    {
        NativeThreadProtocolSP main_thread_sp;
        if (log)
            log->Printf ("NativeProcessLinux::%s() received exec event, code = %d", __FUNCTION__, info->si_code ^ SIGTRAP);

        // Exec clears any pending notifications.
        m_pending_notification_up.reset ();

        // Remove all but the main thread here.  Linux fork creates a new process which only copies the main thread.  Mutexes are in undefined state.
        if (log)
            log->Printf ("NativeProcessLinux::%s exec received, stop tracking all but main thread", __FUNCTION__);

        for (auto thread_sp : m_threads)
        {
            const bool is_main_thread = thread_sp && thread_sp->GetID () == GetID ();
            if (is_main_thread)
            {
                main_thread_sp = thread_sp;
                if (log)
                    log->Printf ("NativeProcessLinux::%s found main thread with tid %" PRIu64 ", keeping", __FUNCTION__, main_thread_sp->GetID ());
            }
            else
            {
                // Tell thread coordinator this thread is dead.
                if (log)
                    log->Printf ("NativeProcessLinux::%s discarding non-main-thread tid %" PRIu64 " due to exec", __FUNCTION__, thread_sp->GetID ());
            }
        }

        m_threads.clear ();

        if (main_thread_sp)
        {
            m_threads.push_back (main_thread_sp);
            SetCurrentThreadID (main_thread_sp->GetID ());
            std::static_pointer_cast<NativeThreadLinux> (main_thread_sp)->SetStoppedByExec ();
        }
        else
        {
            SetCurrentThreadID (LLDB_INVALID_THREAD_ID);
            if (log)
                log->Printf ("NativeProcessLinux::%s pid %" PRIu64 "no main thread found, discarded all threads, we're in a no-thread state!", __FUNCTION__, GetID ());
        }

        // Tell coordinator about about the "new" (since exec) stopped main thread.
        const lldb::tid_t main_thread_tid = GetID ();
        ThreadWasCreated(main_thread_tid);

        // NOTE: ideally these next statements would execute at the same time as the coordinator thread create was executed.
        // Consider a handler that can execute when that happens.
        // Let our delegate know we have just exec'd.
        NotifyDidExec ();

        // If we have a main thread, indicate we are stopped.
        assert (main_thread_sp && "exec called during ptraced process but no main thread metadata tracked");

        // Let the process know we're stopped.
        StopRunningThreads (pid);

        break;
    }

    case (SIGTRAP | (PTRACE_EVENT_EXIT << 8)):
    {
        // The inferior process or one of its threads is about to exit.
        // We don't want to do anything with the thread so we just resume it. In case we
        // want to implement "break on thread exit" functionality, we would need to stop
        // here.

        unsigned long data = 0;
        if (GetEventMessage(pid, &data).Fail())
            data = -1;

        if (log)
        {
            log->Printf ("NativeProcessLinux::%s() received PTRACE_EVENT_EXIT, data = %lx (WIFEXITED=%s,WIFSIGNALED=%s), pid = %" PRIu64 " (%s)",
                         __FUNCTION__,
                         data, WIFEXITED (data) ? "true" : "false", WIFSIGNALED (data) ? "true" : "false",
                         pid,
                    is_main_thread ? "is main thread" : "not main thread");
        }

        if (is_main_thread)
        {
            SetExitStatus (convert_pid_status_to_exit_type (data), convert_pid_status_to_return_code (data), nullptr, true);
        }

        Resume(pid, LLDB_INVALID_SIGNAL_NUMBER);

        break;
    }

    case 0:
    case TRAP_TRACE:  // We receive this on single stepping.
    case TRAP_HWBKPT: // We receive this on watchpoint hit
        if (thread_sp)
        {
            // If a watchpoint was hit, report it
            uint32_t wp_index;
            Error error = thread_sp->GetRegisterContext()->GetWatchpointHitIndex(wp_index, (lldb::addr_t)info->si_addr);
            if (error.Fail() && log)
                log->Printf("NativeProcessLinux::%s() "
                            "received error while checking for watchpoint hits, "
                            "pid = %" PRIu64 " error = %s",
                            __FUNCTION__, pid, error.AsCString());
            if (wp_index != LLDB_INVALID_INDEX32)
            {
                MonitorWatchpoint(pid, thread_sp, wp_index);
                break;
            }
        }
        // Otherwise, report step over
        MonitorTrace(pid, thread_sp);
        break;

    case SI_KERNEL:
    case TRAP_BRKPT:
        MonitorBreakpoint(pid, thread_sp);
        break;

    case SIGTRAP:
    case (SIGTRAP | 0x80):
        if (log)
            log->Printf ("NativeProcessLinux::%s() received unknown SIGTRAP system call stop event, pid %" PRIu64 "tid %" PRIu64 ", resuming", __FUNCTION__, GetID (), pid);

        // Ignore these signals until we know more about them.
        Resume(pid, LLDB_INVALID_SIGNAL_NUMBER);
        break;

    default:
        assert(false && "Unexpected SIGTRAP code!");
        if (log)
            log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 "tid %" PRIu64 " received unhandled SIGTRAP code: 0x%d",
                    __FUNCTION__, GetID (), pid, info->si_code);
        break;
        
    }
}

void
NativeProcessLinux::MonitorTrace(lldb::pid_t pid, NativeThreadProtocolSP thread_sp)
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf("NativeProcessLinux::%s() received trace event, pid = %" PRIu64 " (single stepping)",
                __FUNCTION__, pid);

    if (thread_sp)
        std::static_pointer_cast<NativeThreadLinux>(thread_sp)->SetStoppedByTrace();

    // This thread is currently stopped.
    ThreadDidStop(pid, false);

    // Here we don't have to request the rest of the threads to stop or request a deferred stop.
    // This would have already happened at the time the Resume() with step operation was signaled.
    // At this point, we just need to say we stopped, and the deferred notifcation will fire off
    // once all running threads have checked in as stopped.
    SetCurrentThreadID(pid);
    // Tell the process we have a stop (from software breakpoint).
    StopRunningThreads(pid);
}

void
NativeProcessLinux::MonitorBreakpoint(lldb::pid_t pid, NativeThreadProtocolSP thread_sp)
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf("NativeProcessLinux::%s() received breakpoint event, pid = %" PRIu64,
                __FUNCTION__, pid);

    // This thread is currently stopped.
    ThreadDidStop(pid, false);

    // Mark the thread as stopped at breakpoint.
    if (thread_sp)
    {
        std::static_pointer_cast<NativeThreadLinux>(thread_sp)->SetStoppedByBreakpoint();
        Error error = FixupBreakpointPCAsNeeded(thread_sp);
        if (error.Fail())
            if (log)
                log->Printf("NativeProcessLinux::%s() pid = %" PRIu64 " fixup: %s",
                        __FUNCTION__, pid, error.AsCString());

        if (m_threads_stepping_with_breakpoint.find(pid) != m_threads_stepping_with_breakpoint.end())
            std::static_pointer_cast<NativeThreadLinux>(thread_sp)->SetStoppedByTrace();
    }
    else
        if (log)
            log->Printf("NativeProcessLinux::%s()  pid = %" PRIu64 ": "
                    "warning, cannot process software breakpoint since no thread metadata",
                    __FUNCTION__, pid);


    // We need to tell all other running threads before we notify the delegate about this stop.
    StopRunningThreads(pid);
}

void
NativeProcessLinux::MonitorWatchpoint(lldb::pid_t pid, NativeThreadProtocolSP thread_sp, uint32_t wp_index)
{
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf("NativeProcessLinux::%s() received watchpoint event, "
                    "pid = %" PRIu64 ", wp_index = %" PRIu32,
                    __FUNCTION__, pid, wp_index);

    // This thread is currently stopped.
    ThreadDidStop(pid, false);

    // Mark the thread as stopped at watchpoint.
    // The address is at (lldb::addr_t)info->si_addr if we need it.
    lldbassert(thread_sp && "thread_sp cannot be NULL");
    std::static_pointer_cast<NativeThreadLinux>(thread_sp)->SetStoppedByWatchpoint(wp_index);

    // We need to tell all other running threads before we notify the delegate about this stop.
    StopRunningThreads(pid);
}

void
NativeProcessLinux::MonitorSignal(const siginfo_t *info, lldb::pid_t pid, bool exited)
{
    assert (info && "null info");
    if (!info)
        return;

    const int signo = info->si_signo;
    const bool is_from_llgs = info->si_pid == getpid ();

    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // POSIX says that process behaviour is undefined after it ignores a SIGFPE,
    // SIGILL, SIGSEGV, or SIGBUS *unless* that signal was generated by a
    // kill(2) or raise(3).  Similarly for tgkill(2) on Linux.
    //
    // IOW, user generated signals never generate what we consider to be a
    // "crash".
    //
    // Similarly, ACK signals generated by this monitor.

    Mutex::Locker locker (m_threads_mutex);

    // See if we can find a thread for this signal.
    NativeThreadProtocolSP thread_sp = GetThreadByID (pid);
    if (!thread_sp)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " no thread found for tid %" PRIu64, __FUNCTION__, GetID (), pid);
    }

    // Handle the signal.
    if (info->si_code == SI_TKILL || info->si_code == SI_USER)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() received signal %s (%d) with code %s, (siginfo pid = %d (%s), waitpid pid = %" PRIu64 ")",
                            __FUNCTION__,
                            GetUnixSignals ().GetSignalAsCString (signo),
                            signo,
                            (info->si_code == SI_TKILL ? "SI_TKILL" : "SI_USER"),
                            info->si_pid,
                            is_from_llgs ? "from llgs" : "not from llgs",
                            pid);
    }

    // Check for new thread notification.
    if ((info->si_pid == 0) && (info->si_code == SI_USER))
    {
        // A new thread creation is being signaled. This is one of two parts that come in
        // a non-deterministic order. This code handles the case where the new thread event comes
        // before the event on the parent thread. For the opposite case see code in
        // MonitorSIGTRAP.
        if (log)
            log->Printf ("NativeProcessLinux::%s() pid = %" PRIu64 " tid %" PRIu64 ": new thread notification",
                     __FUNCTION__, GetID (), pid);

        thread_sp = AddThread(pid);
        assert (thread_sp.get() && "failed to create the tracking data for newly created inferior thread");
        // We can now resume the newly created thread.
        std::static_pointer_cast<NativeThreadLinux> (thread_sp)->SetRunning ();
        Resume (pid, LLDB_INVALID_SIGNAL_NUMBER);
        ThreadWasCreated(pid);
        // Done handling.
        return;
    }

    // Check for thread stop notification.
    if (is_from_llgs && (info->si_code == SI_TKILL) && (signo == SIGSTOP))
    {
        // This is a tgkill()-based stop.
        if (thread_sp)
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " tid %" PRIu64 ", thread stopped",
                             __FUNCTION__,
                             GetID (),
                             pid);

            // Check that we're not already marked with a stop reason.
            // Note this thread really shouldn't already be marked as stopped - if we were, that would imply that
            // the kernel signaled us with the thread stopping which we handled and marked as stopped,
            // and that, without an intervening resume, we received another stop.  It is more likely
            // that we are missing the marking of a run state somewhere if we find that the thread was
            // marked as stopped.
            std::shared_ptr<NativeThreadLinux> linux_thread_sp = std::static_pointer_cast<NativeThreadLinux> (thread_sp);
            assert (linux_thread_sp && "linux_thread_sp is null!");

            const StateType thread_state = linux_thread_sp->GetState ();
            if (!StateIsStoppedState (thread_state, false))
            {
                // An inferior thread has stopped because of a SIGSTOP we have sent it.
                // Generally, these are not important stops and we don't want to report them as
                // they are just used to stop other threads when one thread (the one with the
                // *real* stop reason) hits a breakpoint (watchpoint, etc...). However, in the
                // case of an asynchronous Interrupt(), this *is* the real stop reason, so we
                // leave the signal intact if this is the thread that was chosen as the
                // triggering thread.
                if (m_pending_notification_up && m_pending_notification_up->triggering_tid == pid)
                    linux_thread_sp->SetStoppedBySignal(SIGSTOP);
                else
                    linux_thread_sp->SetStoppedBySignal(0);

                SetCurrentThreadID (thread_sp->GetID ());
                ThreadDidStop (thread_sp->GetID (), true);
            }
            else
            {
                if (log)
                {
                    // Retrieve the signal name if the thread was stopped by a signal.
                    int stop_signo = 0;
                    const bool stopped_by_signal = linux_thread_sp->IsStopped (&stop_signo);
                    const char *signal_name = stopped_by_signal ? GetUnixSignals ().GetSignalAsCString (stop_signo) : "<not stopped by signal>";
                    if (!signal_name)
                        signal_name = "<no-signal-name>";

                    log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " tid %" PRIu64 ", thread was already marked as a stopped state (state=%s, signal=%d (%s)), leaving stop signal as is",
                                 __FUNCTION__,
                                 GetID (),
                                 linux_thread_sp->GetID (),
                                 StateAsCString (thread_state),
                                 stop_signo,
                                 signal_name);
                }
                ThreadDidStop (thread_sp->GetID (), false);
            }
        }

        // Done handling.
        return;
    }

    if (log)
        log->Printf ("NativeProcessLinux::%s() received signal %s", __FUNCTION__, GetUnixSignals ().GetSignalAsCString (signo));

    // This thread is stopped.
    ThreadDidStop (pid, false);

    switch (signo)
    {
    case SIGSEGV:
    case SIGILL:
    case SIGFPE:
    case SIGBUS:
        if (thread_sp)
            std::static_pointer_cast<NativeThreadLinux> (thread_sp)->SetCrashedWithException (*info);
        break;
    default:
        // This is just a pre-signal-delivery notification of the incoming signal.
        if (thread_sp)
            std::static_pointer_cast<NativeThreadLinux> (thread_sp)->SetStoppedBySignal (signo);

        break;
    }

    // Send a stop to the debugger after we get all other threads to stop.
    StopRunningThreads (pid);
}

namespace {

struct EmulatorBaton
{
    NativeProcessLinux* m_process;
    NativeRegisterContext* m_reg_context;

    // eRegisterKindDWARF -> RegsiterValue
    std::unordered_map<uint32_t, RegisterValue> m_register_values;

    EmulatorBaton(NativeProcessLinux* process, NativeRegisterContext* reg_context) :
            m_process(process), m_reg_context(reg_context) {}
};

} // anonymous namespace

static size_t
ReadMemoryCallback (EmulateInstruction *instruction,
                    void *baton,
                    const EmulateInstruction::Context &context, 
                    lldb::addr_t addr, 
                    void *dst,
                    size_t length)
{
    EmulatorBaton* emulator_baton = static_cast<EmulatorBaton*>(baton);

    size_t bytes_read;
    emulator_baton->m_process->ReadMemory(addr, dst, length, bytes_read);
    return bytes_read;
}

static bool
ReadRegisterCallback (EmulateInstruction *instruction,
                      void *baton,
                      const RegisterInfo *reg_info,
                      RegisterValue &reg_value)
{
    EmulatorBaton* emulator_baton = static_cast<EmulatorBaton*>(baton);

    auto it = emulator_baton->m_register_values.find(reg_info->kinds[eRegisterKindDWARF]);
    if (it != emulator_baton->m_register_values.end())
    {
        reg_value = it->second;
        return true;
    }

    // The emulator only fill in the dwarf regsiter numbers (and in some case
    // the generic register numbers). Get the full register info from the
    // register context based on the dwarf register numbers.
    const RegisterInfo* full_reg_info = emulator_baton->m_reg_context->GetRegisterInfo(
            eRegisterKindDWARF, reg_info->kinds[eRegisterKindDWARF]);

    Error error = emulator_baton->m_reg_context->ReadRegister(full_reg_info, reg_value);
    if (error.Success())
        return true;

    return false;
}

static bool
WriteRegisterCallback (EmulateInstruction *instruction,
                       void *baton,
                       const EmulateInstruction::Context &context,
                       const RegisterInfo *reg_info,
                       const RegisterValue &reg_value)
{
    EmulatorBaton* emulator_baton = static_cast<EmulatorBaton*>(baton);
    emulator_baton->m_register_values[reg_info->kinds[eRegisterKindDWARF]] = reg_value;
    return true;
}

static size_t
WriteMemoryCallback (EmulateInstruction *instruction,
                     void *baton,
                     const EmulateInstruction::Context &context, 
                     lldb::addr_t addr, 
                     const void *dst,
                     size_t length)
{
    return length;
}

static lldb::addr_t
ReadFlags (NativeRegisterContext* regsiter_context)
{
    const RegisterInfo* flags_info = regsiter_context->GetRegisterInfo(
            eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS);
    return regsiter_context->ReadRegisterAsUnsigned(flags_info, LLDB_INVALID_ADDRESS);
}

Error
NativeProcessLinux::SetupSoftwareSingleStepping(NativeThreadProtocolSP thread_sp)
{
    Error error;
    NativeRegisterContextSP register_context_sp = thread_sp->GetRegisterContext();

    std::unique_ptr<EmulateInstruction> emulator_ap(
        EmulateInstruction::FindPlugin(m_arch, eInstructionTypePCModifying, nullptr));

    if (emulator_ap == nullptr)
        return Error("Instruction emulator not found!");

    EmulatorBaton baton(this, register_context_sp.get());
    emulator_ap->SetBaton(&baton);
    emulator_ap->SetReadMemCallback(&ReadMemoryCallback);
    emulator_ap->SetReadRegCallback(&ReadRegisterCallback);
    emulator_ap->SetWriteMemCallback(&WriteMemoryCallback);
    emulator_ap->SetWriteRegCallback(&WriteRegisterCallback);

    if (!emulator_ap->ReadInstruction())
        return Error("Read instruction failed!");

    bool emulation_result = emulator_ap->EvaluateInstruction(eEmulateInstructionOptionAutoAdvancePC);

    const RegisterInfo* reg_info_pc = register_context_sp->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    const RegisterInfo* reg_info_flags = register_context_sp->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS);

    auto pc_it = baton.m_register_values.find(reg_info_pc->kinds[eRegisterKindDWARF]);
    auto flags_it = baton.m_register_values.find(reg_info_flags->kinds[eRegisterKindDWARF]);

    lldb::addr_t next_pc;
    lldb::addr_t next_flags;
    if (emulation_result)
    {
        assert(pc_it != baton.m_register_values.end() && "Emulation was successfull but PC wasn't updated");
        next_pc = pc_it->second.GetAsUInt64();

        if (flags_it != baton.m_register_values.end())
            next_flags = flags_it->second.GetAsUInt64();
        else
            next_flags = ReadFlags (register_context_sp.get());
    }
    else if (pc_it == baton.m_register_values.end())
    {
        // Emulate instruction failed and it haven't changed PC. Advance PC
        // with the size of the current opcode because the emulation of all
        // PC modifying instruction should be successful. The failure most
        // likely caused by a not supported instruction which don't modify PC.
        next_pc = register_context_sp->GetPC() + emulator_ap->GetOpcode().GetByteSize();
        next_flags = ReadFlags (register_context_sp.get());
    }
    else
    {
        // The instruction emulation failed after it modified the PC. It is an
        // unknown error where we can't continue because the next instruction is
        // modifying the PC but we don't  know how.
        return Error ("Instruction emulation failed unexpectedly.");
    }

    if (m_arch.GetMachine() == llvm::Triple::arm)
    {
        if (next_flags & 0x20)
        {
            // Thumb mode
            error = SetSoftwareBreakpoint(next_pc, 2);
        }
        else
        {
            // Arm mode
            error = SetSoftwareBreakpoint(next_pc, 4);
        }
    }
    else if (m_arch.GetMachine() == llvm::Triple::mips64
            || m_arch.GetMachine() == llvm::Triple::mips64el)
        error = SetSoftwareBreakpoint(next_pc, 4);
    else
    {
        // No size hint is given for the next breakpoint
        error = SetSoftwareBreakpoint(next_pc, 0);
    }

    if (error.Fail())
        return error;

    m_threads_stepping_with_breakpoint.insert({thread_sp->GetID(), next_pc});

    return Error();
}

bool
NativeProcessLinux::SupportHardwareSingleStepping() const
{
    if (m_arch.GetMachine() == llvm::Triple::arm
        || m_arch.GetMachine() == llvm::Triple::mips64 || m_arch.GetMachine() == llvm::Triple::mips64el)
        return false;
    return true;
}

Error
NativeProcessLinux::Resume (const ResumeActionList &resume_actions)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("NativeProcessLinux::%s called: pid %" PRIu64, __FUNCTION__, GetID ());

    bool software_single_step = !SupportHardwareSingleStepping();

    Monitor::ScopedOperationLock monitor_lock(*m_monitor_up);
    Mutex::Locker locker (m_threads_mutex);

    if (software_single_step)
    {
        for (auto thread_sp : m_threads)
        {
            assert (thread_sp && "thread list should not contain NULL threads");

            const ResumeAction *const action = resume_actions.GetActionForThread (thread_sp->GetID (), true);
            if (action == nullptr)
                continue;

            if (action->state == eStateStepping)
            {
                Error error = SetupSoftwareSingleStepping(thread_sp);
                if (error.Fail())
                    return error;
            }
        }
    }

    for (auto thread_sp : m_threads)
    {
        assert (thread_sp && "thread list should not contain NULL threads");

        const ResumeAction *const action = resume_actions.GetActionForThread (thread_sp->GetID (), true);

        if (action == nullptr)
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s no action specified for pid %" PRIu64 " tid %" PRIu64,
                    __FUNCTION__, GetID (), thread_sp->GetID ());
            continue;
        }

        if (log)
        {
            log->Printf ("NativeProcessLinux::%s processing resume action state %s for pid %" PRIu64 " tid %" PRIu64, 
                    __FUNCTION__, StateAsCString (action->state), GetID (), thread_sp->GetID ());
        }

        switch (action->state)
        {
        case eStateRunning:
        {
            // Run the thread, possibly feeding it the signal.
            const int signo = action->signal;
            ResumeThread(thread_sp->GetID (),
                    [=](lldb::tid_t tid_to_resume, bool supress_signal)
                    {
                        std::static_pointer_cast<NativeThreadLinux> (thread_sp)->SetRunning ();
                        // Pass this signal number on to the inferior to handle.
                        const auto resume_result = Resume (tid_to_resume, (signo > 0 && !supress_signal) ? signo : LLDB_INVALID_SIGNAL_NUMBER);
                        if (resume_result.Success())
                            SetState(eStateRunning, true);
                        return resume_result;
                    },
                    false);
            break;
        }

        case eStateStepping:
        {
            // Request the step.
            const int signo = action->signal;
            ResumeThread(thread_sp->GetID (),
                    [=](lldb::tid_t tid_to_step, bool supress_signal)
                    {
                        std::static_pointer_cast<NativeThreadLinux> (thread_sp)->SetStepping ();

                        Error step_result;
                        if (software_single_step)
                            step_result = Resume (tid_to_step, (signo > 0 && !supress_signal) ? signo : LLDB_INVALID_SIGNAL_NUMBER);
                        else
                            step_result = SingleStep (tid_to_step,(signo > 0 && !supress_signal) ? signo : LLDB_INVALID_SIGNAL_NUMBER);

                        assert (step_result.Success() && "SingleStep() failed");
                        if (step_result.Success())
                            SetState(eStateStepping, true);
                        return step_result;
                    },
                    false);
            break;
        }

        case eStateSuspended:
        case eStateStopped:
            lldbassert(0 && "Unexpected state");

        default:
            return Error ("NativeProcessLinux::%s (): unexpected state %s specified for pid %" PRIu64 ", tid %" PRIu64,
                    __FUNCTION__, StateAsCString (action->state), GetID (), thread_sp->GetID ());
        }
    }

    return Error();
}

Error
NativeProcessLinux::Halt ()
{
    Error error;

    if (kill (GetID (), SIGSTOP) != 0)
        error.SetErrorToErrno ();

    return error;
}

Error
NativeProcessLinux::Detach ()
{
    Error error;

    // Tell ptrace to detach from the process.
    if (GetID () != LLDB_INVALID_PROCESS_ID)
        error = Detach (GetID ());

    // Stop monitoring the inferior.
    m_monitor_up->Terminate();

    // No error.
    return error;
}

Error
NativeProcessLinux::Signal (int signo)
{
    Error error;

    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessLinux::%s: sending signal %d (%s) to pid %" PRIu64, 
                __FUNCTION__, signo,  GetUnixSignals ().GetSignalAsCString (signo), GetID ());

    if (kill(GetID(), signo))
        error.SetErrorToErrno();

    return error;
}

Error
NativeProcessLinux::Interrupt ()
{
    // Pick a running thread (or if none, a not-dead stopped thread) as
    // the chosen thread that will be the stop-reason thread.
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    NativeThreadProtocolSP running_thread_sp;
    NativeThreadProtocolSP stopped_thread_sp;
        
    if (log)
        log->Printf ("NativeProcessLinux::%s selecting running thread for interrupt target", __FUNCTION__);

    Monitor::ScopedOperationLock monitor_lock(*m_monitor_up);
    Mutex::Locker locker (m_threads_mutex);

    for (auto thread_sp : m_threads)
    {
        // The thread shouldn't be null but lets just cover that here.
        if (!thread_sp)
            continue;

        // If we have a running or stepping thread, we'll call that the
        // target of the interrupt.
        const auto thread_state = thread_sp->GetState ();
        if (thread_state == eStateRunning ||
            thread_state == eStateStepping)
        {
            running_thread_sp = thread_sp;
            break;
        }
        else if (!stopped_thread_sp && StateIsStoppedState (thread_state, true))
        {
            // Remember the first non-dead stopped thread.  We'll use that as a backup if there are no running threads.
            stopped_thread_sp = thread_sp;
        }
    }

    if (!running_thread_sp && !stopped_thread_sp)
    {
        Error error("found no running/stepping or live stopped threads as target for interrupt");
        if (log)
            log->Printf ("NativeProcessLinux::%s skipping due to error: %s", __FUNCTION__, error.AsCString ());

        return error;
    }

    NativeThreadProtocolSP deferred_signal_thread_sp = running_thread_sp ? running_thread_sp : stopped_thread_sp;

    if (log)
        log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " %s tid %" PRIu64 " chosen for interrupt target",
                     __FUNCTION__,
                     GetID (),
                     running_thread_sp ? "running" : "stopped",
                     deferred_signal_thread_sp->GetID ());

    StopRunningThreads(deferred_signal_thread_sp->GetID());

    return Error();
}

Error
NativeProcessLinux::Kill ()
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessLinux::%s called for PID %" PRIu64, __FUNCTION__, GetID ());

    Error error;

    switch (m_state)
    {
        case StateType::eStateInvalid:
        case StateType::eStateExited:
        case StateType::eStateCrashed:
        case StateType::eStateDetached:
        case StateType::eStateUnloaded:
            // Nothing to do - the process is already dead.
            if (log)
                log->Printf ("NativeProcessLinux::%s ignored for PID %" PRIu64 " due to current state: %s", __FUNCTION__, GetID (), StateAsCString (m_state));
            return error;

        case StateType::eStateConnected:
        case StateType::eStateAttaching:
        case StateType::eStateLaunching:
        case StateType::eStateStopped:
        case StateType::eStateRunning:
        case StateType::eStateStepping:
        case StateType::eStateSuspended:
            // We can try to kill a process in these states.
            break;
    }

    if (kill (GetID (), SIGKILL) != 0)
    {
        error.SetErrorToErrno ();
        return error;
    }

    return error;
}

static Error
ParseMemoryRegionInfoFromProcMapsLine (const std::string &maps_line, MemoryRegionInfo &memory_region_info)
{
    memory_region_info.Clear();

    StringExtractor line_extractor (maps_line.c_str ());

    // Format: {address_start_hex}-{address_end_hex} perms offset  dev   inode   pathname
    // perms: rwxp   (letter is present if set, '-' if not, final character is p=private, s=shared).

    // Parse out the starting address
    lldb::addr_t start_address = line_extractor.GetHexMaxU64 (false, 0);

    // Parse out hyphen separating start and end address from range.
    if (!line_extractor.GetBytesLeft () || (line_extractor.GetChar () != '-'))
        return Error ("malformed /proc/{pid}/maps entry, missing dash between address range");

    // Parse out the ending address
    lldb::addr_t end_address = line_extractor.GetHexMaxU64 (false, start_address);

    // Parse out the space after the address.
    if (!line_extractor.GetBytesLeft () || (line_extractor.GetChar () != ' '))
        return Error ("malformed /proc/{pid}/maps entry, missing space after range");

    // Save the range.
    memory_region_info.GetRange ().SetRangeBase (start_address);
    memory_region_info.GetRange ().SetRangeEnd (end_address);

    // Parse out each permission entry.
    if (line_extractor.GetBytesLeft () < 4)
        return Error ("malformed /proc/{pid}/maps entry, missing some portion of permissions");

    // Handle read permission.
    const char read_perm_char = line_extractor.GetChar ();
    if (read_perm_char == 'r')
        memory_region_info.SetReadable (MemoryRegionInfo::OptionalBool::eYes);
    else
    {
        assert ( (read_perm_char == '-') && "unexpected /proc/{pid}/maps read permission char" );
        memory_region_info.SetReadable (MemoryRegionInfo::OptionalBool::eNo);
    }

    // Handle write permission.
    const char write_perm_char = line_extractor.GetChar ();
    if (write_perm_char == 'w')
        memory_region_info.SetWritable (MemoryRegionInfo::OptionalBool::eYes);
    else
    {
        assert ( (write_perm_char == '-') && "unexpected /proc/{pid}/maps write permission char" );
        memory_region_info.SetWritable (MemoryRegionInfo::OptionalBool::eNo);
    }

    // Handle execute permission.
    const char exec_perm_char = line_extractor.GetChar ();
    if (exec_perm_char == 'x')
        memory_region_info.SetExecutable (MemoryRegionInfo::OptionalBool::eYes);
    else
    {
        assert ( (exec_perm_char == '-') && "unexpected /proc/{pid}/maps exec permission char" );
        memory_region_info.SetExecutable (MemoryRegionInfo::OptionalBool::eNo);
    }

    return Error ();
}

Error
NativeProcessLinux::GetMemoryRegionInfo (lldb::addr_t load_addr, MemoryRegionInfo &range_info)
{
    // FIXME review that the final memory region returned extends to the end of the virtual address space,
    // with no perms if it is not mapped.

    // Use an approach that reads memory regions from /proc/{pid}/maps.
    // Assume proc maps entries are in ascending order.
    // FIXME assert if we find differently.
    Mutex::Locker locker (m_mem_region_cache_mutex);

    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    Error error;

    if (m_supports_mem_region == LazyBool::eLazyBoolNo)
    {
        // We're done.
        error.SetErrorString ("unsupported");
        return error;
    }

    // If our cache is empty, pull the latest.  There should always be at least one memory region
    // if memory region handling is supported.
    if (m_mem_region_cache.empty ())
    {
        error = ProcFileReader::ProcessLineByLine (GetID (), "maps",
             [&] (const std::string &line) -> bool
             {
                 MemoryRegionInfo info;
                 const Error parse_error = ParseMemoryRegionInfoFromProcMapsLine (line, info);
                 if (parse_error.Success ())
                 {
                     m_mem_region_cache.push_back (info);
                     return true;
                 }
                 else
                 {
                     if (log)
                         log->Printf ("NativeProcessLinux::%s failed to parse proc maps line '%s': %s", __FUNCTION__, line.c_str (), error.AsCString ());
                     return false;
                 }
             });

        // If we had an error, we'll mark unsupported.
        if (error.Fail ())
        {
            m_supports_mem_region = LazyBool::eLazyBoolNo;
            return error;
        }
        else if (m_mem_region_cache.empty ())
        {
            // No entries after attempting to read them.  This shouldn't happen if /proc/{pid}/maps
            // is supported.  Assume we don't support map entries via procfs.
            if (log)
                log->Printf ("NativeProcessLinux::%s failed to find any procfs maps entries, assuming no support for memory region metadata retrieval", __FUNCTION__);
            m_supports_mem_region = LazyBool::eLazyBoolNo;
            error.SetErrorString ("not supported");
            return error;
        }

        if (log)
            log->Printf ("NativeProcessLinux::%s read %" PRIu64 " memory region entries from /proc/%" PRIu64 "/maps", __FUNCTION__, static_cast<uint64_t> (m_mem_region_cache.size ()), GetID ());

        // We support memory retrieval, remember that.
        m_supports_mem_region = LazyBool::eLazyBoolYes;
    }
    else
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s reusing %" PRIu64 " cached memory region entries", __FUNCTION__, static_cast<uint64_t> (m_mem_region_cache.size ()));
    }

    lldb::addr_t prev_base_address = 0;

    // FIXME start by finding the last region that is <= target address using binary search.  Data is sorted.
    // There can be a ton of regions on pthreads apps with lots of threads.
    for (auto it = m_mem_region_cache.begin(); it != m_mem_region_cache.end (); ++it)
    {
        MemoryRegionInfo &proc_entry_info = *it;

        // Sanity check assumption that /proc/{pid}/maps entries are ascending.
        assert ((proc_entry_info.GetRange ().GetRangeBase () >= prev_base_address) && "descending /proc/pid/maps entries detected, unexpected");
        prev_base_address = proc_entry_info.GetRange ().GetRangeBase ();

        // If the target address comes before this entry, indicate distance to next region.
        if (load_addr < proc_entry_info.GetRange ().GetRangeBase ())
        {
            range_info.GetRange ().SetRangeBase (load_addr);
            range_info.GetRange ().SetByteSize (proc_entry_info.GetRange ().GetRangeBase () - load_addr);
            range_info.SetReadable (MemoryRegionInfo::OptionalBool::eNo);
            range_info.SetWritable (MemoryRegionInfo::OptionalBool::eNo);
            range_info.SetExecutable (MemoryRegionInfo::OptionalBool::eNo);

            return error;
        }
        else if (proc_entry_info.GetRange ().Contains (load_addr))
        {
            // The target address is within the memory region we're processing here.
            range_info = proc_entry_info;
            return error;
        }

        // The target memory address comes somewhere after the region we just parsed.
    }

    // If we made it here, we didn't find an entry that contained the given address.
    error.SetErrorString ("address comes after final region");

    if (log)
        log->Printf ("NativeProcessLinux::%s failed to find map entry for address 0x%" PRIx64 ": %s", __FUNCTION__, load_addr, error.AsCString ());

    return error;
}

void
NativeProcessLinux::DoStopIDBumped (uint32_t newBumpId)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessLinux::%s(newBumpId=%" PRIu32 ") called", __FUNCTION__, newBumpId);

    {
        Mutex::Locker locker (m_mem_region_cache_mutex);
        if (log)
            log->Printf ("NativeProcessLinux::%s clearing %" PRIu64 " entries from the cache", __FUNCTION__, static_cast<uint64_t> (m_mem_region_cache.size ()));
        m_mem_region_cache.clear ();
    }
}

Error
NativeProcessLinux::AllocateMemory(size_t size, uint32_t permissions, lldb::addr_t &addr)
{
    // FIXME implementing this requires the equivalent of
    // InferiorCallPOSIX::InferiorCallMmap, which depends on
    // functional ThreadPlans working with Native*Protocol.
#if 1
    return Error ("not implemented yet");
#else
    addr = LLDB_INVALID_ADDRESS;

    unsigned prot = 0;
    if (permissions & lldb::ePermissionsReadable)
        prot |= eMmapProtRead;
    if (permissions & lldb::ePermissionsWritable)
        prot |= eMmapProtWrite;
    if (permissions & lldb::ePermissionsExecutable)
        prot |= eMmapProtExec;

    // TODO implement this directly in NativeProcessLinux
    // (and lift to NativeProcessPOSIX if/when that class is
    // refactored out).
    if (InferiorCallMmap(this, addr, 0, size, prot,
                         eMmapFlagsAnon | eMmapFlagsPrivate, -1, 0)) {
        m_addr_to_mmap_size[addr] = size;
        return Error ();
    } else {
        addr = LLDB_INVALID_ADDRESS;
        return Error("unable to allocate %" PRIu64 " bytes of memory with permissions %s", size, GetPermissionsAsCString (permissions));
    }
#endif
}

Error
NativeProcessLinux::DeallocateMemory (lldb::addr_t addr)
{
    // FIXME see comments in AllocateMemory - required lower-level
    // bits not in place yet (ThreadPlans)
    return Error ("not implemented");
}

lldb::addr_t
NativeProcessLinux::GetSharedLibraryInfoAddress ()
{
#if 1
    // punt on this for now
    return LLDB_INVALID_ADDRESS;
#else
    // Return the image info address for the exe module
#if 1
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    ModuleSP module_sp;
    Error error = GetExeModuleSP (module_sp);
    if (error.Fail ())
    {
         if (log)
            log->Warning ("NativeProcessLinux::%s failed to retrieve exe module: %s", __FUNCTION__, error.AsCString ());
        return LLDB_INVALID_ADDRESS;
    }

    if (module_sp == nullptr)
    {
         if (log)
            log->Warning ("NativeProcessLinux::%s exe module returned was NULL", __FUNCTION__);
         return LLDB_INVALID_ADDRESS;
    }

    ObjectFileSP object_file_sp = module_sp->GetObjectFile ();
    if (object_file_sp == nullptr)
    {
         if (log)
            log->Warning ("NativeProcessLinux::%s exe module returned a NULL object file", __FUNCTION__);
         return LLDB_INVALID_ADDRESS;
    }

    return obj_file_sp->GetImageInfoAddress();
#else
    Target *target = &GetTarget();
    ObjectFile *obj_file = target->GetExecutableModule()->GetObjectFile();
    Address addr = obj_file->GetImageInfoAddress(target);

    if (addr.IsValid())
        return addr.GetLoadAddress(target);
    return LLDB_INVALID_ADDRESS;
#endif
#endif // punt on this for now
}

size_t
NativeProcessLinux::UpdateThreads ()
{
    // The NativeProcessLinux monitoring threads are always up to date
    // with respect to thread state and they keep the thread list
    // populated properly. All this method needs to do is return the
    // thread count.
    Mutex::Locker locker (m_threads_mutex);
    return m_threads.size ();
}

bool
NativeProcessLinux::GetArchitecture (ArchSpec &arch) const
{
    arch = m_arch;
    return true;
}

Error
NativeProcessLinux::GetSoftwareBreakpointPCOffset (NativeRegisterContextSP context_sp, uint32_t &actual_opcode_size)
{
    // FIXME put this behind a breakpoint protocol class that can be
    // set per architecture.  Need ARM, MIPS support here.
    static const uint8_t g_aarch64_opcode[] = { 0x00, 0x00, 0x20, 0xd4 };
    static const uint8_t g_i386_opcode [] = { 0xCC };
    static const uint8_t g_mips64_opcode[] = { 0x00, 0x00, 0x00, 0x0d };

    switch (m_arch.GetMachine ())
    {
        case llvm::Triple::aarch64:
            actual_opcode_size = static_cast<uint32_t> (sizeof(g_aarch64_opcode));
            return Error ();

        case llvm::Triple::arm:
            actual_opcode_size = 0; // On arm the PC don't get updated for breakpoint hits
            return Error ();

        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
            actual_opcode_size = static_cast<uint32_t> (sizeof(g_i386_opcode));
            return Error ();

        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
            actual_opcode_size = static_cast<uint32_t> (sizeof(g_mips64_opcode));
            return Error ();
        
        default:
            assert(false && "CPU type not supported!");
            return Error ("CPU type not supported");
    }
}

Error
NativeProcessLinux::SetBreakpoint (lldb::addr_t addr, uint32_t size, bool hardware)
{
    if (hardware)
        return Error ("NativeProcessLinux does not support hardware breakpoints");
    else
        return SetSoftwareBreakpoint (addr, size);
}

Error
NativeProcessLinux::GetSoftwareBreakpointTrapOpcode (size_t trap_opcode_size_hint,
                                                     size_t &actual_opcode_size,
                                                     const uint8_t *&trap_opcode_bytes)
{
    // FIXME put this behind a breakpoint protocol class that can be set per
    // architecture.  Need MIPS support here.
    static const uint8_t g_aarch64_opcode[] = { 0x00, 0x00, 0x20, 0xd4 };
    // The ARM reference recommends the use of 0xe7fddefe and 0xdefe but the
    // linux kernel does otherwise.
    static const uint8_t g_arm_breakpoint_opcode[] = { 0xf0, 0x01, 0xf0, 0xe7 };
    static const uint8_t g_i386_opcode [] = { 0xCC };
    static const uint8_t g_mips64_opcode[] = { 0x00, 0x00, 0x00, 0x0d };
    static const uint8_t g_mips64el_opcode[] = { 0x0d, 0x00, 0x00, 0x00 };
    static const uint8_t g_thumb_breakpoint_opcode[] = { 0x01, 0xde };

    switch (m_arch.GetMachine ())
    {
    case llvm::Triple::aarch64:
        trap_opcode_bytes = g_aarch64_opcode;
        actual_opcode_size = sizeof(g_aarch64_opcode);
        return Error ();

    case llvm::Triple::arm:
        switch (trap_opcode_size_hint)
        {
        case 2:
            trap_opcode_bytes = g_thumb_breakpoint_opcode;
            actual_opcode_size = sizeof(g_thumb_breakpoint_opcode);
            return Error ();
        case 4:
            trap_opcode_bytes = g_arm_breakpoint_opcode;
            actual_opcode_size = sizeof(g_arm_breakpoint_opcode);
            return Error ();
        default:
            assert(false && "Unrecognised trap opcode size hint!");
            return Error ("Unrecognised trap opcode size hint!");
        }

    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        trap_opcode_bytes = g_i386_opcode;
        actual_opcode_size = sizeof(g_i386_opcode);
        return Error ();

    case llvm::Triple::mips64:
        trap_opcode_bytes = g_mips64_opcode;
        actual_opcode_size = sizeof(g_mips64_opcode);
        return Error ();

    case llvm::Triple::mips64el:
        trap_opcode_bytes = g_mips64el_opcode;
        actual_opcode_size = sizeof(g_mips64el_opcode);
        return Error ();

    default:
        assert(false && "CPU type not supported!");
        return Error ("CPU type not supported");
    }
}

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGSEGV(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGSEGV);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGSEGV");
        break;
    case SI_KERNEL:
        // Linux will occasionally send spurious SI_KERNEL codes.
        // (this is poorly documented in sigaction)
        // One way to get this is via unaligned SIMD loads.
        reason = ProcessMessage::eInvalidAddress; // for lack of anything better
        break;
    case SEGV_MAPERR:
        reason = ProcessMessage::eInvalidAddress;
        break;
    case SEGV_ACCERR:
        reason = ProcessMessage::ePrivilegedAddress;
        break;
    }

    return reason;
}
#endif


#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGILL(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGILL);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGILL");
        break;
    case ILL_ILLOPC:
        reason = ProcessMessage::eIllegalOpcode;
        break;
    case ILL_ILLOPN:
        reason = ProcessMessage::eIllegalOperand;
        break;
    case ILL_ILLADR:
        reason = ProcessMessage::eIllegalAddressingMode;
        break;
    case ILL_ILLTRP:
        reason = ProcessMessage::eIllegalTrap;
        break;
    case ILL_PRVOPC:
        reason = ProcessMessage::ePrivilegedOpcode;
        break;
    case ILL_PRVREG:
        reason = ProcessMessage::ePrivilegedRegister;
        break;
    case ILL_COPROC:
        reason = ProcessMessage::eCoprocessorError;
        break;
    case ILL_BADSTK:
        reason = ProcessMessage::eInternalStackError;
        break;
    }

    return reason;
}
#endif

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGFPE(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGFPE);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGFPE");
        break;
    case FPE_INTDIV:
        reason = ProcessMessage::eIntegerDivideByZero;
        break;
    case FPE_INTOVF:
        reason = ProcessMessage::eIntegerOverflow;
        break;
    case FPE_FLTDIV:
        reason = ProcessMessage::eFloatDivideByZero;
        break;
    case FPE_FLTOVF:
        reason = ProcessMessage::eFloatOverflow;
        break;
    case FPE_FLTUND:
        reason = ProcessMessage::eFloatUnderflow;
        break;
    case FPE_FLTRES:
        reason = ProcessMessage::eFloatInexactResult;
        break;
    case FPE_FLTINV:
        reason = ProcessMessage::eFloatInvalidOperation;
        break;
    case FPE_FLTSUB:
        reason = ProcessMessage::eFloatSubscriptRange;
        break;
    }

    return reason;
}
#endif

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGBUS(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGBUS);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGBUS");
        break;
    case BUS_ADRALN:
        reason = ProcessMessage::eIllegalAlignment;
        break;
    case BUS_ADRERR:
        reason = ProcessMessage::eIllegalAddress;
        break;
    case BUS_OBJERR:
        reason = ProcessMessage::eHardwareError;
        break;
    }

    return reason;
}
#endif

Error
NativeProcessLinux::SetWatchpoint (lldb::addr_t addr, size_t size, uint32_t watch_flags, bool hardware)
{
    // The base SetWatchpoint will end up executing monitor operations. Let's lock the monitor
    // for it.
    Monitor::ScopedOperationLock monitor_lock(*m_monitor_up);
    return NativeProcessProtocol::SetWatchpoint(addr, size, watch_flags, hardware);
}

Error
NativeProcessLinux::RemoveWatchpoint (lldb::addr_t addr)
{
    // The base RemoveWatchpoint will end up executing monitor operations. Let's lock the monitor
    // for it.
    Monitor::ScopedOperationLock monitor_lock(*m_monitor_up);
    return NativeProcessProtocol::RemoveWatchpoint(addr);
}

Error
NativeProcessLinux::ReadMemory (lldb::addr_t addr, void *buf, size_t size, size_t &bytes_read)
{
    ReadOperation op(addr, buf, size, bytes_read);
    m_monitor_up->DoOperation(&op);
    return op.GetError ();
}

Error
NativeProcessLinux::ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf, size_t size, size_t &bytes_read)
{
    Error error = ReadMemory(addr, buf, size, bytes_read);
    if (error.Fail()) return error;
    return m_breakpoint_list.RemoveTrapsFromBuffer(addr, buf, size);
}

Error
NativeProcessLinux::WriteMemory(lldb::addr_t addr, const void *buf, size_t size, size_t &bytes_written)
{
    WriteOperation op(addr, buf, size, bytes_written);
    m_monitor_up->DoOperation(&op);
    return op.GetError ();
}

Error
NativeProcessLinux::ReadRegisterValue(lldb::tid_t tid, uint32_t offset, const char* reg_name,
                                      uint32_t size, RegisterValue &value)
{
    ReadRegOperation op(tid, offset, reg_name, value);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::WriteRegisterValue(lldb::tid_t tid, unsigned offset,
                                   const char* reg_name, const RegisterValue &value)
{
    WriteRegOperation op(tid, offset, reg_name, value);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::ReadGPR(lldb::tid_t tid, void *buf, size_t buf_size)
{
    ReadGPROperation op(tid, buf, buf_size);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::ReadFPR(lldb::tid_t tid, void *buf, size_t buf_size)
{
    ReadFPROperation op(tid, buf, buf_size);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::ReadRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset)
{
    ReadRegisterSetOperation op(tid, buf, buf_size, regset);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::ReadHardwareDebugInfo (lldb::tid_t tid, unsigned int &watch_count , unsigned int &break_count)
{
    ReadDBGROperation op(tid, watch_count, break_count);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::WriteHardwareDebugRegs (lldb::tid_t tid, lldb::addr_t *addr_buf, uint32_t *cntrl_buf, int type, int count)
{
    WriteDBGROperation op(tid, addr_buf, cntrl_buf, type, count);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::WriteGPR(lldb::tid_t tid, void *buf, size_t buf_size)
{
    WriteGPROperation op(tid, buf, buf_size);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::WriteFPR(lldb::tid_t tid, void *buf, size_t buf_size)
{
    WriteFPROperation op(tid, buf, buf_size);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::WriteRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset)
{
    WriteRegisterSetOperation op(tid, buf, buf_size, regset);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::Resume (lldb::tid_t tid, uint32_t signo)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    if (log)
        log->Printf ("NativeProcessLinux::%s() resuming thread = %"  PRIu64 " with signal %s", __FUNCTION__, tid,
                                 GetUnixSignals().GetSignalAsCString (signo));
    ResumeOperation op (tid, signo);
    m_monitor_up->DoOperation (&op);
    if (log)
        log->Printf ("NativeProcessLinux::%s() resuming thread = %"  PRIu64 " result = %s", __FUNCTION__, tid, op.GetError().Success() ? "true" : "false");
    return op.GetError();
}

Error
NativeProcessLinux::SingleStep(lldb::tid_t tid, uint32_t signo)
{
    SingleStepOperation op(tid, signo);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::GetSignalInfo(lldb::tid_t tid, void *siginfo)
{
    SiginfoOperation op(tid, siginfo);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::GetEventMessage(lldb::tid_t tid, unsigned long *message)
{
    EventMessageOperation op(tid, message);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

Error
NativeProcessLinux::Detach(lldb::tid_t tid)
{
    if (tid == LLDB_INVALID_THREAD_ID)
        return Error();

    DetachOperation op(tid);
    m_monitor_up->DoOperation(&op);
    return op.GetError();
}

bool
NativeProcessLinux::DupDescriptor(const char *path, int fd, int flags)
{
    int target_fd = open(path, flags, 0666);

    if (target_fd == -1)
        return false;

    if (dup2(target_fd, fd) == -1)
        return false;

    return (close(target_fd) == -1) ? false : true;
}

void
NativeProcessLinux::StartMonitorThread(const InitialOperation &initial_operation, Error &error)
{
    m_monitor_up.reset(new Monitor(initial_operation, this));
    error = m_monitor_up->Initialize();
    if (error.Fail()) {
        m_monitor_up.reset();
    }
}

bool
NativeProcessLinux::HasThreadNoLock (lldb::tid_t thread_id)
{
    for (auto thread_sp : m_threads)
    {
        assert (thread_sp && "thread list should not contain NULL threads");
        if (thread_sp->GetID () == thread_id)
        {
            // We have this thread.
            return true;
        }
    }

    // We don't have this thread.
    return false;
}

NativeThreadProtocolSP
NativeProcessLinux::MaybeGetThreadNoLock (lldb::tid_t thread_id)
{
    // CONSIDER organize threads by map - we can do better than linear.
    for (auto thread_sp : m_threads)
    {
        if (thread_sp->GetID () == thread_id)
            return thread_sp;
    }

    // We don't have this thread.
    return NativeThreadProtocolSP ();
}

bool
NativeProcessLinux::StopTrackingThread (lldb::tid_t thread_id)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);

    if (log)
        log->Printf("NativeProcessLinux::%s (tid: %" PRIu64 ")", __FUNCTION__, thread_id);

    bool found = false;

    Mutex::Locker locker (m_threads_mutex);
    for (auto it = m_threads.begin (); it != m_threads.end (); ++it)
    {
        if (*it && ((*it)->GetID () == thread_id))
        {
            m_threads.erase (it);
            found = true;
            break;
        }
    }

    // If we have a pending notification, remove this from the set.
    if (m_pending_notification_up)
    {
        m_pending_notification_up->wait_for_stop_tids.erase(thread_id);
        SignalIfAllThreadsStopped();
    }

    return found;
}

NativeThreadProtocolSP
NativeProcessLinux::AddThread (lldb::tid_t thread_id)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));

    Mutex::Locker locker (m_threads_mutex);

    if (log)
    {
        log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " adding thread with tid %" PRIu64,
                __FUNCTION__,
                GetID (),
                thread_id);
    }

    assert (!HasThreadNoLock (thread_id) && "attempted to add a thread by id that already exists");

    // If this is the first thread, save it as the current thread
    if (m_threads.empty ())
        SetCurrentThreadID (thread_id);

    NativeThreadProtocolSP thread_sp (new NativeThreadLinux (this, thread_id));
    m_threads.push_back (thread_sp);

    return thread_sp;
}

Error
NativeProcessLinux::FixupBreakpointPCAsNeeded (NativeThreadProtocolSP &thread_sp)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));

    Error error;

    // Get a linux thread pointer.
    if (!thread_sp)
    {
        error.SetErrorString ("null thread_sp");
        if (log)
            log->Printf ("NativeProcessLinux::%s failed: %s", __FUNCTION__, error.AsCString ());
        return error;
    }
    std::shared_ptr<NativeThreadLinux> linux_thread_sp = std::static_pointer_cast<NativeThreadLinux> (thread_sp);

    // Find out the size of a breakpoint (might depend on where we are in the code).
    NativeRegisterContextSP context_sp = linux_thread_sp->GetRegisterContext ();
    if (!context_sp)
    {
        error.SetErrorString ("cannot get a NativeRegisterContext for the thread");
        if (log)
            log->Printf ("NativeProcessLinux::%s failed: %s", __FUNCTION__, error.AsCString ());
        return error;
    }

    uint32_t breakpoint_size = 0;
    error = GetSoftwareBreakpointPCOffset (context_sp, breakpoint_size);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s GetBreakpointSize() failed: %s", __FUNCTION__, error.AsCString ());
        return error;
    }
    else
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s breakpoint size: %" PRIu32, __FUNCTION__, breakpoint_size);
    }

    // First try probing for a breakpoint at a software breakpoint location: PC - breakpoint size.
    const lldb::addr_t initial_pc_addr = context_sp->GetPC ();
    lldb::addr_t breakpoint_addr = initial_pc_addr;
    if (breakpoint_size > 0)
    {
        // Do not allow breakpoint probe to wrap around.
        if (breakpoint_addr >= breakpoint_size)
            breakpoint_addr -= breakpoint_size;
    }

    // Check if we stopped because of a breakpoint.
    NativeBreakpointSP breakpoint_sp;
    error = m_breakpoint_list.GetBreakpoint (breakpoint_addr, breakpoint_sp);
    if (!error.Success () || !breakpoint_sp)
    {
        // We didn't find one at a software probe location.  Nothing to do.
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " no lldb breakpoint found at current pc with adjustment: 0x%" PRIx64, __FUNCTION__, GetID (), breakpoint_addr);
        return Error ();
    }

    // If the breakpoint is not a software breakpoint, nothing to do.
    if (!breakpoint_sp->IsSoftwareBreakpoint ())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " breakpoint found at 0x%" PRIx64 ", not software, nothing to adjust", __FUNCTION__, GetID (), breakpoint_addr);
        return Error ();
    }

    //
    // We have a software breakpoint and need to adjust the PC.
    //

    // Sanity check.
    if (breakpoint_size == 0)
    {
        // Nothing to do!  How did we get here?
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " breakpoint found at 0x%" PRIx64 ", it is software, but the size is zero, nothing to do (unexpected)", __FUNCTION__, GetID (), breakpoint_addr);
        return Error ();
    }

    // Change the program counter.
    if (log)
        log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 ": changing PC from 0x%" PRIx64 " to 0x%" PRIx64, __FUNCTION__, GetID (), linux_thread_sp->GetID (), initial_pc_addr, breakpoint_addr);

    error = context_sp->SetPC (breakpoint_addr);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 ": failed to set PC: %s", __FUNCTION__, GetID (), linux_thread_sp->GetID (), error.AsCString ());
        return error;
    }

    return error;
}

Error
NativeProcessLinux::GetLoadedModuleFileSpec(const char* module_path, FileSpec& file_spec)
{
    char maps_file_name[32];
    snprintf(maps_file_name, sizeof(maps_file_name), "/proc/%" PRIu64 "/maps", GetID());

    FileSpec maps_file_spec(maps_file_name, false);
    if (!maps_file_spec.Exists()) {
        file_spec.Clear();
        return Error("/proc/%" PRIu64 "/maps file doesn't exists!", GetID());
    }

    FileSpec module_file_spec(module_path, true);

    std::ifstream maps_file(maps_file_name);
    std::string maps_data_str((std::istreambuf_iterator<char>(maps_file)), std::istreambuf_iterator<char>());
    StringRef maps_data(maps_data_str.c_str());

    while (!maps_data.empty())
    {
        StringRef maps_row;
        std::tie(maps_row, maps_data) = maps_data.split('\n');

        SmallVector<StringRef, 16> maps_columns;
        maps_row.split(maps_columns, StringRef(" "), -1, false);

        if (maps_columns.size() >= 6)
        {
            file_spec.SetFile(maps_columns[5].str().c_str(), false);
            if (file_spec.GetFilename() == module_file_spec.GetFilename())
                return Error();
        }
    }

    file_spec.Clear();
    return Error("Module file (%s) not found in /proc/%" PRIu64 "/maps file!",
                 module_file_spec.GetFilename().AsCString(), GetID());
}

Error
NativeProcessLinux::ResumeThread(
        lldb::tid_t tid,
        NativeThreadLinux::ResumeThreadFunction request_thread_resume_function,
        bool error_when_already_running)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);

    if (log)
        log->Printf("NativeProcessLinux::%s (tid: %" PRIu64 ", error_when_already_running: %s)",
                __FUNCTION__, tid, error_when_already_running?"true":"false");
    
    auto thread_sp = std::static_pointer_cast<NativeThreadLinux>(GetThreadByID(tid));
    lldbassert(thread_sp != nullptr);

    auto& context = thread_sp->GetThreadContext();
    // Tell the thread to resume if we don't already think it is running.
    const bool is_stopped = StateIsStoppedState(thread_sp->GetState(), true);

    lldbassert(!(error_when_already_running && !is_stopped));

    if (!is_stopped)
    {
        // It's not an error, just a log, if the error_when_already_running flag is not set.
        // This covers cases where, for instance, we're just trying to resume all threads
        // from the user side.
        if (log)
            log->Printf("NativeProcessLinux::%s tid %" PRIu64 " optional resume skipped since it is already running",
                    __FUNCTION__,
                    tid);
        return Error();
    }

    // Before we do the resume below, first check if we have a pending
    // stop notification that is currently waiting for
    // this thread to stop.  This is potentially a buggy situation since
    // we're ostensibly waiting for threads to stop before we send out the
    // pending notification, and here we are resuming one before we send
    // out the pending stop notification.
    if (m_pending_notification_up && log && m_pending_notification_up->wait_for_stop_tids.count (tid) > 0)
    {
        log->Printf("NativeProcessLinux::%s about to resume tid %" PRIu64 " per explicit request but we have a pending stop notification (tid %" PRIu64 ") that is actively waiting for this thread to stop. Valid sequence of events?", __FUNCTION__, tid, m_pending_notification_up->triggering_tid);
    }

    // Request a resume.  We expect this to be synchronous and the system
    // to reflect it is running after this completes.
    const auto error = request_thread_resume_function (tid, false);
    if (error.Success())
        context.request_resume_function = request_thread_resume_function;
    else if (log)
    {
        log->Printf("NativeProcessLinux::%s failed to resume thread tid  %" PRIu64 ": %s",
                         __FUNCTION__, tid, error.AsCString ());
    }

    return error;
}

//===----------------------------------------------------------------------===//

void
NativeProcessLinux::StopRunningThreads(const lldb::tid_t triggering_tid)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);

    if (log)
    {
        log->Printf("NativeProcessLinux::%s about to process event: (triggering_tid: %" PRIu64 ")",
                __FUNCTION__, triggering_tid);
    }

    DoStopThreads(PendingNotificationUP(new PendingNotification(triggering_tid)));

    if (log)
    {
        log->Printf("NativeProcessLinux::%s event processing done", __FUNCTION__);
    }
}

void
NativeProcessLinux::SignalIfAllThreadsStopped()
{
    if (m_pending_notification_up && m_pending_notification_up->wait_for_stop_tids.empty ())
    {
        Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_BREAKPOINTS));

        // Clear any temporary breakpoints we used to implement software single stepping.
        for (const auto &thread_info: m_threads_stepping_with_breakpoint)
        {
            Error error = RemoveBreakpoint (thread_info.second);
            if (error.Fail())
                if (log)
                    log->Printf("NativeProcessLinux::%s() pid = %" PRIu64 " remove stepping breakpoint: %s",
                            __FUNCTION__, thread_info.first, error.AsCString());
        }
        m_threads_stepping_with_breakpoint.clear();

        // Notify the delegate about the stop
        SetCurrentThreadID(m_pending_notification_up->triggering_tid);
        SetState(StateType::eStateStopped, true);
        m_pending_notification_up.reset();
    }
}

void
NativeProcessLinux::RequestStopOnAllRunningThreads()
{
    // Request a stop for all the thread stops that need to be stopped
    // and are not already known to be stopped.  Keep a list of all the
    // threads from which we still need to hear a stop reply.

    ThreadIDSet sent_tids;
    for (const auto &thread_sp: m_threads)
    {
        // We only care about running threads
        if (StateIsStoppedState(thread_sp->GetState(), true))
            continue;

        static_pointer_cast<NativeThreadLinux>(thread_sp)->RequestStop();
        sent_tids.insert (thread_sp->GetID());
    }

    // Set the wait list to the set of tids for which we requested stops.
    m_pending_notification_up->wait_for_stop_tids.swap (sent_tids);
}


Error
NativeProcessLinux::ThreadDidStop (lldb::tid_t tid, bool initiated_by_llgs)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);

    if (log)
        log->Printf("NativeProcessLinux::%s (tid: %" PRIu64 ", %sinitiated by llgs)",
                __FUNCTION__, tid, initiated_by_llgs?"":"not ");

    // Ensure we know about the thread.
    auto thread_sp = std::static_pointer_cast<NativeThreadLinux>(GetThreadByID(tid));
    lldbassert(thread_sp != nullptr);

    // Update the global list of known thread states.  This one is definitely stopped.
    auto& context = thread_sp->GetThreadContext();
    const auto stop_was_requested = context.stop_requested;
    context.stop_requested = false;

    // If we have a pending notification, remove this from the set.
    if (m_pending_notification_up)
    {
        m_pending_notification_up->wait_for_stop_tids.erase(tid);
        SignalIfAllThreadsStopped();
    }

    Error error;
    if (initiated_by_llgs && context.request_resume_function && !stop_was_requested)
    {
        // We can end up here if stop was initiated by LLGS but by this time a
        // thread stop has occurred - maybe initiated by another event.
        if (log)
            log->Printf("Resuming thread %"  PRIu64 " since stop wasn't requested", tid);
        error = context.request_resume_function (tid, true);
        if (error.Fail() && log)
        {
                log->Printf("NativeProcessLinux::%s failed to resume thread tid  %" PRIu64 ": %s",
                        __FUNCTION__, tid, error.AsCString ());
        }
    }
    return error;
}

void
NativeProcessLinux::DoStopThreads(PendingNotificationUP &&notification_up)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);
    if (m_pending_notification_up && log)
    {
        // Yikes - we've already got a pending signal notification in progress.
        // Log this info.  We lose the pending notification here.
        log->Printf("NativeProcessLinux::%s dropping existing pending signal notification for tid %" PRIu64 ", to be replaced with signal for tid %" PRIu64,
                   __FUNCTION__,
                   m_pending_notification_up->triggering_tid,
                   notification_up->triggering_tid);
    }
    m_pending_notification_up = std::move(notification_up);

    RequestStopOnAllRunningThreads();

    SignalIfAllThreadsStopped();
}

void
NativeProcessLinux::ThreadWasCreated (lldb::tid_t tid)
{
    Log *const log = GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);

    if (log)
        log->Printf("NativeProcessLinux::%s (tid: %" PRIu64 ")", __FUNCTION__, tid);

    auto thread_sp = std::static_pointer_cast<NativeThreadLinux>(GetThreadByID(tid));
    lldbassert(thread_sp != nullptr);

    if (m_pending_notification_up && StateIsRunningState(thread_sp->GetState()))
    {
        // We will need to wait for this new thread to stop as well before firing the
        // notification.
        m_pending_notification_up->wait_for_stop_tids.insert(tid);
        thread_sp->RequestStop();
    }
}

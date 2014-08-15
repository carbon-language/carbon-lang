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
#include <linux/unistd.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>

// C++ Includes
#include <fstream>
#include <string>

// Other libraries and framework includes
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/NativeRegisterContext.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Utility/PseudoTerminal.h"

#include "Host/common/NativeBreakpoint.h"
#include "Utility/StringExtractor.h"

#include "Plugins/Process/Utility/LinuxSignals.h"
#include "NativeThreadLinux.h"
#include "ProcFileReader.h"
#include "ProcessPOSIXLog.h"

#define DEBUG_PTRACE_MAXBYTES 20

// Support ptrace extensions even when compiled without required kernel support
#ifndef PT_GETREGS
#ifndef PTRACE_GETREGS
  #define PTRACE_GETREGS 12
#endif
#endif
#ifndef PT_SETREGS
#ifndef PTRACE_SETREGS
  #define PTRACE_SETREGS 13
#endif
#endif
#ifndef PT_GETFPREGS
#ifndef PTRACE_GETFPREGS
  #define PTRACE_GETFPREGS 14
#endif
#endif
#ifndef PT_SETFPREGS
#ifndef PTRACE_SETFPREGS
  #define PTRACE_SETFPREGS 15
#endif
#endif
#ifndef PTRACE_GETREGSET
  #define PTRACE_GETREGSET 0x4204
#endif
#ifndef PTRACE_SETREGSET
  #define PTRACE_SETREGSET 0x4205
#endif
#ifndef PTRACE_GET_THREAD_AREA
  #define PTRACE_GET_THREAD_AREA 25
#endif
#ifndef PTRACE_ARCH_PRCTL
  #define PTRACE_ARCH_PRCTL      30
#endif
#ifndef ARCH_GET_FS
  #define ARCH_SET_GS 0x1001
  #define ARCH_SET_FS 0x1002
  #define ARCH_GET_FS 0x1003
  #define ARCH_GET_GS 0x1004
#endif


// Support hardware breakpoints in case it has not been defined
#ifndef TRAP_HWBKPT
  #define TRAP_HWBKPT 4
#endif

// Try to define a macro to encapsulate the tgkill syscall
// fall back on kill() if tgkill isn't available
#define tgkill(pid, tid, sig)  syscall(SYS_tgkill, pid, tid, sig)

// We disable the tracing of ptrace calls for integration builds to
// avoid the additional indirection and checks.
#ifndef LLDB_CONFIGURATION_BUILDANDINTEGRATION
#define PTRACE(req, pid, addr, data, data_size) \
    PtraceWrapper((req), (pid), (addr), (data), (data_size), #req, __FILE__, __LINE__)
#else
#define PTRACE(req, pid, addr, data, data_size) \
    PtraceWrapper((req), (pid), (addr), (data), (data_size))
#endif

// Private bits we only need internally.
namespace
{
    using namespace lldb;
    using namespace lldb_private;

    const UnixSignals&
    GetUnixSignals ()
    {
        static process_linux::LinuxSignals signals;
        return signals;
    }

    const char *
    GetFilePath(const lldb_private::FileAction *file_action, const char *default_path)
    {
        const char *pts_name = "/dev/pts/";
        const char *path = NULL;

        if (file_action)
        {
            if (file_action->GetAction() == FileAction::eFileActionOpen)
            {
                path = file_action->GetPath ();
                // By default the stdio paths passed in will be pseudo-terminal
                // (/dev/pts). If so, convert to using a different default path
                // instead to redirect I/O to the debugger console. This should
                //  also handle user overrides to /dev/null or a different file.
                if (!path || ::strncmp (path, pts_name, ::strlen (pts_name)) == 0)
                    path = default_path;
            }
        }

        return path;
    }

    Error
    ResolveProcessArchitecture (lldb::pid_t pid, Platform &platform, ArchSpec &arch)
    {
        // Grab process info for the running process.
        ProcessInstanceInfo process_info;
        if (!platform.GetProcessInfo (pid, process_info))
            return lldb_private::Error("failed to get process info");

        // Resolve the executable module.
        ModuleSP exe_module_sp;
        FileSpecList executable_search_paths (Target::GetDefaultExecutableSearchPaths ());
        Error error = platform.ResolveExecutable(
            process_info.GetExecutableFile (),
            platform.GetSystemArchitecture (),
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
    DisplayBytes (lldb_private::StreamString &s, void *bytes, uint32_t count)
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
    PtraceWrapper(int req, lldb::pid_t pid, void *addr, void *data, size_t data_size,
            const char* reqName, const char* file, int line)
    {
        long int result;

        Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PTRACE));

        PtraceDisplayBytes(req, data, data_size);

        errno = 0;
        if (req == PTRACE_GETREGSET || req == PTRACE_SETREGSET)
            result = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), *(unsigned int *)addr, data);
        else
            result = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), addr, data);

        if (log)
            log->Printf("ptrace(%s, %" PRIu64 ", %p, %p, %zu)=%lX called from file %s line %d",
                    reqName, pid, addr, data, data_size, result, file, line);

        PtraceDisplayBytes(req, data, data_size);

        if (log && errno != 0)
        {
            const char* str;
            switch (errno)
            {
            case ESRCH:  str = "ESRCH"; break;
            case EINVAL: str = "EINVAL"; break;
            case EBUSY:  str = "EBUSY"; break;
            case EPERM:  str = "EPERM"; break;
            default:     str = "<unknown>";
            }
            log->Printf("ptrace() failed; errno=%d (%s)", errno, str);
        }

        return result;
    }

#ifdef LLDB_CONFIGURATION_BUILDANDINTEGRATION
    // Wrapper for ptrace when logging is not required.
    // Sets errno to 0 prior to calling ptrace.
    long
    PtraceWrapper(int req, lldb::pid_t pid, void *addr, void *data, size_t data_size)
    {
        long result = 0;
        errno = 0;
        if (req == PTRACE_GETREGSET || req == PTRACE_SETREGSET)
            result = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), *(unsigned int *)addr, data);
        else
            result = ptrace(static_cast<__ptrace_request>(req), static_cast< ::pid_t>(pid), addr, data);
        return result;
    }
#endif

    //------------------------------------------------------------------------------
    // Static implementations of NativeProcessLinux::ReadMemory and
    // NativeProcessLinux::WriteMemory.  This enables mutual recursion between these
    // functions without needed to go thru the thread funnel.

    static lldb::addr_t
    DoReadMemory (
        lldb::pid_t pid,
        lldb::addr_t vm_addr,
        void *buf,
        lldb::addr_t size,
        Error &error)
    {
        // ptrace word size is determined by the host, not the child
        static const unsigned word_size = sizeof(void*);
        unsigned char *dst = static_cast<unsigned char*>(buf);
        lldb::addr_t bytes_read;
        lldb::addr_t remainder;
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
            errno = 0;
            data = PTRACE(PTRACE_PEEKDATA, pid, (void*)vm_addr, NULL, 0);
            if (errno)
            {
                error.SetErrorToErrno();
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

    static lldb::addr_t
    DoWriteMemory(
        lldb::pid_t pid,
        lldb::addr_t vm_addr,
        const void *buf,
        lldb::addr_t size,
        Error &error)
    {
        // ptrace word size is determined by the host, not the child
        static const unsigned word_size = sizeof(void*);
        const unsigned char *src = static_cast<const unsigned char*>(buf);
        lldb::addr_t bytes_written = 0;
        lldb::addr_t remainder;

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
                            (void*)vm_addr, *(unsigned long*)src, data);

                if (PTRACE(PTRACE_POKEDATA, pid, (void*)vm_addr, (void*)data, 0))
                {
                    error.SetErrorToErrno();
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
                            (void*)vm_addr, *(unsigned long*)src, *(unsigned long*)buff);
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
        ReadOperation (
            lldb::addr_t addr,
            void *buff,
            lldb::addr_t size,
            lldb::addr_t &result) :
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
        lldb::addr_t m_size;
        lldb::addr_t &m_result;
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
        WriteOperation (
            lldb::addr_t addr,
            const void *buff,
            lldb::addr_t size,
            lldb::addr_t &result) :
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
        lldb::addr_t m_size;
        lldb::addr_t &m_result;
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
                RegisterValue &value, bool &result)
            : m_tid(tid), m_offset(static_cast<uintptr_t> (offset)), m_reg_name(reg_name),
              m_value(value), m_result(result)
            { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        uintptr_t m_offset;
        const char *m_reg_name;
        RegisterValue &m_value;
        bool &m_result;
    };

    void
    ReadRegOperation::Execute(NativeProcessLinux *monitor)
    {
        Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_REGISTERS));

        // Set errno to zero so that we can detect a failed peek.
        errno = 0;
        lldb::addr_t data = PTRACE(PTRACE_PEEKUSER, m_tid, (void*)m_offset, NULL, 0);
        if (errno)
            m_result = false;
        else
        {
            m_value = data;
            m_result = true;
        }
        if (log)
            log->Printf ("NativeProcessLinux::%s() reg %s: 0x%" PRIx64, __FUNCTION__,
                    m_reg_name, data);
    }

    //------------------------------------------------------------------------------
    /// @class WriteRegOperation
    /// @brief Implements NativeProcessLinux::WriteRegisterValue.
    class WriteRegOperation : public Operation
    {
    public:
        WriteRegOperation(lldb::tid_t tid, unsigned offset, const char *reg_name,
                const RegisterValue &value, bool &result)
            : m_tid(tid), m_offset(offset), m_reg_name(reg_name),
              m_value(value), m_result(result)
            { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        uintptr_t m_offset;
        const char *m_reg_name;
        const RegisterValue &m_value;
        bool &m_result;
    };

    void
    WriteRegOperation::Execute(NativeProcessLinux *monitor)
    {
        void* buf;
        Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_REGISTERS));

        buf = (void*) m_value.GetAsUInt64();

        if (log)
            log->Printf ("NativeProcessLinux::%s() reg %s: %p", __FUNCTION__, m_reg_name, buf);
        if (PTRACE(PTRACE_POKEUSER, m_tid, (void*)m_offset, buf, 0))
            m_result = false;
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class ReadGPROperation
    /// @brief Implements NativeProcessLinux::ReadGPR.
    class ReadGPROperation : public Operation
    {
    public:
        ReadGPROperation(lldb::tid_t tid, void *buf, size_t buf_size, bool &result)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_result(result)
            { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
        bool &m_result;
    };

    void
    ReadGPROperation::Execute(NativeProcessLinux *monitor)
    {
        if (PTRACE(PTRACE_GETREGS, m_tid, NULL, m_buf, m_buf_size) < 0)
            m_result = false;
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class ReadFPROperation
    /// @brief Implements NativeProcessLinux::ReadFPR.
    class ReadFPROperation : public Operation
    {
    public:
        ReadFPROperation(lldb::tid_t tid, void *buf, size_t buf_size, bool &result)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_result(result)
            { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
        bool &m_result;
    };

    void
    ReadFPROperation::Execute(NativeProcessLinux *monitor)
    {
        if (PTRACE(PTRACE_GETFPREGS, m_tid, NULL, m_buf, m_buf_size) < 0)
            m_result = false;
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class ReadRegisterSetOperation
    /// @brief Implements NativeProcessLinux::ReadRegisterSet.
    class ReadRegisterSetOperation : public Operation
    {
    public:
        ReadRegisterSetOperation(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset, bool &result)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_regset(regset), m_result(result)
            { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
        const unsigned int m_regset;
        bool &m_result;
    };

    void
    ReadRegisterSetOperation::Execute(NativeProcessLinux *monitor)
    {
        if (PTRACE(PTRACE_GETREGSET, m_tid, (void *)&m_regset, m_buf, m_buf_size) < 0)
            m_result = false;
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class WriteGPROperation
    /// @brief Implements NativeProcessLinux::WriteGPR.
    class WriteGPROperation : public Operation
    {
    public:
        WriteGPROperation(lldb::tid_t tid, void *buf, size_t buf_size, bool &result)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_result(result)
            { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
        bool &m_result;
    };

    void
    WriteGPROperation::Execute(NativeProcessLinux *monitor)
    {
        if (PTRACE(PTRACE_SETREGS, m_tid, NULL, m_buf, m_buf_size) < 0)
            m_result = false;
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class WriteFPROperation
    /// @brief Implements NativeProcessLinux::WriteFPR.
    class WriteFPROperation : public Operation
    {
    public:
        WriteFPROperation(lldb::tid_t tid, void *buf, size_t buf_size, bool &result)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_result(result)
            { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
        bool &m_result;
    };

    void
    WriteFPROperation::Execute(NativeProcessLinux *monitor)
    {
        if (PTRACE(PTRACE_SETFPREGS, m_tid, NULL, m_buf, m_buf_size) < 0)
            m_result = false;
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class WriteRegisterSetOperation
    /// @brief Implements NativeProcessLinux::WriteRegisterSet.
    class WriteRegisterSetOperation : public Operation
    {
    public:
        WriteRegisterSetOperation(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset, bool &result)
            : m_tid(tid), m_buf(buf), m_buf_size(buf_size), m_regset(regset), m_result(result)
            { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        void *m_buf;
        size_t m_buf_size;
        const unsigned int m_regset;
        bool &m_result;
    };

    void
    WriteRegisterSetOperation::Execute(NativeProcessLinux *monitor)
    {
        if (PTRACE(PTRACE_SETREGSET, m_tid, (void *)&m_regset, m_buf, m_buf_size) < 0)
            m_result = false;
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class ResumeOperation
    /// @brief Implements NativeProcessLinux::Resume.
    class ResumeOperation : public Operation
    {
    public:
        ResumeOperation(lldb::tid_t tid, uint32_t signo, bool &result) :
            m_tid(tid), m_signo(signo), m_result(result) { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        uint32_t m_signo;
        bool &m_result;
    };

    void
    ResumeOperation::Execute(NativeProcessLinux *monitor)
    {
        intptr_t data = 0;

        if (m_signo != LLDB_INVALID_SIGNAL_NUMBER)
            data = m_signo;

        if (PTRACE(PTRACE_CONT, m_tid, NULL, (void*)data, 0))
        {
            Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

            if (log)
                log->Printf ("ResumeOperation (%"  PRIu64 ") failed: %s", m_tid, strerror(errno));
            m_result = false;
        }
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class SingleStepOperation
    /// @brief Implements NativeProcessLinux::SingleStep.
    class SingleStepOperation : public Operation
    {
    public:
        SingleStepOperation(lldb::tid_t tid, uint32_t signo, bool &result)
            : m_tid(tid), m_signo(signo), m_result(result) { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        uint32_t m_signo;
        bool &m_result;
    };

    void
    SingleStepOperation::Execute(NativeProcessLinux *monitor)
    {
        intptr_t data = 0;

        if (m_signo != LLDB_INVALID_SIGNAL_NUMBER)
            data = m_signo;

        if (PTRACE(PTRACE_SINGLESTEP, m_tid, NULL, (void*)data, 0))
            m_result = false;
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class SiginfoOperation
    /// @brief Implements NativeProcessLinux::GetSignalInfo.
    class SiginfoOperation : public Operation
    {
    public:
        SiginfoOperation(lldb::tid_t tid, void *info, bool &result, int &ptrace_err)
            : m_tid(tid), m_info(info), m_result(result), m_err(ptrace_err) { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        void *m_info;
        bool &m_result;
        int &m_err;
    };

    void
    SiginfoOperation::Execute(NativeProcessLinux *monitor)
    {
        if (PTRACE(PTRACE_GETSIGINFO, m_tid, NULL, m_info, 0)) {
            m_result = false;
            m_err = errno;
        }
        else
            m_result = true;
    }

    //------------------------------------------------------------------------------
    /// @class EventMessageOperation
    /// @brief Implements NativeProcessLinux::GetEventMessage.
    class EventMessageOperation : public Operation
    {
    public:
        EventMessageOperation(lldb::tid_t tid, unsigned long *message, bool &result)
            : m_tid(tid), m_message(message), m_result(result) { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        unsigned long *m_message;
        bool &m_result;
    };

    void
    EventMessageOperation::Execute(NativeProcessLinux *monitor)
    {
        if (PTRACE(PTRACE_GETEVENTMSG, m_tid, NULL, m_message, 0))
            m_result = false;
        else
            m_result = true;
    }

    class DetachOperation : public Operation
    {
    public:
        DetachOperation(lldb::tid_t tid, Error &result) : m_tid(tid), m_error(result) { }

        void Execute(NativeProcessLinux *monitor);

    private:
        lldb::tid_t m_tid;
        Error &m_error;
    };

    void
    DetachOperation::Execute(NativeProcessLinux *monitor)
    {
        if (ptrace(PT_DETACH, m_tid, NULL, 0) < 0)
            m_error.SetErrorToErrno();
    }

}

using namespace lldb_private;

// Simple helper function to ensure flags are enabled on the given file
// descriptor.
static bool
EnsureFDFlags(int fd, int flags, Error &error)
{
    int status;

    if ((status = fcntl(fd, F_GETFL)) == -1)
    {
        error.SetErrorToErrno();
        return false;
    }

    if (fcntl(fd, F_SETFL, status | flags) == -1)
    {
        error.SetErrorToErrno();
        return false;
    }

    return true;
}

NativeProcessLinux::OperationArgs::OperationArgs(NativeProcessLinux *monitor)
    : m_monitor(monitor)
{
    sem_init(&m_semaphore, 0, 0);
}

NativeProcessLinux::OperationArgs::~OperationArgs()
{
    sem_destroy(&m_semaphore);
}

NativeProcessLinux::LaunchArgs::LaunchArgs(NativeProcessLinux *monitor,
                                       lldb_private::Module *module,
                                       char const **argv,
                                       char const **envp,
                                       const char *stdin_path,
                                       const char *stdout_path,
                                       const char *stderr_path,
                                       const char *working_dir)
    : OperationArgs(monitor),
      m_module(module),
      m_argv(argv),
      m_envp(envp),
      m_stdin_path(stdin_path),
      m_stdout_path(stdout_path),
      m_stderr_path(stderr_path),
      m_working_dir(working_dir) { }

NativeProcessLinux::LaunchArgs::~LaunchArgs()
{ }

NativeProcessLinux::AttachArgs::AttachArgs(NativeProcessLinux *monitor,
                                       lldb::pid_t pid)
    : OperationArgs(monitor), m_pid(pid) { }

NativeProcessLinux::AttachArgs::~AttachArgs()
{ }

// -----------------------------------------------------------------------------
// Public Static Methods
// -----------------------------------------------------------------------------

lldb_private::Error
NativeProcessLinux::LaunchProcess (
    lldb_private::Module *exe_module,
    lldb_private::ProcessLaunchInfo &launch_info,
    lldb_private::NativeProcessProtocol::NativeDelegate &native_delegate,
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

    const lldb_private::FileAction *file_action;

    // Default of NULL will mean to use existing open file descriptors.
    const char *stdin_path = NULL;
    const char *stdout_path = NULL;
    const char *stderr_path = NULL;

    file_action = launch_info.GetFileActionForFD (STDIN_FILENO);
    stdin_path = GetFilePath (file_action, stdin_path);

    file_action = launch_info.GetFileActionForFD (STDOUT_FILENO);
    stdout_path = GetFilePath (file_action, stdout_path);

    file_action = launch_info.GetFileActionForFD (STDERR_FILENO);
    stderr_path = GetFilePath (file_action, stderr_path);

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

    reinterpret_cast<NativeProcessLinux*> (native_process_sp.get ())->LaunchInferior (
            exe_module,
            launch_info.GetArguments ().GetConstArgumentVector (),
            launch_info.GetEnvironmentEntries ().GetConstArgumentVector (),
            stdin_path,
            stdout_path,
            stderr_path,
            working_dir,
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

lldb_private::Error
NativeProcessLinux::AttachToProcess (
    lldb::pid_t pid,
    lldb_private::NativeProcessProtocol::NativeDelegate &native_delegate,
    NativeProcessProtocolSP &native_process_sp)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log && log->GetMask ().Test (POSIX_LOG_VERBOSE))
        log->Printf ("NativeProcessLinux::%s(pid = %" PRIi64 ")", __FUNCTION__, pid);

    // Grab the current platform architecture.  This should be Linux,
    // since this code is only intended to run on a Linux host.
    PlatformSP platform_sp (Platform::GetDefaultPlatform ());
    if (!platform_sp)
        return Error("failed to get a valid default platform");

    // Retrieve the architecture for the running process.
    ArchSpec process_arch;
    Error error = ResolveProcessArchitecture (pid, *platform_sp.get (), process_arch);
    if (!error.Success ())
        return error;

    native_process_sp.reset (new NativeProcessLinux ());

    if (!native_process_sp->RegisterNativeDelegate (native_delegate))
    {
        native_process_sp.reset (new NativeProcessLinux ());
        error.SetErrorStringWithFormat ("failed to register the native delegate");
        return error;
    }

    reinterpret_cast<NativeProcessLinux*> (native_process_sp.get ())->AttachToInferior (pid, error);
    if (!error.Success ())
    {
        native_process_sp.reset ();
        return error;
    }

    return error;
}

// -----------------------------------------------------------------------------
// Public Instance Methods
// -----------------------------------------------------------------------------

NativeProcessLinux::NativeProcessLinux () :
    NativeProcessProtocol (LLDB_INVALID_PROCESS_ID),
    m_arch (),
    m_operation_thread (LLDB_INVALID_HOST_THREAD),
    m_monitor_thread (LLDB_INVALID_HOST_THREAD),
    m_operation (nullptr),
    m_operation_mutex (),
    m_operation_pending (),
    m_operation_done (),
    m_wait_for_stop_tids (),
    m_wait_for_stop_tids_mutex (),
    m_supports_mem_region (eLazyBoolCalculate),
    m_mem_region_cache (),
    m_mem_region_cache_mutex ()
{
}

//------------------------------------------------------------------------------
/// The basic design of the NativeProcessLinux is built around two threads.
///
/// One thread (@see SignalThread) simply blocks on a call to waitpid() looking
/// for changes in the debugee state.  When a change is detected a
/// ProcessMessage is sent to the associated ProcessLinux instance.  This thread
/// "drives" state changes in the debugger.
///
/// The second thread (@see OperationThread) is responsible for two things 1)
/// launching or attaching to the inferior process, and then 2) servicing
/// operations such as register reads/writes, stepping, etc.  See the comments
/// on the Operation class for more info as to why this is needed.
void
NativeProcessLinux::LaunchInferior (
    Module *module,
    const char *argv[],
    const char *envp[],
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    const char *working_dir,
    lldb_private::Error &error)
{
    if (module)
        m_arch = module->GetArchitecture ();

    SetState(eStateLaunching);

    std::unique_ptr<LaunchArgs> args(
        new LaunchArgs(
            this, module, argv, envp,
            stdin_path, stdout_path, stderr_path,
            working_dir));

    sem_init(&m_operation_pending, 0, 0);
    sem_init(&m_operation_done, 0, 0);

    StartLaunchOpThread(args.get(), error);
    if (!error.Success())
        return;

WAIT_AGAIN:
    // Wait for the operation thread to initialize.
    if (sem_wait(&args->m_semaphore))
    {
        if (errno == EINTR)
            goto WAIT_AGAIN;
        else
        {
            error.SetErrorToErrno();
            return;
        }
    }

    // Check that the launch was a success.
    if (!args->m_error.Success())
    {
        StopOpThread();
        error = args->m_error;
        return;
    }

    // Finally, start monitoring the child process for change in state.
    m_monitor_thread = Host::StartMonitoringChildProcess(
        NativeProcessLinux::MonitorCallback, this, GetID(), true);
    if (!IS_VALID_LLDB_HOST_THREAD(m_monitor_thread))
    {
        error.SetErrorToGenericError();
        error.SetErrorString ("Process attach failed to create monitor thread for NativeProcessLinux::MonitorCallback.");
        return;
    }
}

void
NativeProcessLinux::AttachToInferior (lldb::pid_t pid, lldb_private::Error &error)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    if (log)
        log->Printf ("NativeProcessLinux::%s (pid = %" PRIi64 ")", __FUNCTION__, pid);

    // We can use the Host for everything except the ResolveExecutable portion.
    PlatformSP platform_sp = Platform::GetDefaultPlatform ();
    if (!platform_sp)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s (pid = %" PRIi64 "): no default platform set", __FUNCTION__, pid);
        error.SetErrorString ("no default platform available");
    }

    // Gather info about the process.
    ProcessInstanceInfo process_info;
    platform_sp->GetProcessInfo (pid, process_info);

    // Resolve the executable module
    ModuleSP exe_module_sp;
    FileSpecList executable_search_paths (Target::GetDefaultExecutableSearchPaths());

    error = platform_sp->ResolveExecutable(process_info.GetExecutableFile(),
                                    Host::GetArchitecture(),
                                    exe_module_sp,
                                    executable_search_paths.GetSize() ? &executable_search_paths : NULL);
    if (!error.Success())
        return;

    // Set the architecture to the exe architecture.
    m_arch = exe_module_sp->GetArchitecture();
    if (log)
        log->Printf ("NativeProcessLinux::%s (pid = %" PRIi64 ") detected architecture %s", __FUNCTION__, pid, m_arch.GetArchitectureName ());

    m_pid = pid;
    SetState(eStateAttaching);

    sem_init (&m_operation_pending, 0, 0);
    sem_init (&m_operation_done, 0, 0);

    std::unique_ptr<AttachArgs> args (new AttachArgs (this, pid));

    StartAttachOpThread(args.get (), error);
    if (!error.Success ())
        return;

WAIT_AGAIN:
    // Wait for the operation thread to initialize.
    if (sem_wait (&args->m_semaphore))
    {
        if (errno == EINTR)
            goto WAIT_AGAIN;
        else
        {
            error.SetErrorToErrno ();
            return;
        }
    }

    // Check that the attach was a success.
    if (!args->m_error.Success ())
    {
        StopOpThread ();
        error = args->m_error;
        return;
    }

    // Finally, start monitoring the child process for change in state.
    m_monitor_thread = Host::StartMonitoringChildProcess (
        NativeProcessLinux::MonitorCallback, this, GetID (), true);
    if (!IS_VALID_LLDB_HOST_THREAD (m_monitor_thread))
    {
        error.SetErrorToGenericError ();
        error.SetErrorString ("Process attach failed to create monitor thread for NativeProcessLinux::MonitorCallback.");
        return;
    }
}

NativeProcessLinux::~NativeProcessLinux()
{
    StopMonitor();
}

//------------------------------------------------------------------------------
// Thread setup and tear down.

void
NativeProcessLinux::StartLaunchOpThread(LaunchArgs *args, Error &error)
{
    static const char *g_thread_name = "lldb.process.nativelinux.operation";

    if (IS_VALID_LLDB_HOST_THREAD (m_operation_thread))
        return;

    m_operation_thread =
        Host::ThreadCreate (g_thread_name, LaunchOpThread, args, &error);
}

void *
NativeProcessLinux::LaunchOpThread(void *arg)
{
    LaunchArgs *args = static_cast<LaunchArgs*>(arg);

    if (!Launch(args)) {
        sem_post(&args->m_semaphore);
        return NULL;
    }

    ServeOperation(args);
    return NULL;
}

bool
NativeProcessLinux::Launch(LaunchArgs *args)
{
    NativeProcessLinux *monitor = args->m_monitor;
    assert (monitor && "monitor is NULL");
    if (!monitor)
        return false;

    const char **argv = args->m_argv;
    const char **envp = args->m_envp;
    const char *stdin_path = args->m_stdin_path;
    const char *stdout_path = args->m_stdout_path;
    const char *stderr_path = args->m_stderr_path;
    const char *working_dir = args->m_working_dir;

    lldb_utility::PseudoTerminal terminal;
    const size_t err_len = 1024;
    char err_str[err_len];
    lldb::pid_t pid;
    NativeThreadProtocolSP thread_sp;

    lldb::ThreadSP inferior;
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // Propagate the environment if one is not supplied.
    if (envp == NULL || envp[0] == NULL)
        envp = const_cast<const char **>(environ);

    if ((pid = terminal.Fork(err_str, err_len)) == static_cast<lldb::pid_t> (-1))
    {
        args->m_error.SetErrorToGenericError();
        args->m_error.SetErrorString("Process fork failed.");
        goto FINISH;
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
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior process preparing to fork", __FUNCTION__);

        // Trace this process.
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior process issuing PTRACE_TRACEME", __FUNCTION__);

        if (PTRACE(PTRACE_TRACEME, 0, NULL, NULL, 0) < 0)
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s inferior process PTRACE_TRACEME failed", __FUNCTION__);
            exit(ePtraceFailed);
        }

        // Do not inherit setgid powers.
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior process resetting gid", __FUNCTION__);

        if (setgid(getgid()) != 0)
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s inferior process setgid() failed", __FUNCTION__);
            exit(eSetGidFailed);
        }

        // Attempt to have our own process group.
        // TODO verify if we really want this.
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior process resetting process group", __FUNCTION__);

        if (setpgid(0, 0) != 0)
        {
            if (log)
            {
                const int error_code = errno;
                log->Printf ("NativeProcessLinux::%s inferior setpgid() failed, errno=%d (%s), continuing with existing proccess group %" PRIu64,
                        __FUNCTION__,
                        error_code,
                        strerror (error_code),
                        static_cast<lldb::pid_t> (getpgid (0)));
            }
            // Don't allow this to prevent an inferior exec.
        }

        // Dup file descriptors if needed.
        //
        // FIXME: If two or more of the paths are the same we needlessly open
        // the same file multiple times.
        if (stdin_path != NULL && stdin_path[0])
            if (!DupDescriptor(stdin_path, STDIN_FILENO, O_RDONLY))
                exit(eDupStdinFailed);

        if (stdout_path != NULL && stdout_path[0])
            if (!DupDescriptor(stdout_path, STDOUT_FILENO, O_WRONLY | O_CREAT))
                exit(eDupStdoutFailed);

        if (stderr_path != NULL && stderr_path[0])
            if (!DupDescriptor(stderr_path, STDERR_FILENO, O_WRONLY | O_CREAT))
                exit(eDupStderrFailed);

        // Change working directory
        if (working_dir != NULL && working_dir[0])
          if (0 != ::chdir(working_dir))
              exit(eChdirFailed);

        // Execute.  We should never return.
        execve(argv[0],
               const_cast<char *const *>(argv),
               const_cast<char *const *>(envp));
        exit(eExecFailed);
    }

    // Wait for the child process to trap on its call to execve.
    ::pid_t wpid;
    int status;
    if ((wpid = waitpid(pid, &status, 0)) < 0)
    {
        args->m_error.SetErrorToErrno();

        if (log)
            log->Printf ("NativeProcessLinux::%s waitpid for inferior failed with %s", __FUNCTION__, args->m_error.AsCString ());

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        monitor->SetState (StateType::eStateInvalid);

        goto FINISH;
    }
    else if (WIFEXITED(status))
    {
        // open, dup or execve likely failed for some reason.
        args->m_error.SetErrorToGenericError();
        switch (WEXITSTATUS(status))
        {
            case ePtraceFailed:
                args->m_error.SetErrorString("Child ptrace failed.");
                break;
            case eDupStdinFailed:
                args->m_error.SetErrorString("Child open stdin failed.");
                break;
            case eDupStdoutFailed:
                args->m_error.SetErrorString("Child open stdout failed.");
                break;
            case eDupStderrFailed:
                args->m_error.SetErrorString("Child open stderr failed.");
                break;
            case eChdirFailed:
                args->m_error.SetErrorString("Child failed to set working directory.");
                break;
            case eExecFailed:
                args->m_error.SetErrorString("Child exec failed.");
                break;
            case eSetGidFailed:
                args->m_error.SetErrorString("Child setgid failed.");
                break;
            default:
                args->m_error.SetErrorString("Child returned unknown exit status.");
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
        monitor->SetState (StateType::eStateInvalid);

        goto FINISH;
    }
    assert(WIFSTOPPED(status) && (wpid == static_cast< ::pid_t> (pid)) &&
           "Could not sync with inferior process.");

    if (log)
        log->Printf ("NativeProcessLinux::%s inferior started, now in stopped state", __FUNCTION__);

    if (!SetDefaultPtraceOpts(pid))
    {
        args->m_error.SetErrorToErrno();
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior failed to set default ptrace options: %s",
                    __FUNCTION__,
                    args->m_error.AsCString ());

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        monitor->SetState (StateType::eStateInvalid);

        goto FINISH;
    }

    // Release the master terminal descriptor and pass it off to the
    // NativeProcessLinux instance.  Similarly stash the inferior pid.
    monitor->m_terminal_fd = terminal.ReleaseMasterFileDescriptor();
    monitor->m_pid = pid;

    // Set the terminal fd to be in non blocking mode (it simplifies the
    // implementation of ProcessLinux::GetSTDOUT to have a non-blocking
    // descriptor to read from).
    if (!EnsureFDFlags(monitor->m_terminal_fd, O_NONBLOCK, args->m_error))
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s inferior EnsureFDFlags failed for ensuring terminal O_NONBLOCK setting: %s",
                    __FUNCTION__,
                    args->m_error.AsCString ());

        // Mark the inferior as invalid.
        // FIXME this could really use a new state - eStateLaunchFailure.  For now, using eStateInvalid.
        monitor->SetState (StateType::eStateInvalid);

        goto FINISH;
    }

    if (log)
        log->Printf ("NativeProcessLinux::%s() adding pid = %" PRIu64, __FUNCTION__, pid);

    thread_sp = monitor->AddThread (static_cast<lldb::tid_t> (pid));
    assert (thread_sp && "AddThread() returned a nullptr thread");
    reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetStoppedBySignal (SIGSTOP);
    monitor->SetCurrentThreadID (thread_sp->GetID ());

    // Let our process instance know the thread has stopped.
    monitor->SetState (StateType::eStateStopped);

FINISH:
    if (log)
    {
        if (args->m_error.Success ())
        {
            log->Printf ("NativeProcessLinux::%s inferior launching succeeded", __FUNCTION__);
        }
        else
        {
            log->Printf ("NativeProcessLinux::%s inferior launching failed: %s",
                __FUNCTION__,
                args->m_error.AsCString ());
        }
    }
    return args->m_error.Success();
}

void
NativeProcessLinux::StartAttachOpThread(AttachArgs *args, lldb_private::Error &error)
{
    static const char *g_thread_name = "lldb.process.linux.operation";

    if (IS_VALID_LLDB_HOST_THREAD(m_operation_thread))
        return;

    m_operation_thread =
        Host::ThreadCreate(g_thread_name, AttachOpThread, args, &error);
}

void *
NativeProcessLinux::AttachOpThread(void *arg)
{
    AttachArgs *args = static_cast<AttachArgs*>(arg);

    if (!Attach(args)) {
        sem_post(&args->m_semaphore);
        return NULL;
    }

    ServeOperation(args);
    return NULL;
}

bool
NativeProcessLinux::Attach(AttachArgs *args)
{
    lldb::pid_t pid = args->m_pid;

    NativeProcessLinux *monitor = args->m_monitor;
    lldb::ThreadSP inferior;
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // Use a map to keep track of the threads which we have attached/need to attach.
    Host::TidMap tids_to_attach;
    if (pid <= 1)
    {
        args->m_error.SetErrorToGenericError();
        args->m_error.SetErrorString("Attaching to process 1 is not allowed.");
        goto FINISH;
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
                if (PTRACE(PTRACE_ATTACH, tid, NULL, NULL, 0) < 0)
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
                        args->m_error.SetErrorToErrno();
                        goto FINISH;
                    }
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
                        args->m_error.SetErrorToErrno();
                        goto FINISH;
                    }
                }

                if (!SetDefaultPtraceOpts(tid))
                {
                    args->m_error.SetErrorToErrno();
                    goto FINISH;
                }


                if (log)
                    log->Printf ("NativeProcessLinux::%s() adding tid = %" PRIu64, __FUNCTION__, tid);

                it->second = true;

                // Create the thread, mark it as stopped.
                NativeThreadProtocolSP thread_sp (monitor->AddThread (static_cast<lldb::tid_t> (tid)));
                assert (thread_sp && "AddThread() returned a nullptr");
                reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetStoppedBySignal (SIGSTOP);
                monitor->SetCurrentThreadID (thread_sp->GetID ());
            }

            // move the loop forward
            ++it;
        }
    }

    if (tids_to_attach.size() > 0)
    {
        monitor->m_pid = pid;
        // Let our process instance know the thread has stopped.
        monitor->SetState (StateType::eStateStopped);
    }
    else
    {
        args->m_error.SetErrorToGenericError();
        args->m_error.SetErrorString("No such process.");
    }

 FINISH:
    return args->m_error.Success();
}

bool
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

    return PTRACE(PTRACE_SETOPTIONS, pid, NULL, (void*)ptrace_opts, 0) >= 0;
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

// Main process monitoring waitpid-loop handler.
bool
NativeProcessLinux::MonitorCallback(void *callback_baton,
                                lldb::pid_t pid,
                                bool exited,
                                int signal,
                                int status)
{
    Log *log (GetLogIfAnyCategoriesSet (LIBLLDB_LOG_PROCESS));

    NativeProcessLinux *const process = static_cast<NativeProcessLinux*>(callback_baton);
    assert (process && "process is null");
    if (!process)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " callback_baton was null, can't determine process to use", __FUNCTION__, pid);
        return true;
    }

    // Certain activities differ based on whether the pid is the tid of the main thread.
    const bool is_main_thread = (pid == process->GetID ());

    // Assume we keep monitoring by default.
    bool stop_monitoring = false;

    // Handle when the thread exits.
    if (exited)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s() got exit signal, tid = %"  PRIu64 " (%s main thread)", __FUNCTION__, pid, is_main_thread ? "is" : "is not");

        // This is a thread that exited.  Ensure we're not tracking it anymore.
        const bool thread_found = process->StopTrackingThread (pid);

        if (is_main_thread)
        {
            // We only set the exit status and notify the delegate if we haven't already set the process
            // state to an exited state.  We normally should have received a SIGTRAP | (PTRACE_EVENT_EXIT << 8)
            // for the main thread.
            const bool already_notified = (process->GetState() == StateType::eStateExited) | (process->GetState () == StateType::eStateCrashed);
            if (!already_notified)
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s() tid = %"  PRIu64 " handling main thread exit (%s), expected exit state already set but state was %s instead, setting exit state now", __FUNCTION__, pid, thread_found ? "stopped tracking thread metadata" : "thread metadata not found", StateAsCString (process->GetState ()));
                // The main thread exited.  We're done monitoring.  Report to delegate.
                process->SetExitStatus (convert_pid_status_to_exit_type (status), convert_pid_status_to_return_code (status), nullptr, true);

                // Notify delegate that our process has exited.
                process->SetState (StateType::eStateExited, true);
            }
            else
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s() tid = %"  PRIu64 " main thread now exited (%s)", __FUNCTION__, pid, thread_found ? "stopped tracking thread metadata" : "thread metadata not found");
            }
            return true;
        }
        else
        {
            // Do we want to report to the delegate in this case?  I think not.  If this was an orderly
            // thread exit, we would already have received the SIGTRAP | (PTRACE_EVENT_EXIT << 8) signal,
            // and we would have done an all-stop then.
            if (log)
                log->Printf ("NativeProcessLinux::%s() tid = %"  PRIu64 " handling non-main thread exit (%s)", __FUNCTION__, pid, thread_found ? "stopped tracking thread metadata" : "thread metadata not found");

            // Not the main thread, we keep going.
            return false;
        }
    }

    // Get details on the signal raised.
    siginfo_t info;
    int ptrace_err = 0;

    if (!process->GetSignalInfo (pid, &info, ptrace_err))
    {
        if (ptrace_err == EINVAL)
        {
            // This is the first part of the Linux ptrace group-stop mechanism.
            // The tracer (i.e. NativeProcessLinux) is expected to inject the signal
            // into the tracee (i.e. inferior) at this point.
            if (log)
                log->Printf ("NativeProcessLinux::%s() resuming from group-stop", __FUNCTION__);

            // The inferior process is in 'group-stop', so deliver the stopping signal.
            const bool signal_delivered = process->Resume (pid, info.si_signo);
            if (log)
                log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " group-stop signal delivery of signal 0x%x (%s) - %s", __FUNCTION__, pid, info.si_signo, GetUnixSignals ().GetSignalAsCString (info.si_signo), signal_delivered ? "success" : "failed");

            assert(signal_delivered && "SIGSTOP delivery failed while in 'group-stop' state");

            stop_monitoring = false;
        }
        else
        {
            // ptrace(GETSIGINFO) failed (but not due to group-stop).

            // A return value of ESRCH means the thread/process is no longer on the system,
            // so it was killed somehow outside of our control.  Either way, we can't do anything
            // with it anymore.

            // We stop monitoring if it was the main thread.
            stop_monitoring = is_main_thread;

            // Stop tracking the metadata for the thread since it's entirely off the system now.
            const bool thread_found = process->StopTrackingThread (pid);

            if (log)
                log->Printf ("NativeProcessLinux::%s GetSignalInfo failed: %s, tid = %" PRIu64 ", signal = %d, status = %d (%s, %s, %s)",
                             __FUNCTION__, strerror(ptrace_err), pid, signal, status, ptrace_err == ESRCH ? "thread/process killed" : "unknown reason", is_main_thread ? "is main thread" : "is not main thread", thread_found ? "thread metadata removed" : "thread metadata not found");

            if (is_main_thread)
            {
                // Notify the delegate - our process is not available but appears to have been killed outside
                // our control.  Is eStateExited the right exit state in this case?
                process->SetExitStatus (convert_pid_status_to_exit_type (status), convert_pid_status_to_return_code (status), nullptr, true);
                process->SetState (StateType::eStateExited, true);
            }
            else
            {
                // This thread was pulled out from underneath us.  Anything to do here? Do we want to do an all stop?
                if (log)
                    log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 " non-main thread exit occurred, didn't tell delegate anything since thread disappeared out from underneath us", __FUNCTION__, process->GetID (), pid);
            }
        }
    }
    else
    {
        // We have retrieved the signal info.  Dispatch appropriately.
        if (info.si_signo == SIGTRAP)
            process->MonitorSIGTRAP(&info, pid);
        else
            process->MonitorSignal(&info, pid, exited);

        stop_monitoring = false;
    }

    return stop_monitoring;
}

void
NativeProcessLinux::MonitorSIGTRAP(const siginfo_t *info, lldb::pid_t pid)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    const bool is_main_thread = (pid == GetID ());

    assert(info && info->si_signo == SIGTRAP && "Unexpected child signal!");
    if (!info)
        return;

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
        lldb::tid_t tid = LLDB_INVALID_THREAD_ID;

        unsigned long event_message = 0;
        if (GetEventMessage(pid, &event_message))
            tid = static_cast<lldb::tid_t> (event_message);

        if (log)
            log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " received thread creation event for tid %" PRIu64, __FUNCTION__, pid, tid);

        // If we don't track the thread yet: create it, mark as stopped.
        // If we do track it, this is the wait we needed.  Now resume the new thread.
        // In all cases, resume the current (i.e. main process) thread.
        bool already_tracked = false;
        thread_sp = GetOrCreateThread (tid, already_tracked);
        assert (thread_sp.get() && "failed to get or create the tracking data for newly created inferior thread");

        // If the thread was already tracked, it means the created thread already received its SI_USER notification of creation.
        if (already_tracked)
        {
            // FIXME loops like we want to stop all theads here.
            // StopAllThreads

            // We can now resume the newly created thread since it is fully created.
            reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetRunning ();
            Resume (tid, LLDB_INVALID_SIGNAL_NUMBER);
        }
        else
        {
            // Mark the thread as currently launching.  Need to wait for SIGTRAP clone on the main thread before
            // this thread is ready to go.
            reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetLaunching ();
        }

        // In all cases, we can resume the main thread here.
        Resume (pid, LLDB_INVALID_SIGNAL_NUMBER);
        break;
    }

    case (SIGTRAP | (PTRACE_EVENT_EXEC << 8)):
        if (log)
            log->Printf ("NativeProcessLinux::%s() received exec event, code = %d", __FUNCTION__, info->si_code ^ SIGTRAP);
        // FIXME stop all threads, mark thread stop reason as ThreadStopInfo.reason = eStopReasonExec;
        break;

    case (SIGTRAP | (PTRACE_EVENT_EXIT << 8)):
    {
        // The inferior process or one of its threads is about to exit.
        // Maintain the process or thread in a state of "limbo" until we are
        // explicitly commanded to detach, destroy, resume, etc.
        unsigned long data = 0;
        if (!GetEventMessage(pid, &data))
            data = -1;

        if (log)
        {
            log->Printf ("NativeProcessLinux::%s() received PTRACE_EVENT_EXIT, data = %lx (WIFEXITED=%s,WIFSIGNALED=%s), pid = %" PRIu64 " (%s)",
                         __FUNCTION__,
                         data, WIFEXITED (data) ? "true" : "false", WIFSIGNALED (data) ? "true" : "false",
                         pid,
                    is_main_thread ? "is main thread" : "not main thread");
        }

        // Set the thread to exited.
        if (thread_sp)
            reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetExited ();
        else
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " failed to retrieve thread for tid %" PRIu64", cannot set thread state", __FUNCTION__, GetID (), pid);
        }

        if (is_main_thread)
        {
            SetExitStatus (convert_pid_status_to_exit_type (data), convert_pid_status_to_return_code (data), nullptr, true);
            // Resume the thread so it completely exits.
            Resume (pid, LLDB_INVALID_SIGNAL_NUMBER);
        }
        else
        {
            // FIXME figure out the path where we plan to reap the metadata for the thread.
        }

        break;
    }

    case 0:
    case TRAP_TRACE:
        // We receive this on single stepping.
        if (log)
            log->Printf ("NativeProcessLinux::%s() received trace event, pid = %" PRIu64 " (single stepping)", __FUNCTION__, pid);

        if (thread_sp)
        {
            reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetStoppedBySignal (SIGTRAP);
            SetCurrentThreadID (thread_sp->GetID ());
        }
        else
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " tid %" PRIu64 " single stepping received trace but thread not found", __FUNCTION__, GetID (), pid);
        }

        // Tell the process we have a stop (from single stepping).
        SetState (StateType::eStateStopped, true);
        break;

    case SI_KERNEL:
    case TRAP_BRKPT:
        if (log)
            log->Printf ("NativeProcessLinux::%s() received breakpoint event, pid = %" PRIu64, __FUNCTION__, pid);

        // Mark the thread as stopped at breakpoint.
        if (thread_sp)
        {
            reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetStoppedBySignal (SIGTRAP);
            Error error = FixupBreakpointPCAsNeeded (thread_sp);
            if (error.Fail ())
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s() pid = %" PRIu64 " fixup: %s", __FUNCTION__, pid, error.AsCString ());
            }
        }
        else
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s()  pid = %" PRIu64 ": warning, cannot process software breakpoint since no thread metadata", __FUNCTION__, pid);
        }


        // Tell the process we have a stop from this thread.
        SetCurrentThreadID (pid);
        SetState (StateType::eStateStopped, true);
        break;

    case TRAP_HWBKPT:
        if (log)
            log->Printf ("NativeProcessLinux::%s() received watchpoint event, pid = %" PRIu64, __FUNCTION__, pid);

        // Mark the thread as stopped at watchpoint.
        // The address is at (lldb::addr_t)info->si_addr if we need it.
        if (thread_sp)
            reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetStoppedBySignal (SIGTRAP);
        else
        {
            if (log)
                log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " tid %" PRIu64 ": warning, cannot process hardware breakpoint since no thread metadata", __FUNCTION__, GetID (), pid);
        }

        // Tell the process we have a stop from this thread.
        SetCurrentThreadID (pid);
        SetState (StateType::eStateStopped, true);
        break;

    case SIGTRAP:
    case (SIGTRAP | 0x80):
        if (log)
            log->Printf ("NativeProcessLinux::%s() received system call stop event, pid %" PRIu64 "tid %" PRIu64, __FUNCTION__, GetID (), pid);
        // Ignore these signals until we know more about them.
        Resume(pid, 0);
        break;

    default:
        assert(false && "Unexpected SIGTRAP code!");
        if (log)
            log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 "tid %" PRIu64 " received unhandled SIGTRAP code: 0x%" PRIx64, __FUNCTION__, GetID (), pid, static_cast<uint64_t> (SIGTRAP | (PTRACE_EVENT_CLONE << 8)));
        break;
        
    }
}

void
NativeProcessLinux::MonitorSignal(const siginfo_t *info, lldb::pid_t pid, bool exited)
{
    int signo = info->si_signo;

    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // POSIX says that process behaviour is undefined after it ignores a SIGFPE,
    // SIGILL, SIGSEGV, or SIGBUS *unless* that signal was generated by a
    // kill(2) or raise(3).  Similarly for tgkill(2) on Linux.
    //
    // IOW, user generated signals never generate what we consider to be a
    // "crash".
    //
    // Similarly, ACK signals generated by this monitor.

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
                            (info->si_pid == getpid ()) ? "is monitor" : "is not monitor",
                            pid);
    }

    // Check for new thread notification.
    if ((info->si_pid == 0) && (info->si_code == SI_USER))
    {
        // A new thread creation is being signaled.  This is one of two parts that come in
        // a non-deterministic order.  pid is the thread id.
        if (log)
            log->Printf ("NativeProcessLinux::%s() pid = %" PRIu64 " tid %" PRIu64 ": new thread notification",
                     __FUNCTION__, GetID (), pid);

        // Did we already create the thread?
        bool already_tracked = false;
        thread_sp = GetOrCreateThread (pid, already_tracked);
        assert (thread_sp.get() && "failed to get or create the tracking data for newly created inferior thread");

        // If the thread was already tracked, it means the main thread already received its SIGTRAP for the create.
        if (already_tracked)
        {
            // We can now resume this thread up since it is fully created.
            reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetRunning ();
            Resume (thread_sp->GetID (), LLDB_INVALID_SIGNAL_NUMBER);
        }
        else
        {
            // Mark the thread as currently launching.  Need to wait for SIGTRAP clone on the main thread before
            // this thread is ready to go.
            reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetLaunching ();
        }

        // Done handling.
        return;
    }

    // Check for thread stop notification.
    if ((info->si_pid == getpid ()) && (info->si_code == SI_TKILL) && (signo == SIGSTOP))
    {
        // This is a tgkill()-based stop.
        if (thread_sp)
        {
            // An inferior thread just stopped.  Mark it as such.
            reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetStoppedBySignal (signo);
            SetCurrentThreadID (thread_sp->GetID ());

            // Remove this tid from the wait-for-stop set.
            Mutex::Locker locker (m_wait_for_stop_tids_mutex);

            auto removed_count = m_wait_for_stop_tids.erase (thread_sp->GetID ());
            if (removed_count < 1)
            {
                log->Printf ("NativeProcessLinux::%s() pid = %" PRIu64 " tid %" PRIu64 ": tgkill()-stopped thread not in m_wait_for_stop_tids",
                             __FUNCTION__, GetID (), thread_sp->GetID ());

            }

            // If this is the last thread in the m_wait_for_stop_tids, we need to notify
            // the delegate that a stop has occurred now that every thread that was supposed
            // to stop has stopped.
            if (m_wait_for_stop_tids.empty ())
            {
                if (log)
                {
                    log->Printf ("NativeProcessLinux::%s() pid %" PRIu64 " tid %" PRIu64 ", setting process state to stopped now that all tids marked for stop have completed",
                                 __FUNCTION__,
                                 GetID (),
                                 pid);
                }
                SetState (StateType::eStateStopped, true);
            }
        }

        // Done handling.
        return;
    }

    if (log)
        log->Printf ("NativeProcessLinux::%s() received signal %s", __FUNCTION__, GetUnixSignals ().GetSignalAsCString (signo));

    switch (signo)
    {
    case SIGSEGV:
        {
            lldb::addr_t fault_addr = reinterpret_cast<lldb::addr_t>(info->si_addr);

            // FIXME figure out how to propagate this properly.  Seems like it
            // should go in ThreadStopInfo.
            // We can get more details on the exact nature of the crash here.
            // ProcessMessage::CrashReason reason = GetCrashReasonForSIGSEGV(info);
            if (!exited)
            {
                // This is just a pre-signal-delivery notification of the incoming signal.
                // Send a stop to the debugger.
                if (thread_sp)
                {
                    reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetStoppedBySignal (signo);
                    SetCurrentThreadID (thread_sp->GetID ());
                }
                SetState (StateType::eStateStopped, true);
            }
            else
            {
                if (thread_sp)
                {
                    // FIXME figure out what type this is.
                    const uint64_t exception_type = static_cast<uint64_t> (SIGSEGV);
                    reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetCrashedWithException (exception_type, fault_addr);
                }
                SetState (StateType::eStateCrashed, true);
            }
        }
        break;

    case SIGABRT:
    case SIGILL:
    case SIGFPE:
    case SIGBUS:
        {
            // Break these out into separate cases once I have more data for each type of signal.
            lldb::addr_t fault_addr = reinterpret_cast<lldb::addr_t>(info->si_addr);
            if (!exited)
            {
                // This is just a pre-signal-delivery notification of the incoming signal.
                // Send a stop to the debugger.
                if (thread_sp)
                {
                    reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetStoppedBySignal (signo);
                    SetCurrentThreadID (thread_sp->GetID ());
                }
                SetState (StateType::eStateStopped, true);
            }
            else
            {
                if (thread_sp)
                {
                    // FIXME figure out how to report exit by signal correctly.
                    const uint64_t exception_type = static_cast<uint64_t> (SIGABRT);
                    reinterpret_cast<NativeThreadLinux*> (thread_sp.get ())->SetCrashedWithException (exception_type, fault_addr);
                }
                SetState (StateType::eStateCrashed, true);
            }
        }
        break;

    default:
        if (log)
            log->Printf ("NativeProcessLinux::%s unhandled signal %s (%d)", __FUNCTION__, GetUnixSignals ().GetSignalAsCString (signo), signo);
        break;
    }
}

Error
NativeProcessLinux::Resume (const ResumeActionList &resume_actions)
{
    Error error;

    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf ("NativeProcessLinux::%s called: pid %" PRIu64, __FUNCTION__, GetID ());

    int run_thread_count = 0;
    int stop_thread_count = 0;
    int step_thread_count = 0;

    std::vector<NativeThreadProtocolSP> new_stop_threads;

    Mutex::Locker locker (m_threads_mutex);
    for (auto thread_sp : m_threads)
    {
        assert (thread_sp && "thread list should not contain NULL threads");
        NativeThreadLinux *const linux_thread_p = reinterpret_cast<NativeThreadLinux*> (thread_sp.get ());

        const ResumeAction *const action = resume_actions.GetActionForThread (thread_sp->GetID (), true);
        assert (action && "NULL ResumeAction returned for thread during Resume ()");

        if (log)
        {
            log->Printf ("NativeProcessLinux::%s processing resume action state %s for pid %" PRIu64 " tid %" PRIu64, 
                    __FUNCTION__, StateAsCString (action->state), GetID (), thread_sp->GetID ());
        }

        switch (action->state)
        {
        case eStateRunning:
            // Run the thread, possibly feeding it the signal.
            linux_thread_p->SetRunning ();
            if (action->signal > 0)
            {
                // Resume the thread and deliver the given signal,
                // then mark as delivered.
                Resume (thread_sp->GetID (), action->signal);
                resume_actions.SetSignalHandledForThread (thread_sp->GetID ());
            }
            else
            {
                // Just resume the thread with no signal.
                Resume (thread_sp->GetID (), LLDB_INVALID_SIGNAL_NUMBER);
            }
            ++run_thread_count;
            break;

        case eStateStepping:
            // Note: if we have multiple threads, we may need to stop
            // the other threads first, then step this one.
            linux_thread_p->SetStepping ();
            if (SingleStep (thread_sp->GetID (), 0))
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 " single step succeeded",
                                 __FUNCTION__, GetID (), thread_sp->GetID ());
            }
            else
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 " single step failed",
                                 __FUNCTION__, GetID (), thread_sp->GetID ());
            }
            ++step_thread_count;
            break;

        case eStateSuspended:
        case eStateStopped:
            if (!StateIsStoppedState (linux_thread_p->GetState (), false))
                new_stop_threads.push_back (thread_sp);
            else
            {
                if (log)
                    log->Printf ("NativeProcessLinux::%s no need to stop pid %" PRIu64 " tid %" PRIu64 ", thread state already %s",
                                 __FUNCTION__, GetID (), thread_sp->GetID (), StateAsCString (linux_thread_p->GetState ()));
            }

            ++stop_thread_count;
            break;

        default:
            return Error ("NativeProcessLinux::%s (): unexpected state %s specified for pid %" PRIu64 ", tid %" PRIu64,
                    __FUNCTION__, StateAsCString (action->state), GetID (), thread_sp->GetID ());
        }
    }

    // If any thread was set to run, notify the process state as running.
    if (run_thread_count > 0)
        SetState (StateType::eStateRunning, true);

    // Now do a tgkill SIGSTOP on each thread we want to stop.
    if (!new_stop_threads.empty ())
    {
        // Lock the m_wait_for_stop_tids set so we can fill it with every thread we expect to have stopped.
        Mutex::Locker stop_thread_id_locker (m_wait_for_stop_tids_mutex);
        for (auto thread_sp : new_stop_threads)
        {
            // Send a stop signal to the thread.
            const int result = tgkill (GetID (), thread_sp->GetID (), SIGSTOP);
            if (result != 0)
            {
                // tgkill failed.
                if (log)
                    log->Printf ("NativeProcessLinux::%s error: tgkill SIGSTOP for pid %" PRIu64 " tid %" PRIu64 "failed, retval %d",
                                 __FUNCTION__, GetID (), thread_sp->GetID (), result);
            }
            else
            {
                // tgkill succeeded.  Don't mark the thread state, though.  Let the signal
                // handling mark it.
                if (log)
                    log->Printf ("NativeProcessLinux::%s tgkill SIGSTOP for pid %" PRIu64 " tid %" PRIu64 " succeeded",
                                 __FUNCTION__, GetID (), thread_sp->GetID ());

                // Add it to the set of threads we expect to signal a stop.
                // We won't tell the delegate about it until this list drains to empty.
                m_wait_for_stop_tids.insert (thread_sp->GetID ());
            }
        }
    }

    return error;
}

Error
NativeProcessLinux::Halt ()
{
    Error error;

    // FIXME check if we're already stopped
    const bool is_stopped = false;
    if (is_stopped)
        return error;

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
    StopMonitor ();

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
NativeProcessLinux::AllocateMemory (
    lldb::addr_t size,
    uint32_t permissions,
    lldb::addr_t &addr)
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
NativeProcessLinux::GetSoftwareBreakpointSize (NativeRegisterContextSP context_sp, uint32_t &actual_opcode_size)
{
    // FIXME put this behind a breakpoint protocol class that can be
    // set per architecture.  Need ARM, MIPS support here.
    static const uint8_t g_i386_opcode [] = { 0xCC };

    switch (m_arch.GetMachine ())
    {
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
            actual_opcode_size = static_cast<uint32_t> (sizeof(g_i386_opcode));
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
NativeProcessLinux::GetSoftwareBreakpointTrapOpcode (size_t trap_opcode_size_hint, size_t &actual_opcode_size, const uint8_t *&trap_opcode_bytes)
{
    // FIXME put this behind a breakpoint protocol class that can be
    // set per architecture.  Need ARM, MIPS support here.
    static const uint8_t g_i386_opcode [] = { 0xCC };

    switch (m_arch.GetMachine ())
    {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        trap_opcode_bytes = g_i386_opcode;
        actual_opcode_size = sizeof(g_i386_opcode);
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

void
NativeProcessLinux::ServeOperation(OperationArgs *args)
{
    NativeProcessLinux *monitor = args->m_monitor;

    // We are finised with the arguments and are ready to go.  Sync with the
    // parent thread and start serving operations on the inferior.
    sem_post(&args->m_semaphore);

    for(;;)
    {
        // wait for next pending operation
        if (sem_wait(&monitor->m_operation_pending))
        {
            if (errno == EINTR)
                continue;
            assert(false && "Unexpected errno from sem_wait");
        }

        reinterpret_cast<Operation*>(monitor->m_operation)->Execute(monitor);

        // notify calling thread that operation is complete
        sem_post(&monitor->m_operation_done);
    }
}

void
NativeProcessLinux::DoOperation(void *op)
{
    Mutex::Locker lock(m_operation_mutex);

    m_operation = op;

    // notify operation thread that an operation is ready to be processed
    sem_post(&m_operation_pending);

    // wait for operation to complete
    while (sem_wait(&m_operation_done))
    {
        if (errno == EINTR)
            continue;
        assert(false && "Unexpected errno from sem_wait");
    }
}

Error
NativeProcessLinux::ReadMemory (lldb::addr_t addr, void *buf, lldb::addr_t size, lldb::addr_t &bytes_read)
{
    ReadOperation op(addr, buf, size, bytes_read);
    DoOperation(&op);
    return op.GetError ();
}

Error
NativeProcessLinux::WriteMemory (lldb::addr_t addr, const void *buf, lldb::addr_t size, lldb::addr_t &bytes_written)
{
    WriteOperation op(addr, buf, size, bytes_written);
    DoOperation(&op);
    return op.GetError ();
}

bool
NativeProcessLinux::ReadRegisterValue(lldb::tid_t tid, uint32_t offset, const char* reg_name,
                                  uint32_t size, RegisterValue &value)
{
    bool result;
    ReadRegOperation op(tid, offset, reg_name, value, result);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::WriteRegisterValue(lldb::tid_t tid, unsigned offset,
                                   const char* reg_name, const RegisterValue &value)
{
    bool result;
    WriteRegOperation op(tid, offset, reg_name, value, result);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::ReadGPR(lldb::tid_t tid, void *buf, size_t buf_size)
{
    bool result;
    ReadGPROperation op(tid, buf, buf_size, result);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::ReadFPR(lldb::tid_t tid, void *buf, size_t buf_size)
{
    bool result;
    ReadFPROperation op(tid, buf, buf_size, result);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::ReadRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset)
{
    bool result;
    ReadRegisterSetOperation op(tid, buf, buf_size, regset, result);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::WriteGPR(lldb::tid_t tid, void *buf, size_t buf_size)
{
    bool result;
    WriteGPROperation op(tid, buf, buf_size, result);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::WriteFPR(lldb::tid_t tid, void *buf, size_t buf_size)
{
    bool result;
    WriteFPROperation op(tid, buf, buf_size, result);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::WriteRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset)
{
    bool result;
    WriteRegisterSetOperation op(tid, buf, buf_size, regset, result);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::Resume (lldb::tid_t tid, uint32_t signo)
{
    bool result;
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    if (log)
        log->Printf ("NativeProcessLinux::%s() resuming thread = %"  PRIu64 " with signal %s", __FUNCTION__, tid,
                                 GetUnixSignals().GetSignalAsCString (signo));
    ResumeOperation op (tid, signo, result);
    DoOperation (&op);
    if (log)
        log->Printf ("NativeProcessLinux::%s() resuming result = %s", __FUNCTION__, result ? "true" : "false");
    return result;
}

bool
NativeProcessLinux::SingleStep(lldb::tid_t tid, uint32_t signo)
{
    bool result;
    SingleStepOperation op(tid, signo, result);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::GetSignalInfo(lldb::tid_t tid, void *siginfo, int &ptrace_err)
{
    bool result;
    SiginfoOperation op(tid, siginfo, result, ptrace_err);
    DoOperation(&op);
    return result;
}

bool
NativeProcessLinux::GetEventMessage(lldb::tid_t tid, unsigned long *message)
{
    bool result;
    EventMessageOperation op(tid, message, result);
    DoOperation(&op);
    return result;
}

lldb_private::Error
NativeProcessLinux::Detach(lldb::tid_t tid)
{
    lldb_private::Error error;
    if (tid != LLDB_INVALID_THREAD_ID)
    {
        DetachOperation op(tid, error);
        DoOperation(&op);
    }
    return error;
}

bool
NativeProcessLinux::DupDescriptor(const char *path, int fd, int flags)
{
    int target_fd = open(path, flags, 0666);

    if (target_fd == -1)
        return false;

    return (dup2(target_fd, fd) == -1) ? false : true;
}

void
NativeProcessLinux::StopMonitoringChildProcess()
{
    lldb::thread_result_t thread_result;

    if (IS_VALID_LLDB_HOST_THREAD(m_monitor_thread))
    {
        Host::ThreadCancel(m_monitor_thread, NULL);
        Host::ThreadJoin(m_monitor_thread, &thread_result, NULL);
        m_monitor_thread = LLDB_INVALID_HOST_THREAD;
    }
}

void
NativeProcessLinux::StopMonitor()
{
    StopMonitoringChildProcess();
    StopOpThread();
    sem_destroy(&m_operation_pending);
    sem_destroy(&m_operation_done);

    // TODO: validate whether this still holds, fix up comment.
    // Note: ProcessPOSIX passes the m_terminal_fd file descriptor to
    // Process::SetSTDIOFileDescriptor, which in turn transfers ownership of
    // the descriptor to a ConnectionFileDescriptor object.  Consequently
    // even though still has the file descriptor, we shouldn't close it here.
}

void
NativeProcessLinux::StopOpThread()
{
    lldb::thread_result_t result;

    if (!IS_VALID_LLDB_HOST_THREAD(m_operation_thread))
        return;

    Host::ThreadCancel(m_operation_thread, NULL);
    Host::ThreadJoin(m_operation_thread, &result, NULL);
    m_operation_thread = LLDB_INVALID_HOST_THREAD;
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
    Mutex::Locker locker (m_threads_mutex);
    for (auto it = m_threads.begin (); it != m_threads.end (); ++it)
    {
        if (*it && ((*it)->GetID () == thread_id))
        {
            m_threads.erase (it);
            return true;
        }
    }

    // Didn't find it.
    return false;
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

NativeThreadProtocolSP
NativeProcessLinux::GetOrCreateThread (lldb::tid_t thread_id, bool &created)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));

    Mutex::Locker locker (m_threads_mutex);
    if (log)
    {
        log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " get/create thread with tid %" PRIu64,
                     __FUNCTION__,
                     GetID (),
                     thread_id);
    }

    // Retrieve the thread if it is already getting tracked.
    NativeThreadProtocolSP thread_sp = MaybeGetThreadNoLock (thread_id);
    if (thread_sp)
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 ": thread already tracked, returning",
                         __FUNCTION__,
                         GetID (),
                         thread_id);
        created = false;
        return thread_sp;

    }

    // Create the thread metadata since it isn't being tracked.
    if (log)
        log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 ": thread didn't exist, tracking now",
                     __FUNCTION__,
                     GetID (),
                     thread_id);

    thread_sp.reset (new NativeThreadLinux (this, thread_id));
    m_threads.push_back (thread_sp);
    created = true;
    
    return thread_sp;
}

Error
NativeProcessLinux::FixupBreakpointPCAsNeeded (NativeThreadProtocolSP &thread_sp)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));

    Error error;

    // Get a linux thread pointer.
    if (!thread_sp)
    {
        error.SetErrorString ("null thread_sp");
        if (log)
            log->Printf ("NativeProcessLinux::%s failed: %s", __FUNCTION__, error.AsCString ());
        return error;
    }
    NativeThreadLinux *const linux_thread_p = reinterpret_cast<NativeThreadLinux*> (thread_sp.get());

    // Find out the size of a breakpoint (might depend on where we are in the code).
    NativeRegisterContextSP context_sp = linux_thread_p->GetRegisterContext ();
    if (!context_sp)
    {
        error.SetErrorString ("cannot get a NativeRegisterContext for the thread");
        if (log)
            log->Printf ("NativeProcessLinux::%s failed: %s", __FUNCTION__, error.AsCString ());
        return error;
    }

    uint32_t breakpoint_size = 0;
    error = GetSoftwareBreakpointSize (context_sp, breakpoint_size);
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
    if (breakpoint_size > static_cast<lldb::addr_t> (0))
    {
        // Do not allow breakpoint probe to wrap around.
        if (breakpoint_addr >= static_cast<lldb::addr_t> (breakpoint_size))
            breakpoint_addr -= static_cast<lldb::addr_t> (breakpoint_size);
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
        log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 ": changing PC from 0x%" PRIx64 " to 0x%" PRIx64, __FUNCTION__, GetID (), linux_thread_p->GetID (), initial_pc_addr, breakpoint_addr);

    error = context_sp->SetPC (breakpoint_addr);
    if (error.Fail ())
    {
        if (log)
            log->Printf ("NativeProcessLinux::%s pid %" PRIu64 " tid %" PRIu64 ": failed to set PC: %s", __FUNCTION__, GetID (), linux_thread_p->GetID (), error.AsCString ());
        return error;
    }

    return error;
}

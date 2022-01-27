//===-- GDBRemoteCommunicationServerLLGS.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>

#include "lldb/Host/Config.h"


#include <chrono>
#include <cstring>
#include <limits>
#include <thread>

#include "GDBRemoteCommunicationServerLLGS.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/Debug.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileAction.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/GDBRemote.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/UnimplementedError.h"
#include "lldb/Utility/UriParser.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ScopedPrinter.h"

#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"
#include "lldb/Utility/StringExtractorGDBRemote.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

// GDBRemote Errors

namespace {
enum GDBRemoteServerError {
  // Set to the first unused error number in literal form below
  eErrorFirst = 29,
  eErrorNoProcess = eErrorFirst,
  eErrorResume,
  eErrorExitStatus
};
}

// GDBRemoteCommunicationServerLLGS constructor
GDBRemoteCommunicationServerLLGS::GDBRemoteCommunicationServerLLGS(
    MainLoop &mainloop, const NativeProcessProtocol::Factory &process_factory)
    : GDBRemoteCommunicationServerCommon("gdb-remote.server",
                                         "gdb-remote.server.rx_packet"),
      m_mainloop(mainloop), m_process_factory(process_factory),
      m_current_process(nullptr), m_continue_process(nullptr),
      m_stdio_communication("process.stdio") {
  RegisterPacketHandlers();
}

void GDBRemoteCommunicationServerLLGS::RegisterPacketHandlers() {
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_C,
                                &GDBRemoteCommunicationServerLLGS::Handle_C);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_c,
                                &GDBRemoteCommunicationServerLLGS::Handle_c);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_D,
                                &GDBRemoteCommunicationServerLLGS::Handle_D);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_H,
                                &GDBRemoteCommunicationServerLLGS::Handle_H);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_I,
                                &GDBRemoteCommunicationServerLLGS::Handle_I);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_interrupt,
      &GDBRemoteCommunicationServerLLGS::Handle_interrupt);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_m,
      &GDBRemoteCommunicationServerLLGS::Handle_memory_read);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_M,
                                &GDBRemoteCommunicationServerLLGS::Handle_M);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType__M,
                                &GDBRemoteCommunicationServerLLGS::Handle__M);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType__m,
                                &GDBRemoteCommunicationServerLLGS::Handle__m);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_p,
                                &GDBRemoteCommunicationServerLLGS::Handle_p);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_P,
                                &GDBRemoteCommunicationServerLLGS::Handle_P);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_qC,
                                &GDBRemoteCommunicationServerLLGS::Handle_qC);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qfThreadInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qfThreadInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qFileLoadAddress,
      &GDBRemoteCommunicationServerLLGS::Handle_qFileLoadAddress);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qGetWorkingDir,
      &GDBRemoteCommunicationServerLLGS::Handle_qGetWorkingDir);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QThreadSuffixSupported,
      &GDBRemoteCommunicationServerLLGS::Handle_QThreadSuffixSupported);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QListThreadsInStopReply,
      &GDBRemoteCommunicationServerLLGS::Handle_QListThreadsInStopReply);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qMemoryRegionInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qMemoryRegionInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qMemoryRegionInfoSupported,
      &GDBRemoteCommunicationServerLLGS::Handle_qMemoryRegionInfoSupported);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qProcessInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qProcessInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qRegisterInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qRegisterInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QRestoreRegisterState,
      &GDBRemoteCommunicationServerLLGS::Handle_QRestoreRegisterState);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QSaveRegisterState,
      &GDBRemoteCommunicationServerLLGS::Handle_QSaveRegisterState);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QSetDisableASLR,
      &GDBRemoteCommunicationServerLLGS::Handle_QSetDisableASLR);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QSetWorkingDir,
      &GDBRemoteCommunicationServerLLGS::Handle_QSetWorkingDir);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qsThreadInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qsThreadInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qThreadStopInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qThreadStopInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jThreadsInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_jThreadsInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qWatchpointSupportInfo,
      &GDBRemoteCommunicationServerLLGS::Handle_qWatchpointSupportInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qXfer,
      &GDBRemoteCommunicationServerLLGS::Handle_qXfer);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_s,
                                &GDBRemoteCommunicationServerLLGS::Handle_s);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_stop_reason,
      &GDBRemoteCommunicationServerLLGS::Handle_stop_reason); // ?
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_vAttach,
      &GDBRemoteCommunicationServerLLGS::Handle_vAttach);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_vAttachWait,
      &GDBRemoteCommunicationServerLLGS::Handle_vAttachWait);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qVAttachOrWaitSupported,
      &GDBRemoteCommunicationServerLLGS::Handle_qVAttachOrWaitSupported);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_vAttachOrWait,
      &GDBRemoteCommunicationServerLLGS::Handle_vAttachOrWait);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_vCont,
      &GDBRemoteCommunicationServerLLGS::Handle_vCont);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_vCont_actions,
      &GDBRemoteCommunicationServerLLGS::Handle_vCont_actions);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_vRun,
      &GDBRemoteCommunicationServerLLGS::Handle_vRun);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_x,
      &GDBRemoteCommunicationServerLLGS::Handle_memory_read);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_Z,
                                &GDBRemoteCommunicationServerLLGS::Handle_Z);
  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_z,
                                &GDBRemoteCommunicationServerLLGS::Handle_z);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QPassSignals,
      &GDBRemoteCommunicationServerLLGS::Handle_QPassSignals);

  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jLLDBTraceSupported,
      &GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceSupported);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jLLDBTraceStart,
      &GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceStart);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jLLDBTraceStop,
      &GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceStop);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jLLDBTraceGetState,
      &GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceGetState);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jLLDBTraceGetBinaryData,
      &GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceGetBinaryData);

  RegisterMemberFunctionHandler(StringExtractorGDBRemote::eServerPacketType_g,
                                &GDBRemoteCommunicationServerLLGS::Handle_g);

  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qMemTags,
      &GDBRemoteCommunicationServerLLGS::Handle_qMemTags);

  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QMemTags,
      &GDBRemoteCommunicationServerLLGS::Handle_QMemTags);

  RegisterPacketHandler(StringExtractorGDBRemote::eServerPacketType_k,
                        [this](StringExtractorGDBRemote packet, Status &error,
                               bool &interrupt, bool &quit) {
                          quit = true;
                          return this->Handle_k(packet);
                        });

  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qLLDBSaveCore,
      &GDBRemoteCommunicationServerLLGS::Handle_qSaveCore);
}

void GDBRemoteCommunicationServerLLGS::SetLaunchInfo(const ProcessLaunchInfo &info) {
  m_process_launch_info = info;
}

Status GDBRemoteCommunicationServerLLGS::LaunchProcess() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (!m_process_launch_info.GetArguments().GetArgumentCount())
    return Status("%s: no process command line specified to launch",
                  __FUNCTION__);

  const bool should_forward_stdio =
      m_process_launch_info.GetFileActionForFD(STDIN_FILENO) == nullptr ||
      m_process_launch_info.GetFileActionForFD(STDOUT_FILENO) == nullptr ||
      m_process_launch_info.GetFileActionForFD(STDERR_FILENO) == nullptr;
  m_process_launch_info.SetLaunchInSeparateProcessGroup(true);
  m_process_launch_info.GetFlags().Set(eLaunchFlagDebug);

  if (should_forward_stdio) {
    // Temporarily relax the following for Windows until we can take advantage
    // of the recently added pty support. This doesn't really affect the use of
    // lldb-server on Windows.
#if !defined(_WIN32)
    if (llvm::Error Err = m_process_launch_info.SetUpPtyRedirection())
      return Status(std::move(Err));
#endif
  }

  {
    std::lock_guard<std::recursive_mutex> guard(m_debugged_process_mutex);
    assert(m_debugged_processes.empty() && "lldb-server creating debugged "
                                           "process but one already exists");
    auto process_or =
        m_process_factory.Launch(m_process_launch_info, *this, m_mainloop);
    if (!process_or)
      return Status(process_or.takeError());
    m_continue_process = m_current_process = process_or->get();
    m_debugged_processes[m_current_process->GetID()] = std::move(*process_or);
  }

  SetEnabledExtensions(*m_current_process);

  // Handle mirroring of inferior stdout/stderr over the gdb-remote protocol as
  // needed. llgs local-process debugging may specify PTY paths, which will
  // make these file actions non-null process launch -i/e/o will also make
  // these file actions non-null nullptr means that the traffic is expected to
  // flow over gdb-remote protocol
  if (should_forward_stdio) {
    // nullptr means it's not redirected to file or pty (in case of LLGS local)
    // at least one of stdio will be transferred pty<->gdb-remote we need to
    // give the pty primary handle to this object to read and/or write
    LLDB_LOG(log,
             "pid = {0}: setting up stdout/stderr redirection via $O "
             "gdb-remote commands",
             m_current_process->GetID());

    // Setup stdout/stderr mapping from inferior to $O
    auto terminal_fd = m_current_process->GetTerminalFileDescriptor();
    if (terminal_fd >= 0) {
      LLDB_LOGF(log,
                "ProcessGDBRemoteCommunicationServerLLGS::%s setting "
                "inferior STDIO fd to %d",
                __FUNCTION__, terminal_fd);
      Status status = SetSTDIOFileDescriptor(terminal_fd);
      if (status.Fail())
        return status;
    } else {
      LLDB_LOGF(log,
                "ProcessGDBRemoteCommunicationServerLLGS::%s ignoring "
                "inferior STDIO since terminal fd reported as %d",
                __FUNCTION__, terminal_fd);
    }
  } else {
    LLDB_LOG(log,
             "pid = {0} skipping stdout/stderr redirection via $O: inferior "
             "will communicate over client-provided file descriptors",
             m_current_process->GetID());
  }

  printf("Launched '%s' as process %" PRIu64 "...\n",
         m_process_launch_info.GetArguments().GetArgumentAtIndex(0),
         m_current_process->GetID());

  return Status();
}

Status GDBRemoteCommunicationServerLLGS::AttachToProcess(lldb::pid_t pid) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64,
            __FUNCTION__, pid);

  // Before we try to attach, make sure we aren't already monitoring something
  // else.
  if (!m_debugged_processes.empty())
    return Status("cannot attach to process %" PRIu64
                  " when another process with pid %" PRIu64
                  " is being debugged.",
                  pid, m_current_process->GetID());

  // Try to attach.
  auto process_or = m_process_factory.Attach(pid, *this, m_mainloop);
  if (!process_or) {
    Status status(process_or.takeError());
    llvm::errs() << llvm::formatv("failed to attach to process {0}: {1}\n", pid,
                                  status);
    return status;
  }
  m_continue_process = m_current_process = process_or->get();
  m_debugged_processes[m_current_process->GetID()] = std::move(*process_or);
  SetEnabledExtensions(*m_current_process);

  // Setup stdout/stderr mapping from inferior.
  auto terminal_fd = m_current_process->GetTerminalFileDescriptor();
  if (terminal_fd >= 0) {
    LLDB_LOGF(log,
              "ProcessGDBRemoteCommunicationServerLLGS::%s setting "
              "inferior STDIO fd to %d",
              __FUNCTION__, terminal_fd);
    Status status = SetSTDIOFileDescriptor(terminal_fd);
    if (status.Fail())
      return status;
  } else {
    LLDB_LOGF(log,
              "ProcessGDBRemoteCommunicationServerLLGS::%s ignoring "
              "inferior STDIO since terminal fd reported as %d",
              __FUNCTION__, terminal_fd);
  }

  printf("Attached to process %" PRIu64 "...\n", pid);
  return Status();
}

Status GDBRemoteCommunicationServerLLGS::AttachWaitProcess(
    llvm::StringRef process_name, bool include_existing) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  std::chrono::milliseconds polling_interval = std::chrono::milliseconds(1);

  // Create the matcher used to search the process list.
  ProcessInstanceInfoList exclusion_list;
  ProcessInstanceInfoMatch match_info;
  match_info.GetProcessInfo().GetExecutableFile().SetFile(
      process_name, llvm::sys::path::Style::native);
  match_info.SetNameMatchType(NameMatch::Equals);

  if (include_existing) {
    LLDB_LOG(log, "including existing processes in search");
  } else {
    // Create the excluded process list before polling begins.
    Host::FindProcesses(match_info, exclusion_list);
    LLDB_LOG(log, "placed '{0}' processes in the exclusion list.",
             exclusion_list.size());
  }

  LLDB_LOG(log, "waiting for '{0}' to appear", process_name);

  auto is_in_exclusion_list =
      [&exclusion_list](const ProcessInstanceInfo &info) {
        for (auto &excluded : exclusion_list) {
          if (excluded.GetProcessID() == info.GetProcessID())
            return true;
        }
        return false;
      };

  ProcessInstanceInfoList loop_process_list;
  while (true) {
    loop_process_list.clear();
    if (Host::FindProcesses(match_info, loop_process_list)) {
      // Remove all the elements that are in the exclusion list.
      llvm::erase_if(loop_process_list, is_in_exclusion_list);

      // One match! We found the desired process.
      if (loop_process_list.size() == 1) {
        auto matching_process_pid = loop_process_list[0].GetProcessID();
        LLDB_LOG(log, "found pid {0}", matching_process_pid);
        return AttachToProcess(matching_process_pid);
      }

      // Multiple matches! Return an error reporting the PIDs we found.
      if (loop_process_list.size() > 1) {
        StreamString error_stream;
        error_stream.Format(
            "Multiple executables with name: '{0}' found. Pids: ",
            process_name);
        for (size_t i = 0; i < loop_process_list.size() - 1; ++i) {
          error_stream.Format("{0}, ", loop_process_list[i].GetProcessID());
        }
        error_stream.Format("{0}.", loop_process_list.back().GetProcessID());

        Status error;
        error.SetErrorString(error_stream.GetString());
        return error;
      }
    }
    // No matches, we have not found the process. Sleep until next poll.
    LLDB_LOG(log, "sleep {0} seconds", polling_interval);
    std::this_thread::sleep_for(polling_interval);
  }
}

void GDBRemoteCommunicationServerLLGS::InitializeDelegate(
    NativeProcessProtocol *process) {
  assert(process && "process cannot be NULL");
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  if (log) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s called with "
              "NativeProcessProtocol pid %" PRIu64 ", current state: %s",
              __FUNCTION__, process->GetID(),
              StateAsCString(process->GetState()));
  }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::SendWResponse(
    NativeProcessProtocol *process) {
  assert(process && "process cannot be NULL");
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // send W notification
  auto wait_status = process->GetExitStatus();
  if (!wait_status) {
    LLDB_LOG(log, "pid = {0}, failed to retrieve process exit status",
             process->GetID());

    StreamGDBRemote response;
    response.PutChar('E');
    response.PutHex8(GDBRemoteServerError::eErrorExitStatus);
    return SendPacketNoLock(response.GetString());
  }

  LLDB_LOG(log, "pid = {0}, returning exit type {1}", process->GetID(),
           *wait_status);

  StreamGDBRemote response;
  response.Format("{0:g}", *wait_status);
  return SendPacketNoLock(response.GetString());
}

static void AppendHexValue(StreamString &response, const uint8_t *buf,
                           uint32_t buf_size, bool swap) {
  int64_t i;
  if (swap) {
    for (i = buf_size - 1; i >= 0; i--)
      response.PutHex8(buf[i]);
  } else {
    for (i = 0; i < buf_size; i++)
      response.PutHex8(buf[i]);
  }
}

static llvm::StringRef GetEncodingNameOrEmpty(const RegisterInfo &reg_info) {
  switch (reg_info.encoding) {
  case eEncodingUint:
    return "uint";
  case eEncodingSint:
    return "sint";
  case eEncodingIEEE754:
    return "ieee754";
  case eEncodingVector:
    return "vector";
  default:
    return "";
  }
}

static llvm::StringRef GetFormatNameOrEmpty(const RegisterInfo &reg_info) {
  switch (reg_info.format) {
  case eFormatBinary:
    return "binary";
  case eFormatDecimal:
    return "decimal";
  case eFormatHex:
    return "hex";
  case eFormatFloat:
    return "float";
  case eFormatVectorOfSInt8:
    return "vector-sint8";
  case eFormatVectorOfUInt8:
    return "vector-uint8";
  case eFormatVectorOfSInt16:
    return "vector-sint16";
  case eFormatVectorOfUInt16:
    return "vector-uint16";
  case eFormatVectorOfSInt32:
    return "vector-sint32";
  case eFormatVectorOfUInt32:
    return "vector-uint32";
  case eFormatVectorOfFloat32:
    return "vector-float32";
  case eFormatVectorOfUInt64:
    return "vector-uint64";
  case eFormatVectorOfUInt128:
    return "vector-uint128";
  default:
    return "";
  };
}

static llvm::StringRef GetKindGenericOrEmpty(const RegisterInfo &reg_info) {
  switch (reg_info.kinds[RegisterKind::eRegisterKindGeneric]) {
  case LLDB_REGNUM_GENERIC_PC:
    return "pc";
  case LLDB_REGNUM_GENERIC_SP:
    return "sp";
  case LLDB_REGNUM_GENERIC_FP:
    return "fp";
  case LLDB_REGNUM_GENERIC_RA:
    return "ra";
  case LLDB_REGNUM_GENERIC_FLAGS:
    return "flags";
  case LLDB_REGNUM_GENERIC_ARG1:
    return "arg1";
  case LLDB_REGNUM_GENERIC_ARG2:
    return "arg2";
  case LLDB_REGNUM_GENERIC_ARG3:
    return "arg3";
  case LLDB_REGNUM_GENERIC_ARG4:
    return "arg4";
  case LLDB_REGNUM_GENERIC_ARG5:
    return "arg5";
  case LLDB_REGNUM_GENERIC_ARG6:
    return "arg6";
  case LLDB_REGNUM_GENERIC_ARG7:
    return "arg7";
  case LLDB_REGNUM_GENERIC_ARG8:
    return "arg8";
  default:
    return "";
  }
}

static void CollectRegNums(const uint32_t *reg_num, StreamString &response,
                           bool usehex) {
  for (int i = 0; *reg_num != LLDB_INVALID_REGNUM; ++reg_num, ++i) {
    if (i > 0)
      response.PutChar(',');
    if (usehex)
      response.Printf("%" PRIx32, *reg_num);
    else
      response.Printf("%" PRIu32, *reg_num);
  }
}

static void WriteRegisterValueInHexFixedWidth(
    StreamString &response, NativeRegisterContext &reg_ctx,
    const RegisterInfo &reg_info, const RegisterValue *reg_value_p,
    lldb::ByteOrder byte_order) {
  RegisterValue reg_value;
  if (!reg_value_p) {
    Status error = reg_ctx.ReadRegister(&reg_info, reg_value);
    if (error.Success())
      reg_value_p = &reg_value;
    // else log.
  }

  if (reg_value_p) {
    AppendHexValue(response, (const uint8_t *)reg_value_p->GetBytes(),
                   reg_value_p->GetByteSize(),
                   byte_order == lldb::eByteOrderLittle);
  } else {
    // Zero-out any unreadable values.
    if (reg_info.byte_size > 0) {
      std::basic_string<uint8_t> zeros(reg_info.byte_size, '\0');
      AppendHexValue(response, zeros.data(), zeros.size(), false);
    }
  }
}

static llvm::Optional<json::Object>
GetRegistersAsJSON(NativeThreadProtocol &thread) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  NativeRegisterContext& reg_ctx = thread.GetRegisterContext();

  json::Object register_object;

#ifdef LLDB_JTHREADSINFO_FULL_REGISTER_SET
  const auto expedited_regs =
      reg_ctx.GetExpeditedRegisters(ExpeditedRegs::Full);
#else
  const auto expedited_regs =
      reg_ctx.GetExpeditedRegisters(ExpeditedRegs::Minimal);
#endif
  if (expedited_regs.empty())
    return llvm::None;

  for (auto &reg_num : expedited_regs) {
    const RegisterInfo *const reg_info_p =
        reg_ctx.GetRegisterInfoAtIndex(reg_num);
    if (reg_info_p == nullptr) {
      LLDB_LOGF(log,
                "%s failed to get register info for register index %" PRIu32,
                __FUNCTION__, reg_num);
      continue;
    }

    if (reg_info_p->value_regs != nullptr)
      continue; // Only expedite registers that are not contained in other
                // registers.

    RegisterValue reg_value;
    Status error = reg_ctx.ReadRegister(reg_info_p, reg_value);
    if (error.Fail()) {
      LLDB_LOGF(log, "%s failed to read register '%s' index %" PRIu32 ": %s",
                __FUNCTION__,
                reg_info_p->name ? reg_info_p->name : "<unnamed-register>",
                reg_num, error.AsCString());
      continue;
    }

    StreamString stream;
    WriteRegisterValueInHexFixedWidth(stream, reg_ctx, *reg_info_p,
                                      &reg_value, lldb::eByteOrderBig);

    register_object.try_emplace(llvm::to_string(reg_num),
                                stream.GetString().str());
  }

  return register_object;
}

static const char *GetStopReasonString(StopReason stop_reason) {
  switch (stop_reason) {
  case eStopReasonTrace:
    return "trace";
  case eStopReasonBreakpoint:
    return "breakpoint";
  case eStopReasonWatchpoint:
    return "watchpoint";
  case eStopReasonSignal:
    return "signal";
  case eStopReasonException:
    return "exception";
  case eStopReasonExec:
    return "exec";
  case eStopReasonProcessorTrace:
    return "processor trace";
  case eStopReasonFork:
    return "fork";
  case eStopReasonVFork:
    return "vfork";
  case eStopReasonVForkDone:
    return "vforkdone";
  case eStopReasonInstrumentation:
  case eStopReasonInvalid:
  case eStopReasonPlanComplete:
  case eStopReasonThreadExiting:
  case eStopReasonNone:
    break; // ignored
  }
  return nullptr;
}

static llvm::Expected<json::Array>
GetJSONThreadsInfo(NativeProcessProtocol &process, bool abridged) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  json::Array threads_array;

  // Ensure we can get info on the given thread.
  uint32_t thread_idx = 0;
  for (NativeThreadProtocol *thread;
       (thread = process.GetThreadAtIndex(thread_idx)) != nullptr;
       ++thread_idx) {

    lldb::tid_t tid = thread->GetID();

    // Grab the reason this thread stopped.
    struct ThreadStopInfo tid_stop_info;
    std::string description;
    if (!thread->GetStopReason(tid_stop_info, description))
      return llvm::make_error<llvm::StringError>(
          "failed to get stop reason", llvm::inconvertibleErrorCode());

    const int signum = tid_stop_info.details.signal.signo;
    if (log) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64
                " tid %" PRIu64
                " got signal signo = %d, reason = %d, exc_type = %" PRIu64,
                __FUNCTION__, process.GetID(), tid, signum,
                tid_stop_info.reason, tid_stop_info.details.exception.type);
    }

    json::Object thread_obj;

    if (!abridged) {
      if (llvm::Optional<json::Object> registers = GetRegistersAsJSON(*thread))
        thread_obj.try_emplace("registers", std::move(*registers));
    }

    thread_obj.try_emplace("tid", static_cast<int64_t>(tid));

    if (signum != 0)
      thread_obj.try_emplace("signal", signum);

    const std::string thread_name = thread->GetName();
    if (!thread_name.empty())
      thread_obj.try_emplace("name", thread_name);

    const char *stop_reason = GetStopReasonString(tid_stop_info.reason);
    if (stop_reason)
      thread_obj.try_emplace("reason", stop_reason);

    if (!description.empty())
      thread_obj.try_emplace("description", description);

    if ((tid_stop_info.reason == eStopReasonException) &&
        tid_stop_info.details.exception.type) {
      thread_obj.try_emplace(
          "metype", static_cast<int64_t>(tid_stop_info.details.exception.type));

      json::Array medata_array;
      for (uint32_t i = 0; i < tid_stop_info.details.exception.data_count;
           ++i) {
        medata_array.push_back(
            static_cast<int64_t>(tid_stop_info.details.exception.data[i]));
      }
      thread_obj.try_emplace("medata", std::move(medata_array));
    }
    threads_array.push_back(std::move(thread_obj));
  }
  return threads_array;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::SendStopReplyPacketForThread(
    lldb::tid_t tid) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  // Ensure we have a debugged process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(50);

  LLDB_LOG(log, "preparing packet for pid {0} tid {1}",
           m_current_process->GetID(), tid);

  // Ensure we can get info on the given thread.
  NativeThreadProtocol *thread = m_current_process->GetThreadByID(tid);
  if (!thread)
    return SendErrorResponse(51);

  // Grab the reason this thread stopped.
  struct ThreadStopInfo tid_stop_info;
  std::string description;
  if (!thread->GetStopReason(tid_stop_info, description))
    return SendErrorResponse(52);

  // FIXME implement register handling for exec'd inferiors.
  // if (tid_stop_info.reason == eStopReasonExec) {
  //     const bool force = true;
  //     InitializeRegisters(force);
  // }

  StreamString response;
  // Output the T packet with the thread
  response.PutChar('T');
  int signum = tid_stop_info.details.signal.signo;
  LLDB_LOG(
      log,
      "pid {0}, tid {1}, got signal signo = {2}, reason = {3}, exc_type = {4}",
      m_current_process->GetID(), tid, signum, int(tid_stop_info.reason),
      tid_stop_info.details.exception.type);

  // Print the signal number.
  response.PutHex8(signum & 0xff);

  // Include the tid.
  response.Printf("thread:%" PRIx64 ";", tid);

  // Include the thread name if there is one.
  const std::string thread_name = thread->GetName();
  if (!thread_name.empty()) {
    size_t thread_name_len = thread_name.length();

    if (::strcspn(thread_name.c_str(), "$#+-;:") == thread_name_len) {
      response.PutCString("name:");
      response.PutCString(thread_name);
    } else {
      // The thread name contains special chars, send as hex bytes.
      response.PutCString("hexname:");
      response.PutStringAsRawHex8(thread_name);
    }
    response.PutChar(';');
  }

  // If a 'QListThreadsInStopReply' was sent to enable this feature, we will
  // send all thread IDs back in the "threads" key whose value is a list of hex
  // thread IDs separated by commas:
  //  "threads:10a,10b,10c;"
  // This will save the debugger from having to send a pair of qfThreadInfo and
  // qsThreadInfo packets, but it also might take a lot of room in the stop
  // reply packet, so it must be enabled only on systems where there are no
  // limits on packet lengths.
  if (m_list_threads_in_stop_reply) {
    response.PutCString("threads:");

    uint32_t thread_index = 0;
    NativeThreadProtocol *listed_thread;
    for (listed_thread = m_current_process->GetThreadAtIndex(thread_index);
         listed_thread; ++thread_index,
        listed_thread = m_current_process->GetThreadAtIndex(thread_index)) {
      if (thread_index > 0)
        response.PutChar(',');
      response.Printf("%" PRIx64, listed_thread->GetID());
    }
    response.PutChar(';');

    // Include JSON info that describes the stop reason for any threads that
    // actually have stop reasons. We use the new "jstopinfo" key whose values
    // is hex ascii JSON that contains the thread IDs thread stop info only for
    // threads that have stop reasons. Only send this if we have more than one
    // thread otherwise this packet has all the info it needs.
    if (thread_index > 1) {
      const bool threads_with_valid_stop_info_only = true;
      llvm::Expected<json::Array> threads_info = GetJSONThreadsInfo(
          *m_current_process, threads_with_valid_stop_info_only);
      if (threads_info) {
        response.PutCString("jstopinfo:");
        StreamString unescaped_response;
        unescaped_response.AsRawOstream() << std::move(*threads_info);
        response.PutStringAsRawHex8(unescaped_response.GetData());
        response.PutChar(';');
      } else {
        LLDB_LOG_ERROR(log, threads_info.takeError(),
                       "failed to prepare a jstopinfo field for pid {1}: {0}",
                       m_current_process->GetID());
      }
    }

    uint32_t i = 0;
    response.PutCString("thread-pcs");
    char delimiter = ':';
    for (NativeThreadProtocol *thread;
         (thread = m_current_process->GetThreadAtIndex(i)) != nullptr; ++i) {
      NativeRegisterContext& reg_ctx = thread->GetRegisterContext();

      uint32_t reg_to_read = reg_ctx.ConvertRegisterKindToRegisterNumber(
          eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
      const RegisterInfo *const reg_info_p =
          reg_ctx.GetRegisterInfoAtIndex(reg_to_read);

      RegisterValue reg_value;
      Status error = reg_ctx.ReadRegister(reg_info_p, reg_value);
      if (error.Fail()) {
        LLDB_LOGF(log, "%s failed to read register '%s' index %" PRIu32 ": %s",
                  __FUNCTION__,
                  reg_info_p->name ? reg_info_p->name : "<unnamed-register>",
                  reg_to_read, error.AsCString());
        continue;
      }

      response.PutChar(delimiter);
      delimiter = ',';
      WriteRegisterValueInHexFixedWidth(response, reg_ctx, *reg_info_p,
                                        &reg_value, endian::InlHostByteOrder());
    }

    response.PutChar(';');
  }

  //
  // Expedite registers.
  //

  // Grab the register context.
  NativeRegisterContext& reg_ctx = thread->GetRegisterContext();
  const auto expedited_regs =
      reg_ctx.GetExpeditedRegisters(ExpeditedRegs::Full);

  for (auto &reg_num : expedited_regs) {
    const RegisterInfo *const reg_info_p =
        reg_ctx.GetRegisterInfoAtIndex(reg_num);
    // Only expediate registers that are not contained in other registers.
    if (reg_info_p != nullptr && reg_info_p->value_regs == nullptr) {
      RegisterValue reg_value;
      Status error = reg_ctx.ReadRegister(reg_info_p, reg_value);
      if (error.Success()) {
        response.Printf("%.02x:", reg_num);
        WriteRegisterValueInHexFixedWidth(response, reg_ctx, *reg_info_p,
                                          &reg_value, lldb::eByteOrderBig);
        response.PutChar(';');
      } else {
        LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s failed to read "
                       "register '%s' index %" PRIu32 ": %s",
                  __FUNCTION__,
                  reg_info_p->name ? reg_info_p->name : "<unnamed-register>",
                  reg_num, error.AsCString());
      }
    }
  }

  const char *reason_str = GetStopReasonString(tid_stop_info.reason);
  if (reason_str != nullptr) {
    response.Printf("reason:%s;", reason_str);
  }

  if (!description.empty()) {
    // Description may contains special chars, send as hex bytes.
    response.PutCString("description:");
    response.PutStringAsRawHex8(description);
    response.PutChar(';');
  } else if ((tid_stop_info.reason == eStopReasonException) &&
             tid_stop_info.details.exception.type) {
    response.PutCString("metype:");
    response.PutHex64(tid_stop_info.details.exception.type);
    response.PutCString(";mecount:");
    response.PutHex32(tid_stop_info.details.exception.data_count);
    response.PutChar(';');

    for (uint32_t i = 0; i < tid_stop_info.details.exception.data_count; ++i) {
      response.PutCString("medata:");
      response.PutHex64(tid_stop_info.details.exception.data[i]);
      response.PutChar(';');
    }
  }

  // Include child process PID/TID for forks.
  if (tid_stop_info.reason == eStopReasonFork ||
      tid_stop_info.reason == eStopReasonVFork) {
    assert(bool(m_extensions_supported &
                NativeProcessProtocol::Extension::multiprocess));
    if (tid_stop_info.reason == eStopReasonFork)
      assert(bool(m_extensions_supported &
                  NativeProcessProtocol::Extension::fork));
    if (tid_stop_info.reason == eStopReasonVFork)
      assert(bool(m_extensions_supported &
                  NativeProcessProtocol::Extension::vfork));
    response.Printf("%s:p%" PRIx64 ".%" PRIx64 ";", reason_str,
                    tid_stop_info.details.fork.child_pid,
                    tid_stop_info.details.fork.child_tid);
  }

  return SendPacketNoLock(response.GetString());
}

void GDBRemoteCommunicationServerLLGS::HandleInferiorState_Exited(
    NativeProcessProtocol *process) {
  assert(process && "process cannot be NULL");

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s called", __FUNCTION__);

  PacketResult result = SendStopReasonForState(StateType::eStateExited);
  if (result != PacketResult::Success) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed to send stop "
              "notification for PID %" PRIu64 ", state: eStateExited",
              __FUNCTION__, process->GetID());
  }

  // Close the pipe to the inferior terminal i/o if we launched it and set one
  // up.
  MaybeCloseInferiorTerminalConnection();

  // We are ready to exit the debug monitor.
  m_exit_now = true;
  m_mainloop.RequestTermination();
}

void GDBRemoteCommunicationServerLLGS::HandleInferiorState_Stopped(
    NativeProcessProtocol *process) {
  assert(process && "process cannot be NULL");

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s called", __FUNCTION__);

  // Send the stop reason unless this is the stop after the launch or attach.
  switch (m_inferior_prev_state) {
  case eStateLaunching:
  case eStateAttaching:
    // Don't send anything per debugserver behavior.
    break;
  default:
    // In all other cases, send the stop reason.
    PacketResult result = SendStopReasonForState(StateType::eStateStopped);
    if (result != PacketResult::Success) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s failed to send stop "
                "notification for PID %" PRIu64 ", state: eStateExited",
                __FUNCTION__, process->GetID());
    }
    break;
  }
}

void GDBRemoteCommunicationServerLLGS::ProcessStateChanged(
    NativeProcessProtocol *process, lldb::StateType state) {
  assert(process && "process cannot be NULL");
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  if (log) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s called with "
              "NativeProcessProtocol pid %" PRIu64 ", state: %s",
              __FUNCTION__, process->GetID(), StateAsCString(state));
  }

  switch (state) {
  case StateType::eStateRunning:
    StartSTDIOForwarding();
    break;

  case StateType::eStateStopped:
    // Make sure we get all of the pending stdout/stderr from the inferior and
    // send it to the lldb host before we send the state change notification
    SendProcessOutput();
    // Then stop the forwarding, so that any late output (see llvm.org/pr25652)
    // does not interfere with our protocol.
    StopSTDIOForwarding();
    HandleInferiorState_Stopped(process);
    break;

  case StateType::eStateExited:
    // Same as above
    SendProcessOutput();
    StopSTDIOForwarding();
    HandleInferiorState_Exited(process);
    break;

  default:
    if (log) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s didn't handle state "
                "change for pid %" PRIu64 ", new state: %s",
                __FUNCTION__, process->GetID(), StateAsCString(state));
    }
    break;
  }

  // Remember the previous state reported to us.
  m_inferior_prev_state = state;
}

void GDBRemoteCommunicationServerLLGS::DidExec(NativeProcessProtocol *process) {
  ClearProcessSpecificData();
}

void GDBRemoteCommunicationServerLLGS::NewSubprocess(
    NativeProcessProtocol *parent_process,
    std::unique_ptr<NativeProcessProtocol> child_process) {
  lldb::pid_t child_pid = child_process->GetID();
  assert(child_pid != LLDB_INVALID_PROCESS_ID);
  assert(m_debugged_processes.find(child_pid) == m_debugged_processes.end());
  m_debugged_processes[child_pid] = std::move(child_process);
}

void GDBRemoteCommunicationServerLLGS::DataAvailableCallback() {
  Log *log(GetLogIfAnyCategoriesSet(GDBR_LOG_COMM));

  bool interrupt = false;
  bool done = false;
  Status error;
  while (true) {
    const PacketResult result = GetPacketAndSendResponse(
        std::chrono::microseconds(0), error, interrupt, done);
    if (result == PacketResult::ErrorReplyTimeout)
      break; // No more packets in the queue

    if ((result != PacketResult::Success)) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s processing a packet "
                "failed: %s",
                __FUNCTION__, error.AsCString());
      m_mainloop.RequestTermination();
      break;
    }
  }
}

Status GDBRemoteCommunicationServerLLGS::InitializeConnection(
    std::unique_ptr<Connection> connection) {
  IOObjectSP read_object_sp = connection->GetReadObject();
  GDBRemoteCommunicationServer::SetConnection(std::move(connection));

  Status error;
  m_network_handle_up = m_mainloop.RegisterReadObject(
      read_object_sp, [this](MainLoopBase &) { DataAvailableCallback(); },
      error);
  return error;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::SendONotification(const char *buffer,
                                                    uint32_t len) {
  if ((buffer == nullptr) || (len == 0)) {
    // Nothing to send.
    return PacketResult::Success;
  }

  StreamString response;
  response.PutChar('O');
  response.PutBytesAsRawHex8(buffer, len);

  return SendPacketNoLock(response.GetString());
}

Status GDBRemoteCommunicationServerLLGS::SetSTDIOFileDescriptor(int fd) {
  Status error;

  // Set up the reading/handling of process I/O
  std::unique_ptr<ConnectionFileDescriptor> conn_up(
      new ConnectionFileDescriptor(fd, true));
  if (!conn_up) {
    error.SetErrorString("failed to create ConnectionFileDescriptor");
    return error;
  }

  m_stdio_communication.SetCloseOnEOF(false);
  m_stdio_communication.SetConnection(std::move(conn_up));
  if (!m_stdio_communication.IsConnected()) {
    error.SetErrorString(
        "failed to set connection for inferior I/O communication");
    return error;
  }

  return Status();
}

void GDBRemoteCommunicationServerLLGS::StartSTDIOForwarding() {
  // Don't forward if not connected (e.g. when attaching).
  if (!m_stdio_communication.IsConnected())
    return;

  Status error;
  lldbassert(!m_stdio_handle_up);
  m_stdio_handle_up = m_mainloop.RegisterReadObject(
      m_stdio_communication.GetConnection()->GetReadObject(),
      [this](MainLoopBase &) { SendProcessOutput(); }, error);

  if (!m_stdio_handle_up) {
    // Not much we can do about the failure. Log it and continue without
    // forwarding.
    if (Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS))
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s Failed to set up stdio "
                "forwarding: %s",
                __FUNCTION__, error.AsCString());
  }
}

void GDBRemoteCommunicationServerLLGS::StopSTDIOForwarding() {
  m_stdio_handle_up.reset();
}

void GDBRemoteCommunicationServerLLGS::SendProcessOutput() {
  char buffer[1024];
  ConnectionStatus status;
  Status error;
  while (true) {
    size_t bytes_read = m_stdio_communication.Read(
        buffer, sizeof buffer, std::chrono::microseconds(0), status, &error);
    switch (status) {
    case eConnectionStatusSuccess:
      SendONotification(buffer, bytes_read);
      break;
    case eConnectionStatusLostConnection:
    case eConnectionStatusEndOfFile:
    case eConnectionStatusError:
    case eConnectionStatusNoConnection:
      if (Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS))
        LLDB_LOGF(log,
                  "GDBRemoteCommunicationServerLLGS::%s Stopping stdio "
                  "forwarding as communication returned status %d (error: "
                  "%s)",
                  __FUNCTION__, status, error.AsCString());
      m_stdio_handle_up.reset();
      return;

    case eConnectionStatusInterrupted:
    case eConnectionStatusTimedOut:
      return;
    }
  }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceSupported(
    StringExtractorGDBRemote &packet) {

  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(Status("Process not running."));

  return SendJSONResponse(m_current_process->TraceSupported());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceStop(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(Status("Process not running."));

  packet.ConsumeFront("jLLDBTraceStop:");
  Expected<TraceStopRequest> stop_request =
      json::parse<TraceStopRequest>(packet.Peek(), "TraceStopRequest");
  if (!stop_request)
    return SendErrorResponse(stop_request.takeError());

  if (Error err = m_current_process->TraceStop(*stop_request))
    return SendErrorResponse(std::move(err));

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceStart(
    StringExtractorGDBRemote &packet) {

  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(Status("Process not running."));

  packet.ConsumeFront("jLLDBTraceStart:");
  Expected<TraceStartRequest> request =
      json::parse<TraceStartRequest>(packet.Peek(), "TraceStartRequest");
  if (!request)
    return SendErrorResponse(request.takeError());

  if (Error err = m_current_process->TraceStart(packet.Peek(), request->type))
    return SendErrorResponse(std::move(err));

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceGetState(
    StringExtractorGDBRemote &packet) {

  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(Status("Process not running."));

  packet.ConsumeFront("jLLDBTraceGetState:");
  Expected<TraceGetStateRequest> request =
      json::parse<TraceGetStateRequest>(packet.Peek(), "TraceGetStateRequest");
  if (!request)
    return SendErrorResponse(request.takeError());

  return SendJSONResponse(m_current_process->TraceGetState(request->type));
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jLLDBTraceGetBinaryData(
    StringExtractorGDBRemote &packet) {

  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(Status("Process not running."));

  packet.ConsumeFront("jLLDBTraceGetBinaryData:");
  llvm::Expected<TraceGetBinaryDataRequest> request =
      llvm::json::parse<TraceGetBinaryDataRequest>(packet.Peek(),
                                                   "TraceGetBinaryDataRequest");
  if (!request)
    return SendErrorResponse(Status(request.takeError()));

  if (Expected<std::vector<uint8_t>> bytes =
          m_current_process->TraceGetBinaryData(*request)) {
    StreamGDBRemote response;
    response.PutEscapedBytes(bytes->data(), bytes->size());
    return SendPacketNoLock(response.GetString());
  } else
    return SendErrorResponse(bytes.takeError());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qProcessInfo(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  lldb::pid_t pid = m_current_process->GetID();

  if (pid == LLDB_INVALID_PROCESS_ID)
    return SendErrorResponse(1);

  ProcessInstanceInfo proc_info;
  if (!Host::GetProcessInfo(pid, proc_info))
    return SendErrorResponse(1);

  StreamString response;
  CreateProcessInfoResponse_DebugServerStyle(proc_info, response);
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qC(StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  // Make sure we set the current thread so g and p packets return the data the
  // gdb will expect.
  lldb::tid_t tid = m_current_process->GetCurrentThreadID();
  SetCurrentThreadID(tid);

  NativeThreadProtocol *thread = m_current_process->GetCurrentThread();
  if (!thread)
    return SendErrorResponse(69);

  StreamString response;
  response.Printf("QC%" PRIx64, thread->GetID());

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_k(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  StopSTDIOForwarding();

  if (!m_current_process) {
    LLDB_LOG(log, "No debugged process found.");
    return PacketResult::Success;
  }

  Status error = m_current_process->Kill();
  if (error.Fail())
    LLDB_LOG(log, "Failed to kill debugged process {0}: {1}",
             m_current_process->GetID(), error);

  // No OK response for kill packet.
  // return SendOKResponse ();
  return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QSetDisableASLR(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetDisableASLR:"));
  if (packet.GetU32(0))
    m_process_launch_info.GetFlags().Set(eLaunchFlagDisableASLR);
  else
    m_process_launch_info.GetFlags().Clear(eLaunchFlagDisableASLR);
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QSetWorkingDir(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetWorkingDir:"));
  std::string path;
  packet.GetHexByteString(path);
  m_process_launch_info.SetWorkingDirectory(FileSpec(path));
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qGetWorkingDir(
    StringExtractorGDBRemote &packet) {
  FileSpec working_dir{m_process_launch_info.GetWorkingDirectory()};
  if (working_dir) {
    StreamString response;
    response.PutStringAsRawHex8(working_dir.GetCString());
    return SendPacketNoLock(response.GetString());
  }

  return SendErrorResponse(14);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QThreadSuffixSupported(
    StringExtractorGDBRemote &packet) {
  m_thread_suffix_supported = true;
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QListThreadsInStopReply(
    StringExtractorGDBRemote &packet) {
  m_list_threads_in_stop_reply = true;
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_C(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s called", __FUNCTION__);

  // Ensure we have a native process.
  if (!m_continue_process) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s no debugged process "
              "shared pointer",
              __FUNCTION__);
    return SendErrorResponse(0x36);
  }

  // Pull out the signal number.
  packet.SetFilePos(::strlen("C"));
  if (packet.GetBytesLeft() < 1) {
    // Shouldn't be using a C without a signal.
    return SendIllFormedResponse(packet, "C packet specified without signal.");
  }
  const uint32_t signo =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (signo == std::numeric_limits<uint32_t>::max())
    return SendIllFormedResponse(packet, "failed to parse signal number");

  // Handle optional continue address.
  if (packet.GetBytesLeft() > 0) {
    // FIXME add continue at address support for $C{signo}[;{continue-address}].
    if (*packet.Peek() == ';')
      return SendUnimplementedResponse(packet.GetStringRef().data());
    else
      return SendIllFormedResponse(
          packet, "unexpected content after $C{signal-number}");
  }

  ResumeActionList resume_actions(StateType::eStateRunning,
                                  LLDB_INVALID_SIGNAL_NUMBER);
  Status error;

  // We have two branches: what to do if a continue thread is specified (in
  // which case we target sending the signal to that thread), or when we don't
  // have a continue thread set (in which case we send a signal to the
  // process).

  // TODO discuss with Greg Clayton, make sure this makes sense.

  lldb::tid_t signal_tid = GetContinueThreadID();
  if (signal_tid != LLDB_INVALID_THREAD_ID) {
    // The resume action for the continue thread (or all threads if a continue
    // thread is not set).
    ResumeAction action = {GetContinueThreadID(), StateType::eStateRunning,
                           static_cast<int>(signo)};

    // Add the action for the continue thread (or all threads when the continue
    // thread isn't present).
    resume_actions.Append(action);
  } else {
    // Send the signal to the process since we weren't targeting a specific
    // continue thread with the signal.
    error = m_continue_process->Signal(signo);
    if (error.Fail()) {
      LLDB_LOG(log, "failed to send signal for process {0}: {1}",
               m_continue_process->GetID(), error);

      return SendErrorResponse(0x52);
    }
  }

  // Resume the threads.
  error = m_continue_process->Resume(resume_actions);
  if (error.Fail()) {
    LLDB_LOG(log, "failed to resume threads for process {0}: {1}",
             m_continue_process->GetID(), error);

    return SendErrorResponse(0x38);
  }

  // Don't send an "OK" packet; response is the stopped/exited message.
  return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_c(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s called", __FUNCTION__);

  packet.SetFilePos(packet.GetFilePos() + ::strlen("c"));

  // For now just support all continue.
  const bool has_continue_address = (packet.GetBytesLeft() > 0);
  if (has_continue_address) {
    LLDB_LOG(log, "not implemented for c[address] variant [{0} remains]",
             packet.Peek());
    return SendUnimplementedResponse(packet.GetStringRef().data());
  }

  // Ensure we have a native process.
  if (!m_continue_process) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s no debugged process "
              "shared pointer",
              __FUNCTION__);
    return SendErrorResponse(0x36);
  }

  // Build the ResumeActionList
  ResumeActionList actions(StateType::eStateRunning,
                           LLDB_INVALID_SIGNAL_NUMBER);

  Status error = m_continue_process->Resume(actions);
  if (error.Fail()) {
    LLDB_LOG(log, "c failed for process {0}: {1}", m_continue_process->GetID(),
             error);
    return SendErrorResponse(GDBRemoteServerError::eErrorResume);
  }

  LLDB_LOG(log, "continued process {0}", m_continue_process->GetID());
  // No response required from continue.
  return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_vCont_actions(
    StringExtractorGDBRemote &packet) {
  StreamString response;
  response.Printf("vCont;c;C;s;S");

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_vCont(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
  LLDB_LOGF(log, "GDBRemoteCommunicationServerLLGS::%s handling vCont packet",
            __FUNCTION__);

  packet.SetFilePos(::strlen("vCont"));

  if (packet.GetBytesLeft() == 0) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s missing action from "
              "vCont package",
              __FUNCTION__);
    return SendIllFormedResponse(packet, "Missing action from vCont package");
  }

  // Check if this is all continue (no options or ";c").
  if (::strcmp(packet.Peek(), ";c") == 0) {
    // Move past the ';', then do a simple 'c'.
    packet.SetFilePos(packet.GetFilePos() + 1);
    return Handle_c(packet);
  } else if (::strcmp(packet.Peek(), ";s") == 0) {
    // Move past the ';', then do a simple 's'.
    packet.SetFilePos(packet.GetFilePos() + 1);
    return Handle_s(packet);
  }

  // Ensure we have a native process.
  if (!m_continue_process) {
    LLDB_LOG(log, "no debugged process");
    return SendErrorResponse(0x36);
  }

  ResumeActionList thread_actions;

  while (packet.GetBytesLeft() && *packet.Peek() == ';') {
    // Skip the semi-colon.
    packet.GetChar();

    // Build up the thread action.
    ResumeAction thread_action;
    thread_action.tid = LLDB_INVALID_THREAD_ID;
    thread_action.state = eStateInvalid;
    thread_action.signal = LLDB_INVALID_SIGNAL_NUMBER;

    const char action = packet.GetChar();
    switch (action) {
    case 'C':
      thread_action.signal = packet.GetHexMaxU32(false, 0);
      if (thread_action.signal == 0)
        return SendIllFormedResponse(
            packet, "Could not parse signal in vCont packet C action");
      LLVM_FALLTHROUGH;

    case 'c':
      // Continue
      thread_action.state = eStateRunning;
      break;

    case 'S':
      thread_action.signal = packet.GetHexMaxU32(false, 0);
      if (thread_action.signal == 0)
        return SendIllFormedResponse(
            packet, "Could not parse signal in vCont packet S action");
      LLVM_FALLTHROUGH;

    case 's':
      // Step
      thread_action.state = eStateStepping;
      break;

    default:
      return SendIllFormedResponse(packet, "Unsupported vCont action");
      break;
    }

    // Parse out optional :{thread-id} value.
    if (packet.GetBytesLeft() && (*packet.Peek() == ':')) {
      // Consume the separator.
      packet.GetChar();

      llvm::Expected<lldb::tid_t> tid_ret =
          ReadTid(packet, /*allow_all=*/true, m_continue_process->GetID());
      if (!tid_ret)
        return SendErrorResponse(tid_ret.takeError());

      thread_action.tid = tid_ret.get();
      if (thread_action.tid == StringExtractorGDBRemote::AllThreads)
        thread_action.tid = LLDB_INVALID_THREAD_ID;
    }

    thread_actions.Append(thread_action);
  }

  Status error = m_continue_process->Resume(thread_actions);
  if (error.Fail()) {
    LLDB_LOG(log, "vCont failed for process {0}: {1}",
             m_continue_process->GetID(), error);
    return SendErrorResponse(GDBRemoteServerError::eErrorResume);
  }

  LLDB_LOG(log, "continued process {0}", m_continue_process->GetID());
  // No response required from vCont.
  return PacketResult::Success;
}

void GDBRemoteCommunicationServerLLGS::SetCurrentThreadID(lldb::tid_t tid) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));
  LLDB_LOG(log, "setting current thread id to {0}", tid);

  m_current_tid = tid;
  if (m_current_process)
    m_current_process->SetCurrentThreadID(m_current_tid);
}

void GDBRemoteCommunicationServerLLGS::SetContinueThreadID(lldb::tid_t tid) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));
  LLDB_LOG(log, "setting continue thread id to {0}", tid);

  m_continue_tid = tid;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_stop_reason(
    StringExtractorGDBRemote &packet) {
  // Handle the $? gdbremote command.

  // If no process, indicate error
  if (!m_current_process)
    return SendErrorResponse(02);

  return SendStopReasonForState(m_current_process->GetState());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::SendStopReasonForState(
    lldb::StateType process_state) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  switch (process_state) {
  case eStateAttaching:
  case eStateLaunching:
  case eStateRunning:
  case eStateStepping:
  case eStateDetached:
    // NOTE: gdb protocol doc looks like it should return $OK
    // when everything is running (i.e. no stopped result).
    return PacketResult::Success; // Ignore

  case eStateSuspended:
  case eStateStopped:
  case eStateCrashed: {
    assert(m_current_process != nullptr);
    lldb::tid_t tid = m_current_process->GetCurrentThreadID();
    // Make sure we set the current thread so g and p packets return the data
    // the gdb will expect.
    SetCurrentThreadID(tid);
    return SendStopReplyPacketForThread(tid);
  }

  case eStateInvalid:
  case eStateUnloaded:
  case eStateExited:
    return SendWResponse(m_current_process);

  default:
    LLDB_LOG(log, "pid {0}, current state reporting not handled: {1}",
             m_current_process->GetID(), process_state);
    break;
  }

  return SendErrorResponse(0);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qRegisterInfo(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(68);

  // Ensure we have a thread.
  NativeThreadProtocol *thread = m_current_process->GetThreadAtIndex(0);
  if (!thread)
    return SendErrorResponse(69);

  // Get the register context for the first thread.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();

  // Parse out the register number from the request.
  packet.SetFilePos(strlen("qRegisterInfo"));
  const uint32_t reg_index =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (reg_index == std::numeric_limits<uint32_t>::max())
    return SendErrorResponse(69);

  // Return the end of registers response if we've iterated one past the end of
  // the register set.
  if (reg_index >= reg_context.GetUserRegisterCount())
    return SendErrorResponse(69);

  const RegisterInfo *reg_info = reg_context.GetRegisterInfoAtIndex(reg_index);
  if (!reg_info)
    return SendErrorResponse(69);

  // Build the reginfos response.
  StreamGDBRemote response;

  response.PutCString("name:");
  response.PutCString(reg_info->name);
  response.PutChar(';');

  if (reg_info->alt_name && reg_info->alt_name[0]) {
    response.PutCString("alt-name:");
    response.PutCString(reg_info->alt_name);
    response.PutChar(';');
  }

  response.Printf("bitsize:%" PRIu32 ";", reg_info->byte_size * 8);

  if (!reg_context.RegisterOffsetIsDynamic())
    response.Printf("offset:%" PRIu32 ";", reg_info->byte_offset);

  llvm::StringRef encoding = GetEncodingNameOrEmpty(*reg_info);
  if (!encoding.empty())
    response << "encoding:" << encoding << ';';

  llvm::StringRef format = GetFormatNameOrEmpty(*reg_info);
  if (!format.empty())
    response << "format:" << format << ';';

  const char *const register_set_name =
      reg_context.GetRegisterSetNameForRegisterAtIndex(reg_index);
  if (register_set_name)
    response << "set:" << register_set_name << ';';

  if (reg_info->kinds[RegisterKind::eRegisterKindEHFrame] !=
      LLDB_INVALID_REGNUM)
    response.Printf("ehframe:%" PRIu32 ";",
                    reg_info->kinds[RegisterKind::eRegisterKindEHFrame]);

  if (reg_info->kinds[RegisterKind::eRegisterKindDWARF] != LLDB_INVALID_REGNUM)
    response.Printf("dwarf:%" PRIu32 ";",
                    reg_info->kinds[RegisterKind::eRegisterKindDWARF]);

  llvm::StringRef kind_generic = GetKindGenericOrEmpty(*reg_info);
  if (!kind_generic.empty())
    response << "generic:" << kind_generic << ';';

  if (reg_info->value_regs && reg_info->value_regs[0] != LLDB_INVALID_REGNUM) {
    response.PutCString("container-regs:");
    CollectRegNums(reg_info->value_regs, response, true);
    response.PutChar(';');
  }

  if (reg_info->invalidate_regs && reg_info->invalidate_regs[0]) {
    response.PutCString("invalidate-regs:");
    CollectRegNums(reg_info->invalidate_regs, response, true);
    response.PutChar(';');
  }

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qfThreadInfo(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOG(log, "no process ({0}), returning OK",
             m_current_process ? "invalid process id"
                               : "null m_current_process");
    return SendOKResponse();
  }

  StreamGDBRemote response;
  response.PutChar('m');

  LLDB_LOG(log, "starting thread iteration");
  NativeThreadProtocol *thread;
  uint32_t thread_index;
  for (thread_index = 0,
      thread = m_current_process->GetThreadAtIndex(thread_index);
       thread; ++thread_index,
      thread = m_current_process->GetThreadAtIndex(thread_index)) {
    LLDB_LOG(log, "iterated thread {0}(tid={2})", thread_index,
             thread->GetID());
    if (thread_index > 0)
      response.PutChar(',');
    response.Printf("%" PRIx64, thread->GetID());
  }

  LLDB_LOG(log, "finished thread iteration");
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qsThreadInfo(
    StringExtractorGDBRemote &packet) {
  // FIXME for now we return the full thread list in the initial packet and
  // always do nothing here.
  return SendPacketNoLock("l");
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_g(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Move past packet name.
  packet.SetFilePos(strlen("g"));

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    LLDB_LOG(log, "failed, no thread available");
    return SendErrorResponse(0x15);
  }

  // Get the thread's register context.
  NativeRegisterContext &reg_ctx = thread->GetRegisterContext();

  std::vector<uint8_t> regs_buffer;
  for (uint32_t reg_num = 0; reg_num < reg_ctx.GetUserRegisterCount();
       ++reg_num) {
    const RegisterInfo *reg_info = reg_ctx.GetRegisterInfoAtIndex(reg_num);

    if (reg_info == nullptr) {
      LLDB_LOG(log, "failed to get register info for register index {0}",
               reg_num);
      return SendErrorResponse(0x15);
    }

    if (reg_info->value_regs != nullptr)
      continue; // skip registers that are contained in other registers

    RegisterValue reg_value;
    Status error = reg_ctx.ReadRegister(reg_info, reg_value);
    if (error.Fail()) {
      LLDB_LOG(log, "failed to read register at index {0}", reg_num);
      return SendErrorResponse(0x15);
    }

    if (reg_info->byte_offset + reg_info->byte_size >= regs_buffer.size())
      // Resize the buffer to guarantee it can store the register offsetted
      // data.
      regs_buffer.resize(reg_info->byte_offset + reg_info->byte_size);

    // Copy the register offsetted data to the buffer.
    memcpy(regs_buffer.data() + reg_info->byte_offset, reg_value.GetBytes(),
           reg_info->byte_size);
  }

  // Write the response.
  StreamGDBRemote response;
  response.PutBytesAsRawHex8(regs_buffer.data(), regs_buffer.size());

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_p(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Parse out the register number from the request.
  packet.SetFilePos(strlen("p"));
  const uint32_t reg_index =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (reg_index == std::numeric_limits<uint32_t>::max()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, could not "
              "parse register number from request \"%s\"",
              __FUNCTION__, packet.GetStringRef().data());
    return SendErrorResponse(0x15);
  }

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    LLDB_LOG(log, "failed, no thread available");
    return SendErrorResponse(0x15);
  }

  // Get the thread's register context.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();

  // Return the end of registers response if we've iterated one past the end of
  // the register set.
  if (reg_index >= reg_context.GetUserRegisterCount()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, requested "
              "register %" PRIu32 " beyond register count %" PRIu32,
              __FUNCTION__, reg_index, reg_context.GetUserRegisterCount());
    return SendErrorResponse(0x15);
  }

  const RegisterInfo *reg_info = reg_context.GetRegisterInfoAtIndex(reg_index);
  if (!reg_info) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, requested "
              "register %" PRIu32 " returned NULL",
              __FUNCTION__, reg_index);
    return SendErrorResponse(0x15);
  }

  // Build the reginfos response.
  StreamGDBRemote response;

  // Retrieve the value
  RegisterValue reg_value;
  Status error = reg_context.ReadRegister(reg_info, reg_value);
  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, read of "
              "requested register %" PRIu32 " (%s) failed: %s",
              __FUNCTION__, reg_index, reg_info->name, error.AsCString());
    return SendErrorResponse(0x15);
  }

  const uint8_t *const data =
      static_cast<const uint8_t *>(reg_value.GetBytes());
  if (!data) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed to get data "
              "bytes from requested register %" PRIu32,
              __FUNCTION__, reg_index);
    return SendErrorResponse(0x15);
  }

  // FIXME flip as needed to get data in big/little endian format for this host.
  for (uint32_t i = 0; i < reg_value.GetByteSize(); ++i)
    response.PutHex8(data[i]);

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_P(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Ensure there is more content.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Empty P packet");

  // Parse out the register number from the request.
  packet.SetFilePos(strlen("P"));
  const uint32_t reg_index =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (reg_index == std::numeric_limits<uint32_t>::max()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, could not "
              "parse register number from request \"%s\"",
              __FUNCTION__, packet.GetStringRef().data());
    return SendErrorResponse(0x29);
  }

  // Note debugserver would send an E30 here.
  if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != '='))
    return SendIllFormedResponse(
        packet, "P packet missing '=' char after register number");

  // Parse out the value.
  uint8_t reg_bytes[RegisterValue::kMaxRegisterByteSize];
  size_t reg_size = packet.GetHexBytesAvail(reg_bytes);

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, no thread "
              "available (thread index 0)",
              __FUNCTION__);
    return SendErrorResponse(0x28);
  }

  // Get the thread's register context.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();
  const RegisterInfo *reg_info = reg_context.GetRegisterInfoAtIndex(reg_index);
  if (!reg_info) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, requested "
              "register %" PRIu32 " returned NULL",
              __FUNCTION__, reg_index);
    return SendErrorResponse(0x48);
  }

  // Return the end of registers response if we've iterated one past the end of
  // the register set.
  if (reg_index >= reg_context.GetUserRegisterCount()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, requested "
              "register %" PRIu32 " beyond register count %" PRIu32,
              __FUNCTION__, reg_index, reg_context.GetUserRegisterCount());
    return SendErrorResponse(0x47);
  }

  if (reg_size != reg_info->byte_size)
    return SendIllFormedResponse(packet, "P packet register size is incorrect");

  // Build the reginfos response.
  StreamGDBRemote response;

  RegisterValue reg_value(makeArrayRef(reg_bytes, reg_size),
                          m_current_process->GetArchitecture().GetByteOrder());
  Status error = reg_context.WriteRegister(reg_info, reg_value);
  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, write of "
              "requested register %" PRIu32 " (%s) failed: %s",
              __FUNCTION__, reg_index, reg_info->name, error.AsCString());
    return SendErrorResponse(0x32);
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_H(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Parse out which variant of $H is requested.
  packet.SetFilePos(strlen("H"));
  if (packet.GetBytesLeft() < 1) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, H command "
              "missing {g,c} variant",
              __FUNCTION__);
    return SendIllFormedResponse(packet, "H command missing {g,c} variant");
  }

  const char h_variant = packet.GetChar();
  NativeProcessProtocol *default_process;
  switch (h_variant) {
  case 'g':
    default_process = m_current_process;
    break;

  case 'c':
    default_process = m_continue_process;
    break;

  default:
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, invalid $H variant %c",
        __FUNCTION__, h_variant);
    return SendIllFormedResponse(packet,
                                 "H variant unsupported, should be c or g");
  }

  // Parse out the thread number.
  auto pid_tid = packet.GetPidTid(default_process ? default_process->GetID()
                                                  : LLDB_INVALID_PROCESS_ID);
  if (!pid_tid)
    return SendErrorResponse(llvm::make_error<StringError>(
        inconvertibleErrorCode(), "Malformed thread-id"));

  lldb::pid_t pid = pid_tid->first;
  lldb::tid_t tid = pid_tid->second;

  if (pid == StringExtractorGDBRemote::AllProcesses)
    return SendUnimplementedResponse("Selecting all processes not supported");
  if (pid == LLDB_INVALID_PROCESS_ID)
    return SendErrorResponse(llvm::make_error<StringError>(
        inconvertibleErrorCode(), "No current process and no PID provided"));

  // Check the process ID and find respective process instance.
  auto new_process_it = m_debugged_processes.find(pid);
  if (new_process_it == m_debugged_processes.end())
    return SendErrorResponse(llvm::make_error<StringError>(
        inconvertibleErrorCode(),
        llvm::formatv("No process with PID {0} debugged", pid)));

  // Ensure we have the given thread when not specifying -1 (all threads) or 0
  // (any thread).
  if (tid != LLDB_INVALID_THREAD_ID && tid != 0) {
    NativeThreadProtocol *thread = new_process_it->second->GetThreadByID(tid);
    if (!thread) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerLLGS::%s failed, tid %" PRIu64
                " not found",
                __FUNCTION__, tid);
      return SendErrorResponse(0x15);
    }
  }

  // Now switch the given process and thread type.
  switch (h_variant) {
  case 'g':
    m_current_process = new_process_it->second.get();
    SetCurrentThreadID(tid);
    break;

  case 'c':
    m_continue_process = new_process_it->second.get();
    SetContinueThreadID(tid);
    break;

  default:
    assert(false && "unsupported $H variant - shouldn't get here");
    return SendIllFormedResponse(packet,
                                 "H variant unsupported, should be c or g");
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_I(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  packet.SetFilePos(::strlen("I"));
  uint8_t tmp[4096];
  for (;;) {
    size_t read = packet.GetHexBytesAvail(tmp);
    if (read == 0) {
      break;
    }
    // write directly to stdin *this might block if stdin buffer is full*
    // TODO: enqueue this block in circular buffer and send window size to
    // remote host
    ConnectionStatus status;
    Status error;
    m_stdio_communication.Write(tmp, read, status, &error);
    if (error.Fail()) {
      return SendErrorResponse(0x15);
    }
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_interrupt(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOG(log, "failed, no process available");
    return SendErrorResponse(0x15);
  }

  // Interrupt the process.
  Status error = m_current_process->Interrupt();
  if (error.Fail()) {
    LLDB_LOG(log, "failed for process {0}: {1}", m_current_process->GetID(),
             error);
    return SendErrorResponse(GDBRemoteServerError::eErrorResume);
  }

  LLDB_LOG(log, "stopped process {0}", m_current_process->GetID());

  // No response required from stop all.
  return PacketResult::Success;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_memory_read(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Parse out the memory address.
  packet.SetFilePos(strlen("m"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short m packet");

  // Read the address.  Punting on validation.
  // FIXME replace with Hex U64 read with no default value that fails on failed
  // read.
  const lldb::addr_t read_addr = packet.GetHexMaxU64(false, 0);

  // Validate comma.
  if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != ','))
    return SendIllFormedResponse(packet, "Comma sep missing in m packet");

  // Get # bytes to read.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Length missing in m packet");

  const uint64_t byte_count = packet.GetHexMaxU64(false, 0);
  if (byte_count == 0) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s nothing to read: "
              "zero-length packet",
              __FUNCTION__);
    return SendOKResponse();
  }

  // Allocate the response buffer.
  std::string buf(byte_count, '\0');
  if (buf.empty())
    return SendErrorResponse(0x78);

  // Retrieve the process memory.
  size_t bytes_read = 0;
  Status error = m_current_process->ReadMemoryWithoutTrap(
      read_addr, &buf[0], byte_count, bytes_read);
  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64
              " mem 0x%" PRIx64 ": failed to read. Error: %s",
              __FUNCTION__, m_current_process->GetID(), read_addr,
              error.AsCString());
    return SendErrorResponse(0x08);
  }

  if (bytes_read == 0) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64
              " mem 0x%" PRIx64 ": read 0 of %" PRIu64 " requested bytes",
              __FUNCTION__, m_current_process->GetID(), read_addr, byte_count);
    return SendErrorResponse(0x08);
  }

  StreamGDBRemote response;
  packet.SetFilePos(0);
  char kind = packet.GetChar('?');
  if (kind == 'x')
    response.PutEscapedBytes(buf.data(), byte_count);
  else {
    assert(kind == 'm');
    for (size_t i = 0; i < bytes_read; ++i)
      response.PutHex8(buf[i]);
  }

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle__M(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Parse out the memory address.
  packet.SetFilePos(strlen("_M"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short _M packet");

  const lldb::addr_t size = packet.GetHexMaxU64(false, LLDB_INVALID_ADDRESS);
  if (size == LLDB_INVALID_ADDRESS)
    return SendIllFormedResponse(packet, "Address not valid");
  if (packet.GetChar() != ',')
    return SendIllFormedResponse(packet, "Bad packet");
  Permissions perms = {};
  while (packet.GetBytesLeft() > 0) {
    switch (packet.GetChar()) {
    case 'r':
      perms |= ePermissionsReadable;
      break;
    case 'w':
      perms |= ePermissionsWritable;
      break;
    case 'x':
      perms |= ePermissionsExecutable;
      break;
    default:
      return SendIllFormedResponse(packet, "Bad permissions");
    }
  }

  llvm::Expected<addr_t> addr = m_current_process->AllocateMemory(size, perms);
  if (!addr)
    return SendErrorResponse(addr.takeError());

  StreamGDBRemote response;
  response.PutHex64(*addr);
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle__m(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Parse out the memory address.
  packet.SetFilePos(strlen("_m"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short m packet");

  const lldb::addr_t addr = packet.GetHexMaxU64(false, LLDB_INVALID_ADDRESS);
  if (addr == LLDB_INVALID_ADDRESS)
    return SendIllFormedResponse(packet, "Address not valid");

  if (llvm::Error Err = m_current_process->DeallocateMemory(addr))
    return SendErrorResponse(std::move(Err));

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_M(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Parse out the memory address.
  packet.SetFilePos(strlen("M"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short M packet");

  // Read the address.  Punting on validation.
  // FIXME replace with Hex U64 read with no default value that fails on failed
  // read.
  const lldb::addr_t write_addr = packet.GetHexMaxU64(false, 0);

  // Validate comma.
  if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != ','))
    return SendIllFormedResponse(packet, "Comma sep missing in M packet");

  // Get # bytes to read.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Length missing in M packet");

  const uint64_t byte_count = packet.GetHexMaxU64(false, 0);
  if (byte_count == 0) {
    LLDB_LOG(log, "nothing to write: zero-length packet");
    return PacketResult::Success;
  }

  // Validate colon.
  if ((packet.GetBytesLeft() < 1) || (packet.GetChar() != ':'))
    return SendIllFormedResponse(
        packet, "Comma sep missing in M packet after byte length");

  // Allocate the conversion buffer.
  std::vector<uint8_t> buf(byte_count, 0);
  if (buf.empty())
    return SendErrorResponse(0x78);

  // Convert the hex memory write contents to bytes.
  StreamGDBRemote response;
  const uint64_t convert_count = packet.GetHexBytes(buf, 0);
  if (convert_count != byte_count) {
    LLDB_LOG(log,
             "pid {0} mem {1:x}: asked to write {2} bytes, but only found {3} "
             "to convert.",
             m_current_process->GetID(), write_addr, byte_count, convert_count);
    return SendIllFormedResponse(packet, "M content byte length specified did "
                                         "not match hex-encoded content "
                                         "length");
  }

  // Write the process memory.
  size_t bytes_written = 0;
  Status error = m_current_process->WriteMemory(write_addr, &buf[0], byte_count,
                                                bytes_written);
  if (error.Fail()) {
    LLDB_LOG(log, "pid {0} mem {1:x}: failed to write. Error: {2}",
             m_current_process->GetID(), write_addr, error);
    return SendErrorResponse(0x09);
  }

  if (bytes_written == 0) {
    LLDB_LOG(log, "pid {0} mem {1:x}: wrote 0 of {2} requested bytes",
             m_current_process->GetID(), write_addr, byte_count);
    return SendErrorResponse(0x09);
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qMemoryRegionInfoSupported(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Currently only the NativeProcessProtocol knows if it can handle a
  // qMemoryRegionInfoSupported request, but we're not guaranteed to be
  // attached to a process.  For now we'll assume the client only asks this
  // when a process is being debugged.

  // Ensure we have a process running; otherwise, we can't figure this out
  // since we won't have a NativeProcessProtocol.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Test if we can get any region back when asking for the region around NULL.
  MemoryRegionInfo region_info;
  const Status error = m_current_process->GetMemoryRegionInfo(0, region_info);
  if (error.Fail()) {
    // We don't support memory region info collection for this
    // NativeProcessProtocol.
    return SendUnimplementedResponse("");
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qMemoryRegionInfo(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Ensure we have a process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x15);
  }

  // Parse out the memory address.
  packet.SetFilePos(strlen("qMemoryRegionInfo:"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short qMemoryRegionInfo: packet");

  // Read the address.  Punting on validation.
  const lldb::addr_t read_addr = packet.GetHexMaxU64(false, 0);

  StreamGDBRemote response;

  // Get the memory region info for the target address.
  MemoryRegionInfo region_info;
  const Status error =
      m_current_process->GetMemoryRegionInfo(read_addr, region_info);
  if (error.Fail()) {
    // Return the error message.

    response.PutCString("error:");
    response.PutStringAsRawHex8(error.AsCString());
    response.PutChar(';');
  } else {
    // Range start and size.
    response.Printf("start:%" PRIx64 ";size:%" PRIx64 ";",
                    region_info.GetRange().GetRangeBase(),
                    region_info.GetRange().GetByteSize());

    // Permissions.
    if (region_info.GetReadable() || region_info.GetWritable() ||
        region_info.GetExecutable()) {
      // Write permissions info.
      response.PutCString("permissions:");

      if (region_info.GetReadable())
        response.PutChar('r');
      if (region_info.GetWritable())
        response.PutChar('w');
      if (region_info.GetExecutable())
        response.PutChar('x');

      response.PutChar(';');
    }

    // Flags
    MemoryRegionInfo::OptionalBool memory_tagged =
        region_info.GetMemoryTagged();
    if (memory_tagged != MemoryRegionInfo::eDontKnow) {
      response.PutCString("flags:");
      if (memory_tagged == MemoryRegionInfo::eYes) {
        response.PutCString("mt");
      }
      response.PutChar(';');
    }

    // Name
    ConstString name = region_info.GetName();
    if (name) {
      response.PutCString("name:");
      response.PutStringAsRawHex8(name.GetStringRef());
      response.PutChar(';');
    }
  }

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_Z(StringExtractorGDBRemote &packet) {
  // Ensure we have a process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    LLDB_LOG(log, "failed, no process available");
    return SendErrorResponse(0x15);
  }

  // Parse out software or hardware breakpoint or watchpoint requested.
  packet.SetFilePos(strlen("Z"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(
        packet, "Too short Z packet, missing software/hardware specifier");

  bool want_breakpoint = true;
  bool want_hardware = false;
  uint32_t watch_flags = 0;

  const GDBStoppointType stoppoint_type =
      GDBStoppointType(packet.GetS32(eStoppointInvalid));
  switch (stoppoint_type) {
  case eBreakpointSoftware:
    want_hardware = false;
    want_breakpoint = true;
    break;
  case eBreakpointHardware:
    want_hardware = true;
    want_breakpoint = true;
    break;
  case eWatchpointWrite:
    watch_flags = 1;
    want_hardware = true;
    want_breakpoint = false;
    break;
  case eWatchpointRead:
    watch_flags = 2;
    want_hardware = true;
    want_breakpoint = false;
    break;
  case eWatchpointReadWrite:
    watch_flags = 3;
    want_hardware = true;
    want_breakpoint = false;
    break;
  case eStoppointInvalid:
    return SendIllFormedResponse(
        packet, "Z packet had invalid software/hardware specifier");
  }

  if ((packet.GetBytesLeft() < 1) || packet.GetChar() != ',')
    return SendIllFormedResponse(
        packet, "Malformed Z packet, expecting comma after stoppoint type");

  // Parse out the stoppoint address.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short Z packet, missing address");
  const lldb::addr_t addr = packet.GetHexMaxU64(false, 0);

  if ((packet.GetBytesLeft() < 1) || packet.GetChar() != ',')
    return SendIllFormedResponse(
        packet, "Malformed Z packet, expecting comma after address");

  // Parse out the stoppoint size (i.e. size hint for opcode size).
  const uint32_t size =
      packet.GetHexMaxU32(false, std::numeric_limits<uint32_t>::max());
  if (size == std::numeric_limits<uint32_t>::max())
    return SendIllFormedResponse(
        packet, "Malformed Z packet, failed to parse size argument");

  if (want_breakpoint) {
    // Try to set the breakpoint.
    const Status error =
        m_current_process->SetBreakpoint(addr, size, want_hardware);
    if (error.Success())
      return SendOKResponse();
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
    LLDB_LOG(log, "pid {0} failed to set breakpoint: {1}",
             m_current_process->GetID(), error);
    return SendErrorResponse(0x09);
  } else {
    // Try to set the watchpoint.
    const Status error = m_current_process->SetWatchpoint(
        addr, size, watch_flags, want_hardware);
    if (error.Success())
      return SendOKResponse();
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_WATCHPOINTS));
    LLDB_LOG(log, "pid {0} failed to set watchpoint: {1}",
             m_current_process->GetID(), error);
    return SendErrorResponse(0x09);
  }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_z(StringExtractorGDBRemote &packet) {
  // Ensure we have a process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));
    LLDB_LOG(log, "failed, no process available");
    return SendErrorResponse(0x15);
  }

  // Parse out software or hardware breakpoint or watchpoint requested.
  packet.SetFilePos(strlen("z"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(
        packet, "Too short z packet, missing software/hardware specifier");

  bool want_breakpoint = true;
  bool want_hardware = false;

  const GDBStoppointType stoppoint_type =
      GDBStoppointType(packet.GetS32(eStoppointInvalid));
  switch (stoppoint_type) {
  case eBreakpointHardware:
    want_breakpoint = true;
    want_hardware = true;
    break;
  case eBreakpointSoftware:
    want_breakpoint = true;
    break;
  case eWatchpointWrite:
    want_breakpoint = false;
    break;
  case eWatchpointRead:
    want_breakpoint = false;
    break;
  case eWatchpointReadWrite:
    want_breakpoint = false;
    break;
  default:
    return SendIllFormedResponse(
        packet, "z packet had invalid software/hardware specifier");
  }

  if ((packet.GetBytesLeft() < 1) || packet.GetChar() != ',')
    return SendIllFormedResponse(
        packet, "Malformed z packet, expecting comma after stoppoint type");

  // Parse out the stoppoint address.
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(packet, "Too short z packet, missing address");
  const lldb::addr_t addr = packet.GetHexMaxU64(false, 0);

  if ((packet.GetBytesLeft() < 1) || packet.GetChar() != ',')
    return SendIllFormedResponse(
        packet, "Malformed z packet, expecting comma after address");

  /*
  // Parse out the stoppoint size (i.e. size hint for opcode size).
  const uint32_t size = packet.GetHexMaxU32 (false,
  std::numeric_limits<uint32_t>::max ());
  if (size == std::numeric_limits<uint32_t>::max ())
      return SendIllFormedResponse(packet, "Malformed z packet, failed to parse
  size argument");
  */

  if (want_breakpoint) {
    // Try to clear the breakpoint.
    const Status error =
        m_current_process->RemoveBreakpoint(addr, want_hardware);
    if (error.Success())
      return SendOKResponse();
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_BREAKPOINTS));
    LLDB_LOG(log, "pid {0} failed to remove breakpoint: {1}",
             m_current_process->GetID(), error);
    return SendErrorResponse(0x09);
  } else {
    // Try to clear the watchpoint.
    const Status error = m_current_process->RemoveWatchpoint(addr);
    if (error.Success())
      return SendOKResponse();
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_WATCHPOINTS));
    LLDB_LOG(log, "pid {0} failed to remove watchpoint: {1}",
             m_current_process->GetID(), error);
    return SendErrorResponse(0x09);
  }
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_s(StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  // Ensure we have a process.
  if (!m_continue_process ||
      (m_continue_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(0x32);
  }

  // We first try to use a continue thread id.  If any one or any all set, use
  // the current thread. Bail out if we don't have a thread id.
  lldb::tid_t tid = GetContinueThreadID();
  if (tid == 0 || tid == LLDB_INVALID_THREAD_ID)
    tid = GetCurrentThreadID();
  if (tid == LLDB_INVALID_THREAD_ID)
    return SendErrorResponse(0x33);

  // Double check that we have such a thread.
  // TODO investigate: on MacOSX we might need to do an UpdateThreads () here.
  NativeThreadProtocol *thread = m_continue_process->GetThreadByID(tid);
  if (!thread)
    return SendErrorResponse(0x33);

  // Create the step action for the given thread.
  ResumeAction action = {tid, eStateStepping, LLDB_INVALID_SIGNAL_NUMBER};

  // Setup the actions list.
  ResumeActionList actions;
  actions.Append(action);

  // All other threads stop while we're single stepping a thread.
  actions.SetDefaultThreadActionIfNeeded(eStateStopped, 0);
  Status error = m_continue_process->Resume(actions);
  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s pid %" PRIu64
              " tid %" PRIu64 " Resume() failed with error: %s",
              __FUNCTION__, m_continue_process->GetID(), tid,
              error.AsCString());
    return SendErrorResponse(0x49);
  }

  // No response here - the stop or exit will come from the resulting action.
  return PacketResult::Success;
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
GDBRemoteCommunicationServerLLGS::BuildTargetXml() {
  // Ensure we have a thread.
  NativeThreadProtocol *thread = m_current_process->GetThreadAtIndex(0);
  if (!thread)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No thread available");

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));
  // Get the register context for the first thread.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();

  StreamString response;

  response.Printf("<?xml version=\"1.0\"?>");
  response.Printf("<target version=\"1.0\">");

  response.Printf("<architecture>%s</architecture>",
                  m_current_process->GetArchitecture()
                      .GetTriple()
                      .getArchName()
                      .str()
                      .c_str());

  response.Printf("<feature>");

  const int registers_count = reg_context.GetUserRegisterCount();
  for (int reg_index = 0; reg_index < registers_count; reg_index++) {
    const RegisterInfo *reg_info =
        reg_context.GetRegisterInfoAtIndex(reg_index);

    if (!reg_info) {
      LLDB_LOGF(log,
                "%s failed to get register info for register index %" PRIu32,
                "target.xml", reg_index);
      continue;
    }

    response.Printf("<reg name=\"%s\" bitsize=\"%" PRIu32 "\" regnum=\"%d\" ",
                    reg_info->name, reg_info->byte_size * 8, reg_index);

    if (!reg_context.RegisterOffsetIsDynamic())
      response.Printf("offset=\"%" PRIu32 "\" ", reg_info->byte_offset);

    if (reg_info->alt_name && reg_info->alt_name[0])
      response.Printf("altname=\"%s\" ", reg_info->alt_name);

    llvm::StringRef encoding = GetEncodingNameOrEmpty(*reg_info);
    if (!encoding.empty())
      response << "encoding=\"" << encoding << "\" ";

    llvm::StringRef format = GetFormatNameOrEmpty(*reg_info);
    if (!format.empty())
      response << "format=\"" << format << "\" ";

    const char *const register_set_name =
        reg_context.GetRegisterSetNameForRegisterAtIndex(reg_index);
    if (register_set_name)
      response << "group=\"" << register_set_name << "\" ";

    if (reg_info->kinds[RegisterKind::eRegisterKindEHFrame] !=
        LLDB_INVALID_REGNUM)
      response.Printf("ehframe_regnum=\"%" PRIu32 "\" ",
                      reg_info->kinds[RegisterKind::eRegisterKindEHFrame]);

    if (reg_info->kinds[RegisterKind::eRegisterKindDWARF] !=
        LLDB_INVALID_REGNUM)
      response.Printf("dwarf_regnum=\"%" PRIu32 "\" ",
                      reg_info->kinds[RegisterKind::eRegisterKindDWARF]);

    llvm::StringRef kind_generic = GetKindGenericOrEmpty(*reg_info);
    if (!kind_generic.empty())
      response << "generic=\"" << kind_generic << "\" ";

    if (reg_info->value_regs &&
        reg_info->value_regs[0] != LLDB_INVALID_REGNUM) {
      response.PutCString("value_regnums=\"");
      CollectRegNums(reg_info->value_regs, response, false);
      response.Printf("\" ");
    }

    if (reg_info->invalidate_regs && reg_info->invalidate_regs[0]) {
      response.PutCString("invalidate_regnums=\"");
      CollectRegNums(reg_info->invalidate_regs, response, false);
      response.Printf("\" ");
    }

    response.Printf("/>");
  }

  response.Printf("</feature>");
  response.Printf("</target>");
  return MemoryBuffer::getMemBufferCopy(response.GetString(), "target.xml");
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
GDBRemoteCommunicationServerLLGS::ReadXferObject(llvm::StringRef object,
                                                 llvm::StringRef annex) {
  // Make sure we have a valid process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No process available");
  }

  if (object == "auxv") {
    // Grab the auxv data.
    auto buffer_or_error = m_current_process->GetAuxvData();
    if (!buffer_or_error)
      return llvm::errorCodeToError(buffer_or_error.getError());
    return std::move(*buffer_or_error);
  }

  if (object == "libraries-svr4") {
    auto library_list = m_current_process->GetLoadedSVR4Libraries();
    if (!library_list)
      return library_list.takeError();

    StreamString response;
    response.Printf("<library-list-svr4 version=\"1.0\">");
    for (auto const &library : *library_list) {
      response.Printf("<library name=\"%s\" ",
                      XMLEncodeAttributeValue(library.name.c_str()).c_str());
      response.Printf("lm=\"0x%" PRIx64 "\" ", library.link_map);
      response.Printf("l_addr=\"0x%" PRIx64 "\" ", library.base_addr);
      response.Printf("l_ld=\"0x%" PRIx64 "\" />", library.ld_addr);
    }
    response.Printf("</library-list-svr4>");
    return MemoryBuffer::getMemBufferCopy(response.GetString(), __FUNCTION__);
  }

  if (object == "features" && annex == "target.xml")
    return BuildTargetXml();

  return llvm::make_error<UnimplementedError>();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qXfer(
    StringExtractorGDBRemote &packet) {
  SmallVector<StringRef, 5> fields;
  // The packet format is "qXfer:<object>:<action>:<annex>:offset,length"
  StringRef(packet.GetStringRef()).split(fields, ':', 4);
  if (fields.size() != 5)
    return SendIllFormedResponse(packet, "malformed qXfer packet");
  StringRef &xfer_object = fields[1];
  StringRef &xfer_action = fields[2];
  StringRef &xfer_annex = fields[3];
  StringExtractor offset_data(fields[4]);
  if (xfer_action != "read")
    return SendUnimplementedResponse("qXfer action not supported");
  // Parse offset.
  const uint64_t xfer_offset =
      offset_data.GetHexMaxU64(false, std::numeric_limits<uint64_t>::max());
  if (xfer_offset == std::numeric_limits<uint64_t>::max())
    return SendIllFormedResponse(packet, "qXfer packet missing offset");
  // Parse out comma.
  if (offset_data.GetChar() != ',')
    return SendIllFormedResponse(packet,
                                 "qXfer packet missing comma after offset");
  // Parse out the length.
  const uint64_t xfer_length =
      offset_data.GetHexMaxU64(false, std::numeric_limits<uint64_t>::max());
  if (xfer_length == std::numeric_limits<uint64_t>::max())
    return SendIllFormedResponse(packet, "qXfer packet missing length");

  // Get a previously constructed buffer if it exists or create it now.
  std::string buffer_key = (xfer_object + xfer_action + xfer_annex).str();
  auto buffer_it = m_xfer_buffer_map.find(buffer_key);
  if (buffer_it == m_xfer_buffer_map.end()) {
    auto buffer_up = ReadXferObject(xfer_object, xfer_annex);
    if (!buffer_up)
      return SendErrorResponse(buffer_up.takeError());
    buffer_it = m_xfer_buffer_map
                    .insert(std::make_pair(buffer_key, std::move(*buffer_up)))
                    .first;
  }

  // Send back the response
  StreamGDBRemote response;
  bool done_with_buffer = false;
  llvm::StringRef buffer = buffer_it->second->getBuffer();
  if (xfer_offset >= buffer.size()) {
    // We have nothing left to send.  Mark the buffer as complete.
    response.PutChar('l');
    done_with_buffer = true;
  } else {
    // Figure out how many bytes are available starting at the given offset.
    buffer = buffer.drop_front(xfer_offset);
    // Mark the response type according to whether we're reading the remainder
    // of the data.
    if (xfer_length >= buffer.size()) {
      // There will be nothing left to read after this
      response.PutChar('l');
      done_with_buffer = true;
    } else {
      // There will still be bytes to read after this request.
      response.PutChar('m');
      buffer = buffer.take_front(xfer_length);
    }
    // Now write the data in encoded binary form.
    response.PutEscapedBytes(buffer.data(), buffer.size());
  }

  if (done_with_buffer)
    m_xfer_buffer_map.erase(buffer_it);

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QSaveRegisterState(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Move past packet name.
  packet.SetFilePos(strlen("QSaveRegisterState"));

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    if (m_thread_suffix_supported)
      return SendIllFormedResponse(
          packet, "No thread specified in QSaveRegisterState packet");
    else
      return SendIllFormedResponse(packet,
                                   "No thread was is set with the Hg packet");
  }

  // Grab the register context for the thread.
  NativeRegisterContext& reg_context = thread->GetRegisterContext();

  // Save registers to a buffer.
  DataBufferSP register_data_sp;
  Status error = reg_context.ReadAllRegisterValues(register_data_sp);
  if (error.Fail()) {
    LLDB_LOG(log, "pid {0} failed to save all register values: {1}",
             m_current_process->GetID(), error);
    return SendErrorResponse(0x75);
  }

  // Allocate a new save id.
  const uint32_t save_id = GetNextSavedRegistersID();
  assert((m_saved_registers_map.find(save_id) == m_saved_registers_map.end()) &&
         "GetNextRegisterSaveID() returned an existing register save id");

  // Save the register data buffer under the save id.
  {
    std::lock_guard<std::mutex> guard(m_saved_registers_mutex);
    m_saved_registers_map[save_id] = register_data_sp;
  }

  // Write the response.
  StreamGDBRemote response;
  response.Printf("%" PRIu32, save_id);
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QRestoreRegisterState(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Parse out save id.
  packet.SetFilePos(strlen("QRestoreRegisterState:"));
  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(
        packet, "QRestoreRegisterState packet missing register save id");

  const uint32_t save_id = packet.GetU32(0);
  if (save_id == 0) {
    LLDB_LOG(log, "QRestoreRegisterState packet has malformed save id, "
                  "expecting decimal uint32_t");
    return SendErrorResponse(0x76);
  }

  // Get the thread to use.
  NativeThreadProtocol *thread = GetThreadFromSuffix(packet);
  if (!thread) {
    if (m_thread_suffix_supported)
      return SendIllFormedResponse(
          packet, "No thread specified in QRestoreRegisterState packet");
    else
      return SendIllFormedResponse(packet,
                                   "No thread was is set with the Hg packet");
  }

  // Grab the register context for the thread.
  NativeRegisterContext &reg_context = thread->GetRegisterContext();

  // Retrieve register state buffer, then remove from the list.
  DataBufferSP register_data_sp;
  {
    std::lock_guard<std::mutex> guard(m_saved_registers_mutex);

    // Find the register set buffer for the given save id.
    auto it = m_saved_registers_map.find(save_id);
    if (it == m_saved_registers_map.end()) {
      LLDB_LOG(log,
               "pid {0} does not have a register set save buffer for id {1}",
               m_current_process->GetID(), save_id);
      return SendErrorResponse(0x77);
    }
    register_data_sp = it->second;

    // Remove it from the map.
    m_saved_registers_map.erase(it);
  }

  Status error = reg_context.WriteAllRegisterValues(register_data_sp);
  if (error.Fail()) {
    LLDB_LOG(log, "pid {0} failed to restore all register values: {1}",
             m_current_process->GetID(), error);
    return SendErrorResponse(0x77);
  }

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_vAttach(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Consume the ';' after vAttach.
  packet.SetFilePos(strlen("vAttach"));
  if (!packet.GetBytesLeft() || packet.GetChar() != ';')
    return SendIllFormedResponse(packet, "vAttach missing expected ';'");

  // Grab the PID to which we will attach (assume hex encoding).
  lldb::pid_t pid = packet.GetU32(LLDB_INVALID_PROCESS_ID, 16);
  if (pid == LLDB_INVALID_PROCESS_ID)
    return SendIllFormedResponse(packet,
                                 "vAttach failed to parse the process id");

  // Attempt to attach.
  LLDB_LOGF(log,
            "GDBRemoteCommunicationServerLLGS::%s attempting to attach to "
            "pid %" PRIu64,
            __FUNCTION__, pid);

  Status error = AttachToProcess(pid);

  if (error.Fail()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed to attach to "
              "pid %" PRIu64 ": %s\n",
              __FUNCTION__, pid, error.AsCString());
    return SendErrorResponse(error);
  }

  // Notify we attached by sending a stop packet.
  return SendStopReasonForState(m_current_process->GetState());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_vAttachWait(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Consume the ';' after the identifier.
  packet.SetFilePos(strlen("vAttachWait"));

  if (!packet.GetBytesLeft() || packet.GetChar() != ';')
    return SendIllFormedResponse(packet, "vAttachWait missing expected ';'");

  // Allocate the buffer for the process name from vAttachWait.
  std::string process_name;
  if (!packet.GetHexByteString(process_name))
    return SendIllFormedResponse(packet,
                                 "vAttachWait failed to parse process name");

  LLDB_LOG(log, "attempting to attach to process named '{0}'", process_name);

  Status error = AttachWaitProcess(process_name, false);
  if (error.Fail()) {
    LLDB_LOG(log, "failed to attach to process named '{0}': {1}", process_name,
             error);
    return SendErrorResponse(error);
  }

  // Notify we attached by sending a stop packet.
  return SendStopReasonForState(m_current_process->GetState());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qVAttachOrWaitSupported(
    StringExtractorGDBRemote &packet) {
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_vAttachOrWait(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Consume the ';' after the identifier.
  packet.SetFilePos(strlen("vAttachOrWait"));

  if (!packet.GetBytesLeft() || packet.GetChar() != ';')
    return SendIllFormedResponse(packet, "vAttachOrWait missing expected ';'");

  // Allocate the buffer for the process name from vAttachWait.
  std::string process_name;
  if (!packet.GetHexByteString(process_name))
    return SendIllFormedResponse(packet,
                                 "vAttachOrWait failed to parse process name");

  LLDB_LOG(log, "attempting to attach to process named '{0}'", process_name);

  Status error = AttachWaitProcess(process_name, true);
  if (error.Fail()) {
    LLDB_LOG(log, "failed to attach to process named '{0}': {1}", process_name,
             error);
    return SendErrorResponse(error);
  }

  // Notify we attached by sending a stop packet.
  return SendStopReasonForState(m_current_process->GetState());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_vRun(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  llvm::StringRef s = packet.GetStringRef();
  if (!s.consume_front("vRun;"))
    return SendErrorResponse(8);

  llvm::SmallVector<llvm::StringRef, 16> argv;
  s.split(argv, ';');

  for (llvm::StringRef hex_arg : argv) {
    StringExtractor arg_ext{hex_arg};
    std::string arg;
    arg_ext.GetHexByteString(arg);
    m_process_launch_info.GetArguments().AppendArgument(arg);
    LLDB_LOGF(log, "LLGSPacketHandler::%s added arg: \"%s\"", __FUNCTION__,
              arg.c_str());
  }

  if (!argv.empty()) {
    m_process_launch_info.GetExecutableFile().SetFile(
        m_process_launch_info.GetArguments()[0].ref(), FileSpec::Style::native);
    m_process_launch_error = LaunchProcess();
    if (m_process_launch_error.Success())
      return SendStopReasonForState(m_current_process->GetState());
    LLDB_LOG(log, "failed to launch exe: {0}", m_process_launch_error);
  }
  return SendErrorResponse(8);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_D(StringExtractorGDBRemote &packet) {
  StopSTDIOForwarding();

  lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;

  // Consume the ';' after D.
  packet.SetFilePos(1);
  if (packet.GetBytesLeft()) {
    if (packet.GetChar() != ';')
      return SendIllFormedResponse(packet, "D missing expected ';'");

    // Grab the PID from which we will detach (assume hex encoding).
    pid = packet.GetU32(LLDB_INVALID_PROCESS_ID, 16);
    if (pid == LLDB_INVALID_PROCESS_ID)
      return SendIllFormedResponse(packet, "D failed to parse the process id");
  }

  // Detach forked children if their PID was specified *or* no PID was requested
  // (i.e. detach-all packet).
  llvm::Error detach_error = llvm::Error::success();
  bool detached = false;
  for (auto it = m_debugged_processes.begin();
       it != m_debugged_processes.end();) {
    if (pid == LLDB_INVALID_PROCESS_ID || pid == it->first) {
      if (llvm::Error e = it->second->Detach().ToError())
        detach_error = llvm::joinErrors(std::move(detach_error), std::move(e));
      else {
        if (it->second.get() == m_current_process)
          m_current_process = nullptr;
        if (it->second.get() == m_continue_process)
          m_continue_process = nullptr;
        it = m_debugged_processes.erase(it);
        detached = true;
        continue;
      }
    }
    ++it;
  }

  if (detach_error)
    return SendErrorResponse(std::move(detach_error));
  if (!detached)
    return SendErrorResponse(Status("PID %" PRIu64 " not traced", pid));
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qThreadStopInfo(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  packet.SetFilePos(strlen("qThreadStopInfo"));
  const lldb::tid_t tid = packet.GetHexMaxU64(false, LLDB_INVALID_THREAD_ID);
  if (tid == LLDB_INVALID_THREAD_ID) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s failed, could not "
              "parse thread id from request \"%s\"",
              __FUNCTION__, packet.GetStringRef().data());
    return SendErrorResponse(0x15);
  }
  return SendStopReplyPacketForThread(tid);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_jThreadsInfo(
    StringExtractorGDBRemote &) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_THREAD));

  // Ensure we have a debugged process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(50);
  LLDB_LOG(log, "preparing packet for pid {0}", m_current_process->GetID());

  StreamString response;
  const bool threads_with_valid_stop_info_only = false;
  llvm::Expected<json::Value> threads_info =
      GetJSONThreadsInfo(*m_current_process, threads_with_valid_stop_info_only);
  if (!threads_info) {
    LLDB_LOG_ERROR(log, threads_info.takeError(),
                   "failed to prepare a packet for pid {1}: {0}",
                   m_current_process->GetID());
    return SendErrorResponse(52);
  }

  response.AsRawOstream() << *threads_info;
  StreamGDBRemote escaped_response;
  escaped_response.PutEscapedBytes(response.GetData(), response.GetSize());
  return SendPacketNoLock(escaped_response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qWatchpointSupportInfo(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_current_process ||
      m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)
    return SendErrorResponse(68);

  packet.SetFilePos(strlen("qWatchpointSupportInfo"));
  if (packet.GetBytesLeft() == 0)
    return SendOKResponse();
  if (packet.GetChar() != ':')
    return SendErrorResponse(67);

  auto hw_debug_cap = m_current_process->GetHardwareDebugSupportInfo();

  StreamGDBRemote response;
  if (hw_debug_cap == llvm::None)
    response.Printf("num:0;");
  else
    response.Printf("num:%d;", hw_debug_cap->second);

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qFileLoadAddress(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_current_process ||
      m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)
    return SendErrorResponse(67);

  packet.SetFilePos(strlen("qFileLoadAddress:"));
  if (packet.GetBytesLeft() == 0)
    return SendErrorResponse(68);

  std::string file_name;
  packet.GetHexByteString(file_name);

  lldb::addr_t file_load_address = LLDB_INVALID_ADDRESS;
  Status error =
      m_current_process->GetFileLoadAddress(file_name, file_load_address);
  if (error.Fail())
    return SendErrorResponse(69);

  if (file_load_address == LLDB_INVALID_ADDRESS)
    return SendErrorResponse(1); // File not loaded

  StreamGDBRemote response;
  response.PutHex64(file_load_address);
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QPassSignals(
    StringExtractorGDBRemote &packet) {
  std::vector<int> signals;
  packet.SetFilePos(strlen("QPassSignals:"));

  // Read sequence of hex signal numbers divided by a semicolon and optionally
  // spaces.
  while (packet.GetBytesLeft() > 0) {
    int signal = packet.GetS32(-1, 16);
    if (signal < 0)
      return SendIllFormedResponse(packet, "Failed to parse signal number.");
    signals.push_back(signal);

    packet.SkipSpaces();
    char separator = packet.GetChar();
    if (separator == '\0')
      break; // End of string
    if (separator != ';')
      return SendIllFormedResponse(packet, "Invalid separator,"
                                            " expected semicolon.");
  }

  // Fail if we don't have a current process.
  if (!m_current_process)
    return SendErrorResponse(68);

  Status error = m_current_process->IgnoreSignals(signals);
  if (error.Fail())
    return SendErrorResponse(69);

  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qMemTags(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Ensure we have a process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(1);
  }

  // We are expecting
  // qMemTags:<hex address>,<hex length>:<hex type>

  // Address
  packet.SetFilePos(strlen("qMemTags:"));
  const char *current_char = packet.Peek();
  if (!current_char || *current_char == ',')
    return SendIllFormedResponse(packet, "Missing address in qMemTags packet");
  const lldb::addr_t addr = packet.GetHexMaxU64(/*little_endian=*/false, 0);

  // Length
  char previous_char = packet.GetChar();
  current_char = packet.Peek();
  // If we don't have a separator or the length field is empty
  if (previous_char != ',' || (current_char && *current_char == ':'))
    return SendIllFormedResponse(packet,
                                 "Invalid addr,length pair in qMemTags packet");

  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(
        packet, "Too short qMemtags: packet (looking for length)");
  const size_t length = packet.GetHexMaxU64(/*little_endian=*/false, 0);

  // Type
  const char *invalid_type_err = "Invalid type field in qMemTags: packet";
  if (packet.GetBytesLeft() < 1 || packet.GetChar() != ':')
    return SendIllFormedResponse(packet, invalid_type_err);

  // Type is a signed integer but packed into the packet as its raw bytes.
  // However, our GetU64 uses strtoull which allows +/-. We do not want this.
  const char *first_type_char = packet.Peek();
  if (first_type_char && (*first_type_char == '+' || *first_type_char == '-'))
    return SendIllFormedResponse(packet, invalid_type_err);

  // Extract type as unsigned then cast to signed.
  // Using a uint64_t here so that we have some value outside of the 32 bit
  // range to use as the invalid return value.
  uint64_t raw_type =
      packet.GetU64(std::numeric_limits<uint64_t>::max(), /*base=*/16);

  if ( // Make sure the cast below would be valid
      raw_type > std::numeric_limits<uint32_t>::max() ||
      // To catch inputs like "123aardvark" that will parse but clearly aren't
      // valid in this case.
      packet.GetBytesLeft()) {
    return SendIllFormedResponse(packet, invalid_type_err);
  }

  // First narrow to 32 bits otherwise the copy into type would take
  // the wrong 4 bytes on big endian.
  uint32_t raw_type_32 = raw_type;
  int32_t type = reinterpret_cast<int32_t &>(raw_type_32);

  StreamGDBRemote response;
  std::vector<uint8_t> tags;
  Status error = m_current_process->ReadMemoryTags(type, addr, length, tags);
  if (error.Fail())
    return SendErrorResponse(1);

  // This m is here in case we want to support multi part replies in the future.
  // In the same manner as qfThreadInfo/qsThreadInfo.
  response.PutChar('m');
  response.PutBytesAsRawHex8(tags.data(), tags.size());
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_QMemTags(
    StringExtractorGDBRemote &packet) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Ensure we have a process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)) {
    LLDB_LOGF(
        log,
        "GDBRemoteCommunicationServerLLGS::%s failed, no process available",
        __FUNCTION__);
    return SendErrorResponse(1);
  }

  // We are expecting
  // QMemTags:<hex address>,<hex length>:<hex type>:<tags as hex bytes>

  // Address
  packet.SetFilePos(strlen("QMemTags:"));
  const char *current_char = packet.Peek();
  if (!current_char || *current_char == ',')
    return SendIllFormedResponse(packet, "Missing address in QMemTags packet");
  const lldb::addr_t addr = packet.GetHexMaxU64(/*little_endian=*/false, 0);

  // Length
  char previous_char = packet.GetChar();
  current_char = packet.Peek();
  // If we don't have a separator or the length field is empty
  if (previous_char != ',' || (current_char && *current_char == ':'))
    return SendIllFormedResponse(packet,
                                 "Invalid addr,length pair in QMemTags packet");

  if (packet.GetBytesLeft() < 1)
    return SendIllFormedResponse(
        packet, "Too short QMemtags: packet (looking for length)");
  const size_t length = packet.GetHexMaxU64(/*little_endian=*/false, 0);

  // Type
  const char *invalid_type_err = "Invalid type field in QMemTags: packet";
  if (packet.GetBytesLeft() < 1 || packet.GetChar() != ':')
    return SendIllFormedResponse(packet, invalid_type_err);

  // Our GetU64 uses strtoull which allows leading +/-, we don't want that.
  const char *first_type_char = packet.Peek();
  if (first_type_char && (*first_type_char == '+' || *first_type_char == '-'))
    return SendIllFormedResponse(packet, invalid_type_err);

  // The type is a signed integer but is in the packet as its raw bytes.
  // So parse first as unsigned then cast to signed later.
  // We extract to 64 bit, even though we only expect 32, so that we've
  // got some invalid value we can check for.
  uint64_t raw_type =
      packet.GetU64(std::numeric_limits<uint64_t>::max(), /*base=*/16);
  if (raw_type > std::numeric_limits<uint32_t>::max())
    return SendIllFormedResponse(packet, invalid_type_err);

  // First narrow to 32 bits. Otherwise the copy below would get the wrong
  // 4 bytes on big endian.
  uint32_t raw_type_32 = raw_type;
  int32_t type = reinterpret_cast<int32_t &>(raw_type_32);

  // Tag data
  if (packet.GetBytesLeft() < 1 || packet.GetChar() != ':')
    return SendIllFormedResponse(packet,
                                 "Missing tag data in QMemTags: packet");

  // Must be 2 chars per byte
  const char *invalid_data_err = "Invalid tag data in QMemTags: packet";
  if (packet.GetBytesLeft() % 2)
    return SendIllFormedResponse(packet, invalid_data_err);

  // This is bytes here and is unpacked into target specific tags later
  // We cannot assume that number of bytes == length here because the server
  // can repeat tags to fill a given range.
  std::vector<uint8_t> tag_data;
  // Zero length writes will not have any tag data
  // (but we pass them on because it will still check that tagging is enabled)
  if (packet.GetBytesLeft()) {
    size_t byte_count = packet.GetBytesLeft() / 2;
    tag_data.resize(byte_count);
    size_t converted_bytes = packet.GetHexBytes(tag_data, 0);
    if (converted_bytes != byte_count) {
      return SendIllFormedResponse(packet, invalid_data_err);
    }
  }

  Status status =
      m_current_process->WriteMemoryTags(type, addr, length, tag_data);
  return status.Success() ? SendOKResponse() : SendErrorResponse(1);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerLLGS::Handle_qSaveCore(
    StringExtractorGDBRemote &packet) {
  // Fail if we don't have a current process.
  if (!m_current_process ||
      (m_current_process->GetID() == LLDB_INVALID_PROCESS_ID))
    return SendErrorResponse(Status("Process not running."));

  std::string path_hint;

  StringRef packet_str{packet.GetStringRef()};
  assert(packet_str.startswith("qSaveCore"));
  if (packet_str.consume_front("qSaveCore;")) {
    for (auto x : llvm::split(packet_str, ';')) {
      if (x.consume_front("path-hint:"))
        StringExtractor(x).GetHexByteString(path_hint);
      else
        return SendErrorResponse(Status("Unsupported qSaveCore option"));
    }
  }

  llvm::Expected<std::string> ret = m_current_process->SaveCore(path_hint);
  if (!ret)
    return SendErrorResponse(ret.takeError());

  StreamString response;
  response.PutCString("core-path:");
  response.PutStringAsRawHex8(ret.get());
  return SendPacketNoLock(response.GetString());
}

void GDBRemoteCommunicationServerLLGS::MaybeCloseInferiorTerminalConnection() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Tell the stdio connection to shut down.
  if (m_stdio_communication.IsConnected()) {
    auto connection = m_stdio_communication.GetConnection();
    if (connection) {
      Status error;
      connection->Disconnect(&error);

      if (error.Success()) {
        LLDB_LOGF(log,
                  "GDBRemoteCommunicationServerLLGS::%s disconnect process "
                  "terminal stdio - SUCCESS",
                  __FUNCTION__);
      } else {
        LLDB_LOGF(log,
                  "GDBRemoteCommunicationServerLLGS::%s disconnect process "
                  "terminal stdio - FAIL: %s",
                  __FUNCTION__, error.AsCString());
      }
    }
  }
}

NativeThreadProtocol *GDBRemoteCommunicationServerLLGS::GetThreadFromSuffix(
    StringExtractorGDBRemote &packet) {
  // We have no thread if we don't have a process.
  if (!m_current_process ||
      m_current_process->GetID() == LLDB_INVALID_PROCESS_ID)
    return nullptr;

  // If the client hasn't asked for thread suffix support, there will not be a
  // thread suffix. Use the current thread in that case.
  if (!m_thread_suffix_supported) {
    const lldb::tid_t current_tid = GetCurrentThreadID();
    if (current_tid == LLDB_INVALID_THREAD_ID)
      return nullptr;
    else if (current_tid == 0) {
      // Pick a thread.
      return m_current_process->GetThreadAtIndex(0);
    } else
      return m_current_process->GetThreadByID(current_tid);
  }

  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_THREAD));

  // Parse out the ';'.
  if (packet.GetBytesLeft() < 1 || packet.GetChar() != ';') {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s gdb-remote parse "
              "error: expected ';' prior to start of thread suffix: packet "
              "contents = '%s'",
              __FUNCTION__, packet.GetStringRef().data());
    return nullptr;
  }

  if (!packet.GetBytesLeft())
    return nullptr;

  // Parse out thread: portion.
  if (strncmp(packet.Peek(), "thread:", strlen("thread:")) != 0) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerLLGS::%s gdb-remote parse "
              "error: expected 'thread:' but not found, packet contents = "
              "'%s'",
              __FUNCTION__, packet.GetStringRef().data());
    return nullptr;
  }
  packet.SetFilePos(packet.GetFilePos() + strlen("thread:"));
  const lldb::tid_t tid = packet.GetHexMaxU64(false, 0);
  if (tid != 0)
    return m_current_process->GetThreadByID(tid);

  return nullptr;
}

lldb::tid_t GDBRemoteCommunicationServerLLGS::GetCurrentThreadID() const {
  if (m_current_tid == 0 || m_current_tid == LLDB_INVALID_THREAD_ID) {
    // Use whatever the debug process says is the current thread id since the
    // protocol either didn't specify or specified we want any/all threads
    // marked as the current thread.
    if (!m_current_process)
      return LLDB_INVALID_THREAD_ID;
    return m_current_process->GetCurrentThreadID();
  }
  // Use the specific current thread id set by the gdb remote protocol.
  return m_current_tid;
}

uint32_t GDBRemoteCommunicationServerLLGS::GetNextSavedRegistersID() {
  std::lock_guard<std::mutex> guard(m_saved_registers_mutex);
  return m_next_saved_registers_id++;
}

void GDBRemoteCommunicationServerLLGS::ClearProcessSpecificData() {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  LLDB_LOG(log, "clearing {0} xfer buffers", m_xfer_buffer_map.size());
  m_xfer_buffer_map.clear();
}

FileSpec
GDBRemoteCommunicationServerLLGS::FindModuleFile(const std::string &module_path,
                                                 const ArchSpec &arch) {
  if (m_current_process) {
    FileSpec file_spec;
    if (m_current_process
            ->GetLoadedModuleFileSpec(module_path.c_str(), file_spec)
            .Success()) {
      if (FileSystem::Instance().Exists(file_spec))
        return file_spec;
    }
  }

  return GDBRemoteCommunicationServerCommon::FindModuleFile(module_path, arch);
}

std::string GDBRemoteCommunicationServerLLGS::XMLEncodeAttributeValue(
    llvm::StringRef value) {
  std::string result;
  for (const char &c : value) {
    switch (c) {
    case '\'':
      result += "&apos;";
      break;
    case '"':
      result += "&quot;";
      break;
    case '<':
      result += "&lt;";
      break;
    case '>':
      result += "&gt;";
      break;
    default:
      result += c;
      break;
    }
  }
  return result;
}

llvm::Expected<lldb::tid_t> GDBRemoteCommunicationServerLLGS::ReadTid(
    StringExtractorGDBRemote &packet, bool allow_all, lldb::pid_t default_pid) {
  assert(m_current_process);
  assert(m_current_process->GetID() != LLDB_INVALID_PROCESS_ID);

  auto pid_tid = packet.GetPidTid(default_pid);
  if (!pid_tid)
    return llvm::make_error<StringError>(inconvertibleErrorCode(),
                                         "Malformed thread-id");

  lldb::pid_t pid = pid_tid->first;
  lldb::tid_t tid = pid_tid->second;

  if (!allow_all && pid == StringExtractorGDBRemote::AllProcesses)
    return llvm::make_error<StringError>(
        inconvertibleErrorCode(),
        llvm::formatv("PID value {0} not allowed", pid == 0 ? 0 : -1));

  if (!allow_all && tid == StringExtractorGDBRemote::AllThreads)
    return llvm::make_error<StringError>(
        inconvertibleErrorCode(),
        llvm::formatv("TID value {0} not allowed", tid == 0 ? 0 : -1));

  if (pid != StringExtractorGDBRemote::AllProcesses) {
    if (pid != m_current_process->GetID())
      return llvm::make_error<StringError>(
          inconvertibleErrorCode(), llvm::formatv("PID {0} not debugged", pid));
  }

  return tid;
}

std::vector<std::string> GDBRemoteCommunicationServerLLGS::HandleFeatures(
    const llvm::ArrayRef<llvm::StringRef> client_features) {
  std::vector<std::string> ret =
      GDBRemoteCommunicationServerCommon::HandleFeatures(client_features);
  ret.insert(ret.end(), {
                            "QThreadSuffixSupported+",
                            "QListThreadsInStopReply+",
                            "qXfer:features:read+",
                        });

  // report server-only features
  using Extension = NativeProcessProtocol::Extension;
  Extension plugin_features = m_process_factory.GetSupportedExtensions();
  if (bool(plugin_features & Extension::pass_signals))
    ret.push_back("QPassSignals+");
  if (bool(plugin_features & Extension::auxv))
    ret.push_back("qXfer:auxv:read+");
  if (bool(plugin_features & Extension::libraries_svr4))
    ret.push_back("qXfer:libraries-svr4:read+");
  if (bool(plugin_features & Extension::memory_tagging))
    ret.push_back("memory-tagging+");
  if (bool(plugin_features & Extension::savecore))
    ret.push_back("qSaveCore+");

  // check for client features
  m_extensions_supported = {};
  for (llvm::StringRef x : client_features)
    m_extensions_supported |=
        llvm::StringSwitch<Extension>(x)
            .Case("multiprocess+", Extension::multiprocess)
            .Case("fork-events+", Extension::fork)
            .Case("vfork-events+", Extension::vfork)
            .Default({});

  m_extensions_supported &= plugin_features;

  // fork & vfork require multiprocess
  if (!bool(m_extensions_supported & Extension::multiprocess))
    m_extensions_supported &= ~(Extension::fork | Extension::vfork);

  // report only if actually supported
  if (bool(m_extensions_supported & Extension::multiprocess))
    ret.push_back("multiprocess+");
  if (bool(m_extensions_supported & Extension::fork))
    ret.push_back("fork-events+");
  if (bool(m_extensions_supported & Extension::vfork))
    ret.push_back("vfork-events+");

  for (auto &x : m_debugged_processes)
    SetEnabledExtensions(*x.second);
  return ret;
}

void GDBRemoteCommunicationServerLLGS::SetEnabledExtensions(
    NativeProcessProtocol &process) {
  NativeProcessProtocol::Extension flags = m_extensions_supported;
  assert(!bool(flags & ~m_process_factory.GetSupportedExtensions()));
  process.SetEnabledExtensions(flags);
}

std::string
lldb_private::process_gdb_remote::LLGSArgToURL(llvm::StringRef url_arg,
                                               bool reverse_connect) {
  // Try parsing the argument as URL.
  if (llvm::Optional<URI> url = URI::Parse(url_arg)) {
    if (reverse_connect)
      return url_arg.str();

    // Translate the scheme from LLGS notation to ConnectionFileDescriptor.
    // If the scheme doesn't match any, pass it through to support using CFD
    // schemes directly.
    std::string new_url = llvm::StringSwitch<std::string>(url->scheme)
                              .Case("tcp", "listen")
                              .Case("unix", "unix-accept")
                              .Case("unix-abstract", "unix-abstract-accept")
                              .Default(url->scheme.str());
    llvm::append_range(new_url, url_arg.substr(url->scheme.size()));
    return new_url;
  }

  std::string host_port = url_arg.str();
  // If host_and_port starts with ':', default the host to be "localhost" and
  // expect the remainder to be the port.
  if (url_arg.startswith(":"))
    host_port.insert(0, "localhost");

  // Try parsing the (preprocessed) argument as host:port pair.
  if (!llvm::errorToBool(Socket::DecodeHostAndPort(host_port).takeError()))
    return (reverse_connect ? "connect://" : "listen://") + host_port;

  // If none of the above applied, interpret the argument as UNIX socket path.
  return (reverse_connect ? "unix-connect://" : "unix-accept://") +
         url_arg.str();
}

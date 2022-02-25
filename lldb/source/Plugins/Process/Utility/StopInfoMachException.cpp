//===-- StopInfoMachException.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StopInfoMachException.h"

#include "lldb/lldb-forward.h"

#if defined(__APPLE__)
// Needed for the EXC_RESOURCE interpretation macros
#include <kern/exc_resource.h>
#endif

#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

/// Information about a pointer-authentication related instruction.
struct PtrauthInstructionInfo {
  bool IsAuthenticated;
  bool IsLoad;
  bool DoesBranch;
};

/// Get any pointer-authentication related information about the instruction
/// at address \p at_addr.
static llvm::Optional<PtrauthInstructionInfo>
GetPtrauthInstructionInfo(Target &target, const ArchSpec &arch,
                          const Address &at_addr) {
  const char *plugin_name = nullptr;
  const char *flavor = nullptr;
  AddressRange range_bounds(at_addr, 4);
  const bool prefer_file_cache = true;
  DisassemblerSP disassembler_sp = Disassembler::DisassembleRange(
      arch, plugin_name, flavor, target, range_bounds, prefer_file_cache);
  if (!disassembler_sp)
    return llvm::None;

  InstructionList &insn_list = disassembler_sp->GetInstructionList();
  InstructionSP insn = insn_list.GetInstructionAtIndex(0);
  if (!insn)
    return llvm::None;

  return PtrauthInstructionInfo{insn->IsAuthenticated(), insn->IsLoad(),
                                insn->DoesBranch()};
}

/// Describe the load address of \p addr using the format filename:line:col.
static void DescribeAddressBriefly(Stream &strm, const Address &addr,
                                   Target &target) {
  strm.Printf("at address=0x%" PRIx64, addr.GetLoadAddress(&target));
  StreamString s;
  if (addr.GetDescription(s, target, eDescriptionLevelBrief))
    strm.Printf(" %s", s.GetString().data());
  strm.Printf(".\n");
}

bool StopInfoMachException::DeterminePtrauthFailure(ExecutionContext &exe_ctx) {
  bool IsBreakpoint = m_value == 6; // EXC_BREAKPOINT
  bool IsBadAccess = m_value == 1;  // EXC_BAD_ACCESS
  if (!IsBreakpoint && !IsBadAccess)
    return false;

  // Check that we have a live process.
  if (!exe_ctx.HasProcessScope() || !exe_ctx.HasThreadScope() ||
      !exe_ctx.HasTargetScope())
    return false;

  Thread &thread = *exe_ctx.GetThreadPtr();
  StackFrameSP current_frame = thread.GetStackFrameAtIndex(0);
  if (!current_frame)
    return false;

  Target &target = *exe_ctx.GetTargetPtr();
  Process &process = *exe_ctx.GetProcessPtr();
  ABISP abi_sp = process.GetABI();
  const ArchSpec &arch = target.GetArchitecture();
  assert(abi_sp && "Missing ABI info");

  // Check for a ptrauth-enabled target.
  const bool ptrauth_enabled_target =
      arch.GetCore() == ArchSpec::eCore_arm_arm64e;
  if (!ptrauth_enabled_target)
    return false;

  // Set up a stream we can write a diagnostic into.
  StreamString strm;
  auto emit_ptrauth_prologue = [&](uint64_t at_address) {
    strm.Printf("EXC_BAD_ACCESS (code=%" PRIu64 ", address=0x%" PRIx64 ")\n",
                m_exc_code, at_address);
    strm.Printf("Note: Possible pointer authentication failure detected.\n");
  };

  // Check if we have a "brk 0xc47x" trap, where the value that failed to
  // authenticate is in x16.
  Address current_address = current_frame->GetFrameCodeAddress();
  if (IsBreakpoint) {
    RegisterContext *reg_ctx = exe_ctx.GetRegisterContext();
    if (!reg_ctx)
      return false;

    const RegisterInfo *X16Info = reg_ctx->GetRegisterInfoByName("x16");
    RegisterValue X16Val;
    if (!reg_ctx->ReadRegister(X16Info, X16Val))
      return false;
    uint64_t bad_address = X16Val.GetAsUInt64();

    uint64_t fixed_bad_address = abi_sp->FixCodeAddress(bad_address);
    Address brk_address;
    if (!target.ResolveLoadAddress(fixed_bad_address, brk_address))
      return false;

    auto brk_ptrauth_info =
        GetPtrauthInstructionInfo(target, arch, current_address);
    if (brk_ptrauth_info && brk_ptrauth_info->IsAuthenticated) {
      emit_ptrauth_prologue(bad_address);
      strm.Printf("Found value that failed to authenticate ");
      DescribeAddressBriefly(strm, brk_address, target);
      m_description = std::string(strm.GetString());
      return true;
    }
    return false;
  }

  assert(IsBadAccess && "Handle EXC_BAD_ACCESS only after this point");

  // Check that we have the "bad address" from an EXC_BAD_ACCESS.
  if (m_exc_data_count < 2)
    return false;

  // Ok, we know the Target is valid and that it describes a ptrauth-enabled
  // device. Now, we need to determine whether this exception was caused by a
  // ptrauth failure.

  uint64_t bad_address = m_exc_subcode;
  uint64_t fixed_bad_address = abi_sp->FixCodeAddress(bad_address);
  uint64_t current_pc = current_address.GetLoadAddress(&target);

  // Detect: LDRAA, LDRAB (Load Register, with pointer authentication).
  //
  // If an authenticated load results in an exception, the instruction at the
  // current PC should be one of LDRAx.
  if (bad_address != current_pc && fixed_bad_address != current_pc) {
    auto ptrauth_info =
        GetPtrauthInstructionInfo(target, arch, current_address);
    if (ptrauth_info && ptrauth_info->IsAuthenticated && ptrauth_info->IsLoad) {
      emit_ptrauth_prologue(bad_address);
      strm.Printf("Found authenticated load instruction ");
      DescribeAddressBriefly(strm, current_address, target);
      m_description = std::string(strm.GetString());
      return true;
    }
  }

  // Detect: BLRAA, BLRAAZ, BLRAB, BLRABZ (Branch with Link to Register, with
  // pointer authentication).
  //
  // TODO: Detect: BRAA, BRAAZ, BRAB, BRABZ (Branch to Register, with pointer
  // authentication). At a minimum, this requires call site info support for
  // indirect calls.
  //
  // If an authenticated call or tail call results in an exception, stripping
  // the bad address should give the current PC, which points to the address
  // we tried to branch to.
  if (bad_address != current_pc && fixed_bad_address == current_pc) {
    if (StackFrameSP parent_frame = thread.GetStackFrameAtIndex(1)) {
      addr_t return_pc =
          parent_frame->GetFrameCodeAddress().GetLoadAddress(&target);
      Address blr_address;
      if (!target.ResolveLoadAddress(return_pc - 4, blr_address))
        return false;

      auto blr_ptrauth_info =
          GetPtrauthInstructionInfo(target, arch, blr_address);
      if (blr_ptrauth_info && blr_ptrauth_info->IsAuthenticated &&
          blr_ptrauth_info->DoesBranch) {
        emit_ptrauth_prologue(bad_address);
        strm.Printf("Found authenticated indirect branch ");
        DescribeAddressBriefly(strm, blr_address, target);
        m_description = std::string(strm.GetString());
        return true;
      }
    }
  }

  // TODO: Detect: RETAA, RETAB (Return from subroutine, with pointer
  // authentication).
  //
  // Is there a motivating, non-malicious code snippet that corrupts LR?

  return false;
}

const char *StopInfoMachException::GetDescription() {
  if (!m_description.empty())
    return m_description.c_str();
  if (GetValue() == eStopReasonInvalid)
    return "invalid stop reason!";

  ExecutionContext exe_ctx(m_thread_wp.lock());
  Target *target = exe_ctx.GetTargetPtr();
  const llvm::Triple::ArchType cpu =
      target ? target->GetArchitecture().GetMachine()
             : llvm::Triple::UnknownArch;

  const char *exc_desc = nullptr;
  const char *code_label = "code";
  const char *code_desc = nullptr;
  const char *subcode_label = "subcode";
  const char *subcode_desc = nullptr;

#if defined(__APPLE__)
  char code_desc_buf[32];
  char subcode_desc_buf[32];
#endif

  switch (m_value) {
  case 1: // EXC_BAD_ACCESS
    exc_desc = "EXC_BAD_ACCESS";
    subcode_label = "address";
    switch (cpu) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      switch (m_exc_code) {
      case 0xd:
        code_desc = "EXC_I386_GPFLT";
        m_exc_data_count = 1;
        break;
      }
      break;
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      switch (m_exc_code) {
      case 0x101:
        code_desc = "EXC_ARM_DA_ALIGN";
        break;
      case 0x102:
        code_desc = "EXC_ARM_DA_DEBUG";
        break;
      }
      break;

    case llvm::Triple::aarch64:
      if (DeterminePtrauthFailure(exe_ctx))
        return m_description.c_str();
      break;

    default:
      break;
    }
    break;

  case 2: // EXC_BAD_INSTRUCTION
    exc_desc = "EXC_BAD_INSTRUCTION";
    switch (cpu) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      if (m_exc_code == 1)
        code_desc = "EXC_I386_INVOP";
      break;

    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      if (m_exc_code == 1)
        code_desc = "EXC_ARM_UNDEFINED";
      break;

    default:
      break;
    }
    break;

  case 3: // EXC_ARITHMETIC
    exc_desc = "EXC_ARITHMETIC";
    switch (cpu) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      switch (m_exc_code) {
      case 1:
        code_desc = "EXC_I386_DIV";
        break;
      case 2:
        code_desc = "EXC_I386_INTO";
        break;
      case 3:
        code_desc = "EXC_I386_NOEXT";
        break;
      case 4:
        code_desc = "EXC_I386_EXTOVR";
        break;
      case 5:
        code_desc = "EXC_I386_EXTERR";
        break;
      case 6:
        code_desc = "EXC_I386_EMERR";
        break;
      case 7:
        code_desc = "EXC_I386_BOUND";
        break;
      case 8:
        code_desc = "EXC_I386_SSEEXTERR";
        break;
      }
      break;

    default:
      break;
    }
    break;

  case 4: // EXC_EMULATION
    exc_desc = "EXC_EMULATION";
    break;

  case 5: // EXC_SOFTWARE
    exc_desc = "EXC_SOFTWARE";
    if (m_exc_code == 0x10003) {
      subcode_desc = "EXC_SOFT_SIGNAL";
      subcode_label = "signo";
    }
    break;

  case 6: // EXC_BREAKPOINT
  {
    exc_desc = "EXC_BREAKPOINT";
    switch (cpu) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      switch (m_exc_code) {
      case 1:
        code_desc = "EXC_I386_SGL";
        break;
      case 2:
        code_desc = "EXC_I386_BPT";
        break;
      }
      break;

    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      switch (m_exc_code) {
      case 0x101:
        code_desc = "EXC_ARM_DA_ALIGN";
        break;
      case 0x102:
        code_desc = "EXC_ARM_DA_DEBUG";
        break;
      case 1:
        code_desc = "EXC_ARM_BREAKPOINT";
        break;
      // FIXME temporary workaround, exc_code 0 does not really mean
      // EXC_ARM_BREAKPOINT
      case 0:
        code_desc = "EXC_ARM_BREAKPOINT";
        break;
      }
      break;

    case llvm::Triple::aarch64:
      if (DeterminePtrauthFailure(exe_ctx))
        return m_description.c_str();
      break;

    default:
      break;
    }
  } break;

  case 7:
    exc_desc = "EXC_SYSCALL";
    break;

  case 8:
    exc_desc = "EXC_MACH_SYSCALL";
    break;

  case 9:
    exc_desc = "EXC_RPC_ALERT";
    break;

  case 10:
    exc_desc = "EXC_CRASH";
    break;
  case 11:
    exc_desc = "EXC_RESOURCE";
#if defined(__APPLE__)
    {
      int resource_type = EXC_RESOURCE_DECODE_RESOURCE_TYPE(m_exc_code);

      code_label = "limit";
      code_desc = code_desc_buf;
      subcode_label = "observed";
      subcode_desc = subcode_desc_buf;

      switch (resource_type) {
      case RESOURCE_TYPE_CPU:
        exc_desc = "EXC_RESOURCE RESOURCE_TYPE_CPU";
        snprintf(code_desc_buf, sizeof(code_desc_buf), "%d%%",
                 (int)EXC_RESOURCE_CPUMONITOR_DECODE_PERCENTAGE(m_exc_code));
        snprintf(subcode_desc_buf, sizeof(subcode_desc_buf), "%d%%",
                 (int)EXC_RESOURCE_CPUMONITOR_DECODE_PERCENTAGE_OBSERVED(
                     m_exc_subcode));
        break;
      case RESOURCE_TYPE_WAKEUPS:
        exc_desc = "EXC_RESOURCE RESOURCE_TYPE_WAKEUPS";
        snprintf(
            code_desc_buf, sizeof(code_desc_buf), "%d w/s",
            (int)EXC_RESOURCE_CPUMONITOR_DECODE_WAKEUPS_PERMITTED(m_exc_code));
        snprintf(subcode_desc_buf, sizeof(subcode_desc_buf), "%d w/s",
                 (int)EXC_RESOURCE_CPUMONITOR_DECODE_WAKEUPS_OBSERVED(
                     m_exc_subcode));
        break;
      case RESOURCE_TYPE_MEMORY:
        exc_desc = "EXC_RESOURCE RESOURCE_TYPE_MEMORY";
        snprintf(code_desc_buf, sizeof(code_desc_buf), "%d MB",
                 (int)EXC_RESOURCE_HWM_DECODE_LIMIT(m_exc_code));
        subcode_desc = nullptr;
        subcode_label = "unused";
        break;
#if defined(RESOURCE_TYPE_IO)
      // RESOURCE_TYPE_IO is introduced in macOS SDK 10.12.
      case RESOURCE_TYPE_IO:
        exc_desc = "EXC_RESOURCE RESOURCE_TYPE_IO";
        snprintf(code_desc_buf, sizeof(code_desc_buf), "%d MB",
                 (int)EXC_RESOURCE_IO_DECODE_LIMIT(m_exc_code));
        snprintf(subcode_desc_buf, sizeof(subcode_desc_buf), "%d MB",
                 (int)EXC_RESOURCE_IO_OBSERVED(m_exc_subcode));
        ;
        break;
#endif
      }
    }
#endif
    break;
  case 12:
    exc_desc = "EXC_GUARD";
    break;
  }

  StreamString strm;

  if (exc_desc)
    strm.PutCString(exc_desc);
  else
    strm.Printf("EXC_??? (%" PRIu64 ")", m_value);

  if (m_exc_data_count >= 1) {
    if (code_desc)
      strm.Printf(" (%s=%s", code_label, code_desc);
    else
      strm.Printf(" (%s=%" PRIu64, code_label, m_exc_code);
  }

  if (m_exc_data_count >= 2) {
    if (subcode_desc)
      strm.Printf(", %s=%s", subcode_label, subcode_desc);
    else
      strm.Printf(", %s=0x%" PRIx64, subcode_label, m_exc_subcode);
  }

  if (m_exc_data_count > 0)
    strm.PutChar(')');

  m_description = std::string(strm.GetString());
  return m_description.c_str();
}

static StopInfoSP GetStopInfoForHardwareBP(Thread &thread, Target *target,
                                           uint32_t exc_data_count,
                                           uint64_t exc_sub_code,
                                           uint64_t exc_sub_sub_code) {
  // Try hardware watchpoint.
  if (target) {
    // The exc_sub_code indicates the data break address.
    lldb::WatchpointSP wp_sp =
        target->GetWatchpointList().FindByAddress((lldb::addr_t)exc_sub_code);
    if (wp_sp && wp_sp->IsEnabled()) {
      // Debugserver may piggyback the hardware index of the fired watchpoint
      // in the exception data. Set the hardware index if that's the case.
      if (exc_data_count >= 3)
        wp_sp->SetHardwareIndex((uint32_t)exc_sub_sub_code);
      return StopInfo::CreateStopReasonWithWatchpointID(thread, wp_sp->GetID());
    }
  }

  // Try hardware breakpoint.
  ProcessSP process_sp(thread.GetProcess());
  if (process_sp) {
    // The exc_sub_code indicates the data break address.
    lldb::BreakpointSiteSP bp_sp =
        process_sp->GetBreakpointSiteList().FindByAddress(
            (lldb::addr_t)exc_sub_code);
    if (bp_sp && bp_sp->IsEnabled()) {
      // Debugserver may piggyback the hardware index of the fired breakpoint
      // in the exception data. Set the hardware index if that's the case.
      if (exc_data_count >= 3)
        bp_sp->SetHardwareIndex((uint32_t)exc_sub_sub_code);
      return StopInfo::CreateStopReasonWithBreakpointSiteID(thread,
                                                            bp_sp->GetID());
    }
  }

  return nullptr;
}

StopInfoSP StopInfoMachException::CreateStopReasonWithMachException(
    Thread &thread, uint32_t exc_type, uint32_t exc_data_count,
    uint64_t exc_code, uint64_t exc_sub_code, uint64_t exc_sub_sub_code,
    bool pc_already_adjusted, bool adjust_pc_if_needed) {
  if (exc_type == 0)
    return StopInfoSP();

  uint32_t pc_decrement = 0;
  ExecutionContext exe_ctx(thread.shared_from_this());
  Target *target = exe_ctx.GetTargetPtr();
  const llvm::Triple::ArchType cpu =
      target ? target->GetArchitecture().GetMachine()
             : llvm::Triple::UnknownArch;

  switch (exc_type) {
  case 1: // EXC_BAD_ACCESS
  case 2: // EXC_BAD_INSTRUCTION
  case 3: // EXC_ARITHMETIC
  case 4: // EXC_EMULATION
    break;

  case 5:                    // EXC_SOFTWARE
    if (exc_code == 0x10003) // EXC_SOFT_SIGNAL
    {
      if (exc_sub_code == 5) {
        // On MacOSX, a SIGTRAP can signify that a process has called exec,
        // so we should check with our dynamic loader to verify.
        ProcessSP process_sp(thread.GetProcess());
        if (process_sp) {
          DynamicLoader *dynamic_loader = process_sp->GetDynamicLoader();
          if (dynamic_loader && dynamic_loader->ProcessDidExec()) {
            // The program was re-exec'ed
            return StopInfo::CreateStopReasonWithExec(thread);
          }
        }
      }
      return StopInfo::CreateStopReasonWithSignal(thread, exc_sub_code);
    }
    break;

  case 6: // EXC_BREAKPOINT
  {
    bool is_actual_breakpoint = false;
    bool is_trace_if_actual_breakpoint_missing = false;
    switch (cpu) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      if (exc_code == 1) // EXC_I386_SGL
      {
        if (!exc_sub_code) {
          // This looks like a plain trap.
          // Have to check if there is a breakpoint here as well.  When you
          // single-step onto a trap, the single step stops you not to trap.
          // Since we also do that check below, let's just use that logic.
          is_actual_breakpoint = true;
          is_trace_if_actual_breakpoint_missing = true;
        } else {
          if (StopInfoSP stop_info =
                  GetStopInfoForHardwareBP(thread, target, exc_data_count,
                                           exc_sub_code, exc_sub_sub_code))
            return stop_info;
        }
      } else if (exc_code == 2 || // EXC_I386_BPT
                 exc_code == 3)   // EXC_I386_BPTFLT
      {
        // KDP returns EXC_I386_BPTFLT for trace breakpoints
        if (exc_code == 3)
          is_trace_if_actual_breakpoint_missing = true;

        is_actual_breakpoint = true;
        if (!pc_already_adjusted)
          pc_decrement = 1;
      }
      break;

    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      if (exc_code == 0x102) // EXC_ARM_DA_DEBUG
      {
        // It's a watchpoint, then, if the exc_sub_code indicates a
        // known/enabled data break address from our watchpoint list.
        lldb::WatchpointSP wp_sp;
        if (target)
          wp_sp = target->GetWatchpointList().FindByAddress(
              (lldb::addr_t)exc_sub_code);
        if (wp_sp && wp_sp->IsEnabled()) {
          // Debugserver may piggyback the hardware index of the fired
          // watchpoint in the exception data. Set the hardware index if
          // that's the case.
          if (exc_data_count >= 3)
            wp_sp->SetHardwareIndex((uint32_t)exc_sub_sub_code);
          return StopInfo::CreateStopReasonWithWatchpointID(thread,
                                                            wp_sp->GetID());
        } else {
          is_actual_breakpoint = true;
          is_trace_if_actual_breakpoint_missing = true;
        }
      } else if (exc_code == 1) // EXC_ARM_BREAKPOINT
      {
        is_actual_breakpoint = true;
        is_trace_if_actual_breakpoint_missing = true;
      } else if (exc_code == 0) // FIXME not EXC_ARM_BREAKPOINT but a kernel
                                // is currently returning this so accept it
                                // as indicating a breakpoint until the
                                // kernel is fixed
      {
        is_actual_breakpoint = true;
        is_trace_if_actual_breakpoint_missing = true;
      }
      break;

    case llvm::Triple::aarch64_32:
    case llvm::Triple::aarch64: {
      if (exc_code == 1 && exc_sub_code == 0) // EXC_ARM_BREAKPOINT
      {
        // This is hit when we single instruction step aka MDSCR_EL1 SS bit 0
        // is set
        is_actual_breakpoint = false;
        is_trace_if_actual_breakpoint_missing = true;
      }
      if (exc_code == 0x102) // EXC_ARM_DA_DEBUG
      {
        // It's a watchpoint, then, if the exc_sub_code indicates a
        // known/enabled data break address from our watchpoint list.
        lldb::WatchpointSP wp_sp;
        if (target)
          wp_sp = target->GetWatchpointList().FindByAddress(
              (lldb::addr_t)exc_sub_code);
        if (wp_sp && wp_sp->IsEnabled()) {
          // Debugserver may piggyback the hardware index of the fired
          // watchpoint in the exception data. Set the hardware index if
          // that's the case.
          if (exc_data_count >= 3)
            wp_sp->SetHardwareIndex((uint32_t)exc_sub_sub_code);
          return StopInfo::CreateStopReasonWithWatchpointID(thread,
                                                            wp_sp->GetID());
        }
        // EXC_ARM_DA_DEBUG seems to be reused for EXC_BREAKPOINT as well as
        // EXC_BAD_ACCESS
        if (thread.GetTemporaryResumeState() == eStateStepping)
          return StopInfo::CreateStopReasonToTrace(thread);
      }
      // It looks like exc_sub_code has the 4 bytes of the instruction that
      // triggered the exception, i.e. our breakpoint opcode
      is_actual_breakpoint = exc_code == 1;
      break;
    }

    default:
      break;
    }

    if (is_actual_breakpoint) {
      RegisterContextSP reg_ctx_sp(thread.GetRegisterContext());
      addr_t pc = reg_ctx_sp->GetPC() - pc_decrement;

      ProcessSP process_sp(thread.CalculateProcess());

      lldb::BreakpointSiteSP bp_site_sp;
      if (process_sp)
        bp_site_sp = process_sp->GetBreakpointSiteList().FindByAddress(pc);
      if (bp_site_sp && bp_site_sp->IsEnabled()) {
        // Update the PC if we were asked to do so, but only do so if we find
        // a breakpoint that we know about cause this could be a trap
        // instruction in the code
        if (pc_decrement > 0 && adjust_pc_if_needed)
          reg_ctx_sp->SetPC(pc);

        // If the breakpoint is for this thread, then we'll report the hit,
        // but if it is for another thread, we can just report no reason.  We
        // don't need to worry about stepping over the breakpoint here, that
        // will be taken care of when the thread resumes and notices that
        // there's a breakpoint under the pc. If we have an operating system
        // plug-in, we might have set a thread specific breakpoint using the
        // operating system thread ID, so we can't make any assumptions about
        // the thread ID so we must always report the breakpoint regardless
        // of the thread.
        if (bp_site_sp->ValidForThisThread(thread) ||
            thread.GetProcess()->GetOperatingSystem() != nullptr)
          return StopInfo::CreateStopReasonWithBreakpointSiteID(
              thread, bp_site_sp->GetID());
        else if (is_trace_if_actual_breakpoint_missing)
          return StopInfo::CreateStopReasonToTrace(thread);
        else
          return StopInfoSP();
      }

      // Don't call this a trace if we weren't single stepping this thread.
      if (is_trace_if_actual_breakpoint_missing &&
          thread.GetTemporaryResumeState() == eStateStepping) {
        return StopInfo::CreateStopReasonToTrace(thread);
      }
    }
  } break;

  case 7:  // EXC_SYSCALL
  case 8:  // EXC_MACH_SYSCALL
  case 9:  // EXC_RPC_ALERT
  case 10: // EXC_CRASH
    break;
  }

  return StopInfoSP(new StopInfoMachException(thread, exc_type, exc_data_count,
                                              exc_code, exc_sub_code));
}

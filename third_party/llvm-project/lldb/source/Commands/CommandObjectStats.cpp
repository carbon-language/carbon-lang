//===-- CommandObjectStats.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectStats.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

class CommandObjectStatsEnable : public CommandObjectParsed {
public:
  CommandObjectStatsEnable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "enable",
                            "Enable statistics collection", nullptr,
                            eCommandProcessMustBePaused) {}

  ~CommandObjectStatsEnable() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (DebuggerStats::GetCollectingStats()) {
      result.AppendError("statistics already enabled");
      return false;
    }

    DebuggerStats::SetCollectingStats(true);
    result.SetStatus(eReturnStatusSuccessFinishResult);
    return true;
  }
};

class CommandObjectStatsDisable : public CommandObjectParsed {
public:
  CommandObjectStatsDisable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "disable",
                            "Disable statistics collection", nullptr,
                            eCommandProcessMustBePaused) {}

  ~CommandObjectStatsDisable() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!DebuggerStats::GetCollectingStats()) {
      result.AppendError("need to enable statistics before disabling them");
      return false;
    }

    DebuggerStats::SetCollectingStats(false);
    result.SetStatus(eReturnStatusSuccessFinishResult);
    return true;
  }
};

#define LLDB_OPTIONS_statistics_dump
#include "CommandOptions.inc"

class CommandObjectStatsDump : public CommandObjectParsed {
  class CommandOptions : public Options {
  public:
    CommandOptions() { OptionParsingStarting(nullptr); }

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'a':
        m_all_targets = true;
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }
      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_all_targets = false;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_statistics_dump_options);
    }

    bool m_all_targets = false;
  };

public:
  CommandObjectStatsDump(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "statistics dump", "Dump metrics in JSON format",
            "statistics dump [<options>]", eCommandRequiresTarget) {}

  ~CommandObjectStatsDump() override = default;

  Options *GetOptions() override { return &m_options; }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    Target *target = nullptr;
    if (!m_options.m_all_targets)
      target = m_exe_ctx.GetTargetPtr();

    result.AppendMessageWithFormatv(
        "{0:2}", DebuggerStats::ReportStatistics(GetDebugger(), target));
    result.SetStatus(eReturnStatusSuccessFinishResult);
    return true;
  }

  CommandOptions m_options;
};

CommandObjectStats::CommandObjectStats(CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "statistics",
                             "Print statistics about a debugging session",
                             "statistics <subcommand> [<subcommand-options>]") {
  LoadSubCommand("enable",
                 CommandObjectSP(new CommandObjectStatsEnable(interpreter)));
  LoadSubCommand("disable",
                 CommandObjectSP(new CommandObjectStatsDisable(interpreter)));
  LoadSubCommand("dump",
                 CommandObjectSP(new CommandObjectStatsDump(interpreter)));
}

CommandObjectStats::~CommandObjectStats() = default;

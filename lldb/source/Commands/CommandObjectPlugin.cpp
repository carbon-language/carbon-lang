//===-- CommandObjectPlugin.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectPlugin.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

class CommandObjectPluginLoad : public CommandObjectParsed {
public:
  CommandObjectPluginLoad(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "plugin load",
                            "Import a dylib that implements an LLDB plugin.",
                            nullptr) {
    CommandArgumentEntry arg1;
    CommandArgumentData cmd_arg;

    // Define the first (and only) variant of this arg.
    cmd_arg.arg_type = eArgTypeFilename;
    cmd_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(cmd_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
  }

  ~CommandObjectPluginLoad() override = default;

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    CommandCompletions::InvokeCommonCompletionCallbacks(
        GetCommandInterpreter(), CommandCompletions::eDiskFileCompletion,
        request, nullptr);
  }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    size_t argc = command.GetArgumentCount();

    if (argc != 1) {
      result.AppendError("'plugin load' requires one argument");
      return false;
    }

    Status error;

    FileSpec dylib_fspec(command[0].ref());
    FileSystem::Instance().Resolve(dylib_fspec);

    if (GetDebugger().LoadPlugin(dylib_fspec, error))
      result.SetStatus(eReturnStatusSuccessFinishResult);
    else {
      result.AppendError(error.AsCString());
    }

    return result.Succeeded();
  }
};

CommandObjectPlugin::CommandObjectPlugin(CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "plugin",
                             "Commands for managing LLDB plugins.",
                             "plugin <subcommand> [<subcommand-options>]") {
  LoadSubCommand("load",
                 CommandObjectSP(new CommandObjectPluginLoad(interpreter)));
}

CommandObjectPlugin::~CommandObjectPlugin() = default;

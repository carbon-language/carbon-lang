//===-- CommandAlias.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandAlias_h_
#define liblldb_CommandAlias_h_

// C Includes
// C++ Includes
#include <memory>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-forward.h"
#include "lldb/Interpreter/Args.h"

namespace lldb_private {
class CommandAlias
{
public:
    typedef std::unique_ptr<CommandAlias> UniquePointer;

    static bool
    ProcessAliasOptionsArgs (lldb::CommandObjectSP &cmd_obj_sp,
                             const char *options_args,
                             OptionArgVectorSP &option_arg_vector_sp);

    CommandAlias (lldb::CommandObjectSP cmd_sp = nullptr,
                  OptionArgVectorSP args_sp = nullptr);
    
    void
    GetAliasHelp (StreamString &help_string);
    
    bool
    IsValid ()
    {
        return m_underlying_command_sp && m_option_args_sp;
    }
    
    explicit operator bool ()
    {
        return IsValid();
    }
    
    lldb::CommandObjectSP GetUnderlyingCommand() { return m_underlying_command_sp; }
    OptionArgVectorSP GetOptionArguments() { return m_option_args_sp; }
    
private:
    lldb::CommandObjectSP m_underlying_command_sp;
    OptionArgVectorSP m_option_args_sp ;
};
} // namespace lldb_private

#endif // liblldb_CommandAlias_h_

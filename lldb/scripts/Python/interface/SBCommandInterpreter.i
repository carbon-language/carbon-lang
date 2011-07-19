//===-- SWIG Interface for SBCommandInterpreter -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBCommandInterpreter
{
public:
    enum
    {
        eBroadcastBitThreadShouldExit       = (1 << 0),
        eBroadcastBitResetPrompt            = (1 << 1),
        eBroadcastBitQuitCommandReceived    = (1 << 2),           // User entered quit 
        eBroadcastBitAsynchronousOutputData = (1 << 3),
        eBroadcastBitAsynchronousErrorData  = (1 << 4)
    };

    SBCommandInterpreter (const lldb::SBCommandInterpreter &rhs);
    
    ~SBCommandInterpreter ();

    static const char * 
    GetArgumentTypeAsCString (const lldb::CommandArgumentType arg_type);
    
    static const char *
    GetArgumentDescriptionAsCString (const lldb::CommandArgumentType arg_type);
    
    bool
    IsValid() const;

    bool
    CommandExists (const char *cmd);

    bool
    AliasExists (const char *cmd);

    lldb::SBBroadcaster
    GetBroadcaster ();

    bool
    HasCommands ();

    bool
    HasAliases ();

    bool
    HasAliasOptions ();

    lldb::SBProcess
    GetProcess ();

    ssize_t
    WriteToScriptInterpreter (const char *src);

    ssize_t
    WriteToScriptInterpreter (const char *src, size_t src_len);

    void
    SourceInitFileInHomeDirectory (lldb::SBCommandReturnObject &result);

    void
    SourceInitFileInCurrentWorkingDirectory (lldb::SBCommandReturnObject &result);

    lldb::ReturnStatus
    HandleCommand (const char *command_line, lldb::SBCommandReturnObject &result, bool add_to_history = false);

    int
    HandleCompletion (const char *current_line,
                      const char *cursor,
                      const char *last_char,
                      int match_start_point,
                      int max_return_elements,
                      lldb::SBStringList &matches);
};

} // namespace lldb

//===-- CommandObjectQuit.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectQuit_h_
#define liblldb_CommandObjectQuit_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectQuit
//-------------------------------------------------------------------------

// SPECIAL NOTE!! The CommandObjectQuit is special, because the actual function to execute
// when the user types 'quit' is passed (via function pointer) to the Command Interpreter when it
// is constructed.  The function pointer is then stored in this CommandObjectQuit, and is invoked
// via the CommandObjectQuit::Execute function.  This is the only command object that works this
// way; it was done this way because different Command Interpreter callers may want or need different things
// to be done in order to shut down properly.

class CommandObjectQuit : public CommandObject
{
public:

    CommandObjectQuit ();

    virtual
    ~CommandObjectQuit ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectQuit_h_

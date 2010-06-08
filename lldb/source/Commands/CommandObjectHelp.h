//===-- CommandObjectHelp.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectHelp_h_
#define liblldb_CommandObjectHelp_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectHelp
//-------------------------------------------------------------------------

class CommandObjectHelp : public CommandObject
{
public:

    CommandObjectHelp ();

    virtual
    ~CommandObjectHelp ();

    bool
    OldExecute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);
    
    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

    virtual int
    HandleCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      CommandInterpreter *interpreter,
                      StringList &matches);

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectHelp_h_

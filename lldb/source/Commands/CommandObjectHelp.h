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

    virtual bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result);

    virtual int
    HandleCompletion (CommandInterpreter &interpreter,
                      Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      StringList &matches);

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectHelp_h_

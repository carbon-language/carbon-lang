//===-- CommandObjectSyntax.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectSyntax_h_
#define liblldb_CommandObjectSyntax_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectSyntax
//-------------------------------------------------------------------------

class CommandObjectSyntax : public CommandObject
{
public:

    CommandObjectSyntax (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectSyntax ();
    
    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);


};

} // namespace lldb_private

#endif  // liblldb_CommandObjectSyntax_h_

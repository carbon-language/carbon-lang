//===-- CommandObjectSet.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectSet_h_
#define liblldb_CommandObjectSet_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectSet
//-------------------------------------------------------------------------

class CommandObjectSet : public CommandObject
{
public:

    CommandObjectSet ();

    virtual
    ~CommandObjectSet ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectSet_h_

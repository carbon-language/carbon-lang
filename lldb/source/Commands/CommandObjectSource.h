//===-- CommandObjectSource.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectSource_h_
#define liblldb_CommandObjectSource_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Core/STLUtils.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectSource
//-------------------------------------------------------------------------

class CommandObjectSource : public CommandObject
{
public:

    CommandObjectSource ();

    virtual
    ~CommandObjectSource ();

    STLStringArray &
    GetCommands ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectSource_h_

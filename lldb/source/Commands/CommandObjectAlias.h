//===-- CommandObjectAlias.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectAlias_h_
#define liblldb_CommandObjectAlias_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectAlias
//-------------------------------------------------------------------------

class CommandObjectAlias : public CommandObject
{
public:

    CommandObjectAlias ();

    virtual
    ~CommandObjectAlias ();

    virtual bool
    Execute (Args& command,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectAlias_h_

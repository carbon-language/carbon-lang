//===-- CommandObjectVersion.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectVersion_h_
#define liblldb_CommandObjectVersion_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectVersion
//-------------------------------------------------------------------------

class CommandObjectVersion : public CommandObject
{
public:

    CommandObjectVersion (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectVersion ();

    virtual bool
    Execute (Args& args,
             CommandReturnObject &result);

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectVersion_h_

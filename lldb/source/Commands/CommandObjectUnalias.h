//===-- CommandObjectUnalias.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectUnalias_h_
#define liblldb_CommandObjectUnalias_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectUnalias
//-------------------------------------------------------------------------

class CommandObjectUnalias : public CommandObject
{
public:

    CommandObjectUnalias ();

    virtual
    ~CommandObjectUnalias ();

    virtual bool
    Execute (Args& args,
             CommandContext *context,
             CommandInterpreter *interpreter,
             CommandReturnObject &result);

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectUnalias_h_

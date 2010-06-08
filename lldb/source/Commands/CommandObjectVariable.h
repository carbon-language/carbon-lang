//===-- CommandObjectVariable.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectVariable_h_
#define liblldb_CommandObjectVariable_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectImage
//-------------------------------------------------------------------------

class CommandObjectVariable : public CommandObjectMultiword
{
public:

    CommandObjectVariable (CommandInterpreter *iterpreter);

    virtual
    ~CommandObjectVariable ();

private:
    //------------------------------------------------------------------
    // For CommandObjectVariable only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (CommandObjectVariable);
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectVariable_h_

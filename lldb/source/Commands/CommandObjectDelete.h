//===-- CommandObjectDelete.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectDelete_h_
#define liblldb_CommandObjectDelete_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectCrossref.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectDelete
//-------------------------------------------------------------------------

class CommandObjectDelete : public CommandObjectCrossref
{
public:
    CommandObjectDelete ();

    virtual
    ~CommandObjectDelete ();

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectDelete_h_

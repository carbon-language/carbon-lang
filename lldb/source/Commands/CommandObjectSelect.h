//===-- CommandObjectSelect.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectSelect_h_
#define liblldb_CommandObjectSelect_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectCrossref.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectSelect
//-------------------------------------------------------------------------

class CommandObjectSelect : public CommandObjectCrossref
{
public:
    CommandObjectSelect ();

    virtual
    ~CommandObjectSelect ();

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectSelect_h_

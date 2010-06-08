//===-- CommandObjectInfo.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectInfo_h_
#define liblldb_CommandObjectInfo_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectCrossref.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectInfo
//-------------------------------------------------------------------------

class CommandObjectInfo : public CommandObjectCrossref
{
public:
    CommandObjectInfo ();

    virtual
    ~CommandObjectInfo ();

};

} // namespace lldb_private

#endif  // liblldb_CommandObjectInfo_h_

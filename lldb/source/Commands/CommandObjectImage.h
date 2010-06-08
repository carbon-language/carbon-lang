//===-- CommandObjectImage.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectImage_h_
#define liblldb_CommandObjectImage_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectMultiword.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectImage
//-------------------------------------------------------------------------

class CommandObjectImage : public CommandObjectMultiword
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    CommandObjectImage(CommandInterpreter *interpreter);
    virtual
    ~CommandObjectImage();

private:
    //------------------------------------------------------------------
    // For CommandObjectImage only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (CommandObjectImage);
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectImage_h_

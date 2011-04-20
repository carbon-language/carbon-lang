//===-- CommandObjectRegexCommand.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectRegexCommand_h_
#define liblldb_CommandObjectRegexCommand_h_

// C Includes
// C++ Includes
#include <list>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/RegularExpression.h"
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectRegexCommand
//-------------------------------------------------------------------------

class CommandObjectRegexCommand : public CommandObject
{
public:

    CommandObjectRegexCommand (CommandInterpreter &interpreter,
                               const char *name, 
                               const char *help, 
                               const char *syntax, 
                               uint32_t max_matches);
    
    virtual
    ~CommandObjectRegexCommand ();

    virtual bool
    Execute (Args& command,
             CommandReturnObject &result);

    virtual bool
    WantsRawCommandString() { return true; }

    virtual bool
    ExecuteRawCommandString (const char *command,
                             CommandReturnObject &result);


    bool
    AddRegexCommand (const char *re_cstr, const char *command_cstr);

    bool
    HasRegexEntries () const
    {
        return !m_entries.empty();
    }

protected:
    struct Entry
    {
        RegularExpression regex;
        std::string command;
    };

    typedef std::list<Entry> EntryCollection;
    const uint32_t m_max_matches;
    EntryCollection m_entries;

private:
    DISALLOW_COPY_AND_ASSIGN (CommandObjectRegexCommand);
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectRegexCommand_h_

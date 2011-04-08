//===-- StringList.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StringList_h_
#define liblldb_StringList_h_

#include <stdint.h>

#include "lldb/Core/STLUtils.h"
#include "lldb/lldb-forward.h"

namespace lldb_private {

class StringList
{
public:

    StringList ();

    StringList (const char *str);

    StringList (const char **strv, int strc);
    
    virtual
    ~StringList ();

    void
    AppendString (const char *str);

    void
    AppendString (const char *str, size_t str_len);

    void
    AppendList (const char ** strv, int strc);

    void
    AppendList (StringList strings);

    bool
    ReadFileLines (FileSpec &input_file);
    
    uint32_t
    GetSize () const;

    const char *
    GetStringAtIndex (size_t idx) const;

    void
    Clear ();

    void
    LongestCommonPrefix (std::string &common_prefix);

    void
    InsertStringAtIndex (size_t id, const char *str);

    void
    DeleteStringAtIndex (size_t id);

    void
    RemoveBlankLines ();

    size_t
    SplitIntoLines (const char *lines, size_t len);

private:

    STLStringArray m_strings;
};

} // namespace lldb_private

#endif // liblldb_StringList_h_

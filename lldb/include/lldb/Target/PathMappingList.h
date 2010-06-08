//===-- PathMappingList.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PathMappingList_h_
#define liblldb_PathMappingList_h_

// C Includes
// C++ Includes
#include <map>
#include <vector>
// Other libraries and framework includes
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
// Project includes

namespace lldb_private {

class PathMappingList
{
public:

    typedef void (*ChangedCallback) (const PathMappingList &path_list,
                                     void *baton);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    PathMappingList (ChangedCallback callback,
                     void *callback_baton);

    virtual
    ~PathMappingList ();

    void
    Append (const ConstString &path, const ConstString &replacement, bool notify);

    void
    Clear (bool notify);

    void
    Dump (Stream *s);

    size_t
    GetSize ();

    void
    Insert (const ConstString &path, 
            const ConstString &replacement, 
            uint32_t insert_idx,
            bool notify);

    bool
    Remove (off_t index, bool notify);

    bool
    RemapPath (const ConstString &path, ConstString &new_path);

protected:
    typedef std::pair <ConstString, ConstString> pair;
    typedef std::vector <pair> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    collection m_pairs;
    ChangedCallback m_callback;
    void * m_callback_baton;
private:
    //------------------------------------------------------------------
    // For PathMappingList only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (PathMappingList);
};

} // namespace lldb_private

#endif  // liblldb_PathMappingList_h_

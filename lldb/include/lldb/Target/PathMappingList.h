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

    PathMappingList (const PathMappingList &rhs);

    virtual
    ~PathMappingList ();

    const PathMappingList &
    operator =(const PathMappingList &rhs);

    void
    Append (const ConstString &path, const ConstString &replacement, bool notify);

    void
    Clear (bool notify);

    void
    Dump (Stream *s);

    size_t
    GetSize ();

    bool
    GetPathsAtIndex (uint32_t idx, ConstString &path, ConstString &new_path) const;
    
    void
    Insert (const ConstString &path, 
            const ConstString &replacement, 
            uint32_t insert_idx,
            bool notify);

    bool
    Remove (off_t index, bool notify);

    bool
    Remove (const ConstString &path, bool notify);

    bool
    Replace (const ConstString &path, 
             const ConstString &new_path, 
             bool notify);
    
    bool
    RemapPath (const ConstString &path, ConstString &new_path);

    uint32_t
    FindIndexForPath (const ConstString &path) const;

protected:
    typedef std::pair <ConstString, ConstString> pair;
    typedef std::vector <pair> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

    iterator
    FindIteratorForPath (const ConstString &path);
    
    const_iterator
    FindIteratorForPath (const ConstString &path) const;

    collection m_pairs;
    ChangedCallback m_callback;
    void * m_callback_baton;
};

} // namespace lldb_private

#endif  // liblldb_PathMappingList_h_

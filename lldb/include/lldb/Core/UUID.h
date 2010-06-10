//===-- UUID.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_UUID_h_
#define liblldb_UUID_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private.h"

namespace lldb_private {

class UUID
{
public:
    typedef uint8_t ValueType[16];

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    UUID ();
    UUID (const UUID& rhs);
    UUID (const void *uuid_bytes, uint32_t num_uuid_bytes);

    ~UUID ();

    const UUID&
    operator=(const UUID& rhs);

    void
    Clear ();

    void
    Dump (Stream *s) const;

    const void *
    GetBytes() const;

    static size_t
    GetByteSize();

    bool
    IsValid () const;

    void
    SetBytes (const void *uuid_bytes);

    char *
    GetAsCString (char *dst, size_t dst_len) const;

    size_t
    SetfromCString (const char *c_str);

protected:
    //------------------------------------------------------------------
    // Classes that inherit from UUID can see and modify these
    //------------------------------------------------------------------
    ValueType m_uuid;
};

bool operator == (const UUID &lhs, const UUID &rhs);
bool operator != (const UUID &lhs, const UUID &rhs);
bool operator <  (const UUID &lhs, const UUID &rhs);
bool operator <= (const UUID &lhs, const UUID &rhs);
bool operator >  (const UUID &lhs, const UUID &rhs);
bool operator >= (const UUID &lhs, const UUID &rhs);

} // namespace lldb_private

#endif  // liblldb_UUID_h_

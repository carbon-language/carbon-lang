//===-- UserID.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/UserID.h"
#include "lldb/Core/Stream.h"

using namespace lldb;
using namespace lldb_private;

UserID::UserID (user_id_t uid) :
    m_uid(uid)
{
}

UserID::~UserID ()
{
}

void
UserID::Clear ()
{
    m_uid = LLDB_INVALID_UID;
}


user_id_t
UserID::GetID () const
{
    return m_uid;
}

void
UserID::SetID (user_id_t uid)
{
    m_uid = uid;
}

UserID::IDMatches::IDMatches (user_id_t uid) :
    m_uid(uid)
{
}

bool
UserID::IDMatches::operator() (const UserID& rhs) const
{
    return m_uid == rhs.GetID();
}

Stream&
lldb_private::operator << (Stream& strm, const UserID& uid)
{
    strm.Printf("{0x%8.8x}", uid.GetID());
    return strm;
}

bool
lldb_private::operator== (const UserID& lhs, const UserID& rhs)
{
    return lhs.GetID() == rhs.GetID();
}

bool
lldb_private::operator!= (const UserID& lhs, const UserID& rhs)
{
    return lhs.GetID() != rhs.GetID();
}


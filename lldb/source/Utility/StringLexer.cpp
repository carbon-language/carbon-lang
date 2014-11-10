//===--------------------- StringLexer.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StringLexer.h"

#include <algorithm>
#include <assert.h>

using namespace lldb_utility;

StringLexer::StringLexer (std::string s) :
    m_data(s),
    m_position(0)
{ }

StringLexer::StringLexer (const StringLexer& rhs) :
    m_data(rhs.m_data),
    m_position(rhs.m_position)
{ }

StringLexer::Character
StringLexer::Peek ()
{
    return m_data[m_position];
}

bool
StringLexer::NextIf (Character c)
{
    auto val = Peek();
    if (val == c)
    {
        Next();
        return true;
    }
    return false;
}

StringLexer::Character
StringLexer::Next ()
{
    auto val = Peek();
    Consume();
    return val;
}

bool
StringLexer::HasAtLeast (Size s)
{
    return (m_data.size() - m_position) >= s;
}

void
StringLexer::PutBack (Size s)
{
    assert (m_position >= s);
    m_position -= s;
}

bool
StringLexer::HasAny (Character c)
{
    return m_data.find(c, m_position) != std::string::npos;
}

void
StringLexer::Consume()
{
    m_position++;
}

StringLexer&
StringLexer::operator = (const StringLexer& rhs)
{
    if (this != &rhs)
    {
        m_data = rhs.m_data;
        m_position = rhs.m_position;
    }
    return *this;
}

//===--------------------- StringLexer.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StringLexer.h"

using namespace lldb_utility;

StringLexer::StringLexer (std::string s) :
m_data(s),
m_position(0),
m_putback_data()
{ }

StringLexer::StringLexer (const StringLexer& rhs) :
m_data(rhs.m_data),
m_position(rhs.m_position),
m_putback_data(rhs.m_putback_data)
{ }

StringLexer::Character
StringLexer::Peek ()
{
    if (m_putback_data.empty())
        return m_data[m_position];
    else
        return m_putback_data.front();
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
    return m_data.size()-m_position >= s;
}


void
StringLexer::PutBack (Character c)
{
    m_putback_data.push_back(c);
}

bool
StringLexer::HasAny (Character c)
{
    return m_data.find(c, m_position) != std::string::npos;
}

void
StringLexer::Consume()
{
    if (m_putback_data.empty())
        m_position++;
    else
        m_putback_data.pop_front();
}

StringLexer&
StringLexer::operator = (const StringLexer& rhs)
{
    if (this != &rhs)
    {
        m_data = rhs.m_data;
        m_position = rhs.m_position;
        m_putback_data = rhs.m_putback_data;
    }
    return *this;
}

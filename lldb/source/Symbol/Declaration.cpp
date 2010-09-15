//===-- Declaration.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/Declaration.h"
#include "lldb/Core/Stream.h"

using namespace lldb_private;

Declaration::Declaration() :
    m_file(),
    m_line(0),
    m_column(0)
{
}

Declaration::Declaration(const FileSpec& f, uint32_t l, uint32_t c) :
    m_file(f),
    m_line(l),
    m_column(c)
{
}

Declaration::Declaration(const Declaration& rhs) :
    m_file(rhs.m_file),
    m_line(rhs.m_line),
    m_column(rhs.m_column)
{
}

Declaration::Declaration(const Declaration* decl_ptr) :
    m_file(),
    m_line(0),
    m_column(0)
{
    if (decl_ptr != NULL)
        *this = *decl_ptr;
}

bool
Declaration::IsValid() const
{
    return m_file && m_line != 0;
}

void
Declaration::Clear()
{
    m_file.Clear();
    m_line= 0;
    m_column = 0;
}

void
Declaration::Dump(Stream *s, bool show_fullpaths) const
{
    if (m_file)
    {
        *s << ", decl = ";
        if (show_fullpaths)
            *s << m_file;
        else
            *s << m_file.GetFilename();
        if (m_line > 0)
            s->Printf(":%u", m_line);
        if (m_column > 0)
            s->Printf(":%u", m_column);
    }
    else
    {
        if (m_line > 0)
        {
            s->Printf(", line = %u", m_line);
            if (m_column > 0)
                s->Printf(":%u", m_column);
        }
        else if (m_column > 0)
            s->Printf(", column = %u", m_column);
    }
}

void
Declaration::DumpStopContext (Stream *s, bool show_fullpaths) const
{
    if (m_file)
    {
        if (show_fullpaths || s->GetVerbose())
            *s << m_file;
        else
            m_file.GetFilename().Dump(s);

        if (m_line > 0)
            s->Printf(":%u", m_line);
        if (m_column > 0)
            s->Printf(":%u", m_column);
    }
    else
    {
        s->Printf(" line %u", m_line);
        if (m_column > 0)
            s->Printf(":%u", m_column);
    }
}

uint32_t
Declaration::GetColumn() const
{
    return m_column;
}

FileSpec&
Declaration::GetFile()
{
    return m_file;
}

const FileSpec&
Declaration::GetFile() const
{
    return m_file;
}

uint32_t
Declaration::GetLine() const
{
    return m_line;
}

size_t
Declaration::MemorySize() const
{
    return sizeof(Declaration);
}

void
Declaration::SetColumn(uint32_t col)
{
    m_column = col;
}

void
Declaration::SetFile(const FileSpec& file)
{
    m_file = file;
}

void
Declaration::SetLine(uint32_t line)
{
    m_line = line;
}



int
Declaration::Compare(const Declaration& a, const Declaration& b)
{
    int result = FileSpec::Compare(a.m_file, b.m_file, true);
    if (result)
        return result;
    if (a.m_line < b.m_line)
        return -1;
    else if (a.m_line > b.m_line)
        return 1;
    if (a.m_column < b.m_column)
        return -1;
    else if (a.m_column > b.m_column)
        return 1;
    return 0;
}

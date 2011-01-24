//===-- StreamFile.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/StreamFile.h"
#include <stdio.h>

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// StreamFile constructor
//----------------------------------------------------------------------
StreamFile::StreamFile () :
    Stream (),
    m_file (NULL),
    m_close_file (false),
    m_path_name ()
{
}

StreamFile::StreamFile(uint32_t flags, uint32_t addr_size, ByteOrder byte_order, FILE *f) :
    Stream (flags, addr_size, byte_order),
    m_file (f),
    m_close_file (false),
    m_path_name ()
{
}

StreamFile::StreamFile(FILE *f, bool tranfer_ownership) :
    Stream (),
    m_file (f),
    m_close_file (tranfer_ownership),
    m_path_name ()
{
}

StreamFile::StreamFile(uint32_t flags, uint32_t addr_size, ByteOrder byte_order, const char *path, const char *permissions) :
    Stream (flags, addr_size, byte_order),
    m_file (NULL),
    m_close_file(false),
    m_path_name (path)
{
    Open(path, permissions);
}

StreamFile::StreamFile(const char *path, const char *permissions) :
    Stream (),
    m_file (NULL),
    m_close_file(false),
    m_path_name (path)
{
    Open(path, permissions);
}


StreamFile::~StreamFile()
{
    Close ();
}

void
StreamFile::Close ()
{
    if (m_close_file && m_file != NULL)
        ::fclose (m_file);
    m_file = NULL;
    m_close_file = false;
}

bool
StreamFile::Open (const char *path, const char *permissions)
{
    Close();
    if (path && path[0])
    {
        if ((m_path_name.size() == 0)
            || (m_path_name.compare(path) != 0))
            m_path_name = path;
        m_file = ::fopen (path, permissions);
        if (m_file != NULL)
            m_close_file = true;
    }
    return m_file != NULL;
}

void
StreamFile::SetLineBuffered ()
{
    if (m_file != NULL)
        setlinebuf (m_file);
}

void
StreamFile::Flush ()
{
    if (m_file)
        ::fflush (m_file);
}

int
StreamFile::Write (const void *s, size_t length)
{
    if (m_file)
        return ::fwrite (s, 1, length, m_file);
    return 0;
}

FILE *
StreamFile::GetFileHandle()
{
    return m_file;
}

void
StreamFile::SetFileHandle (FILE *file, bool close_file)
{
    Close();
    m_file = file;
    m_close_file = close_file;
}

const char *
StreamFile::GetFilePathname ()
{
    if (m_path_name.size() == 0)
        return NULL;
    else
        return m_path_name.c_str();
}

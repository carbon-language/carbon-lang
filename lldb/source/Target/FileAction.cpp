//===-- FileAction.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <fcntl.h>

#if defined(_WIN32)
#include "lldb/Host/Windows/win32.h" // For O_NOCTTY
#endif

#include "lldb/Target/FileAction.h"

using namespace lldb_private;

//----------------------------------------------------------------------------
// FileAction member functions
//----------------------------------------------------------------------------

FileAction::FileAction() : m_action(eFileActionNone), m_fd(-1), m_arg(-1), m_path() {}

void FileAction::Clear()
{
    m_action = eFileActionNone;
    m_fd = -1;
    m_arg = -1;
    m_path.clear();
}

const char *FileAction::GetPath() const
{
    if (m_path.empty())
        return NULL;
    return m_path.c_str();
}

bool FileAction::Open(int fd, const char *path, bool read, bool write)
{
    if ((read || write) && fd >= 0 && path && path[0])
    {
        m_action = eFileActionOpen;
        m_fd = fd;
        if (read && write)
            m_arg = O_NOCTTY | O_CREAT | O_RDWR;
        else if (read)
            m_arg = O_NOCTTY | O_RDONLY;
        else
            m_arg = O_NOCTTY | O_CREAT | O_WRONLY;
        m_path.assign(path);
        return true;
    }
    else
    {
        Clear();
    }
    return false;
}

bool FileAction::Close(int fd)
{
    Clear();
    if (fd >= 0)
    {
        m_action = eFileActionClose;
        m_fd = fd;
    }
    return m_fd >= 0;
}

bool FileAction::Duplicate(int fd, int dup_fd)
{
    Clear();
    if (fd >= 0 && dup_fd >= 0)
    {
        m_action = eFileActionDuplicate;
        m_fd = fd;
        m_arg = dup_fd;
    }
    return m_fd >= 0;
}

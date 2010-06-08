//===-- DataBufferMemoryMap.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "lldb/Core/DataBufferMemoryMap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Host/Host.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// Default Constructor
//----------------------------------------------------------------------
DataBufferMemoryMap::DataBufferMemoryMap() :
    m_mmap_addr(NULL),
    m_mmap_size(0),
    m_data(NULL),
    m_size(0),
    m_error()
{
}

//----------------------------------------------------------------------
// Virtual destructor since this class inherits from a pure virtual
// base class.
//----------------------------------------------------------------------
DataBufferMemoryMap::~DataBufferMemoryMap()
{
    Clear();
}

//----------------------------------------------------------------------
// Return a pointer to the bytes owned by this object, or NULL if
// the object contains no bytes.
//----------------------------------------------------------------------
uint8_t *
DataBufferMemoryMap::GetBytes()
{
    return m_data;
}

//----------------------------------------------------------------------
// Return a const pointer to the bytes owned by this object, or NULL
// if the object contains no bytes.
//----------------------------------------------------------------------
const uint8_t *
DataBufferMemoryMap::GetBytes() const
{
    return m_data;
}

//----------------------------------------------------------------------
// Return the number of bytes this object currently contains.
//----------------------------------------------------------------------
size_t
DataBufferMemoryMap::GetByteSize() const
{
    return m_size;
}

//----------------------------------------------------------------------
// Reverts this object to an empty state by unmapping any memory
// that is currently owned.
//----------------------------------------------------------------------
void
DataBufferMemoryMap::Clear()
{
    if (m_mmap_addr != NULL)
    {
        ::munmap((void *)m_mmap_addr, m_mmap_size);
        m_mmap_addr = NULL;
        m_mmap_size = 0;
        m_data = NULL;
        m_size = 0;
    }
    m_error.Clear();
}


const Error &
DataBufferMemoryMap::GetError() const
{
    return m_error;
}

//----------------------------------------------------------------------
// Memory map "length" bytes from "file" starting "offset"
// bytes into the file. If "length" is set to SIZE_T_MAX, then
// map as many bytes as possible.
//
// Returns the number of bytes mapped starting from the requested
// offset.
//----------------------------------------------------------------------
size_t
DataBufferMemoryMap::MemoryMapFromFileSpec (const FileSpec* file, off_t offset, size_t length)
{
    if (file != NULL)
    {
        char path[PATH_MAX];
        if (file->GetPath(path, sizeof(path)))
        {
            int fd = ::open(path, O_RDONLY, 0);
            if (fd >= 0)
            {
                MemoryMapFromFileDescriptor (fd, offset, length);
                ::close(fd);
                return GetByteSize();
            }
            else
            {
                m_error.SetErrorToErrno();
                return 0;
            }
        }
    }
    // We should only get here if there was an error
    Clear();
    return 0;
}


//----------------------------------------------------------------------
// The file descriptor FD is assumed to already be opened as read only
// and the STAT structure is assumed to a valid pointer and already
// containing valid data from a call to stat().
//
// Memory map FILE_LENGTH bytes in FILE starting FILE_OFFSET bytes into
// the file. If FILE_LENGTH is set to SIZE_T_MAX, then map as many bytes
// as possible.
//
// RETURNS
//  Number of bytes mapped starting from the requested offset.
//----------------------------------------------------------------------
size_t
DataBufferMemoryMap::MemoryMapFromFileDescriptor (int fd, off_t offset, size_t length)
{
    Clear();
    if (fd >= 0)
    {
        struct stat stat;
        if (::fstat(fd, &stat) == 0)
        {
            if ((stat.st_mode & S_IFREG) && (stat.st_size > offset))
            {
                if (length == SIZE_T_MAX)
                    length = stat.st_size - offset;

                // Cap the length if too much data was requested
                if (length > stat.st_size - offset)
                    length = stat.st_size - offset;

                if (length > 0)
                {
                    m_mmap_addr = (uint8_t *)::mmap(NULL, length, PROT_READ, MAP_FILE | MAP_SHARED, fd, offset);

                    if (m_mmap_addr == (void*)-1)
                    {
                        m_error.SetErrorToErrno ();
                        if (m_error.GetError() == EINVAL)
                        {
                            // We may still have a shot at memory mapping if we align things correctly
                            size_t page_offset = offset % Host::GetPageSize();
                            if (page_offset != 0)
                            {
                                m_mmap_addr = (uint8_t *)::mmap(NULL, length + page_offset, PROT_READ, MAP_FILE | MAP_SHARED, fd, offset - page_offset);
                                if (m_mmap_addr == (void*)-1)
                                {
                                    // Failed to map file
                                    m_mmap_addr = NULL;
                                }
                                else if (m_mmap_addr != NULL)
                                {
                                    // We recovered and were able to memory map
                                    // after we aligned things to page boundaries
                                    m_error.Clear ();

                                    // Save the actual mmap'ed size
                                    m_mmap_size = length + page_offset;
                                    // Our data is at an offset into the the mapped data
                                    m_data = m_mmap_addr + page_offset;
                                    // Our pretend size is the size that was requestd
                                    m_size = length;
                                }
                            }
                        }
                    }
                    else
                    {
                        // We were able to map the requested data in one chunk
                        // where our mmap and actual data are the same.
                        m_mmap_size = length;
                        m_data = m_mmap_addr;
                        m_size = length;
                    }
                }
            }
        }

        ::close (fd);
    }
    return GetByteSize ();
}

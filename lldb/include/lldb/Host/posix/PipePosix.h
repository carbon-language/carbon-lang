//===-- PipePosix.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_posix_PipePosix_h_
#define liblldb_Host_posix_PipePosix_h_
#if defined(__cplusplus)

#include <stdarg.h>
#include <stdio.h>
#include <sys/types.h>

#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Pipe Pipe.h "lldb/Host/posix/PipePosix.h"
/// @brief A posix-based implementation of Pipe, a class that abtracts
///        unix style pipes.
///
/// A class that abstracts the LLDB core from host pipe functionality.
//----------------------------------------------------------------------
class Pipe
{
public:
    static int kInvalidDescriptor;
    
    Pipe();
    
    ~Pipe();
    
    bool
    Open(bool child_processes_inherit = false);

    bool
    IsValid() const;
    
    bool
    ReadDescriptorIsValid() const;

    bool
    WriteDescriptorIsValid() const;

    int
    GetReadFileDescriptor() const;
    
    int
    GetWriteFileDescriptor() const;
    
    // Close both descriptors
    void
    Close();

    bool
    CloseReadFileDescriptor();
    
    bool
    CloseWriteFileDescriptor();

    int
    ReleaseReadFileDescriptor();
    
    int
    ReleaseWriteFileDescriptor();

    size_t
    Read (void *buf, size_t size);

    size_t
    Write (const void *buf, size_t size);
private:
    int m_fds[2];
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Host_posix_PipePosix_h_

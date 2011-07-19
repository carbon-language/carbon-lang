//===-- SWIG Interface for SBStream -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

namespace lldb {

class SBStream
{
public:

    SBStream ();
    
    ~SBStream ();

    bool
    IsValid() const;

    // If this stream is not redirected to a file, it will maintain a local
    // cache for the stream data which can be accessed using this accessor.
    const char *
    GetData ();

    // If this stream is not redirected to a file, it will maintain a local
    // cache for the stream output whose length can be accessed using this 
    // accessor.
    size_t
    GetSize();

    void
    Printf (const char *format, ...);

    void
    RedirectToFile (const char *path, bool append);

    void
    RedirectToFileHandle (FILE *fh, bool transfer_fh_ownership);

    void
    RedirectToFileDescriptor (int fd, bool transfer_fh_ownership);

    // If the stream is redirected to a file, forget about the file and if
    // ownership of the file was transfered to this object, close the file.
    // If the stream is backed by a local cache, clear this cache.
    void
    Clear ();
};

} // namespace lldb

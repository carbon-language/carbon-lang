//===-- SBStream.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBStream_h_
#define LLDB_SBStream_h_

#include <stdio.h>

#include "lldb/API/SBDefines.h"

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

protected:
    friend class SBAddress;
    friend class SBBlock;
    friend class SBBreakpoint;
    friend class SBBreakpointLocation;
    friend class SBCompileUnit;
    friend class SBData;
    friend class SBEvent;
    friend class SBFrame;
    friend class SBFunction;
    friend class SBInstruction;
    friend class SBInstructionList;
    friend class SBModule;
    friend class SBSourceManager_impl;
    friend class SBSymbol;
    friend class SBSymbolContext;
    friend class SBTarget;
    friend class SBThread;
    friend class SBValue;
    friend class SBCommandReturnObject;

#ifndef SWIG

    lldb_private::Stream *
    operator->();

    lldb_private::Stream *
    get();

    lldb_private::Stream &
    ref();

#endif

private:

    DISALLOW_COPY_AND_ASSIGN (SBStream);
    std::auto_ptr<lldb_private::Stream> m_opaque_ap;
    bool m_is_file;
};

} // namespace lldb

#endif // LLDB_SBStream_h_

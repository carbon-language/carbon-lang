//===-- StreamFile.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamFile_h_
#define liblldb_StreamFile_h_

// C Includes
// C++ Includes

#include <string>

// Other libraries and framework includes
// Project includes

#include "lldb/Core/Stream.h"

namespace lldb_private {

class StreamFile : public Stream
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    StreamFile ();

    StreamFile (uint32_t flags, uint32_t addr_size, lldb::ByteOrder byte_order, FILE *f);

    StreamFile (FILE *f, bool tranfer_ownership = false);

    StreamFile (uint32_t flags, uint32_t addr_size, lldb::ByteOrder byte_order, const char *path, const char *permissions = "w");

    StreamFile (const char *path, const char *permissions = "w");

    virtual
    ~StreamFile();

    void
    Close ();

    bool
    Open (const char *path, const char *permissions = "w");

    virtual void
    Flush ();

    virtual int
    Write (const void *s, size_t length);

    FILE *
    GetFileHandle ();

    void
    SetFileHandle (FILE *file, bool close_file);

    const char *
    GetFilePathname ();
    
    void
    SetLineBuffered();

protected:
    //------------------------------------------------------------------
    // Classes that inherit from StreamFile can see and modify these
    //------------------------------------------------------------------
    FILE* m_file;           ///< File handle to dump to.
    bool m_close_file;
    std::string m_path_name;
    
private:
    DISALLOW_COPY_AND_ASSIGN (StreamFile);
};

} // namespace lldb_private

#endif  // liblldb_StreamFile_h_

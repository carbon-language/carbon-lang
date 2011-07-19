//===-- SWIG Interface for SBFileSpec ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBFileSpec
{
public:
    SBFileSpec ();

    SBFileSpec (const lldb::SBFileSpec &rhs);

    SBFileSpec (const char *path);// Deprected, use SBFileSpec (const char *path, bool resolve)

    SBFileSpec (const char *path, bool resolve);

    ~SBFileSpec ();

    bool
    IsValid() const;

    bool
    Exists () const;

    bool
    ResolveExecutableLocation ();

    const char *
    GetFilename() const;

    const char *
    GetDirectory() const;

    uint32_t
    GetPath (char *dst_path, size_t dst_len) const;

    static int
    ResolvePath (const char *src_path, char *dst_path, size_t dst_len);

    bool
    GetDescription (lldb::SBStream &description) const;
};

} // namespace lldb

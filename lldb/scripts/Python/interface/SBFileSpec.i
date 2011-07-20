//===-- SWIG Interface for SBFileSpec ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a file specfication that divides the path into a directory and
basename.  The string values of the paths are put into uniqued string pools
for fast comparisons and efficient memory usage.

For example, the following code

        lineEntry = context.GetLineEntry()
        self.expect(lineEntry.GetFileSpec().GetDirectory(), 'The line entry should have the correct directory',
                    exe=False,
            substrs = [self.mydir])
        self.expect(lineEntry.GetFileSpec().GetFilename(), 'The line entry should have the correct filename',
                    exe=False,
            substrs = ['main.c'])
        self.assertTrue(lineEntry.GetLine() == self.line,
                        'The line entry's line number should match ')

gets the line entry from the symbol context when a thread is stopped.
It gets the file spec corresponding to the line entry and checks that
the filename and the directory matches wat we expect.
") SBFileSpec;
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

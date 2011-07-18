//===-- SWIG Interface for SBLineEntry --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Specifies an association with a contiguous range of instructions and
a source file location. SBCompileUnit contains SBLineEntry(s).

See also SBCompileUnit for example usage of SBLineEntry API."
) SBLineEntry;
class SBLineEntry
{
public:

    SBLineEntry ();

    SBLineEntry (const lldb::SBLineEntry &rhs);

    ~SBLineEntry ();

    lldb::SBAddress
    GetStartAddress () const;

    lldb::SBAddress
    GetEndAddress () const;

    bool
    IsValid () const;

    lldb::SBFileSpec
    GetFileSpec () const;

    uint32_t
    GetLine () const;

    uint32_t
    GetColumn () const;

    bool
    GetDescription (lldb::SBStream &description);
};

} // namespace lldb

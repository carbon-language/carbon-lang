//===-- SWIG Interface for SBDeclaration --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

    %feature("docstring",
    "Specifies an association with a line and column for a variable."
    ) SBDeclaration;
    class SBDeclaration
    {
        public:

        SBDeclaration ();

        SBDeclaration (const lldb::SBDeclaration &rhs);

        ~SBDeclaration ();

        bool
        IsValid () const;

        explicit operator bool() const;

        lldb::SBFileSpec
        GetFileSpec () const;

        uint32_t
        GetLine () const;

        uint32_t
        GetColumn () const;

        bool
        GetDescription (lldb::SBStream &description);

        void
        SetFileSpec (lldb::SBFileSpec filespec);

        void
        SetLine (uint32_t line);

        void
        SetColumn (uint32_t column);

        bool
        operator == (const lldb::SBDeclaration &rhs) const;

        bool
        operator != (const lldb::SBDeclaration &rhs) const;

        STRING_EXTENSION(SBDeclaration)

#ifdef SWIGPYTHON
        %pythoncode %{
            file = property(GetFileSpec, None, doc='''A read only property that returns an lldb object that represents the file (lldb.SBFileSpec) for this line entry.''')
            line = property(GetLine, None, doc='''A read only property that returns the 1 based line number for this line entry, a return value of zero indicates that no line information is available.''')
            column = property(GetColumn, None, doc='''A read only property that returns the 1 based column number for this line entry, a return value of zero indicates that no column information is available.''')
        %}
#endif
    };

} // namespace lldb

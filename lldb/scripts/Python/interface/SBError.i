//===-- SWIG Interface for SBError ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBError {
public:
    SBError ();

    SBError (const lldb::SBError &rhs);

    ~SBError();

    const char *
    GetCString () const;

    void
    Clear ();

    bool
    Fail () const;

    bool
    Success () const;

    uint32_t
    GetError () const;

    lldb::ErrorType
    GetType () const;

    void
    SetError (uint32_t err, lldb::ErrorType type);

    void
    SetErrorToErrno ();

    void
    SetErrorToGenericError ();

    void
    SetErrorString (const char *err_str);

    int
    SetErrorStringWithFormat (const char *format, ...);

    bool
    IsValid () const;

    bool
    GetDescription (lldb::SBStream &description);
};

} // namespace lldb

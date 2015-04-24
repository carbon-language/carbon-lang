//===-- SWIG Interface for SBUnixSignals ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Allows you to manipulate LLDB's signal disposition"
) SBUnixSignals;
class SBUnixSignals
{
public:
    SBUnixSignals ();

    SBUnixSignals (const lldb::SBUnixSignals &rhs);

    ~SBUnixSignals();

    void
    Clear ();

    bool
    IsValid () const;

    const char *
    GetSignalAsCString (int32_t signo) const;

    int32_t
    GetSignalNumberFromName (const char *name) const;

    bool
    GetShouldSuppress (int32_t signo) const;

    bool
    SetShouldSuppress (int32_t signo,
                       bool value);

    bool
    GetShouldStop (int32_t signo) const;

    bool
    SetShouldStop (int32_t signo,
                   bool value);

    bool
    GetShouldNotify (int32_t signo) const;

    bool
    SetShouldNotify (int32_t signo, bool value);

    int32_t
    GetNumSignals () const;

    int32_t
    GetSignalAtIndex (int32_t index) const;

    %pythoncode %{
        def get_unix_signals_list(self):
            signals = []
            for idx in range(0, self.GetNumSignals()):
                signals.append(self.GetSignalAtIndex(sig))
            return signals

        __swig_getmethods__["signals"] = get_unix_signals_list
        if _newclass: threads = property(get_unix_signals_list, None, doc='''A read only property that returns a list() of valid signal numbers for this platform.''')
    %}
};

}  // namespace lldb

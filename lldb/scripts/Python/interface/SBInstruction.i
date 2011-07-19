//===-- SWIG Interface for SBInstruction ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

// There's a lot to be fixed here, but need to wait for underlying insn implementation
// to be revised & settle down first.

namespace lldb {

class SBInstruction
{
public:

    SBInstruction ();

    SBInstruction (const SBInstruction &rhs);
    
    ~SBInstruction ();

    bool
    IsValid();

    SBAddress
    GetAddress();

    size_t
    GetByteSize ();

    bool
    DoesBranch ();

    void
    Print (FILE *out);

    bool
    GetDescription (lldb::SBStream &description);

    bool
    EmulateWithFrame (lldb::SBFrame &frame, uint32_t evaluate_options);

    bool
    DumpEmulation (const char * triple); // triple is to specify the architecture, e.g. 'armv6' or 'arm-apple-darwin'
    
    bool
    TestEmulation (lldb::SBStream &output_stream, const char *test_file);
};

} // namespace lldb

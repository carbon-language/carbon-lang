//===-- SBInstruction.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBInstruction_h_
#define LLDB_SBInstruction_h_

#include "lldb/API/SBDefines.h"

#include <stdio.h>

// There's a lot to be fixed here, but need to wait for underlying insn implementation
// to be revised & settle down first.

namespace lldb {

class SBInstruction
{
public:

    SBInstruction ();

    SBInstruction (const SBInstruction &rhs);
    
#ifndef SWIG
    const SBInstruction &
    operator = (const SBInstruction &rhs);
#endif

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
    EmulateWithFrame (lldb::SBFrame &frame, bool auto_advance_pc);

    bool
    DumpEmulation (const char * triple); // triple is to specify the architecture, e.g. 'armv6' or 'arm-apple-darwin'
    
    bool
    TestEmulation (lldb::SBStream &output_stream, const char *test_file);

protected:
    friend class SBInstructionList;

    SBInstruction (const lldb::InstructionSP &inst_sp);

    void
    SetOpaque (const lldb::InstructionSP &inst_sp);

private:

    lldb::InstructionSP  m_opaque_sp;
};


} // namespace lldb

#endif // LLDB_SBInstruction_h_

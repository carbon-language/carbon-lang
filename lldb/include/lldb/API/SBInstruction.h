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

//class lldb_private::Disassembler::Instruction;

namespace lldb {

class SBInstruction
{
public:

    //SBInstruction (lldb_private::Disassembler::Instruction *lldb_insn);

    SBInstruction ();

    ~SBInstruction ();

    //bool
    //IsValid();

    //size_t
    //GetByteSize ();

    //void
    //SetByteSize (size_t byte_size);

    //bool
    //DoesBranch ();

    void
    Print (FILE *out);

    //bool
    //GetDescription (lldb::SBStream &description);

private:

    //lldb_private::Disassembler::Instruction::SharedPtr  m_opaque_sp;


};


} // namespace lldb

#endif // LLDB_SBInstruction_h_

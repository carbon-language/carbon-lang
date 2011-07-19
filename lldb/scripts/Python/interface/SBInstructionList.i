//===-- SWIG Interface for SBInstructionList --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

namespace lldb {

class SBInstructionList
{
public:

    SBInstructionList ();

    SBInstructionList (const SBInstructionList &rhs);
    
    ~SBInstructionList ();

    bool
    IsValid () const;

    size_t
    GetSize ();

    lldb::SBInstruction
    GetInstructionAtIndex (uint32_t idx);

    void
    Clear ();

    void
    AppendInstruction (lldb::SBInstruction inst);

    void
    Print (FILE *out);

    bool
    GetDescription (lldb::SBStream &description);
    
    bool
    DumpEmulationForAllInstructions (const char *triple);
};

} // namespace lldb

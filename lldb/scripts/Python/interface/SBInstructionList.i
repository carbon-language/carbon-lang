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

%feature("docstring",
"Represents a list of machine instructions.  SBFunction and SBSymbol have
GetInstructions() methods which return SBInstructionList instances.

SBInstructionList supports instruction (SBInstruction instance) iteration.
For example (see also SBDebugger for a more complete example),

def disassemble_instructions (insts):
    for i in insts:
        print i

defines a function which takes an SBInstructionList instance and prints out
the machine instructions in assembly format."
) SBInstructionList;
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

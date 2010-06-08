//===-- SBInstructionList.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBInstructionList_h_
#define LLDB_SBInstructionList_h_

#include <LLDB/SBDefines.h>

namespace lldb {

class SBInstructionList
{
public:

    SBInstructionList ();

    ~SBInstructionList ();

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

private:

    // If we have an instruction list, it will need to be backed by an
    // lldb_private class that contains the list, we can't inherit from
    // std::vector here...
    //std::vector <SBInstruction> m_insn_list;

};


} // namespace lldb

#endif // LLDB_SBInstructionList_h_

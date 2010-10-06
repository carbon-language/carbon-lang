//===-- SBInstructionList.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBInstructionList.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Stream.h"

using namespace lldb;
using namespace lldb_private;


SBInstructionList::SBInstructionList () :
    m_opaque_sp()
{
}

SBInstructionList::~SBInstructionList ()
{
}

size_t
SBInstructionList::GetSize ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetInstructionList().GetSize();
    return 0;
}

SBInstruction
SBInstructionList::GetInstructionAtIndex (uint32_t idx)
{
    SBInstruction inst;
    if (m_opaque_sp && idx < m_opaque_sp->GetInstructionList().GetSize())
        inst.SetOpaque (m_opaque_sp->GetInstructionList().GetInstructionAtIndex (idx));
    return inst;
}

void
SBInstructionList::Clear ()
{
    m_opaque_sp.reset();
}

void
SBInstructionList::AppendInstruction (SBInstruction insn)
{
}

void
SBInstructionList::SetDisassembler (const lldb::DisassemblerSP &opaque_sp)
{
    m_opaque_sp = opaque_sp;
}

void
SBInstructionList::Print (FILE *out)
{
    if (out == NULL)
        return;
}


bool
SBInstructionList::GetDescription (lldb::SBStream &description)
{
    if (m_opaque_sp)
    {
        size_t num_instructions = GetSize ();
        if (num_instructions)
        {
            // Call the ref() to make sure a stream is created if one deesn't 
            // exist already inside description...
            Stream &sref = description.ref();
            for (size_t i=0; i<num_instructions; ++i)
            {
                Instruction *inst = m_opaque_sp->GetInstructionList().GetInstructionAtIndex (i).get();
                if (inst == NULL)
                    break;
                inst->Dump (&sref, true, NULL, 0, NULL, false);
                sref.EOL();
            }
            return true;
        }
    }
    return false;
}



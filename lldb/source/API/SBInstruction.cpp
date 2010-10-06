//===-- SBInstruction.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBInstruction.h"

#include "lldb/API/SBAddress.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/API/SBStream.h"

#include "lldb/Core/Disassembler.h"
#include "lldb/Core/StreamFile.h"

using namespace lldb;
using namespace lldb_private;

SBInstruction::SBInstruction ()
{
}

SBInstruction::SBInstruction (const lldb::InstructionSP& inst_sp) :
    m_opaque_sp (inst_sp)
{
}

SBInstruction::~SBInstruction ()
{
}

bool
SBInstruction::IsValid()
{
    return (m_opaque_sp.get() != NULL);
}

SBAddress
SBInstruction::GetAddress()
{
    SBAddress sb_addr;
    if (m_opaque_sp && m_opaque_sp->GetAddress().IsValid())
        sb_addr.SetAddress(&m_opaque_sp->GetAddress());
    return sb_addr;
}

size_t
SBInstruction::GetByteSize ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetByteSize();
    return 0;
}

bool
SBInstruction::DoesBranch ()
{
    if (m_opaque_sp)
        return m_opaque_sp->DoesBranch ();
    return false;
}

void
SBInstruction::SetOpaque (const lldb::InstructionSP &inst_sp)
{
    m_opaque_sp = inst_sp;
}

bool
SBInstruction::GetDescription (lldb::SBStream &s)
{
    if (m_opaque_sp)
    {
        // Use the "ref()" instead of the "get()" accessor in case the SBStream 
        // didn't have a stream already created, one will get created...
        m_opaque_sp->Dump (&s.ref(), true, NULL, 0, NULL, false);
        return true;
    }
    return false;
}

void
SBInstruction::Print (FILE *out)
{
    if (out == NULL)
        return;

    if (m_opaque_sp)
    {
        StreamFile out_stream (out);
        m_opaque_sp->Dump (&out_stream, true, NULL, 0, NULL, false);
    }
}

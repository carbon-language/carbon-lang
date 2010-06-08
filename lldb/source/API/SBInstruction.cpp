//===-- SBInstruction.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBInstruction.h"

#include "lldb/Core/Disassembler.h"

using namespace lldb;
using namespace lldb_private;

//SBInstruction::SBInstruction (lldb_private::Disassembler::Instruction *lldb_insn) :
//    m_lldb_object_sp (lldb_insn);
//{
//}

SBInstruction::SBInstruction ()
{
}

SBInstruction::~SBInstruction ()
{
}

//bool
//SBInstruction::IsValid()
//{
//    return (m_lldb_object_sp.get() != NULL);
//}

//size_t
//SBInstruction::GetByteSize ()
//{
//    if (IsValid())
//    {
//        return m_lldb_object_sp->GetByteSize();
//    }
//    return 0;
//}

//void
//SBInstruction::SetByteSize (size_T byte_size)
//{
//    if (IsValid ())
//    {
//        m_lldb_object_sp->SetByteSize (byte_size);
//    }
//}

//bool
//SBInstruction::DoesBranch ()
//{
//    if (IsValid ())
//    {
//        return m_lldb_object_sp->DoesBranch ();
//    }
//    return false;
//}

void
SBInstruction::Print (FILE *out)
{
    if (out == NULL)
        return;

    //StreamFile out_strem (out);

    //m_lldb_object_sp->Dump (out, LLDB_INVALID_ADDRESS, NULL, 0);
}

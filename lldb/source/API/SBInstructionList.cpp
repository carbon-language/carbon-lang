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

using namespace lldb;


SBInstructionList::SBInstructionList ()
{
}

SBInstructionList::~SBInstructionList ()
{
}

size_t
SBInstructionList::GetSize ()
{
    return 0;
}

SBInstruction
SBInstructionList::GetInstructionAtIndex (uint32_t idx)
{
    SBInstruction inst;
    return inst;
}

void
SBInstructionList::Clear ()
{
}

void
SBInstructionList::AppendInstruction (SBInstruction insn)
{
}

void
SBInstructionList::Print (FILE *out)
{
    if (out == NULL)
        return;
}


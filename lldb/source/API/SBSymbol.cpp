//===-- SBSymbol.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

SBSymbol::SBSymbol () :
    m_opaque_ptr (NULL)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBSymbol::SBSymbol () ==> this = %p", this);
}

SBSymbol::SBSymbol (lldb_private::Symbol *lldb_object_ptr) :
    m_opaque_ptr (lldb_object_ptr)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
    {
        SBStream sstr;
        GetDescription (sstr);
        log->Printf ("SBSymbol::SBSymbol (lldb_private::Symbol *lldb_object_ptr) lldb_object_ptr = %p ==> "
                     "this = %p (%s)", lldb_object_ptr, this, sstr.GetData());
    }
}

SBSymbol::~SBSymbol ()
{
    m_opaque_ptr = NULL;
}

bool
SBSymbol::IsValid () const
{
    return m_opaque_ptr != NULL;
}

const char *
SBSymbol::GetName() const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBSymbol::GetName ()");

    if (m_opaque_ptr)
    {
        if (log)
            log->Printf ("SBSymbol::GetName ==> %s", m_opaque_ptr->GetMangled().GetName().AsCString());
        return m_opaque_ptr->GetMangled().GetName().AsCString();
    }
    
    if (log)
        log->Printf ("SBSymbol::GetName ==> NULL");

    return NULL;
}

const char *
SBSymbol::GetMangledName () const
{
    if (m_opaque_ptr)
        return m_opaque_ptr->GetMangled().GetMangledName().AsCString();
    return NULL;
}


bool
SBSymbol::operator == (const SBSymbol &rhs) const
{
    return m_opaque_ptr == rhs.m_opaque_ptr;
}

bool
SBSymbol::operator != (const SBSymbol &rhs) const
{
    return m_opaque_ptr != rhs.m_opaque_ptr;
}

bool
SBSymbol::GetDescription (SBStream &description)
{
    if (m_opaque_ptr)
    {
        description.ref();
        m_opaque_ptr->GetDescription (description.get(), 
                                      lldb::eDescriptionLevelFull, NULL);
    }
    else
        description.Printf ("No value");
    
    return true;
}



SBInstructionList
SBSymbol::GetInstructions (SBTarget target)
{
    SBInstructionList sb_instructions;
    if (m_opaque_ptr)
    {
        ExecutionContext exe_ctx;
        if (target.IsValid())
            target->CalculateExecutionContext (exe_ctx);
        const AddressRange *symbol_range = m_opaque_ptr->GetAddressRangePtr();
        if (symbol_range)
        {
            Module *module = symbol_range->GetBaseAddress().GetModule();
            if (module)
            {
                sb_instructions.SetDisassembler (Disassembler::DisassembleRange (module->GetArchitecture (),
                                                                                 exe_ctx,
                                                                                 *symbol_range));
            }
        }
    }
    return sb_instructions;
}


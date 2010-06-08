//===-- SBModule.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBModule.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/Core/Module.h"

using namespace lldb;


SBModule::SBModule () :
    m_lldb_object_sp ()
{
}

SBModule::SBModule (const lldb::ModuleSP& module_sp) :
    m_lldb_object_sp (module_sp)
{
}

SBModule::~SBModule ()
{
}

bool
SBModule::IsValid () const
{
    return m_lldb_object_sp.get() != NULL;
}

SBFileSpec
SBModule::GetFileSpec () const
{
    SBFileSpec file_spec;
    if (m_lldb_object_sp)
        file_spec.SetFileSpec(m_lldb_object_sp->GetFileSpec());
    return file_spec;
}

const uint8_t *
SBModule::GetUUIDBytes () const
{
    if (m_lldb_object_sp)
        return (const uint8_t *)m_lldb_object_sp->GetUUID().GetBytes();
    return NULL;
}


bool
SBModule::operator == (const SBModule &rhs) const
{
    if (m_lldb_object_sp)
        return m_lldb_object_sp.get() == rhs.m_lldb_object_sp.get();
    return false;
}

bool
SBModule::operator != (const SBModule &rhs) const
{
    if (m_lldb_object_sp)
        return m_lldb_object_sp.get() != rhs.m_lldb_object_sp.get();
    return false;
}

lldb::ModuleSP &
SBModule::operator *()
{
    return m_lldb_object_sp;
}

lldb_private::Module *
SBModule::operator ->()
{
    return m_lldb_object_sp.get();
}

const lldb_private::Module *
SBModule::operator ->() const
{
    return m_lldb_object_sp.get();
}

lldb_private::Module *
SBModule::get()
{
    return m_lldb_object_sp.get();
}

const lldb_private::Module *
SBModule::get() const
{
    return m_lldb_object_sp.get();
}


void
SBModule::SetModule (const lldb::ModuleSP& module_sp)
{
    m_lldb_object_sp = module_sp;
}


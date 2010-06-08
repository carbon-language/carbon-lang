//===-- ModuleChild.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ModuleChild.h"

using namespace lldb_private;

ModuleChild::ModuleChild (Module* module) :
    m_module(module)
{
}

ModuleChild::ModuleChild (const ModuleChild& rhs) :
    m_module(rhs.m_module)
{
}

ModuleChild::~ModuleChild()
{
}

const ModuleChild&
ModuleChild::operator= (const ModuleChild& rhs)
{
    if (this != &rhs)
        m_module = rhs.m_module;
    return *this;
}

Module *
ModuleChild::GetModule ()
{
    return m_module;
}

Module *
ModuleChild::GetModule () const
{
    return m_module;
}

void
ModuleChild::SetModule (Module *module)
{
    m_module = module;
}

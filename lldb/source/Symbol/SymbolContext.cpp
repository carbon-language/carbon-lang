//===-- SymbolContext.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Target.h"
#include "lldb/Symbol/SymbolVendor.h"

using namespace lldb;
using namespace lldb_private;

SymbolContext::SymbolContext() :
    target_sp   (),
    module_sp   (),
    comp_unit   (NULL),
    function    (NULL),
    block       (NULL),
    line_entry  (),
    symbol      (NULL)
{
}

SymbolContext::SymbolContext(const ModuleSP& m, CompileUnit *cu, Function *f, Block *b, LineEntry *le, Symbol *s) :
    target_sp   (),
    module_sp   (m),
    comp_unit   (cu),
    function    (f),
    block       (b),
    line_entry  (),
    symbol      (s)
{
    if (le)
        line_entry = *le;
}

SymbolContext::SymbolContext(const TargetSP &t, const ModuleSP& m, CompileUnit *cu, Function *f, Block *b, LineEntry *le, Symbol *s) :
    target_sp   (t),
    module_sp   (m),
    comp_unit   (cu),
    function    (f),
    block       (b),
    line_entry  (),
    symbol      (s)
{
    if (le)
        line_entry = *le;
}

SymbolContext::SymbolContext(const SymbolContext& rhs) :
    target_sp   (rhs.target_sp),
    module_sp   (rhs.module_sp),
    comp_unit   (rhs.comp_unit),
    function    (rhs.function),
    block       (rhs.block),
    line_entry  (rhs.line_entry),
    symbol      (rhs.symbol)
{
}


SymbolContext::SymbolContext (SymbolContextScope *sc_scope) :
    target_sp   (),
    module_sp   (),
    comp_unit   (NULL),
    function    (NULL),
    block       (NULL),
    line_entry  (),
    symbol      (NULL)
{
    sc_scope->CalculateSymbolContext (this);
}

const SymbolContext&
SymbolContext::operator= (const SymbolContext& rhs)
{
    if (this != &rhs)
    {
        target_sp   = rhs.target_sp;
        module_sp   = rhs.module_sp;
        comp_unit   = rhs.comp_unit;
        function    = rhs.function;
        block       = rhs.block;
        line_entry  = rhs.line_entry;
        symbol      = rhs.symbol;
    }
    return *this;
}

void
SymbolContext::Clear()
{
    target_sp.reset();
    module_sp.reset();
    comp_unit   = NULL;
    function    = NULL;
    block       = NULL;
    line_entry.Clear();
    symbol      = NULL;
}

void
SymbolContext::DumpStopContext
(
    Stream *s,
    ExecutionContextScope *exe_scope,
    const Address &addr,
    bool show_module
) const
{
    Process *process = NULL;
    if (exe_scope)
        process = exe_scope->CalculateProcess();
    addr_t load_addr = addr.GetLoadAddress (process);

    if (show_module && module_sp)
    {
        *s << module_sp->GetFileSpec().GetFilename() << '`';
    }

    if (function != NULL)
    {
        if (function->GetMangled().GetName())
            function->GetMangled().GetName().Dump(s);

        const addr_t func_load_addr = function->GetAddressRange().GetBaseAddress().GetLoadAddress(process);
        if (load_addr > func_load_addr)
            s->Printf(" + %llu", load_addr - func_load_addr);

        if (block != NULL)
        {
            s->IndentMore();
            block->DumpStopContext(s, this);
            s->IndentLess();
        }
        else
        {
            if (line_entry.IsValid())
            {
                s->PutCString(" at ");
                if (line_entry.DumpStopContext(s))
                    return;
            }
        }
    }
    else if (symbol != NULL)
    {
        symbol->GetMangled().GetName().Dump(s);

        if (symbol->GetAddressRangePtr())
        {
            const addr_t sym_load_addr = symbol->GetAddressRangePtr()->GetBaseAddress().GetLoadAddress(process);
            if (load_addr > sym_load_addr)
                s->Printf(" + %llu", load_addr - sym_load_addr);
        }
    }
    else
    {
        addr.Dump(s, exe_scope, Address::DumpStyleModuleWithFileAddress);
    }
}

void
SymbolContext::Dump(Stream *s, Process *process) const
{
    *s << (void *)this << ": ";
    s->Indent();
    s->PutCString("SymbolContext");
    s->IndentMore();
    s->EOL();
    s->IndentMore();
    s->Indent();
    *s << "Module    = " << (void *)module_sp.get() << ' ';
    if (module_sp)
        module_sp->GetFileSpec().Dump(s);
    s->EOL();
    s->Indent();
    *s << "CompileUnit  = " << (void *)comp_unit;
    if (comp_unit != NULL)
        *s << " {" << comp_unit->GetID() << "} " << *(dynamic_cast<FileSpec*> (comp_unit));
    s->EOL();
    s->Indent();
    *s << "Function  = " << (void *)function;
    if (function != NULL)
    {
        *s << " {" << function->GetID() << "} ";/// << function->GetType()->GetName();
//      Type* func_type = function->Type();
//      if (func_type)
//      {
//          s->EOL();
//          const UserDefType* func_udt = func_type->GetUserDefinedType().get();
//          if (func_udt)
//          {
//              s->IndentMore();
//              func_udt->Dump(s, func_type);
//              s->IndentLess();
//          }
//      }
    }
    s->EOL();
    s->Indent();
    *s << "Block     = " << (void *)block;
    if (block != NULL)
        *s << " {" << block->GetID() << '}';
    // Dump the block and pass it a negative depth to we print all the parent blocks
    //if (block != NULL)
    //  block->Dump(s, function->GetFileAddress(), INT_MIN);
    s->EOL();
    s->Indent();
    *s << "LineEntry = ";
    line_entry.Dump (s, process, true, Address::DumpStyleLoadAddress, Address::DumpStyleModuleWithFileAddress, true);
    s->EOL();
    s->Indent();
    *s << "Symbol    = " << (void *)symbol;
    if (symbol != NULL && symbol->GetMangled())
        *s << ' ' << symbol->GetMangled().GetName().AsCString();
    s->EOL();
    s->IndentLess();
    s->IndentLess();
}

bool
lldb_private::operator== (const SymbolContext& lhs, const SymbolContext& rhs)
{
    return lhs.target_sp.get() == rhs.target_sp.get() &&
           lhs.module_sp.get() == rhs.module_sp.get() &&
           lhs.comp_unit    == rhs.comp_unit &&
           lhs.function     == rhs.function &&
           LineEntry::Compare(lhs.line_entry, rhs.line_entry) == 0 &&
           lhs.symbol       == rhs.symbol;
}

bool
lldb_private::operator!= (const SymbolContext& lhs, const SymbolContext& rhs)
{
    return lhs.target_sp.get() != rhs.target_sp.get() ||
           lhs.module_sp.get() != rhs.module_sp.get() ||
           lhs.comp_unit    != rhs.comp_unit ||
           lhs.function     != rhs.function ||
           LineEntry::Compare(lhs.line_entry, rhs.line_entry) != 0 ||
           lhs.symbol       != rhs.symbol;
}

bool
SymbolContext::GetAddressRange (uint32_t scope, AddressRange &range) const
{
    if ((scope & eSymbolContextLineEntry) && line_entry.IsValid())
    {
        range = line_entry.range;
        return true;
    }
    else if ((scope & eSymbolContextFunction) && function != NULL)
    {
        range = function->GetAddressRange();
        return true;
    }
    else if ((scope & eSymbolContextSymbol) && symbol != NULL && symbol->GetAddressRangePtr())
    {
        range = *symbol->GetAddressRangePtr();

        if (range.GetByteSize() == 0)
        {
            if (module_sp)
            {
                ObjectFile *objfile = module_sp->GetObjectFile();
                if (objfile)
                {
                    Symtab *symtab = objfile->GetSymtab();
                    if (symtab)
                        range.SetByteSize(symtab->CalculateSymbolSize (symbol));
                }
            }
        }
        return true;
    }
    range.Clear();
    return false;
}


Function *
SymbolContext::FindFunctionByName (const char *name) const
{
    ConstString name_const_str (name);
    if (function != NULL)
    {
        // FIXME: Look in the class of the current function, if it exists,
        // for methods matching name.
    }

    //
    if (comp_unit != NULL)
    {
        // Make sure we've read in all the functions.  We should be able to check and see
        // if there's one by this name present before we do this...
        module_sp->GetSymbolVendor()->ParseCompileUnitFunctions(*this);
        uint32_t func_idx;
        lldb::FunctionSP func_sp;
        for (func_idx = 0; (func_sp = comp_unit->GetFunctionAtIndex(func_idx)) != NULL; ++func_idx)
        {
            if (func_sp->GetMangled().GetName() == name_const_str)
                return func_sp.get();
        }
    }
    if (module_sp != NULL)
    {
        SymbolContextList sc_matches;
        if (module_sp->FindFunctions (name_const_str, false, sc_matches) > 0)
        {
            SymbolContext sc;
            sc_matches.GetContextAtIndex (0, sc);
            return sc.function;
        }
    }

    if (target_sp)
    {
        SymbolContextList sc_matches;
        if (target_sp->GetImages().FindFunctions (name_const_str, sc_matches) > 0)
        {
            SymbolContext sc;
            sc_matches.GetContextAtIndex (0, sc);
            return sc.function;
        }
    }

    return NULL;
}

lldb::VariableSP
SymbolContext::FindVariableByName (const char *name) const
{
    lldb::VariableSP return_value;
    return return_value;
}

lldb::TypeSP
SymbolContext::FindTypeByName (const char *name) const
{
    lldb::TypeSP return_value;
    return return_value;
}

//----------------------------------------------------------------------
//
//  SymbolContextList
//
//----------------------------------------------------------------------


SymbolContextList::SymbolContextList() :
    m_symbol_contexts()
{
}

SymbolContextList::~SymbolContextList()
{
}

void
SymbolContextList::Append(const SymbolContext& sc)
{
    m_symbol_contexts.push_back(sc);
}

void
SymbolContextList::Clear()
{
    m_symbol_contexts.clear();
}

void
SymbolContextList::Dump(Stream *s, Process *process) const
{

    *s << (void *)this << ": ";
    s->Indent();
    s->PutCString("SymbolContextList");
    s->EOL();
    s->IndentMore();

    collection::const_iterator pos, end = m_symbol_contexts.end();
    for (pos = m_symbol_contexts.begin(); pos != end; ++pos)
    {
        pos->Dump(s, process);
    }
    s->IndentLess();
}

bool
SymbolContextList::GetContextAtIndex(uint32_t idx, SymbolContext& sc) const
{
    if (idx < m_symbol_contexts.size())
    {
        sc = m_symbol_contexts[idx];
        return true;
    }
    return false;
}

bool
SymbolContextList::RemoveContextAtIndex (uint32_t idx)
{
    if (idx < m_symbol_contexts.size())
    {
        m_symbol_contexts.erase(m_symbol_contexts.begin() + idx);
        return true;
    }
    return false;
}

uint32_t
SymbolContextList::GetSize() const
{
    return m_symbol_contexts.size();
}

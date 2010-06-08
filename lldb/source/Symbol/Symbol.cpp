//===-- Symbol.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/Symbol.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;


Symbol::Symbol() :
    UserID (),
    m_mangled (),
    m_type (eSymbolTypeInvalid),
    m_type_data (0),
    m_type_data_resolved (false),
    m_is_synthetic (false),
    m_is_debug (false),
    m_is_external (false),
    m_size_is_sibling (false),
    m_size_is_synthesized (false),
    m_searched_for_function (false),
    m_addr_range (),
    m_flags (),
    m_function (NULL)
{
}

Symbol::Symbol
(
    user_id_t symID,
    const char *name,
    bool name_is_mangled,
    SymbolType type,
    bool external,
    bool is_debug,
    bool is_trampoline,
    bool is_artificial,
    const Section* section,
    addr_t offset,
    uint32_t size,
    uint32_t flags
) :
    UserID (symID),
    m_mangled (name, name_is_mangled),
    m_type (type),
    m_type_data (0),
    m_type_data_resolved (false),
    m_is_synthetic (is_artificial),
    m_is_debug (is_debug),
    m_is_external (external),
    m_size_is_sibling (false),
    m_size_is_synthesized (false),
    m_searched_for_function (false),
    m_addr_range (section, offset, size),
    m_flags (flags),
    m_function (NULL)
{
}

Symbol::Symbol
(
    user_id_t symID,
    const char *name,
    bool name_is_mangled,
    SymbolType type,
    bool external,
    bool is_debug,
    bool is_trampoline,
    bool is_artificial,
    const AddressRange &range,
    uint32_t flags
) :
    UserID (symID),
    m_mangled (name, name_is_mangled),
    m_type (type),
    m_type_data (0),
    m_type_data_resolved (false),
    m_is_synthetic (is_artificial),
    m_is_debug (is_debug),
    m_is_external (external),
    m_size_is_sibling (false),
    m_size_is_synthesized (false),
    m_searched_for_function (false),
    m_addr_range (range),
    m_flags (flags),
    m_function (NULL)
{
}

Symbol::Symbol(const Symbol& rhs):
    UserID (rhs),
    m_mangled (rhs.m_mangled),
    m_type (rhs.m_type),
    m_type_data (rhs.m_type_data),
    m_type_data_resolved (rhs.m_type_data_resolved),
    m_is_synthetic (rhs.m_is_synthetic),
    m_is_debug (rhs.m_is_debug),
    m_is_external (rhs.m_is_external),
    m_size_is_sibling (rhs.m_size_is_sibling),
    m_size_is_synthesized (false),
    m_searched_for_function (false),
    m_addr_range (rhs.m_addr_range),
    m_flags (rhs.m_flags),
    m_function (NULL)
{
}

const Symbol&
Symbol::operator= (const Symbol& rhs)
{
    if (this != &rhs)
    {
        UserID::operator= (rhs);
        m_mangled = rhs.m_mangled;
        m_type = rhs.m_type;
        m_type_data = rhs.m_type_data;
        m_type_data_resolved = rhs.m_type_data_resolved;
        m_is_synthetic = rhs.m_is_synthetic;
        m_is_debug = rhs.m_is_debug;
        m_is_external = rhs.m_is_external;
        m_size_is_sibling = rhs.m_size_is_sibling;
        m_size_is_synthesized = rhs.m_size_is_sibling;
        m_searched_for_function = rhs.m_searched_for_function;
        m_addr_range = rhs.m_addr_range;
        m_flags = rhs.m_flags;
        m_function = rhs.m_function;
    }
    return *this;
}

AddressRange &
Symbol::GetAddressRangeRef()
{
    return m_addr_range;
}

const AddressRange &
Symbol::GetAddressRangeRef() const
{
    return m_addr_range;
}

AddressRange *
Symbol::GetAddressRangePtr()
{
    if (m_addr_range.GetBaseAddress().GetSection())
        return &m_addr_range;
    return NULL;
}

const AddressRange *
Symbol::GetAddressRangePtr() const
{
    if (m_addr_range.GetBaseAddress().GetSection())
        return &m_addr_range;
    return NULL;
}

bool
Symbol::GetSizeIsSibling() const
{
    return m_size_is_sibling;
}

bool
Symbol::GetSizeIsSynthesized() const
{
    return m_size_is_synthesized;
}

uint32_t
Symbol::GetSiblingIndex() const
{
    return m_size_is_sibling ? m_addr_range.GetByteSize() : 0;
}

uint32_t
Symbol::GetFlags() const
{
    return m_flags;
}

void
Symbol::SetFlags (uint32_t flags)
{
    m_flags = flags;
}

SymbolType
Symbol::GetType() const
{
    return m_type;
}

void
Symbol::SetType(SymbolType type)
{
    m_type = type;
}

bool
Symbol::IsSynthetic () const
{
    return m_is_synthetic;
}

void
Symbol::SetIsSynthetic (bool b)
{
    m_is_synthetic = b;
}

void
Symbol::SetSizeIsSynthesized(bool b)
{
    m_size_is_synthesized = b;
}


bool
Symbol::IsDebug() const
{
    return m_is_debug;
}

void
Symbol::SetDebug (bool b)
{
    m_is_debug = b;
}

bool
Symbol::IsExternal() const
{
    return m_is_external;
}

void
Symbol::SetExternal(bool b)
{
    m_is_external = b;
}

bool
Symbol::IsTrampoline () const
{
    return m_type == eSymbolTypeTrampoline;
}

uint32_t
Symbol::GetByteSize() const
{
    return m_addr_range.GetByteSize();
}

void
Symbol::SetByteSize (uint32_t size)
{
    m_addr_range.SetByteSize(size);
}

void
Symbol::SetSizeIsSibling (bool b)
{
    m_size_is_sibling = b;
}

void
Symbol::Dump(Stream *s, Process *process, uint32_t index) const
{
//  s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
//  s->Indent();
//  s->Printf("Symbol[%5u] %6u %c%c %-12s ",
    s->Printf("[%5u] %6u %c%c%c %-12s ",
              index,
              GetID(),
              m_is_debug ? 'D' : ' ',
              m_is_synthetic ? 'S' : ' ',
              m_is_external ? 'X' : ' ',
              GetTypeAsString());

    const Section *section = m_addr_range.GetBaseAddress().GetSection();
    if (section != NULL)
    {
        if (!m_addr_range.GetBaseAddress().Dump(s, NULL, Address::DumpStyleFileAddress))
            s->Printf("%*s", 18, "");

        s->PutChar(' ');

        if (!m_addr_range.GetBaseAddress().Dump(s, process, Address::DumpStyleLoadAddress))
            s->Printf("%*s", 18, "");

        const char *format = m_size_is_sibling ?
                            " Sibling -> [%5llu] 0x%8.8x %s\n":
                            " 0x%16.16llx 0x%8.8x %s\n";
        s->Printf(  format,
                    m_addr_range.GetByteSize(),
                    m_flags,
                    m_mangled.GetName().AsCString(""));
    }
    else
    {
        const char *format = m_size_is_sibling ?
                            "0x%16.16llx                    Sibling -> [%5llu] 0x%8.8x %s\n":
                            "0x%16.16llx                    0x%16.16llx 0x%8.8x %s\n";
        s->Printf(  format,
                    m_addr_range.GetBaseAddress().GetOffset(),
                    m_addr_range.GetByteSize(),
                    m_flags,
                    m_mangled.GetName().AsCString(""));
    }
}

const Mangled&
Symbol::GetMangled() const
{
    return m_mangled;
}

Mangled&
Symbol::GetMangled()
{
    return m_mangled;
}

Address &
Symbol::GetValue()
{
    return m_addr_range.GetBaseAddress();
}

const Address &
Symbol::GetValue() const
{
    return m_addr_range.GetBaseAddress();
}

void
Symbol::SetValue (Address &value)
{
    m_addr_range.GetBaseAddress() = value;
}

Function *
Symbol::GetFunction ()
{
    if (m_function == NULL && !m_searched_for_function)
    {
        m_searched_for_function = true;
        Module *module = m_addr_range.GetBaseAddress().GetModule();
        if (module)
        {
            SymbolContext sc;
            if (module->ResolveSymbolContextForAddress(m_addr_range.GetBaseAddress(), eSymbolContextFunction, sc))
                m_function = sc.function;
        }
    }
    return m_function;
}

uint32_t
Symbol::GetPrologueByteSize ()
{
    if (m_type == eSymbolTypeCode || m_type == eSymbolTypeFunction)
    {
        if (!m_type_data_resolved)
        {
            m_type_data_resolved = true;
            Module *module = m_addr_range.GetBaseAddress().GetModule();
            SymbolContext sc;
            if (module && module->ResolveSymbolContextForAddress (m_addr_range.GetBaseAddress(),
                                                                  eSymbolContextLineEntry,
                                                                  sc))
            {
                m_type_data = sc.line_entry.range.GetByteSize();
            }
            else
            {
                // TODO: expose something in Process to figure out the
                // size of a function prologue.
            }
        }
        return m_type_data;
    }
    return 0;
}

void
Symbol::SetValue (const AddressRange &range)
{
    m_addr_range = range;
}


void
Symbol::SetValue(addr_t value)
{
    m_addr_range.GetBaseAddress().SetSection(NULL);
    m_addr_range.GetBaseAddress().SetOffset(value);
}


bool
Symbol::Compare(const ConstString& name, SymbolType type) const
{
    if (m_type == eSymbolTypeAny || m_type == type)
        return m_mangled.GetMangledName() == name || m_mangled.GetDemangledName() == name;
    return false;
}

#define ENUM_TO_CSTRING(x)  case eSymbolType##x: return #x;

const char *
Symbol::GetTypeAsString() const
{
    switch (m_type)
    {
    ENUM_TO_CSTRING(Invalid);
    ENUM_TO_CSTRING(Absolute);
    ENUM_TO_CSTRING(Extern);
    ENUM_TO_CSTRING(Code);
    ENUM_TO_CSTRING(Data);
    ENUM_TO_CSTRING(Trampoline);
    ENUM_TO_CSTRING(Runtime);
    ENUM_TO_CSTRING(Exception);
    ENUM_TO_CSTRING(SourceFile);
    ENUM_TO_CSTRING(HeaderFile);
    ENUM_TO_CSTRING(ObjectFile);
    ENUM_TO_CSTRING(Function);
    ENUM_TO_CSTRING(FunctionEnd);
    ENUM_TO_CSTRING(CommonBlock);
    ENUM_TO_CSTRING(Block);
    ENUM_TO_CSTRING(Static);
    ENUM_TO_CSTRING(Global);
    ENUM_TO_CSTRING(Local);
    ENUM_TO_CSTRING(Param);
    ENUM_TO_CSTRING(Variable);
    ENUM_TO_CSTRING(VariableType);
    ENUM_TO_CSTRING(LineEntry);
    ENUM_TO_CSTRING(LineHeader);
    ENUM_TO_CSTRING(ScopeBegin);
    ENUM_TO_CSTRING(ScopeEnd);
    ENUM_TO_CSTRING(Additional);
    ENUM_TO_CSTRING(Compiler);
    ENUM_TO_CSTRING(Instrumentation);
    ENUM_TO_CSTRING(Undefined);
    default:
        break;
    }
    return "<unknown SymbolType>";
}


//===-- Address.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//static size_t
//ReadBytes (ExecutionContextScope *exe_scope, const Address &address, void *dst, size_t dst_len)
//{
//    if (exe_scope == NULL)
//        return 0;
//
//    lldb::AddressType addr_type = eAddressTypeInvalid;
//    addr_t addr = LLDB_INVALID_ADDRESS;
//
//    Process *process = exe_scope->CalculateProcess();
//
//    if (process && process->IsAlive())
//    {
//        addr = address.GetLoadAddress(process);
//        if (addr != LLDB_INVALID_ADDRESS)
//            addr_type = eAddressTypeLoad;
//    }
//
//    if (addr == LLDB_INVALID_ADDRESS)
//    {
//        addr = address.GetFileAddress();
//        if (addr != LLDB_INVALID_ADDRESS)
//            addr_type = eAddressTypeFile;
//    }
//
//    if (addr_type == eAddressTypeInvalid)
//        return false;
//
//    Target *target = exe_scope->CalculateTarget();
//    if (target)
//    {
//        Error error;
//        ObjectFile *objfile = NULL;
//        if (address.GetModule())
//            objfile = address.GetModule()->GetObjectFile();
//        return target->ReadMemory (addr_type, addr, dst, dst_len, error, objfile);
//    }
//    return 0;
//}

static size_t
ReadBytes (ExecutionContextScope *exe_scope, const Address &address, void *dst, size_t dst_len)
{
    if (exe_scope == NULL)
        return 0;

    Target *target = exe_scope->CalculateTarget();
    if (target)
    {
        Error error;
        return target->ReadMemory (address, dst, dst_len, error);
    }
    return 0;
}

static bool
GetByteOrderAndAddressSize (ExecutionContextScope *exe_scope, const Address &address, ByteOrder& byte_order, uint32_t& addr_size)
{
    byte_order = eByteOrderInvalid;
    addr_size = 0;
    if (exe_scope == NULL)
        return false;

    Process *process = exe_scope->CalculateProcess();
    if (process)
    {
        byte_order = process->GetByteOrder();
        addr_size = process->GetAddressByteSize();
    }

    if (byte_order == eByteOrderInvalid || addr_size == 0)
    {
        Module *module = address.GetModule();
        if (module)
        {
            byte_order = module->GetArchitecture().GetDefaultEndian();
            addr_size = module->GetArchitecture().GetAddressByteSize();
        }
    }
    return byte_order != eByteOrderInvalid && addr_size != 0;
}

static uint64_t
ReadUIntMax64 (ExecutionContextScope *exe_scope, const Address &address, uint32_t byte_size, bool &success)
{
    uint64_t uval64 = 0;
    if (exe_scope == NULL || byte_size > sizeof(uint64_t))
    {
        success = false;
        return 0;
    }
    uint64_t buf;

    success = ReadBytes (exe_scope, address, &buf, byte_size) == byte_size;
    if (success)
    {
        ByteOrder byte_order = eByteOrderInvalid;
        uint32_t addr_size = 0;
        if (GetByteOrderAndAddressSize (exe_scope, address, byte_order, addr_size))
        {
            DataExtractor data (&buf, sizeof(buf), byte_order, addr_size);
            uint32_t offset = 0;
            uval64 = data.GetU64(&offset);
        }
        else
            success = false;
    }
    return uval64;
}

static bool
ReadAddress (ExecutionContextScope *exe_scope, const Address &address, uint32_t pointer_size, Address &deref_so_addr)
{
    if (exe_scope == NULL)
        return false;


    bool success = false;
    addr_t deref_addr = ReadUIntMax64 (exe_scope, address, pointer_size, success);
    if (success)
    {
        Process *process = exe_scope->CalculateProcess();
        if (process && process->IsAlive())
        {
            if (!process->ResolveLoadAddress (deref_addr, deref_so_addr))
            {
                deref_so_addr.SetSection(NULL);
                deref_so_addr.SetOffset(deref_addr);
            }
        }
        else
        {
            Target *target = exe_scope->CalculateTarget();
            if (target == NULL)
                return false;

            if (!target->GetImages().ResolveFileAddress(deref_addr, deref_so_addr))
            {
                deref_so_addr.SetSection(NULL);
                deref_so_addr.SetOffset(deref_addr);
            }
        }
        return true;
    }
    return false;
}

static bool
DumpUInt (ExecutionContextScope *exe_scope, const Address &address, uint32_t byte_size, Stream* strm)
{
    if (exe_scope == NULL || byte_size == 0)
        return 0;
    std::vector<uint8_t> buf(byte_size, 0);

    if (ReadBytes (exe_scope, address, &buf[0], buf.size()) == buf.size())
    {
        ByteOrder byte_order = eByteOrderInvalid;
        uint32_t addr_size = 0;
        if (GetByteOrderAndAddressSize (exe_scope, address, byte_order, addr_size))
        {
            DataExtractor data (&buf.front(), buf.size(), byte_order, addr_size);

            data.Dump (strm,
                       0,                 // Start offset in "data"
                       eFormatHex,        // Print as characters
                       buf.size(),        // Size of item
                       1,                 // Items count
                       UINT32_MAX,        // num per line
                       LLDB_INVALID_ADDRESS,// base address
                       0,                 // bitfield bit size
                       0);                // bitfield bit offset

            return true;
        }
    }
    return false;
}


static size_t
ReadCStringFromMemory (ExecutionContextScope *exe_scope, const Address &address, Stream *strm)
{
    if (exe_scope == NULL)
        return 0;
    const size_t k_buf_len = 256;
    char buf[k_buf_len+1];
    buf[k_buf_len] = '\0'; // NULL terminate

    // Byte order and adderss size don't matter for C string dumping..
    DataExtractor data (buf, sizeof(buf), eByteOrderHost, 4);
    size_t total_len = 0;
    size_t bytes_read;
    Address curr_address(address);
    strm->PutChar ('"');
    while ((bytes_read = ReadBytes (exe_scope, curr_address, buf, k_buf_len)) > 0)
    {
        size_t len = strlen(buf);
        if (len == 0)
            break;
        if (len > bytes_read)
            len = bytes_read;

        data.Dump (strm,
                   0,                 // Start offset in "data"
                   eFormatChar,       // Print as characters
                   1,                 // Size of item (1 byte for a char!)
                   len,               // How many bytes to print?
                   UINT32_MAX,        // num per line
                   LLDB_INVALID_ADDRESS,// base address
                   0,                 // bitfield bit size

                   0);                // bitfield bit offset

        total_len += bytes_read;

        if (len < k_buf_len)
            break;
        curr_address.SetOffset (curr_address.GetOffset() + bytes_read);
    }
    strm->PutChar ('"');
    return total_len;
}

Address::Address (addr_t address, const SectionList * sections) :
    m_section (NULL),
    m_offset (LLDB_INVALID_ADDRESS)
{
    ResolveAddressUsingFileSections(address, sections);
}

const Address&
Address::operator= (const Address& rhs)
{
    if (this != &rhs)
    {
        m_section = rhs.m_section;
        m_offset = rhs.m_offset;
    }
    return *this;
}

bool
Address::ResolveAddressUsingFileSections (addr_t addr, const SectionList *sections)
{
    if (sections)
        m_section = sections->FindSectionContainingFileAddress(addr).get();
    else
        m_section = NULL;

    if (m_section != NULL)
    {
        assert( m_section->ContainsFileAddress(addr) );
        m_offset = addr - m_section->GetFileAddress();
        return true;    // Successfully transformed addr into a section offset address
    }

    m_offset = addr;
    return false;       // Failed to resolve this address to a section offset value
}

//bool
//Address::ResolveAddressUsingLoadSections (addr_t addr, const SectionList *sections)
//{
//    if (sections)
//        m_section = sections->FindSectionContainingLoadAddress(addr).get();
//    else
//        m_section = NULL;
//
//    if (m_section != NULL)
//    {
//        assert( m_section->ContainsLoadAddress(addr) );
//        m_offset = addr - m_section->GetLoadBaseAddress();
//        return true;    // Successfully transformed addr into a section offset address
//    }
//
//    m_offset = addr;
//    return false;       // Failed to resolve this address to a section offset value
//}
//
Module *
Address::GetModule () const
{
    if (m_section)
        return m_section->GetModule();
    return NULL;
}

//addr_t
//Address::Address() const
//{
//  addr_t addr = GetLoadAddress();
//  if (addr != LLDB_INVALID_ADDRESS)
//      return addr;
//  return GetFileAddress();
//}
//

addr_t
Address::GetFileAddress () const
{
    if (m_section != NULL)
    {
        addr_t sect_file_addr = m_section->GetFileAddress();
        if (sect_file_addr == LLDB_INVALID_ADDRESS)
        {
            // Section isn't resolved, we can't return a valid file address
            return LLDB_INVALID_ADDRESS;
        }
        // We have a valid file range, so we can return the file based
        // address by adding the file base address to our offset
        return sect_file_addr + m_offset;
    }
    // No section, we just return the offset since it is the value in this case
    return m_offset;
}

addr_t
Address::GetLoadAddress (Process *process) const
{
    if (m_section != NULL)
    {
        if (process)
        {
            addr_t sect_load_addr = m_section->GetLoadBaseAddress (process);

            if (sect_load_addr != LLDB_INVALID_ADDRESS)
            {
                // We have a valid file range, so we can return the file based
                // address by adding the file base address to our offset
                return sect_load_addr + m_offset;
            }
        }
       // The section isn't resolved or no process was supplied so we can't
       // return a valid file address.
       return LLDB_INVALID_ADDRESS;
    }
    // No section, we just return the offset since it is the value in this case
    return m_offset;
}

bool
Address::Dump (Stream *s, ExecutionContextScope *exe_scope, DumpStyle style, DumpStyle fallback_style, uint32_t addr_size) const
{
    // If the section was NULL, only load address is going to work.
    if (m_section == NULL)
        style = DumpStyleLoadAddress;

    Target *target = NULL;
    Process *process = NULL;
    if (exe_scope)
    {
        target = exe_scope->CalculateTarget();
        process = exe_scope->CalculateProcess();
    }
    // If addr_byte_size is UINT32_MAX, then determine the correct address
    // byte size for the process or default to the size of addr_t
    if (addr_size == UINT32_MAX)
    {
        if (process)
            addr_size = process->GetAddressByteSize ();
        else
            addr_size = sizeof(addr_t);
    }

    lldb_private::Address so_addr;
    switch (style)
    {
    case DumpStyleInvalid:
        return false;

    case DumpStyleSectionNameOffset:
        if (m_section != NULL)
        {
            m_section->DumpName(s);
            s->Printf (" + %llu", m_offset);
        }
        else
        {
            s->Address(m_offset, addr_size);
        }
        break;

    case DumpStyleSectionPointerOffset:
        s->Printf("(Section *)%.*p + ", (int)sizeof(void*) * 2, m_section);
        s->Address(m_offset, addr_size);
        break;

    case DumpStyleModuleWithFileAddress:
        s->Printf("%s[", m_section->GetModule()->GetFileSpec().GetFilename().AsCString());
        // Fall through
    case DumpStyleFileAddress:
        {
            addr_t file_addr = GetFileAddress();
            if (file_addr == LLDB_INVALID_ADDRESS)
            {
                if (fallback_style != DumpStyleInvalid)
                    return Dump (s, exe_scope, fallback_style, DumpStyleInvalid, addr_size);
                return false;
            }
            s->Address (file_addr, addr_size);
            if (style == DumpStyleModuleWithFileAddress)
                s->PutChar(']');
        }
        break;

    case DumpStyleLoadAddress:
        {
            addr_t load_addr = GetLoadAddress (process);
            if (load_addr == LLDB_INVALID_ADDRESS)
            {
                if (fallback_style != DumpStyleInvalid)
                    return Dump (s, exe_scope, fallback_style, DumpStyleInvalid, addr_size);
                return false;
            }
            s->Address (load_addr, addr_size);
        }
        break;

    case DumpStyleResolvedDescription:
    case DumpStyleResolvedDescriptionNoModule:
        if (IsSectionOffset())
        {
            lldb::AddressType addr_type = eAddressTypeLoad;
            addr_t addr = GetLoadAddress (process);
            if (addr == LLDB_INVALID_ADDRESS)
            {
                addr = GetFileAddress();
                addr_type = eAddressTypeFile;
            }

            uint32_t pointer_size = 4;
            lldb_private::Module *module = GetModule();
            if (process)
                pointer_size = process->GetAddressByteSize();
            else if (module)
                pointer_size = module->GetArchitecture().GetAddressByteSize();

            bool showed_info = false;
            const Section *section = GetSection();
            if (section)
            {
                SectionType sect_type = section->GetType();
                switch (sect_type)
                {
                case eSectionTypeDataCString:
                    // Read the C string from memory and display it
                    showed_info = true;
                    ReadCStringFromMemory (exe_scope, *this, s);
                    break;

                case eSectionTypeDataCStringPointers:
                    {
                        if (ReadAddress (exe_scope, *this, pointer_size, so_addr))
                        {
#if VERBOSE_OUTPUT
                            s->PutCString("(char *)");
                            so_addr.Dump(s, exe_scope, DumpStyleLoadAddress, DumpStyleFileAddress);
                            s->PutCString(": ");
#endif
                            showed_info = true;
                            ReadCStringFromMemory (exe_scope, so_addr, s);
                        }
                    }
                    break;

                case eSectionTypeDataObjCMessageRefs:
                    {
                        if (ReadAddress (exe_scope, *this, pointer_size, so_addr))
                        {
                            if (target && so_addr.IsSectionOffset())
                            {
                                lldb_private::SymbolContext func_sc;
                                target->GetImages().ResolveSymbolContextForAddress (so_addr,
                                                                                    eSymbolContextEverything,
                                                                                    func_sc);
                                if (func_sc.function || func_sc.symbol)
                                {
                                    showed_info = true;
#if VERBOSE_OUTPUT
                                    s->PutCString ("(objc_msgref *) -> { (func*)");
                                    so_addr.Dump(s, exe_scope, DumpStyleLoadAddress, DumpStyleFileAddress);
#else
                                    s->PutCString ("{ ");
#endif
                                    Address cstr_addr(*this);
                                    cstr_addr.SetOffset(cstr_addr.GetOffset() + pointer_size);
                                    func_sc.DumpStopContext(s, exe_scope, so_addr, true, false);
                                    if (ReadAddress (exe_scope, cstr_addr, pointer_size, so_addr))
                                    {
#if VERBOSE_OUTPUT
                                        s->PutCString("), (char *)");
                                        so_addr.Dump(s, exe_scope, DumpStyleLoadAddress, DumpStyleFileAddress);
                                        s->PutCString(" (");
#else
                                        s->PutCString(", ");
#endif
                                        ReadCStringFromMemory (exe_scope, so_addr, s);
                                    }
#if VERBOSE_OUTPUT
                                    s->PutCString(") }");
#else
                                    s->PutCString(" }");
#endif
                                }
                            }
                        }
                    }
                    break;

                case eSectionTypeDataObjCCFStrings:
                    {
                        Address cfstring_data_addr(*this);
                        cfstring_data_addr.SetOffset(cfstring_data_addr.GetOffset() + (2 * pointer_size));
                        if (ReadAddress (exe_scope, cfstring_data_addr, pointer_size, so_addr))
                        {
#if VERBOSE_OUTPUT
                            s->PutCString("(CFString *) ");
                            cfstring_data_addr.Dump(s, exe_scope, DumpStyleLoadAddress, DumpStyleFileAddress);
                            s->PutCString(" -> @");
#else
                            s->PutChar('@');
#endif
                            if (so_addr.Dump(s, exe_scope, DumpStyleResolvedDescription))
                                showed_info = true;
                        }
                    }
                    break;

                case eSectionTypeData4:
                    // Read the 4 byte data and display it
                    showed_info = true;
                    s->PutCString("(uint32_t) ");
                    DumpUInt (exe_scope, *this, 4, s);
                    break;

                case eSectionTypeData8:
                    // Read the 8 byte data and display it
                    showed_info = true;
                    s->PutCString("(uint64_t) ");
                    DumpUInt (exe_scope, *this, 8, s);
                    break;

                case eSectionTypeData16:
                    // Read the 16 byte data and display it
                    showed_info = true;
                    s->PutCString("(uint128_t) ");
                    DumpUInt (exe_scope, *this, 16, s);
                    break;

                case eSectionTypeDataPointers:
                    // Read the pointer data and display it
                    {
                        if (ReadAddress (exe_scope, *this, pointer_size, so_addr))
                        {
                            s->PutCString ("(void *)");
                            so_addr.Dump(s, exe_scope, DumpStyleLoadAddress, DumpStyleFileAddress);

                            showed_info = true;
                            if (so_addr.IsSectionOffset())
                            {
                                lldb_private::SymbolContext pointer_sc;
                                if (target)
                                {
                                    target->GetImages().ResolveSymbolContextForAddress (so_addr,
                                                                                        eSymbolContextEverything,
                                                                                        pointer_sc);
                                    if (pointer_sc.function || pointer_sc.symbol)
                                    {
                                        s->PutCString(": ");
                                        pointer_sc.DumpStopContext(s, exe_scope, so_addr, false, false);
                                    }
                                }
                            }
                        }
                    }
                    break;

                default:
                    break;
                }
            }

            if (!showed_info)
            {
                if (module)
                {
                    lldb_private::SymbolContext sc;
                    module->ResolveSymbolContextForAddress(*this, eSymbolContextEverything, sc);
                    if (sc.function || sc.symbol)
                    {
                        bool show_stop_context = true;
                        bool show_module = (style == DumpStyleResolvedDescription);
                        if (sc.function == NULL && sc.symbol != NULL)
                        {
                            // If we have just a symbol make sure it is in the right section
                            if (sc.symbol->GetAddressRangePtr())
                            {
                                if (sc.symbol->GetAddressRangePtr()->GetBaseAddress().GetSection() != GetSection())
                                {
                                    // don't show the module if the symbol is a trampoline symbol
                                    show_stop_context = false;
                                }
                            }
                        }
                        if (show_stop_context)
                        {
                            // We have a function or a symbol from the same
                            // sections as this address.
                            sc.DumpStopContext(s, exe_scope, *this, show_module, false);
                        }
                        else
                        {
                            // We found a symbol but it was in a different
                            // section so it isn't the symbol we should be
                            // showing, just show the section name + offset
                            Dump (s, exe_scope, DumpStyleSectionNameOffset);
                        }
                    }
                }
            }
        }
        else
        {
            if (fallback_style != DumpStyleInvalid)
                return Dump (s, exe_scope, fallback_style, DumpStyleInvalid, addr_size);
            return false;
        }
        break;

    case DumpStyleDetailedSymbolContext:
        if (IsSectionOffset())
        {
            lldb::AddressType addr_type = eAddressTypeLoad;
            addr_t addr = GetLoadAddress (process);
            if (addr == LLDB_INVALID_ADDRESS)
            {
                addr = GetFileAddress();
                addr_type = eAddressTypeFile;
            }

            lldb_private::Module *module = GetModule();
            if (module)
            {
                lldb_private::SymbolContext sc;
                module->ResolveSymbolContextForAddress(*this, eSymbolContextEverything, sc);
                if (sc.symbol)
                {
                    // If we have just a symbol make sure it is in the same section
                    // as our address. If it isn't, then we might have just found
                    // the last symbol that came before the address that we are 
                    // looking up that has nothing to do with our address lookup.
                    if (sc.symbol->GetAddressRangePtr() && sc.symbol->GetAddressRangePtr()->GetBaseAddress().GetSection() != GetSection())
                        sc.symbol = NULL;
                }
                sc.GetDescription(s, eDescriptionLevelBrief, process);
            }
        }
        if (fallback_style != DumpStyleInvalid)
            return Dump (s, exe_scope, fallback_style, DumpStyleInvalid, addr_size);
        return false;
        break;
    }

    return true;
}

//Stream& operator << (Stream& s, const Address& so_addr)
//{
//    so_addr.Dump(&s, Address::DumpStyleSectionNameOffset);
//    return s;
//}
//
void
Address::CalculateSymbolContext (SymbolContext *sc)
{
    sc->Clear();
    // Absolute addresses don't have enough information to reconstruct even their target.
    if (m_section == NULL)
        return;

    if (m_section->GetModule())
    {
        sc->module_sp = m_section->GetModule()->GetSP();
        if (sc->module_sp)
            sc->module_sp->ResolveSymbolContextForAddress (*this, eSymbolContextEverything, *sc);
    }
}

void
Address::DumpDebug(Stream *s) const
{
    *s << (void *)this << ": " << "Address";
    if (m_section != NULL)
    {
        *s << ", section = " << (void *)m_section << " (" << m_section->GetName() << "), offset = " << m_offset;
    }
    else
    {
        *s << ", vm_addr = " << m_offset;
    }
    s->EOL();
}

int
Address::CompareFileAddress (const Address& a, const Address& b)
{
    addr_t a_file_addr = a.GetFileAddress();
    addr_t b_file_addr = b.GetFileAddress();
    if (a_file_addr < b_file_addr)
        return -1;
    if (a_file_addr > b_file_addr)
        return +1;
    return 0;
}


int
Address::CompareLoadAddress (const Address& a, const Address& b, Process *process)
{
    assert (process != NULL);
    addr_t a_load_addr = a.GetLoadAddress (process);
    addr_t b_load_addr = b.GetLoadAddress (process);
    if (a_load_addr < b_load_addr)
        return -1;
    if (a_load_addr > b_load_addr)
        return +1;
    return 0;
}

int
Address::CompareModulePointerAndOffset (const Address& a, const Address& b)
{
    Module *a_module = a.GetModule ();
    Module *b_module = b.GetModule ();
    if (a_module < b_module)
        return -1;
    if (a_module > b_module)
        return +1;
    // Modules are the same, just compare the file address since they should
    // be unique
    addr_t a_file_addr = a.GetFileAddress();
    addr_t b_file_addr = b.GetFileAddress();
    if (a_file_addr < b_file_addr)
        return -1;
    if (a_file_addr > b_file_addr)
        return +1;
    return 0;
}


size_t
Address::MemorySize () const
{
    // Noting special for the memory size of a single Address object,
    // it is just the size of itself.
    return sizeof(Address);
}


/// The only comparisons that make sense are the load addresses
//bool
//lldb::operator< (const Address& lhs, const Address& rhs)
//{
//    lldb::addr_t lhs_addr = lhs.GetLoadAddress();
//    lldb::addr_t rhs_addr = rhs.GetLoadAddress();
//
//    if (lhs_addr == rhs_addr)
//    {
//        lhs_addr = lhs.GetFileAddress();
//        rhs_addr = rhs.GetFileAddress();
//    }
//    return lhs_addr < rhs_addr;
//}
//
//bool
//lldb::operator<=    (const Address& lhs, const Address& rhs)
//{
//    lldb::addr_t lhs_addr = lhs.GetLoadAddress();
//    lldb::addr_t rhs_addr = rhs.GetLoadAddress();
//
//    if (lhs_addr == rhs_addr)
//    {
//        lhs_addr = lhs.GetFileAddress();
//        rhs_addr = rhs.GetFileAddress();
//    }
//    return lhs_addr <= rhs_addr;
//}
//
//bool
//lldb::operator> (const Address& lhs, const Address& rhs)
//{
//    lldb::addr_t lhs_addr = lhs.GetLoadAddress();
//    lldb::addr_t rhs_addr = rhs.GetLoadAddress();
//
//    if (lhs_addr == rhs_addr)
//    {
//        lhs_addr = lhs.GetFileAddress();
//        rhs_addr = rhs.GetFileAddress();
//    }
//    return lhs_addr > rhs_addr;
//}
//
//bool
//lldb::operator>=    (const Address& lhs, const Address& rhs)
//{
//    lldb::addr_t lhs_addr = lhs.GetLoadAddress();
//    lldb::addr_t rhs_addr = rhs.GetLoadAddress();
//
//    if (lhs_addr == rhs_addr)
//    {
//        lhs_addr = lhs.GetFileAddress();
//        rhs_addr = rhs.GetFileAddress();
//    }
//    return lhs_addr >= rhs_addr;
//}
//

// The operator == checks for exact equality only (same section, same offset)
bool
lldb_private::operator== (const Address& a, const Address& rhs)
{
    return  a.GetSection() == rhs.GetSection() &&
            a.GetOffset()  == rhs.GetOffset();
}
// The operator != checks for exact inequality only (differing section, or
// different offset)
bool
lldb_private::operator!= (const Address& a, const Address& rhs)
{
    return  a.GetSection() != rhs.GetSection() ||
            a.GetOffset()  != rhs.GetOffset();
}

bool
Address::IsLinkedAddress () const
{
    return m_section && m_section->GetLinkedSection();
}


void
Address::ResolveLinkedAddress ()
{
    if (m_section)
    {
        const Section *linked_section = m_section->GetLinkedSection();
        if (linked_section)
        {
            m_offset += m_section->GetLinkedOffset();
            m_section = linked_section;
        }
    }
}

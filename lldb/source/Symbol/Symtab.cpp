//===-- Symtab.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <map>

#include "lldb/Core/Module.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Timer.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symtab.h"

using namespace lldb;
using namespace lldb_private;



Symtab::Symtab(ObjectFile *objfile) :
    m_objfile (objfile),
    m_symbols (),
    m_addr_indexes (),
    m_name_to_index (),
    m_mutex (Mutex::eMutexTypeRecursive),
    m_addr_indexes_computed (false),
    m_name_indexes_computed (false)
{
}

Symtab::~Symtab()
{
}

void
Symtab::Reserve(uint32_t count)
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    m_symbols.reserve (count);
}

Symbol *
Symtab::Resize(uint32_t count)
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    m_symbols.resize (count);
    return &m_symbols[0];
}

uint32_t
Symtab::AddSymbol(const Symbol& symbol)
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    uint32_t symbol_idx = m_symbols.size();
    m_name_to_index.Clear();
    m_addr_indexes.clear();
    m_symbols.push_back(symbol);
    m_addr_indexes_computed = false;
    m_name_indexes_computed = false;
    return symbol_idx;
}

size_t
Symtab::GetNumSymbols() const
{
    Mutex::Locker locker (m_mutex);
    return m_symbols.size();
}

void
Symtab::Dump (Stream *s, Target *target, SortOrder sort_order)
{
    Mutex::Locker locker (m_mutex);

//    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    const FileSpec &file_spec = m_objfile->GetFileSpec();
    const char * object_name = NULL;
    if (m_objfile->GetModule())
        object_name = m_objfile->GetModule()->GetObjectName().GetCString();

    if (file_spec)
        s->Printf("Symtab, file = %s/%s%s%s%s, num_symbols = %u",
        file_spec.GetDirectory().AsCString(),
        file_spec.GetFilename().AsCString(),
        object_name ? "(" : "",
        object_name ? object_name : "",
        object_name ? ")" : "",
        m_symbols.size());
    else
        s->Printf("Symtab, num_symbols = %u", m_symbols.size());

    if (!m_symbols.empty())
    {
        switch (sort_order)
        {
        case eSortOrderNone:
            {
                s->PutCString (":\n");
                DumpSymbolHeader (s);
                const_iterator begin = m_symbols.begin();
                const_iterator end = m_symbols.end();
                for (const_iterator pos = m_symbols.begin(); pos != end; ++pos)
                {
                    s->Indent();
                    pos->Dump(s, target, std::distance(begin, pos));
                }
            }
            break;

        case eSortOrderByName:
            {
                // Although we maintain a lookup by exact name map, the table
                // isn't sorted by name. So we must make the ordered symbol list
                // up ourselves.
                s->PutCString (" (sorted by name):\n");
                DumpSymbolHeader (s);
                typedef std::multimap<const char*, const Symbol *, CStringCompareFunctionObject> CStringToSymbol;
                CStringToSymbol name_map;
                for (const_iterator pos = m_symbols.begin(), end = m_symbols.end(); pos != end; ++pos)
                {
                    const char *name = pos->GetMangled().GetName(Mangled::ePreferDemangled).AsCString();
                    if (name && name[0])
                        name_map.insert (std::make_pair(name, &(*pos)));
                }
                
                for (CStringToSymbol::const_iterator pos = name_map.begin(), end = name_map.end(); pos != end; ++pos)
                {
                    s->Indent();
                    pos->second->Dump (s, target, pos->second - &m_symbols[0]);
                }
            }
            break;
            
        case eSortOrderByAddress:
            s->PutCString (" (sorted by address):\n");
            DumpSymbolHeader (s);
            if (!m_addr_indexes_computed)
                InitAddressIndexes();
            const size_t num_symbols = GetNumSymbols();
            std::vector<uint32_t>::const_iterator pos;
            std::vector<uint32_t>::const_iterator end = m_addr_indexes.end();
            for (pos = m_addr_indexes.begin(); pos != end; ++pos)
            {
                uint32_t idx = *pos;
                if (idx < num_symbols)
                {
                    s->Indent();
                    m_symbols[idx].Dump(s, target, idx);
                }
            }
            break;
        }
    }
}

void
Symtab::Dump(Stream *s, Target *target, std::vector<uint32_t>& indexes) const
{
    Mutex::Locker locker (m_mutex);

    const size_t num_symbols = GetNumSymbols();
    //s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->Printf("Symtab %u symbol indexes (%u symbols total):\n", indexes.size(), m_symbols.size());
    s->IndentMore();

    if (!indexes.empty())
    {
        std::vector<uint32_t>::const_iterator pos;
        std::vector<uint32_t>::const_iterator end = indexes.end();
        DumpSymbolHeader (s);
        for (pos = indexes.begin(); pos != end; ++pos)
        {
            uint32_t idx = *pos;
            if (idx < num_symbols)
            {
                s->Indent();
                m_symbols[idx].Dump(s, target, idx);
            }
        }
    }
    s->IndentLess ();
}

void
Symtab::DumpSymbolHeader (Stream *s)
{
    s->Indent("               Debug symbol\n");
    s->Indent("               |Synthetic symbol\n");
    s->Indent("               ||Externally Visible\n");
    s->Indent("               |||\n");
    s->Indent("Index   UserID DSX Type         File Address/Value Load Address       Size               Flags      Name\n");
    s->Indent("------- ------ --- ------------ ------------------ ------------------ ------------------ ---------- ----------------------------------\n");
}


static int
CompareSymbolID (const void *key, const void *p)
{
    const user_id_t match_uid = *(user_id_t*) key;
    const user_id_t symbol_uid = ((Symbol *)p)->GetID();
    if (match_uid < symbol_uid)
        return -1;
    if (match_uid > symbol_uid)
        return 1;
    return 0;
}

Symbol *
Symtab::FindSymbolByID (lldb::user_id_t symbol_uid) const
{
    Mutex::Locker locker (m_mutex);

    Symbol *symbol = (Symbol*)::bsearch (&symbol_uid, 
                                         &m_symbols[0], 
                                         m_symbols.size(), 
                                         (uint8_t *)&m_symbols[1] - (uint8_t *)&m_symbols[0], 
                                         CompareSymbolID);
    return symbol;
}


Symbol *
Symtab::SymbolAtIndex(uint32_t idx)
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    if (idx < m_symbols.size())
        return &m_symbols[idx];
    return NULL;
}


const Symbol *
Symtab::SymbolAtIndex(uint32_t idx) const
{
    // Clients should grab the mutex from this symbol table and lock it manually
    // when calling this function to avoid performance issues.
    if (idx < m_symbols.size())
        return &m_symbols[idx];
    return NULL;
}

//----------------------------------------------------------------------
// InitNameIndexes
//----------------------------------------------------------------------
void
Symtab::InitNameIndexes()
{
    // Protected function, no need to lock mutex...
    if (!m_name_indexes_computed)
    {
        m_name_indexes_computed = true;
        Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
        // Create the name index vector to be able to quickly search by name
        const size_t count = m_symbols.size();
        assert(m_objfile != NULL);
        assert(m_objfile->GetModule() != NULL);

#if 1
        m_name_to_index.Reserve (count);
#else
        // TODO: benchmark this to see if we save any memory. Otherwise we
        // will always keep the memory reserved in the vector unless we pull
        // some STL swap magic and then recopy...
        uint32_t actual_count = 0;
        for (const_iterator pos = m_symbols.begin(), end = m_symbols.end();
             pos != end; 
             ++pos)
        {
            const Mangled &mangled = pos->GetMangled();
            if (mangled.GetMangledName())
                ++actual_count;
            
            if (mangled.GetDemangledName())
                ++actual_count;
        }

        m_name_to_index.Reserve (actual_count);
#endif

        UniqueCStringMap<uint32_t>::Entry entry;

        for (entry.value = 0; entry.value < count; ++entry.value)
        {
            const Symbol *symbol = &m_symbols[entry.value];

            // Don't let trampolines get into the lookup by name map
            // If we ever need the trampoline symbols to be searchable by name
            // we can remove this and then possibly add a new bool to any of the
            // Symtab functions that lookup symbols by name to indicate if they
            // want trampolines.
            if (symbol->IsTrampoline())
                continue;

            const Mangled &mangled = symbol->GetMangled();
            entry.cstring = mangled.GetMangledName().GetCString();
            if (entry.cstring && entry.cstring[0])
                m_name_to_index.Append (entry);

            entry.cstring = mangled.GetDemangledName().GetCString();
            if (entry.cstring && entry.cstring[0])
                m_name_to_index.Append (entry);
        }
        m_name_to_index.Sort();
    }
}

uint32_t
Symtab::AppendSymbolIndexesWithType (SymbolType symbol_type, std::vector<uint32_t>& indexes, uint32_t start_idx, uint32_t end_index) const
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();

    const uint32_t count = std::min<uint32_t> (m_symbols.size(), end_index);

    for (uint32_t i = start_idx; i < count; ++i)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type)
            indexes.push_back(i);
    }

    return indexes.size() - prev_size;
}

uint32_t
Symtab::AppendSymbolIndexesWithTypeAndFlagsValue (SymbolType symbol_type, uint32_t flags_value, std::vector<uint32_t>& indexes, uint32_t start_idx, uint32_t end_index) const
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();

    const uint32_t count = std::min<uint32_t> (m_symbols.size(), end_index);

    for (uint32_t i = start_idx; i < count; ++i)
    {
        if ((symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type) && m_symbols[i].GetFlags() == flags_value)
            indexes.push_back(i);
    }

    return indexes.size() - prev_size;
}

uint32_t
Symtab::AppendSymbolIndexesWithType (SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& indexes, uint32_t start_idx, uint32_t end_index) const
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();

    const uint32_t count = std::min<uint32_t> (m_symbols.size(), end_index);

    for (uint32_t i = start_idx; i < count; ++i)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type)
        {
            if (CheckSymbolAtIndex(i, symbol_debug_type, symbol_visibility))
                indexes.push_back(i);
        }
    }

    return indexes.size() - prev_size;
}


uint32_t
Symtab::GetIndexForSymbol (const Symbol *symbol) const
{
    const Symbol *first_symbol = &m_symbols[0];
    if (symbol >= first_symbol && symbol < first_symbol + m_symbols.size())
        return symbol - first_symbol;
    return UINT32_MAX;
}

struct SymbolSortInfo
{
    const bool sort_by_load_addr;
    const Symbol *symbols;
};

namespace {
    struct SymbolIndexComparator {
        const std::vector<Symbol>& symbols;
        SymbolIndexComparator(const std::vector<Symbol>& s) : symbols(s) { }
        bool operator()(uint32_t index_a, uint32_t index_b) {
            addr_t value_a;
            addr_t value_b;
            if (symbols[index_a].GetValue().GetSection() == symbols[index_b].GetValue().GetSection()) {
                value_a = symbols[index_a].GetValue ().GetOffset();
                value_b = symbols[index_b].GetValue ().GetOffset();
            } else {
                value_a = symbols[index_a].GetValue ().GetFileAddress();
                value_b = symbols[index_b].GetValue ().GetFileAddress();
            }

            if (value_a == value_b) {
                // The if the values are equal, use the original symbol user ID
                lldb::user_id_t uid_a = symbols[index_a].GetID();
                lldb::user_id_t uid_b = symbols[index_b].GetID();
                if (uid_a < uid_b)
                    return true;
                if (uid_a > uid_b)
                    return false;
                return false;
            } else if (value_a < value_b)
                return true;
        
            return false;
        }
    };
}

void
Symtab::SortSymbolIndexesByValue (std::vector<uint32_t>& indexes, bool remove_duplicates) const
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__,__PRETTY_FUNCTION__);
    // No need to sort if we have zero or one items...
    if (indexes.size() <= 1)
        return;

    // Sort the indexes in place using std::stable_sort.
    // NOTE: The use of std::stable_sort instead of std::sort here is strictly for performance,
    // not correctness.  The indexes vector tends to be "close" to sorted, which the
    // stable sort handles better.
    std::stable_sort(indexes.begin(), indexes.end(), SymbolIndexComparator(m_symbols));

    // Remove any duplicates if requested
    if (remove_duplicates)
        std::unique(indexes.begin(), indexes.end());
}

uint32_t
Symtab::AppendSymbolIndexesWithName (const ConstString& symbol_name, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    if (symbol_name)
    {
        const size_t old_size = indexes.size();
        if (!m_name_indexes_computed)
            InitNameIndexes();

        const char *symbol_cstr = symbol_name.GetCString();
        const UniqueCStringMap<uint32_t>::Entry *entry_ptr;

        for (entry_ptr = m_name_to_index.FindFirstValueForName (symbol_cstr);
             entry_ptr!= NULL;
             entry_ptr = m_name_to_index.FindNextValueForName (symbol_cstr, entry_ptr))
        {
            indexes.push_back (entry_ptr->value);
        }
        return indexes.size() - old_size;
    }
    return 0;
}

uint32_t
Symtab::AppendSymbolIndexesWithName (const ConstString& symbol_name, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    if (symbol_name)
    {
        const size_t old_size = indexes.size();
        if (!m_name_indexes_computed)
            InitNameIndexes();

        const char *symbol_cstr = symbol_name.GetCString();
        const UniqueCStringMap<uint32_t>::Entry *entry_ptr;
        for (entry_ptr = m_name_to_index.FindFirstValueForName (symbol_cstr);
             entry_ptr!= NULL;
             entry_ptr = m_name_to_index.FindNextValueForName (symbol_cstr, entry_ptr))
        {
            if (CheckSymbolAtIndex(entry_ptr->value, symbol_debug_type, symbol_visibility))
                indexes.push_back (entry_ptr->value);
        }
        return indexes.size() - old_size;
    }
    return 0;
}

uint32_t
Symtab::AppendSymbolIndexesWithNameAndType (const ConstString& symbol_name, SymbolType symbol_type, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    if (AppendSymbolIndexesWithName(symbol_name, indexes) > 0)
    {
        std::vector<uint32_t>::iterator pos = indexes.begin();
        while (pos != indexes.end())
        {
            if (symbol_type == eSymbolTypeAny || m_symbols[*pos].GetType() == symbol_type)
                ++pos;
            else
                indexes.erase(pos);
        }
    }
    return indexes.size();
}

uint32_t
Symtab::AppendSymbolIndexesWithNameAndType (const ConstString& symbol_name, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    if (AppendSymbolIndexesWithName(symbol_name, symbol_debug_type, symbol_visibility, indexes) > 0)
    {
        std::vector<uint32_t>::iterator pos = indexes.begin();
        while (pos != indexes.end())
        {
            if (symbol_type == eSymbolTypeAny || m_symbols[*pos].GetType() == symbol_type)
                ++pos;
            else
                indexes.erase(pos);
        }
    }
    return indexes.size();
}


uint32_t
Symtab::AppendSymbolIndexesMatchingRegExAndType (const RegularExpression &regexp, SymbolType symbol_type, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();
    uint32_t sym_end = m_symbols.size();

    for (int i = 0; i < sym_end; i++)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type)
        {
            const char *name = m_symbols[i].GetMangled().GetName().AsCString();
            if (name)
            {
                if (regexp.Execute (name))
                    indexes.push_back(i);
            }
        }
    }
    return indexes.size() - prev_size;

}

uint32_t
Symtab::AppendSymbolIndexesMatchingRegExAndType (const RegularExpression &regexp, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& indexes)
{
    Mutex::Locker locker (m_mutex);

    uint32_t prev_size = indexes.size();
    uint32_t sym_end = m_symbols.size();

    for (int i = 0; i < sym_end; i++)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[i].GetType() == symbol_type)
        {
            if (CheckSymbolAtIndex(i, symbol_debug_type, symbol_visibility) == false)
                continue;

            const char *name = m_symbols[i].GetMangled().GetName().AsCString();
            if (name)
            {
                if (regexp.Execute (name))
                    indexes.push_back(i);
            }
        }
    }
    return indexes.size() - prev_size;

}

Symbol *
Symtab::FindSymbolWithType (SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, uint32_t& start_idx)
{
    Mutex::Locker locker (m_mutex);

    const size_t count = m_symbols.size();
    for (uint32_t idx = start_idx; idx < count; ++idx)
    {
        if (symbol_type == eSymbolTypeAny || m_symbols[idx].GetType() == symbol_type)
        {
            if (CheckSymbolAtIndex(idx, symbol_debug_type, symbol_visibility))
            {
                start_idx = idx;
                return &m_symbols[idx];
            }
        }
    }
    return NULL;
}

size_t
Symtab::FindAllSymbolsWithNameAndType (const ConstString &name, SymbolType symbol_type, std::vector<uint32_t>& symbol_indexes)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    // Initialize all of the lookup by name indexes before converting NAME
    // to a uniqued string NAME_STR below.
    if (!m_name_indexes_computed)
        InitNameIndexes();

    if (name)
    {
        // The string table did have a string that matched, but we need
        // to check the symbols and match the symbol_type if any was given.
        AppendSymbolIndexesWithNameAndType (name, symbol_type, symbol_indexes);
    }
    return symbol_indexes.size();
}

size_t
Symtab::FindAllSymbolsWithNameAndType (const ConstString &name, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& symbol_indexes)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    // Initialize all of the lookup by name indexes before converting NAME
    // to a uniqued string NAME_STR below.
    if (!m_name_indexes_computed)
        InitNameIndexes();

    if (name)
    {
        // The string table did have a string that matched, but we need
        // to check the symbols and match the symbol_type if any was given.
        AppendSymbolIndexesWithNameAndType (name, symbol_type, symbol_debug_type, symbol_visibility, symbol_indexes);
    }
    return symbol_indexes.size();
}

size_t
Symtab::FindAllSymbolsMatchingRexExAndType (const RegularExpression &regex, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility, std::vector<uint32_t>& symbol_indexes)
{
    Mutex::Locker locker (m_mutex);

    AppendSymbolIndexesMatchingRegExAndType(regex, symbol_type, symbol_debug_type, symbol_visibility, symbol_indexes);
    return symbol_indexes.size();
}

Symbol *
Symtab::FindFirstSymbolWithNameAndType (const ConstString &name, SymbolType symbol_type, Debug symbol_debug_type, Visibility symbol_visibility)
{
    Mutex::Locker locker (m_mutex);

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s", __PRETTY_FUNCTION__);
    if (!m_name_indexes_computed)
        InitNameIndexes();

    if (name)
    {
        std::vector<uint32_t> matching_indexes;
        // The string table did have a string that matched, but we need
        // to check the symbols and match the symbol_type if any was given.
        if (AppendSymbolIndexesWithNameAndType (name, symbol_type, symbol_debug_type, symbol_visibility, matching_indexes))
        {
            std::vector<uint32_t>::const_iterator pos, end = matching_indexes.end();
            for (pos = matching_indexes.begin(); pos != end; ++pos)
            {
                Symbol *symbol = SymbolAtIndex(*pos);

                if (symbol->Compare(name, symbol_type))
                    return symbol;
            }
        }
    }
    return NULL;
}

typedef struct
{
    const Symtab *symtab;
    const addr_t file_addr;
    Symbol *match_symbol;
    const uint32_t *match_index_ptr;
    addr_t match_offset;
} SymbolSearchInfo;

static int
SymbolWithFileAddress (SymbolSearchInfo *info, const uint32_t *index_ptr)
{
    const Symbol *curr_symbol = info->symtab->SymbolAtIndex (index_ptr[0]);
    if (curr_symbol == NULL)
        return -1;

    const addr_t info_file_addr = info->file_addr;

    // lldb::Symbol::GetAddressRangePtr() will only return a non NULL address
    // range if the symbol has a section!
    const AddressRange *curr_range = curr_symbol->GetAddressRangePtr();
    if (curr_range)
    {
        const addr_t curr_file_addr = curr_range->GetBaseAddress().GetFileAddress();
        if (info_file_addr < curr_file_addr)
            return -1;
        if (info_file_addr > curr_file_addr)
            return +1;
        info->match_symbol = const_cast<Symbol *>(curr_symbol);
        info->match_index_ptr = index_ptr;
        return 0;
    }

    return -1;
}

static int
SymbolWithClosestFileAddress (SymbolSearchInfo *info, const uint32_t *index_ptr)
{
    const Symbol *symbol = info->symtab->SymbolAtIndex (index_ptr[0]);
    if (symbol == NULL)
        return -1;

    const addr_t info_file_addr = info->file_addr;
    const AddressRange *curr_range = symbol->GetAddressRangePtr();
    if (curr_range)
    {
        const addr_t curr_file_addr = curr_range->GetBaseAddress().GetFileAddress();
        if (info_file_addr < curr_file_addr)
            return -1;

        // Since we are finding the closest symbol that is greater than or equal
        // to 'info->file_addr' we set the symbol here. This will get set
        // multiple times, but after the search is done it will contain the best
        // symbol match
        info->match_symbol = const_cast<Symbol *>(symbol);
        info->match_index_ptr = index_ptr;
        info->match_offset = info_file_addr - curr_file_addr;

        if (info_file_addr > curr_file_addr)
            return +1;
        return 0;
    }
    return -1;
}

static SymbolSearchInfo
FindIndexPtrForSymbolContainingAddress(Symtab* symtab, addr_t file_addr, const uint32_t* indexes, uint32_t num_indexes)
{
    SymbolSearchInfo info = { symtab, file_addr, NULL, NULL, 0 };
    ::bsearch (&info, 
               indexes, 
               num_indexes, 
               sizeof(uint32_t), 
               (ComparisonFunction)SymbolWithClosestFileAddress);
    return info;
}


void
Symtab::InitAddressIndexes()
{
    // Protected function, no need to lock mutex...
    if (!m_addr_indexes_computed && !m_symbols.empty())
    {
        m_addr_indexes_computed = true;
#if 0
        // The old was to add only code, trampoline or data symbols...
        AppendSymbolIndexesWithType (eSymbolTypeCode, m_addr_indexes);
        AppendSymbolIndexesWithType (eSymbolTypeTrampoline, m_addr_indexes);
        AppendSymbolIndexesWithType (eSymbolTypeData, m_addr_indexes);
#else
        // The new way adds all symbols with valid addresses that are section
        // offset.
        const_iterator begin = m_symbols.begin();
        const_iterator end = m_symbols.end();
        for (const_iterator pos = m_symbols.begin(); pos != end; ++pos)
        {
            if (pos->GetAddressRangePtr())
                m_addr_indexes.push_back (std::distance(begin, pos));
        }
#endif
        SortSymbolIndexesByValue (m_addr_indexes, false);
        m_addr_indexes.push_back (UINT32_MAX);   // Terminator for bsearch since we might need to look at the next symbol
    }
}

size_t
Symtab::CalculateSymbolSize (Symbol *symbol)
{
    Mutex::Locker locker (m_mutex);

    if (m_symbols.empty())
        return 0;

    // Make sure this symbol is from this symbol table...
    if (symbol < &m_symbols.front() || symbol > &m_symbols.back())
        return 0;

    // See if this symbol already has a byte size?
    size_t byte_size = symbol->GetByteSize();

    if (byte_size)
    {
        // It does, just return it
        return byte_size;
    }

    // Else if this is an address based symbol, figure out the delta between
    // it and the next address based symbol
    if (symbol->GetAddressRangePtr())
    {
        if (!m_addr_indexes_computed)
            InitAddressIndexes();
        const size_t num_addr_indexes = m_addr_indexes.size();
        SymbolSearchInfo info = FindIndexPtrForSymbolContainingAddress(this, symbol->GetAddressRangePtr()->GetBaseAddress().GetFileAddress(), &m_addr_indexes.front(), num_addr_indexes);
        if (info.match_index_ptr != NULL)
        {
            const lldb::addr_t curr_file_addr = symbol->GetAddressRangePtr()->GetBaseAddress().GetFileAddress();
            // We can figure out the address range of all symbols except the
            // last one by taking the delta between the current symbol and
            // the next symbol

            for (uint32_t addr_index = info.match_index_ptr - &m_addr_indexes.front() + 1;
                 addr_index < num_addr_indexes;
                 ++addr_index)
            {
                Symbol *next_symbol = SymbolAtIndex(m_addr_indexes[addr_index]);
                if (next_symbol == NULL)
                    break;

                assert (next_symbol->GetAddressRangePtr());
                const lldb::addr_t next_file_addr = next_symbol->GetAddressRangePtr()->GetBaseAddress().GetFileAddress();
                if (next_file_addr > curr_file_addr)
                {
                    byte_size = next_file_addr - curr_file_addr;
                    symbol->GetAddressRangePtr()->SetByteSize(byte_size);
                    symbol->SetSizeIsSynthesized(true);
                    break;
                }
            }
        }
    }
    return byte_size;
}

Symbol *
Symtab::FindSymbolWithFileAddress (addr_t file_addr)
{
    Mutex::Locker locker (m_mutex);

    if (!m_addr_indexes_computed)
        InitAddressIndexes();

    SymbolSearchInfo info = { this, file_addr, NULL, NULL, 0 };

    uint32_t* match = (uint32_t*)::bsearch (&info, 
                                            &m_addr_indexes[0], 
                                            m_addr_indexes.size(), 
                                            sizeof(uint32_t), 
                                            (ComparisonFunction)SymbolWithFileAddress);
    if (match)
        return SymbolAtIndex (*match);
    return NULL;
}


Symbol *
Symtab::FindSymbolContainingFileAddress (addr_t file_addr, const uint32_t* indexes, uint32_t num_indexes)
{
    Mutex::Locker locker (m_mutex);

    SymbolSearchInfo info = { this, file_addr, NULL, NULL, 0 };

    ::bsearch (&info, 
               indexes, 
               num_indexes, 
               sizeof(uint32_t), 
               (ComparisonFunction)SymbolWithClosestFileAddress);

    if (info.match_symbol)
    {
        if (info.match_offset == 0)
        {
            // We found an exact match!
            return info.match_symbol;
        }

        const size_t symbol_byte_size = CalculateSymbolSize(info.match_symbol);
        
        if (symbol_byte_size == 0)
        {
            // We weren't able to find the size of the symbol so lets just go 
            // with that match we found in our search...
            return info.match_symbol;
        }

        // We were able to figure out a symbol size so lets make sure our 
        // offset puts "file_addr" in the symbol's address range.
        if (info.match_offset < symbol_byte_size)
            return info.match_symbol;
    }
    return NULL;
}

Symbol *
Symtab::FindSymbolContainingFileAddress (addr_t file_addr)
{
    Mutex::Locker locker (m_mutex);

    if (!m_addr_indexes_computed)
        InitAddressIndexes();

    return FindSymbolContainingFileAddress (file_addr, &m_addr_indexes[0], m_addr_indexes.size());
}


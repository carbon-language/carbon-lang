//===-- Symtab.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef liblldb_Symtab_h_
#define liblldb_Symtab_h_

#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Symbol/Symbol.h"

namespace lldb_private {

class Symtab
{
public:
                        Symtab(ObjectFile *objfile);
                        ~Symtab();

            void        Reserve (uint32_t count);
            Symbol *    Resize (uint32_t count);
            uint32_t    AddSymbol(const Symbol& symbol);
            size_t      GetNumSymbols() const;
            void        Dump(Stream *s, Process *process) const;
            void        Dump(Stream *s, Process *process, std::vector<uint32_t>& indexes) const;

            Symbol *    FindSymbolByID (lldb::user_id_t uid) const;
            Symbol *    SymbolAtIndex (uint32_t idx);
    const   Symbol *    SymbolAtIndex (uint32_t idx) const;
            Symbol *    FindSymbolWithType (lldb::SymbolType symbol_type, uint32_t &start_idx);
    const   Symbol *    FindSymbolWithType (lldb::SymbolType symbol_type, uint32_t &start_idx) const;
            uint32_t    AppendSymbolIndexesWithType (lldb::SymbolType symbol_type, std::vector<uint32_t>& matches, uint32_t start_idx = 0, uint32_t end_index = UINT_MAX) const;
            uint32_t    AppendSymbolIndexesWithName (const ConstString& symbol_name, std::vector<uint32_t>& matches);
            uint32_t    AppendSymbolIndexesWithNameAndType (const ConstString& symbol_name, lldb::SymbolType symbol_type, std::vector<uint32_t>& matches);
            uint32_t    AppendSymbolIndexesMatchingRegExAndType (const RegularExpression &regex, lldb::SymbolType symbol_type, std::vector<uint32_t>& indexes);
            size_t      FindAllSymbolsWithNameAndType (const ConstString &name, lldb::SymbolType symbol_type, std::vector<uint32_t>& symbol_indexes);
            size_t      FindAllSymbolsMatchingRexExAndType (const RegularExpression &regex, lldb::SymbolType symbol_type, std::vector<uint32_t>& symbol_indexes);
            Symbol *    FindFirstSymbolWithNameAndType (const ConstString &name, lldb::SymbolType symbol_type = lldb::eSymbolTypeAny);
            Symbol *    FindSymbolWithFileAddress (lldb::addr_t file_addr);
//            Symbol *    FindSymbolContainingAddress (const Address& value, const uint32_t* indexes, uint32_t num_indexes);
//            Symbol *    FindSymbolContainingAddress (const Address& value);
            Symbol *    FindSymbolContainingFileAddress (lldb::addr_t file_addr, const uint32_t* indexes, uint32_t num_indexes);
            Symbol *    FindSymbolContainingFileAddress (lldb::addr_t file_addr);
            size_t      CalculateSymbolSize (Symbol *symbol);

            void        SortSymbolIndexesByValue (std::vector<uint32_t>& indexes, bool remove_duplicates) const;

    static  void        DumpSymbolHeader (Stream *s);

protected:
    typedef std::vector<Symbol>         collection;
    typedef collection::iterator        iterator;
    typedef collection::const_iterator  const_iterator;

            void        InitNameIndexes ();
            void        InitAddressIndexes ();

    ObjectFile *        m_objfile;
    collection          m_symbols;
    std::vector<uint32_t> m_addr_indexes;
    UniqueCStringMap<uint32_t> m_name_to_index;

private:
    DISALLOW_COPY_AND_ASSIGN (Symtab);
};

} // namespace lldb_private

#endif  // liblldb_Symtab_h_

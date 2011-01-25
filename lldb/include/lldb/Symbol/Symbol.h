//===-- Symbol.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Symbol_h_
#define liblldb_Symbol_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/UserID.h"
#include "lldb/Symbol/SymbolContextScope.h"

namespace lldb_private {

class Symbol :
    public UserID,   // Used to uniquely identify this symbol in its symbol table
    public SymbolContextScope
{
public:
    // ObjectFile readers can classify their symbol table entries and searches can be made
    // on specific types where the symbol values will have drastically different meanings
    // and sorting requirements.
    Symbol();

    Symbol (lldb::user_id_t symID,
            const char *name,
            bool name_is_mangled,
            lldb::SymbolType type,
            bool external,
            bool is_debug,
            bool is_trampoline,
            bool is_artificial,
            const Section* section,
            lldb::addr_t value,
            uint32_t size,
            uint32_t flags);

    Symbol (lldb::user_id_t symID,
            const char *name,
            bool name_is_mangled,
            lldb::SymbolType type,
            bool external,
            bool is_debug,
            bool is_trampoline,
            bool is_artificial,
            const AddressRange &range,
            uint32_t flags);

    Symbol (const Symbol& rhs);

    const Symbol&
    operator= (const Symbol& rhs);

    bool
    Compare (const ConstString& name, lldb::SymbolType type) const;

    void
    Dump (Stream *s, Target *target, uint32_t index) const;

    AddressRange *
    GetAddressRangePtr ();

    const AddressRange *
    GetAddressRangePtr () const;

    AddressRange &
    GetAddressRangeRef() { return m_addr_range; }

    const AddressRange &
    GetAddressRangeRef() const { return m_addr_range; }

    const ConstString &
    GetName () { return m_mangled.GetName(); }

    Mangled&
    GetMangled () { return m_mangled; }

    const Mangled&
    GetMangled () const { return m_mangled; }

    bool
    GetSizeIsSibling () const { return m_size_is_sibling; }

    bool
    GetSizeIsSynthesized() const { return m_size_is_synthesized; }

    uint32_t
    GetSiblingIndex () const;

    lldb::addr_t
    GetByteSize () const { return m_addr_range.GetByteSize(); }

    lldb::SymbolType
    GetType () const { return m_type; }

    void
    SetType (lldb::SymbolType type) { m_type = type; }

    const char *
    GetTypeAsString () const;

    uint32_t
    GetFlags () const { return m_flags; }

    void
    SetFlags (uint32_t flags) { m_flags = flags; }

    void
    GetDescription (Stream *s, lldb::DescriptionLevel level, Target *target) const;

    Function *
    GetFunction ();

    Address &
    GetValue () { return m_addr_range.GetBaseAddress(); }

    const Address &
    GetValue () const { return m_addr_range.GetBaseAddress(); }

    bool
    IsSynthetic () const { return m_is_synthetic; }

    void
    SetIsSynthetic (bool b) { m_is_synthetic = b; }

    void
    SetSizeIsSynthesized(bool b) { m_size_is_synthesized = b; }

    bool
    IsDebug () const { return m_is_debug; }

    void
    SetDebug (bool b) { m_is_debug = b; }

    bool
    IsExternal () const { return m_is_external; }

    void
    SetExternal (bool b) { m_is_external = b; }

    bool
    IsTrampoline () const;

    void
    SetByteSize (uint32_t size) { m_addr_range.SetByteSize(size); }

    void
    SetSizeIsSibling (bool b) { m_size_is_sibling = b; }

    void
    SetValue (Address &value) { m_addr_range.GetBaseAddress() = value; }

    void
    SetValue (const AddressRange &range) { m_addr_range = range; }

    void
    SetValue (lldb::addr_t value);

    // If m_type is "Code" or "Function" then this will return the prologue size
    // in bytes, else it will return zero.
    uint32_t
    GetPrologueByteSize ();

    //------------------------------------------------------------------
    /// @copydoc SymbolContextScope::CalculateSymbolContext(SymbolContext*)
    ///
    /// @see SymbolContextScope
    //------------------------------------------------------------------
    virtual void
    CalculateSymbolContext (SymbolContext *sc);

    //------------------------------------------------------------------
    /// @copydoc SymbolContextScope::DumpSymbolContext(Stream*)
    ///
    /// @see SymbolContextScope
    //------------------------------------------------------------------
    virtual void
    DumpSymbolContext (Stream *s);

protected:

    Mangled         m_mangled;              // uniqued symbol name/mangled name pair
    lldb::SymbolType m_type;                 // symbol type
    uint16_t        m_type_data;            // data specific to m_type
    uint16_t        m_type_data_resolved:1, // True if the data in m_type_data has already been calculated
                    m_is_synthetic:1,       // non-zero if this symbol is not actually in the symbol table, but synthesized from other info in the object file.
                    m_is_debug:1,           // non-zero if this symbol is debug information in a symbol
                    m_is_external:1,        // non-zero if this symbol is globally visible
                    m_size_is_sibling:1,    // m_size contains the index of this symbol's sibling
                    m_size_is_synthesized:1,// non-zero if this symbol's size was calculated using a delta between this symbol and the next
                    m_searched_for_function:1;// non-zero if we have looked for the function associated with this symbol already.
    AddressRange    m_addr_range;           // Contains the value, or the section offset address when the value is an address in a section, and the size (if any)
    uint32_t        m_flags;                // A copy of the flags from the original symbol table, the ObjectFile plug-in can interpret these
    Function *      m_function;
};

} // namespace lldb_private

#endif  // liblldb_Symbol_h_

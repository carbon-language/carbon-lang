//===-- SymbolContext.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef liblldb_SymbolContext_h_
#define liblldb_SymbolContext_h_

#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/LineEntry.h"

namespace lldb_private {

class SymbolContextScope;
//----------------------------------------------------------------------
/// @class SymbolContext SymbolContext.h "lldb/Symbol/SymbolContext.h"
/// @brief Defines a symbol context baton that can be handed other debug
/// core functions.
///
/// Many debugger functions require a context when doing lookups. This
/// class provides a common structure that can be used as the result
/// of a query that can contain a single result. Examples of such
/// queries include
///     @li Looking up a load address.
//----------------------------------------------------------------------
class SymbolContext
{
public:

    //------------------------------------------------------------------
    /// Default constructor.
    ///
    /// Initialize all pointer members to NULL and all struct members
    /// to their default state.
    //------------------------------------------------------------------
    SymbolContext ();

    //------------------------------------------------------------------
    /// Construct with an object that knows how to reconstruct its
    /// symbol context.
    ///
    /// @param[in] sc_scope
    ///     A symbol context scope object that knows how to reconstruct
    ///     it's context.
    //------------------------------------------------------------------
    explicit 
    SymbolContext (SymbolContextScope *sc_scope);

    //------------------------------------------------------------------
    /// Construct with module, and optional compile unit, function,
    /// block, line table, line entry and symbol.
    ///
    /// Initialize all pointer to the specified values.
    ///
    /// @param[in] module
    ///     A Module pointer to the module for this context.
    ///
    /// @param[in] comp_unit
    ///     A CompileUnit pointer to the compile unit for this context.
    ///
    /// @param[in] function
    ///     A Function pointer to the function for this context.
    ///
    /// @param[in] block
    ///     A Block pointer to the deepest block for this context.
    ///
    /// @param[in] line_entry
    ///     A LineEntry pointer to the line entry for this context.
    ///
    /// @param[in] symbol
    ///     A Symbol pointer to the symbol for this context.
    //------------------------------------------------------------------
    explicit
    SymbolContext (const lldb::TargetSP &target_sp,
                   const lldb::ModuleSP &module_sp,
                   CompileUnit *comp_unit = NULL,
                   Function *function = NULL,
                   Block *block = NULL,
                   LineEntry *line_entry = NULL,
                   Symbol *symbol = NULL);

    // This version sets the target to a NULL TargetSP if you don't know it.
    explicit
    SymbolContext (const lldb::ModuleSP &module_sp,
                   CompileUnit *comp_unit = NULL,
                   Function *function = NULL,
                   Block *block = NULL,
                   LineEntry *line_entry = NULL,
                   Symbol *symbol = NULL);

    //------------------------------------------------------------------
    /// Copy constructor
    ///
    /// Makes a copy of the another SymbolContext object \a rhs.
    ///
    /// @param[in] rhs
    ///     A const SymbolContext object reference to copy.
    //------------------------------------------------------------------
    SymbolContext (const SymbolContext& rhs);

    //------------------------------------------------------------------
    /// Assignment operator.
    ///
    /// Copies the address value from another SymbolContext object \a
    /// rhs into \a this object.
    ///
    /// @param[in] rhs
    ///     A const SymbolContext object reference to copy.
    ///
    /// @return
    ///     A const SymbolContext object reference to \a this.
    //------------------------------------------------------------------
    const SymbolContext&
    operator= (const SymbolContext& rhs);

    //------------------------------------------------------------------
    /// Clear the object's state.
    ///
    /// Resets all pointer members to NULL, and clears any class objects
    /// to their default state.
    //------------------------------------------------------------------
    void
    Clear ();

    //------------------------------------------------------------------
    /// Dump a description of this object to a Stream.
    ///
    /// Dump a description of the contents of this object to the
    /// supplied stream \a s.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    //------------------------------------------------------------------
    void
    Dump (Stream *s, Target *target) const;

    //------------------------------------------------------------------
    /// Dump the stop context in this object to a Stream.
    ///
    /// Dump the best description of this object to the stream. The
    /// information displayed depends on the amount and quality of the
    /// information in this context. If a module, function, file and
    /// line number are available, they will be dumped. If only a
    /// module and function or symbol name with offset is available,
    /// that will be ouput. Else just the address at which the target
    /// was stopped will be displayed.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    ///
    /// @param[in] so_addr
    ///     The resolved section offset address.
    //------------------------------------------------------------------
    void
    DumpStopContext (Stream *s,
                     ExecutionContextScope *exe_scope,
                     const Address &so_addr,
                     bool show_fullpaths,
                     bool show_module,
                     bool show_inlined_frames) const;

    //------------------------------------------------------------------
    /// Get the address range contained within a symbol context.
    ///
    /// Address range priority is as follows:
    ///     - line_entry address range if line_entry is valid
    ///     - function address range if function is not NULL
    ///     - symbol address range if symbol is not NULL
    ///
    /// @param[out] range
    ///     An address range object that will be filled in if \b true
    ///     is returned.
    ///
    /// @return
    ///     \b True if this symbol context contains items that describe
    ///     an address range, \b false otherwise.
    //------------------------------------------------------------------
    bool
    GetAddressRange (uint32_t scope, AddressRange &range) const;


    void
    GetDescription(Stream *s, 
                   lldb::DescriptionLevel level, 
                   Target *target) const;
    
    uint32_t
    GetResolvedMask () const;


    //------------------------------------------------------------------
    /// Find a function matching the given name, working out from this
    /// symbol context.
    ///
    /// @return
    ///     The number of symbol contexts found.
    //------------------------------------------------------------------
    size_t
    FindFunctionsByName (const ConstString &name, 
                         bool append, 
                         SymbolContextList &sc_list) const;


    ClangNamespaceDecl
    FindNamespace (const ConstString &name) const;

    //------------------------------------------------------------------
    /// Find a variable matching the given name, working out from this
    /// symbol context.
    ///
    /// @return
    ///     A shared pointer to the variable found.
    //------------------------------------------------------------------
    //lldb::VariableSP
    //FindVariableByName (const char *name) const;

    //------------------------------------------------------------------
    /// Find a type matching the given name, working out from this
    /// symbol context.
    ///
    /// @return
    ///     A shared pointer to the variable found.
    //------------------------------------------------------------------
    lldb::TypeSP
    FindTypeByName (const ConstString &name) const;

    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    lldb::TargetSP  target_sp;  ///< The Target for a given query
    lldb::ModuleSP  module_sp;  ///< The Module for a given query
    CompileUnit *   comp_unit;  ///< The CompileUnit for a given query
    Function *      function;   ///< The Function for a given query
    Block *         block;      ///< The Block for a given query
    LineEntry       line_entry; ///< The LineEntry for a given query
    Symbol *        symbol;     ///< The Symbol for a given query
};

//----------------------------------------------------------------------
/// @class SymbolContextList SymbolContext.h "lldb/Symbol/SymbolContext.h"
/// @brief Defines a list of symbol context objects.
///
/// This class provides a common structure that can be used to contain
/// the result of a query that can contain a multiple results. Examples
/// of such queries include:
///     @li Looking up a function by name.
///     @li Finding all addressses for a specified file and line number.
//----------------------------------------------------------------------
class SymbolContextList
{
public:
    //------------------------------------------------------------------
    /// Default constructor.
    ///
    /// Initialize with an empty list.
    //------------------------------------------------------------------
    SymbolContextList ();

    //------------------------------------------------------------------
    /// Destructor.
    //------------------------------------------------------------------
    ~SymbolContextList ();

    //------------------------------------------------------------------
    /// Append a new symbol context to the list.
    ///
    /// @param[in] sc
    ///     A symbol context to append to the list.
    //------------------------------------------------------------------
    void
    Append(const SymbolContext& sc);

    //------------------------------------------------------------------
    /// Clear the object's state.
    ///
    /// Clears the symbol context list.
    //------------------------------------------------------------------
    void
    Clear();

    //------------------------------------------------------------------
    /// Dump a description of this object to a Stream.
    ///
    /// Dump a description of the contents of each symbol context in
    /// the list to the supplied stream \a s.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    //------------------------------------------------------------------
    void
    Dump(Stream *s, Target *target) const;

    //------------------------------------------------------------------
    /// Get accessor for a symbol context at index \a idx.
    ///
    /// Dump a description of the contents of each symbol context in
    /// the list to the supplied stream \a s.
    ///
    /// @param[in] idx
    ///     The zero based index into the symbol context list.
    ///
    /// @param[out] sc
    ///     A reference to the symbol context to fill in.
    ///
    /// @return
    ///     Returns \b true if \a idx was a valid index into this
    ///     symbol context list and \a sc was filled in, \b false
    ///     otherwise.
    //------------------------------------------------------------------
    bool
    GetContextAtIndex(uint32_t idx, SymbolContext& sc) const;

    bool
    RemoveContextAtIndex (uint32_t idx);
    //------------------------------------------------------------------
    /// Get accessor for a symbol context list size.
    ///
    /// @return
    ///     Returns the number of symbol context objects in the list.
    //------------------------------------------------------------------
    uint32_t
    GetSize() const;

protected:
    typedef std::vector<SymbolContext> collection; ///< The collection type for the list.

    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    collection m_symbol_contexts; ///< The list of symbol contexts.
};

bool operator== (const SymbolContext& lhs, const SymbolContext& rhs);
bool operator!= (const SymbolContext& lhs, const SymbolContext& rhs);

} // namespace lldb_private

#endif  // liblldb_SymbolContext_h_

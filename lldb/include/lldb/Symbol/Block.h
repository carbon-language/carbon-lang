//===-- Block.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Block_h_
#define liblldb_Block_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/VMRange.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Symbol/SymbolContext.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Block Block.h "lldb/Symbol/Block.h"
/// @brief A class that describes a single lexical block.
///
/// A Function object owns a BlockList object which owns one or more
/// Block objects. The BlockList object contains a section offset
/// address range, and Block objects contain one or more ranges
/// which are offsets into that range. Blocks are can have discontiguous
/// ranges within the BlockList adress range, and each block can
/// contain child blocks each with their own sets of ranges.
///
/// Each block has a variable list that represents local, argument, and
/// static variables that are scoped to the block.
///
/// Inlined functions are representated by attaching a
/// InlineFunctionInfo shared pointer object to a block. Inlined
/// functions are represented as named blocks.
//----------------------------------------------------------------------
class Block :
    public UserID,
    public SymbolContextScope
{
public:
    friend class Function;
    friend class BlockList;
    //------------------------------------------------------------------
    /// Enumeration values for special and invalid Block User ID
    /// values.
    //------------------------------------------------------------------
    enum
    {
        RootID = LLDB_INVALID_UID - 1,  ///< The Block UID for the root block
        InvalidID = LLDB_INVALID_UID        ///< Invalid Block UID.
    };

    //------------------------------------------------------------------
    /// Construct with a User ID \a uid, \a depth.
    ///
    /// Initialize this block with the specified UID \a uid. The
    /// \a depth in the \a block_list is used to represent the parent,
    /// sibling, and child block information and also allows for partial
    /// parsing at the block level.
    ///
    /// @param[in] uid
    ///     The UID for a given block. This value is given by the
    ///     SymbolFile plug-in and can be any value that helps the
    ///     SymbolFile plug-in to match this block back to the debug
    ///     information data that it parses for further or more in
    ///     depth parsing. Common values would be the index into a
    ///     table, or an offset into the debug information.
    ///
    /// @param[in] depth
    ///     The integer depth of this block in the block list hierarchy.
    ///
    /// @param[in] block_list
    ///     The block list that this object belongs to.
    ///
    /// @see BlockList
    //------------------------------------------------------------------
    Block (lldb::user_id_t uid, uint32_t depth, BlockList* block_list);

    //------------------------------------------------------------------
    /// Copy constructor.
    ///
    /// Makes a copy of the another Block object \a rhs.
    ///
    /// @param[in] rhs
    ///     A const Block object reference to copy.
    //------------------------------------------------------------------
    Block (const Block& rhs);

    //------------------------------------------------------------------
    /// Destructor.
    //------------------------------------------------------------------
    ~Block ();

    //------------------------------------------------------------------
    /// Assignment operator.
    ///
    /// Copies the block value from another Block object \a rhs
    /// into \a this object.
    ///
    /// @param[in] rhs
    ///     A const Block object reference to copy.
    ///
    /// @return
    ///     A const Block object reference to \a this.
    //------------------------------------------------------------------
    const Block&
    operator= (const Block& rhs);

    //------------------------------------------------------------------
    /// Add a child to this object.
    ///
    /// @param[in] uid
    ///     The UID for a given block. This value is given by the
    ///     SymbolFile plug-in and can be any value that helps the
    ///     SymbolFile plug-in to match this block back to the debug
    ///     information data that it parses for further or more in
    ///     depth parsing. Common values would be the index into a
    ///     table, or an offset into the debug information.
    ///
    /// @return
    ///     Returns \a uid if the child was successfully added to this
    ///     block, or Block::InvalidID on failure.
    //------------------------------------------------------------------
    lldb::user_id_t
    AddChild (lldb::user_id_t uid);

    //------------------------------------------------------------------
    /// Add a new offset range to this block.
    ///
    /// @param[in] start_offset
    ///     An offset into this Function's address range that
    ///     describes the start address of a range for this block.
    ///
    /// @param[in] end_offset
    ///     An offset into this Function's address range that
    ///     describes the end address of a range for this block.
    //------------------------------------------------------------------
    void
    AddRange(lldb::addr_t start_offset, lldb::addr_t end_offset);

    //------------------------------------------------------------------
    /// @copydoc SymbolContextScope::CalculateSymbolContext(SymbolContext*)
    ///
    /// @see SymbolContextScope
    //------------------------------------------------------------------
    virtual void
    CalculateSymbolContext(SymbolContext* sc);

    //------------------------------------------------------------------
    /// Check if an offset is in one of the block offset ranges.
    ///
    /// @param[in] range_offset
    ///     An offset into the Function's address range.
    ///
    /// @return
    ///     Returns \b true if \a range_offset falls in one of this
    ///     block's ranges, \b false otherwise.
    //------------------------------------------------------------------
    bool
    Contains (lldb::addr_t range_offset) const;

    //------------------------------------------------------------------
    /// Check if a offset range is in one of the block offset ranges.
    ///
    /// @param[in] range
    ///     An offset range into the Function's address range.
    ///
    /// @return
    ///     Returns \b true if \a range falls in one of this
    ///     block's ranges, \b false otherwise.
    //------------------------------------------------------------------
    bool
    Contains (const VMRange& range) const;

    bool
    ContainsBlockWithID (lldb::user_id_t block_id) const;

    //------------------------------------------------------------------
    /// Dump the block contents.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    ///
    /// @param[in] base_addr
    ///     The resolved start address of the Function's address
    ///     range. This should be resolved as the file or load address
    ///     prior to passing the value into this function for dumping.
    ///
    /// @param[in] depth
    ///     Limit the number of levels deep that this function should
    ///     print as this block can contain child blocks. Specify
    ///     INT_MAX to dump all child blocks.
    ///
    /// @param[in] show_context
    ///     If \b true, variables will dump their context information.
    //------------------------------------------------------------------
    void
    Dump (Stream *s, lldb::addr_t base_addr, int32_t depth, bool show_context) const;

    void
    DumpStopContext (Stream *s, const SymbolContext *sc);

    //------------------------------------------------------------------
    /// @copydoc SymbolContextScope::DumpSymbolContext(Stream*)
    ///
    /// @see SymbolContextScope
    //------------------------------------------------------------------
    virtual void
    DumpSymbolContext(Stream *s);

    //------------------------------------------------------------------
    /// Get the parent block's UID.
    ///
    /// @return
    ///     The UID of the parent block, or Block::InvalidID
    ///     if this block has no parent.
    //------------------------------------------------------------------
    lldb::user_id_t
    GetParentUID () const;

    //------------------------------------------------------------------
    /// Get the sibling block's UID.
    ///
    /// @return
    ///     The UID of the sibling block, or Block::InvalidID
    ///     if this block has no sibling.
    //------------------------------------------------------------------
    lldb::user_id_t
    GetSiblingUID () const;

    //------------------------------------------------------------------
    /// Get the first child block's UID.
    ///
    /// @return
    ///     The UID of the first child block, or Block::InvalidID
    ///     if this block has no first child.
    //------------------------------------------------------------------
    lldb::user_id_t
    GetFirstChildUID () const;

    //------------------------------------------------------------------
    /// Get the variable list for this block and optionally all child
    /// blocks if \a get_child_variables is \b true.
    ///
    /// @param[in] get_child_variables
    ///     If \b true, all variables from all child blocks will be
    ///     added to the variable list.
    ///
    /// @param[in] can_create
    ///     If \b true, the variables can be parsed if they already
    ///     haven't been, else the current state of the block will be
    ///     returned. Passing \b true for this parameter can be used
    ///     to see the current state of what has been parsed up to this
    ///     point.
    ///
    /// @return
    ///     A variable list shared pointer that contains all variables
    ///     for this block.
    //------------------------------------------------------------------
    lldb::VariableListSP
    GetVariableList (bool get_child_variables, bool can_create);


    //------------------------------------------------------------------
    /// Appends the variables from this block, and optionally from all
    /// parent blocks, to \a variable_list.
    ///
    /// @param[in] can_create
    ///     If \b true, the variables can be parsed if they already
    ///     haven't been, else the current state of the block will be
    ///     returned. Passing \b true for this parameter can be used
    ///     to see the current state of what has been parsed up to this
    ///     point.
    ///
    /// @param[in] get_parent_variables
    ///     If \b true, all variables from all parent blocks will be
    ///     added to the variable list.
    ///
    /// @param[in/out] variable_list
    ///     All variables in this block, and optionally all parent
    ///     blocks will be added to this list.
    ///
    /// @return
    ///     The number of variable that were appended to \a
    ///     variable_list.
    //------------------------------------------------------------------
    uint32_t
    AppendVariables(bool can_create, bool get_parent_variables, VariableList *variable_list);

    //------------------------------------------------------------------
    /// Get accessor for any inlined function information.
    ///
    /// @return
    ///     A pointer to any inlined function information, or NULL if
    ///     this is a regular block.
    //------------------------------------------------------------------
    InlineFunctionInfo*
    InlinedFunctionInfo ();

    //------------------------------------------------------------------
    /// Get const accessor for any inlined function information.
    ///
    /// @return
    ///     A cpmst pointer to any inlined function information, or NULL
    ///     if this is a regular block.
    //------------------------------------------------------------------
    const InlineFunctionInfo*
    InlinedFunctionInfo () const;

    //------------------------------------------------------------------
    /// Get the memory cost of this object.
    ///
    /// Returns the cost of this object plus any owned objects from the
    /// ranges, variables, and inline function information.
    ///
    /// @return
    ///     The number of bytes that this object occupies in memory.
    //------------------------------------------------------------------
    size_t
    MemorySize() const;

    //------------------------------------------------------------------
    /// Set accessor for any inlined function information.
    ///
    /// @param[in] name
    ///     The method name for the inlined function. This value should
    ///     not be NULL.
    ///
    /// @param[in] mangled
    ///     The mangled method name for the inlined function. This can
    ///     be NULL if there is no mangled name for an inlined function
    ///     or if the name is the same as \a name.
    ///
    /// @param[in] decl_ptr
    ///     A optional pointer to declaration information for the
    ///     inlined function information. This value can be NULL to
    ///     indicate that no declaration information is available.
    ///
    /// @param[in] call_decl_ptr
    ///     Optional calling location declaration information that
    ///     describes from where this inlined function was called.
    //------------------------------------------------------------------
    void
    SetInlinedFunctionInfo (const char *name,
                            const char *mangled,
                            const Declaration *decl_ptr,
                            const Declaration *call_decl_ptr);

    //------------------------------------------------------------------
    /// Set accessor for the variable list.
    ///
    /// Called by the SymbolFile plug-ins after they have parsed the
    /// variable lists and are ready to hand ownership of the list over
    /// to this object.
    ///
    /// @param[in] variable_list_sp
    ///     A shared pointer to a VariableList.
    //------------------------------------------------------------------
    void
    SetVariableList (lldb::VariableListSP& variable_list_sp);

protected:
    //------------------------------------------------------------------
    /// Get accessor for the integer block depth value.
    ///
    /// @return
    ///     The integer depth of this block in the block hiearchy.
    //------------------------------------------------------------------
    uint32_t Depth () const;

    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    BlockList *m_block_list; ///< The block list, one of which is this one
    uint32_t m_depth; ///< The depth of this block where zero is the root block
    VMRange::collection m_ranges; ///< A list of address offset ranges relative to the function's section/offset address.
    lldb::InlineFunctionInfoSP m_inlineInfoSP; ///< Inlined function information.
    lldb::VariableListSP m_variables; ///< The variable list for all local, static and paramter variables scoped to this block.
    // TOOD: add a Type* list
};

//----------------------------------------------------------------------
/// @class BlockList Block.h "lldb/Symbol/Block.h"
/// @brief A class that contains a heirachical collection of lexical
/// block objects where one block is the root.
///
/// A collection of Block objects is managed by this class. All access
/// to the block data is made through the block_uid of each block. This
/// facilitates partial parsing and can enable block specific data to
/// only be parsed when the data is asked for (variables, params, types,
/// etc).
//----------------------------------------------------------------------

class BlockList
{
public:
    friend class Block;
    typedef std::vector<Block> collection;///< Our block collection type.

    //------------------------------------------------------------------
    /// Construct with \a function and section offset based address
    /// range.
    ///
    /// @param[in] function
    ///     A const Function object that owns this block list.
    ///
    /// @param[in] range
    ///     A section offset based address range object.
    //------------------------------------------------------------------
    BlockList (Function *function, const AddressRange& range);

    //------------------------------------------------------------------
    /// Destructor.
    //------------------------------------------------------------------
    ~BlockList ();

    //------------------------------------------------------------------
    /// Add a child block to a parent block.
    ///
    /// Adds a new child to a parent block. The UID values for
    /// blocks are created by the SymbolFile plug-ins and should have
    /// values that facilitate correlating an existing Block object
    /// with information in the debug information file. Typically
    /// a table index, or a debug information offset is used.
    ///
    /// @param[in] parent_uid
    ///     The UID for a the existing parent block that will have
    ///     a new child, whose UID is \a child_uid, added to its
    ///     child list.
    ///
    /// @param[in] child_uid
    ///     The UID for the new child block.
    ///
    /// @return
    ///     Returns \a child_uid if the child was successfully added
    ///     to the parent \a parent_uid, or Block::InvalidID on
    ///     failure (if the parent doesn't exist).
    //------------------------------------------------------------------
    lldb::user_id_t
    AddChild (lldb::user_id_t parent_uid, lldb::user_id_t child_uid);

    //------------------------------------------------------------------
    /// Add a child block to a parent block.
    ///
    /// Adds a new child to a parent block. The UID values for
    /// blocks are created by the SymbolFile plug-ins and should have
    /// values that facilitate correlating an existing Block object
    /// with information in the debug information file. Typically
    /// a table index, or a debug information offset is used.
    ///
    /// @param[in] block_uid
    ///     The UID for a the existing block that will get the
    ///     new range.
    ///
    /// @param[in] start_offset
    ///     An offset into this object's address range that
    ///     describes the start address of a range for \a block_uid.
    ///
    /// @param[in] end_offset
    ///     An offset into this object's address range that
    ///     describes the end address of a range for for \a block_uid.
    ///
    /// @return
    ///     Returns \b true if the range was successfully added to
    ///     the block whose UID is \a block_uid, \b false otherwise.
    //------------------------------------------------------------------
    bool
    AddRange (lldb::user_id_t block_uid, lldb::addr_t start_offset, lldb::addr_t end_offset);

//    const Block *
//    FindDeepestBlockForAddress (const Address &addr);

    //------------------------------------------------------------------
    /// Get accessor for the section offset based address range.
    ///
    /// All Block objects contained in a BlockList are relative to
    /// the base address in this object.
    ///
    /// @return
    ///     Returns a reference to the section offset based address
    ///     range object.
    //------------------------------------------------------------------
    AddressRange &
    GetAddressRange ();

    //------------------------------------------------------------------
    /// Get const accessor for the section offset based address range.
    ///
    /// All Block objects contained in a BlockList are relative to
    /// the base address in this object.
    ///
    /// @return
    ///     Returns a const reference to the section offset based
    ///     address range object.
    //------------------------------------------------------------------
    const AddressRange &
    GetAddressRange () const;

    //------------------------------------------------------------------
    /// Dump the block list contents.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    ///
    /// @param[in] block_uid
    ///     The UID of the block in the block list to dump. If this
    ///     value is Block::RootID, then the entire block list will
    ///     dumped as long as \a depth is set to a large enough value.
    ///
    /// @param[in] depth
    ///     Limit the number of levels deep that this function should
    ///     print as the block whose UID is \a block_uid can contain
    ///     child blocks. Specify INT_MAX to dump all child blocks.
    ///
    /// @param[in] show_context
    ///     If \b true, variables will dump their context information.
    //------------------------------------------------------------------
    void
    Dump (Stream *s, lldb::user_id_t block_uid, uint32_t depth, bool show_context) const;

    //------------------------------------------------------------------
    /// Get a block object pointer by block UID.
    ///
    /// @param[in] block_uid
    ///     The UID of the block to retrieve.
    ///
    /// @return
    ///     A pointer to the block object, or NULL if \a block_uid
    ///     doesn't exist in the block list.
    //------------------------------------------------------------------
    Block *
    GetBlockByID (lldb::user_id_t block_uid);

    //------------------------------------------------------------------
    /// Get a const block object pointer by block UID.
    ///
    /// @param[in] block_uid
    ///     The UID of the block to retrieve.
    ///
    /// @return
    ///     A const pointer to the block object, or NULL if \a block_uid
    ///     doesn't exist in the block list.
    //------------------------------------------------------------------
    const Block *
    GetBlockByID (lldb::user_id_t block_uid) const;

    //------------------------------------------------------------------
    /// Get a function object pointer for the block list.
    ///
    /// @return
    ///     A pointer to the function object.
    //------------------------------------------------------------------
    Function *
    GetFunction ();

    //------------------------------------------------------------------
    /// Get a const function object pointer for the block list.
    ///
    /// @return
    ///     A const pointer to the function object.
    //------------------------------------------------------------------
    const Function *
    GetFunction () const;

    //------------------------------------------------------------------
    /// Get the first child block UID for the block whose UID is \a
    /// block_uid.
    ///
    /// @param[in] block_uid
    ///     The UID of the block we wish to access information for.
    ///
    /// @return
    ///     The UID of the first child block, or Block::InvalidID
    ///     if this block has no children, or if \a block_uid is not
    ///     a valid block ID for this block list.
    //------------------------------------------------------------------
    lldb::user_id_t
    GetFirstChild (lldb::user_id_t block_uid) const;

    //------------------------------------------------------------------
    /// Get the parent block UID for the block whose UID is \a
    /// block_uid.
    ///
    /// @param[in] block_uid
    ///     The UID of the block we wish to access information for.
    ///
    /// @return
    ///     The UID of the parent block, or Block::InvalidID
    ///     if this block has no parent, or if \a block_uid is not
    ///     a valid block ID for this block list.
    //------------------------------------------------------------------
    lldb::user_id_t
    GetParent (lldb::user_id_t block_uid) const;

    //------------------------------------------------------------------
    /// Get the sibling block UID for the block whose UID is \a
    /// block_uid.
    ///
    /// @param[in] block_uid
    ///     The UID of the block we wish to access information for.
    ///
    /// @return
    ///     The UID of the sibling block, or Block::InvalidID
    ///     if this block has no sibling, or if \a block_uid is not
    ///     a valid block ID for this block list.
    //------------------------------------------------------------------
    lldb::user_id_t
    GetSibling (lldb::user_id_t block_uid) const;

    //------------------------------------------------------------------
    /// Get the variable list for the block whose UID is \a block_uid.
    ///
    /// @param[in] block_uid
    ///     The UID of the block we wish to access information for.
    ///
    /// @param[in] can_create
    ///     If \b true, the variable list can be parsed on demand. If
    ///     \b false, the variable list contained in this object will
    ///     be returned.
    ///
    /// @return
    ///     The variable list shared pointer which may contain a NULL
    ///     variable list object.
    //------------------------------------------------------------------
    lldb::VariableListSP
    GetVariableList (lldb::user_id_t block_uid, bool get_child_variables, bool can_create);

    //------------------------------------------------------------------
    /// Check if the block list is empty.
    ///
    /// @return
    ///     Returns \b true if the block list is empty, \b false
    ///     otherwise.
    //------------------------------------------------------------------
    bool
    IsEmpty () const;

    //------------------------------------------------------------------
    /// Get the memory cost of this object.
    ///
    /// Returns the cost of this object plus any owned objects (address
    /// range, and contains Block objects).
    ///
    /// @return
    ///     The number of bytes that this object occupies in memory.
    //------------------------------------------------------------------
    size_t
    MemorySize () const;

    //------------------------------------------------------------------
    /// Set the variable list for the block whose UID is \a block_uid.
    ///
    /// @param[in] block_uid
    ///     The UID of the block we wish to set information for.
    ///
    /// @param[in] variable_list_sp
    ///     A shared pointer to list of variables.
    ///
    /// @return
    ///     Returns \b true if the variable list was successfully added
    ///     to the block, \b false otherwise.
    //------------------------------------------------------------------
    bool
    SetVariableList (lldb::user_id_t block_uid, lldb::VariableListSP& variable_list_sp);

    //------------------------------------------------------------------
    /// Set the inlined function info for the block whose UID is \a
    /// block_uid.
    ///
    /// @param[in] block_uid
    ///     The UID of the block we wish to set information for.
    ///
    /// @param[in] name
    ///     The method name for the inlined function. This value should
    ///     not be NULL.
    ///
    /// @param[in] mangled
    ///     The mangled method name for the inlined function. This can
    ///     be NULL if there is no mangled name for an inlined function
    ///     or if the name is the same as \a name.
    ///
    /// @param[in] decl_ptr
    ///     A optional pointer to declaration information for the
    ///     inlined function information. This value can be NULL to
    ///     indicate that no declaration information is available.
    ///
    /// @param[in] call_decl_ptr
    ///     Optional calling location declaration information that
    ///     describes from where this inlined function was called.
    ///
    /// @return
    ///     Returns \b true if the inline function info was successfully
    ///     associated with the block, \b false otherwise.
    //------------------------------------------------------------------
    bool
    SetInlinedFunctionInfo (lldb::user_id_t block_uid, const char *name, const char *mangled, const Declaration *decl_ptr, const Declaration *call_decl_ptr);

protected:
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    Function *m_function;   ///< A pointer to the function that owns this block list.
    AddressRange m_range;   ///< The section offset based address range.
    collection  m_blocks;   ///< A contiguous array of block objects.

    bool
    BlockContainsBlockWithID (const lldb::user_id_t block_id, const lldb::user_id_t find_block_id) const;

private:

    DISALLOW_COPY_AND_ASSIGN (BlockList);
};

} // namespace lldb_private

#endif  // liblldb_Block_h_

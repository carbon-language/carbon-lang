//===-- Flags.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Flags_h_
#define liblldb_Flags_h_
#if defined(__cplusplus)


#include <stdint.h>
#include <unistd.h>

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Flags Flags.h "lldb/Core/Flags.h"
/// @brief A class to manage flag bits.
///
/// The Flags class does bits.
//----------------------------------------------------------------------
class Flags
{
public:
    //----------------------------------------------------------------------
    /// The value type for flag bits is a 32 bit unsigned integer type.
    //----------------------------------------------------------------------
    typedef uint32_t ValueType;

    //----------------------------------------------------------------------
    /// Construct with initial flag bit values.
    ///
    /// Constructs this object with \a bits as the initial value for all
    /// of the flag bits.
    ///
    /// @param[in] bits
    ///     The initial value for all flag bits.
    //----------------------------------------------------------------------
    Flags (ValueType bits = 0);

    //----------------------------------------------------------------------
    /// Copy constructor.
    ///
    /// Construct and copy the flag bits from \a rhs.
    ///
    /// @param[in] rhs
    ///     A const Flags object reference to copy.
    //----------------------------------------------------------------------
    Flags (const Flags& rhs);

    //----------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual in case this class is subclassed.
    //----------------------------------------------------------------------
    virtual
    ~Flags ();

    //----------------------------------------------------------------------
    /// Get accessor for all flag bits.
    ///
    /// @return
    ///     Returns all of the flag bits as a Flags::ValueType.
    //----------------------------------------------------------------------
    ValueType
    GetAllFlagBits () const;

    size_t
    GetBitSize() const;

    //----------------------------------------------------------------------
    /// Set accessor for all flag bits.
    ///
    /// @param[in] bits
    ///     The bits with which to replace all of the current flag bits.
    //----------------------------------------------------------------------
    void
    SetAllFlagBits (ValueType bits);

    //----------------------------------------------------------------------
    /// Clear one or more flag bits.
    ///
    /// @param[in] bits
    ///     A bitfield containing one or more flag bits.
    ///
    /// @return
    ///     The new flag bits after clearing all bits from \a bits.
    //----------------------------------------------------------------------
    ValueType
    Clear (ValueType bits);

    //----------------------------------------------------------------------
    /// Set one or more flag bits.
    ///
    /// @param[in] bits
    ///     A bitfield containing one or more flag bits.
    ///
    /// @return
    ///     The new flag bits after setting all bits from \a bits.
    //----------------------------------------------------------------------
    ValueType
    Set (ValueType bits);

    //----------------------------------------------------------------------
    /// Test one or more flag bits.
    ///
    /// @return
    ///     \b true if \b any flag bits in \a bits are set, \b false
    ///     otherwise.
    //----------------------------------------------------------------------
    bool
    IsSet (ValueType bits) const;

    //----------------------------------------------------------------------
    /// Test one or more flag bits.
    ///
    /// @return
    ///     \b true if \b all flag bits in \a bits are clear, \b false
    ///     otherwise.
    //----------------------------------------------------------------------
    bool
    IsClear (ValueType bits) const;

    //----------------------------------------------------------------------
    /// Get the number of zero bits in \a m_flags.
    ///
    /// @return
    ///     The number of bits that are set to 0 in the current flags.
    //----------------------------------------------------------------------
    size_t
    ClearCount () const;

    //----------------------------------------------------------------------
    /// Get the number of one bits in \a m_flags.
    ///
    /// @return
    ///     The number of bits that are set to 1 in the current flags.
    //----------------------------------------------------------------------
    size_t
    SetCount () const;

protected:
    ValueType   m_flags;    ///< The flag bits.
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Flags_h_

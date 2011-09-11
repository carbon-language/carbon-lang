//===-- ConstString.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ConstString_h_
#define liblldb_ConstString_h_
#if defined(__cplusplus)

#include <assert.h>

#include "lldb/lldb-private.h"
#include "llvm/ADT/StringRef.h"


namespace lldb_private {

//----------------------------------------------------------------------
/// @class ConstString ConstString.h "lldb/Core/ConstString.h"
/// @brief A uniqued constant string class.
///
/// Provides an efficient way to store strings as uniqued ref counted
/// strings. Since the strings are uniqued, finding strings that are
/// equal to one another is very fast (pointer compares). It also allows
/// for many common strings from many different sources to be shared to
/// keep the memory footprint low.
//----------------------------------------------------------------------
class ConstString
{
public:
    //------------------------------------------------------------------
    /// Default constructor
    ///
    /// Initializes the string to an empty string.
    //------------------------------------------------------------------
    ConstString ():
        m_string (NULL)
    {
    }


    //------------------------------------------------------------------
    /// Copy constructor
    ///
    /// Copies the string value in \a rhs and retains an extra reference
    /// to the string value in the string pool.
    ///
    /// @param[in] rhs
    ///     Another string object to copy.
    //------------------------------------------------------------------
    ConstString (const ConstString& rhs) :
        m_string (rhs.m_string)
    {
    }

    //------------------------------------------------------------------
    /// Construct with C String value
    ///
    /// Constructs this object with a C string by looking to see if the
    /// C string already exists in the global string pool. If it does
    /// exist, it retains an extra reference to the string in the string
    /// pool. If it doesn't exist, it is added to the string pool with
    /// a reference count of 1.
    ///
    /// @param[in] cstr
    ///     A NULL terminated C string to add to the string pool.
    //------------------------------------------------------------------
    explicit ConstString (const char *cstr);

    //------------------------------------------------------------------
    /// Construct with C String value with max length
    ///
    /// Constructs this object with a C string with a length. If
    /// \a max_cstr_len is greater than the actual length of the string,
    /// the string length will be truncated. This allows substrings to
    /// be created without the need to NULL terminate the string as it
    /// is passed into this function.
    ///
    /// If the C string already exists in the global string pool, it
    /// retains an extra reference to the string in the string
    /// pool. If it doesn't exist, it is added to the string pool with
    /// a reference count of 1.
    ///
    /// @param[in] cstr
    ///     A NULL terminated C string to add to the string pool.
    ///
    /// @param[in] max_cstr_len
    ///     The max length of \a cstr. If the string length of \a cstr
    ///     is less than \a max_cstr_len, then the string will be
    ///     truncated. If the string length of \a cstr is greater than
    ///     \a max_cstr_len, then only max_cstr_len bytes will be used
    ///     from \a cstr.
    //------------------------------------------------------------------
    explicit ConstString (const char *cstr, size_t max_cstr_len);

    //------------------------------------------------------------------
    /// Destructor
    ///
    /// Decrements the reference count on the contained string, and if
    /// the resulting reference count is zero, then the string is removed
    /// from the global string pool. If the reference count is still
    /// greater than zero, the string will remain in the string pool
    /// until the last reference is released by other ConstString objects.
    //------------------------------------------------------------------
    ~ConstString ()
    {
    }


    //----------------------------------------------------------------------
    /// C string equality function object for CStrings contains in the
    /// same StringPool only. (binary predicate).
    //----------------------------------------------------------------------
    struct StringIsEqual
    {
        //--------------------------------------------------------------
        /// C equality test.
        ///
        /// Two C strings are equal when they are contained in ConstString
        /// objects when their pointer values are equal to each other.
        ///
        /// @return
        ///     Returns \b true if the C string in \a lhs is equal to
        ///     the C string value in \a rhs, \b false otherwise.
        //--------------------------------------------------------------
        bool operator()(const char* lhs, const char* rhs) const
        {
            return lhs == rhs;
        }
    };

    //------------------------------------------------------------------
    /// Convert to bool operator.
    ///
    /// This allows code to check a ConstString object to see if it
    /// contains a valid string using code such as:
    ///
    /// @code
    /// ConstString str(...);
    /// if (str)
    /// { ...
    /// @endcode
    ///
    /// @return
    ///     /b True this object contains a valid non-empty C string, \b 
    ///     false otherwise.
    //------------------------------------------------------------------
    operator bool() const
    {
        return m_string && m_string[0];
    }

    //------------------------------------------------------------------
    /// Assignment operator
    ///
    /// Assigns the string in this object with the value from \a rhs
    /// and increments the reference count of that string.
    ///
    /// The previously contained string will be get its reference count
    /// decremented and removed from the string pool if its reference
    /// count reaches zero.
    ///
    /// @param[in] rhs
    ///     Another string object to copy into this object.
    ///
    /// @return
    ///     A const reference to this object.
    //------------------------------------------------------------------
    const ConstString&
    operator = (const ConstString& rhs)
    {
        m_string = rhs.m_string;
        return *this;
    }

    //------------------------------------------------------------------
    /// Equal to operator
    ///
    /// Returns true if this string is equal to the string in \a rhs.
    /// This operation is very fast as it results in a pointer
    /// comparison since all strings are in a uniqued and reference
    /// counted string pool.
    ///
    /// @param[in] rhs
    ///     Another string object to compare this object to.
    ///
    /// @return
    ///     @li \b true if this object is equal to \a rhs.
    ///     @li \b false if this object is not equal to \a rhs.
    //------------------------------------------------------------------
    bool
    operator == (const ConstString& rhs) const
    {
        // We can do a pointer compare to compare these strings since they
        // must come from the same pool in order to be equal.
        return m_string == rhs.m_string;
    }

    //------------------------------------------------------------------
    /// Not equal to operator
    ///
    /// Returns true if this string is not equal to the string in \a rhs.
    /// This operation is very fast as it results in a pointer
    /// comparison since all strings are in a uniqued and reference
    /// counted string pool.
    ///
    /// @param[in] rhs
    ///     Another string object to compare this object to.
    ///
    /// @return
    ///     @li \b true if this object is not equal to \a rhs.
    ///     @li \b false if this object is equal to \a rhs.
    //------------------------------------------------------------------
    bool
    operator != (const ConstString& rhs) const
    {
        return m_string != rhs.m_string;
    }

    bool
    operator < (const ConstString& rhs) const;

    //------------------------------------------------------------------
    /// Get the string value as a C string.
    ///
    /// Get the value of the contained string as a NULL terminated C
    /// string value.
    ///
    /// If \a value_if_empty is NULL, then NULL will be returned.
    ///
    /// @return
    ///     Returns \a value_if_empty if the string is empty, otherwise
    ///     the C string value contained in this object.
    //------------------------------------------------------------------
    const char *
    AsCString(const char *value_if_empty = NULL) const
    {
        if (m_string == NULL)
            return value_if_empty;
        return m_string;
    }

    llvm::StringRef
    GetStringRef () const
    {
        return llvm::StringRef (m_string, GetLength());
    }
    
    const char *
    GetCString () const
    {
        return m_string;
    }


    size_t
    GetLength () const;
    //------------------------------------------------------------------
    /// Clear this object's state.
    ///
    /// Clear any contained string and reset the value to the an empty
    /// string value.
    ///
    /// The previously contained string will be get its reference count
    /// decremented and removed from the string pool if its reference
    /// count reaches zero.
    //------------------------------------------------------------------
    void
    Clear ()
    {
        m_string = NULL;
    }

    //------------------------------------------------------------------
    /// Compare two string objects.
    ///
    /// Compares the C string values contained in \a lhs and \a rhs and
    /// returns an integer result.
    ///
    /// @param[in] lhs
    ///     The Left Hand Side const ConstString object reference.
    ///
    /// @param[in] rhs
    ///     The Right Hand Side const ConstString object reference.
    ///
    /// @return
    ///     @li -1 if lhs < rhs
    ///     @li 0 if lhs == rhs
    ///     @li 1 if lhs > rhs
    //------------------------------------------------------------------
    static int
    Compare (const ConstString& lhs, const ConstString& rhs);

    //------------------------------------------------------------------
    /// Dump the object description to a stream.
    ///
    /// Dump the string value to the stream \a s. If the contained string
    /// is empty, print \a value_if_empty to the stream instead. If
    /// \a value_if_empty is NULL, then nothing will be dumped to the
    /// stream.
    ///
    /// @param[in] s
    ///     The stream that will be used to dump the object description.
    ///
    /// @param[in] value_if_empty
    ///     The value to dump if the string is empty. If NULL, nothing
    ///     will be output to the stream.
    //------------------------------------------------------------------
    void
    Dump (Stream *s, const char *value_if_empty = NULL) const;

    //------------------------------------------------------------------
    /// Dump the object debug description to a stream.
    ///
    /// Dump the string value and the reference count to the stream \a
    /// s.
    ///
    /// @param[in] s
    ///     The stream that will be used to dump the object description.
    //------------------------------------------------------------------
    void
    DumpDebug (Stream *s) const;

    //------------------------------------------------------------------
    /// Test for empty string.
    ///
    /// @return
    ///     @li \b true if the contained string is empty.
    ///     @li \b false if the contained string is not empty.
    //------------------------------------------------------------------
    bool
    IsEmpty () const
    {
        return m_string == NULL || m_string[0] == '\0';
    }


    //------------------------------------------------------------------
    /// Set the C string value.
    ///
    /// Set the string value in the object by uniquing the \a cstr
    /// string value in our global string pool.
    ///
    /// If the C string already exists in the global string pool, it
    /// finds the current entry and retains an extra reference to the
    /// string in the string pool. If it doesn't exist, it is added to
    /// the string pool with a reference count of 1.
    ///
    /// @param[in] cstr
    ///     A NULL terminated C string to add to the string pool.
    //------------------------------------------------------------------
    void
    SetCString (const char *cstr);

    void
    SetCStringWithMangledCounterpart (const char *demangled, const ConstString &mangled);

    bool
    GetMangledCounterpart (ConstString &counterpart) const;

    //------------------------------------------------------------------
    /// Set the C string value with length.
    ///
    /// Set the string value in the object by uniquing \a cstr_len bytes
    /// starting at the \a cstr string value in our global string pool.
    /// If trim is true, then \a cstr_len indicates a maximum length of
    /// the CString and if the actual length of the string is less, then
    /// it will be trimmed. If trim is false, then this allows strings
    /// with NULL characters to be added to the string pool.
    ///
    /// If the C string already exists in the global string pool, it
    /// retains an extra reference to the string in the string
    /// pool. If it doesn't exist, it is added to the string pool with
    /// a reference count of 1.
    ///
    /// @param[in] cstr
    ///     A NULL terminated C string to add to the string pool.
    ///
    /// @param[in] cstr_len
    ///     The absolute length of the C string if \a trim is false,
    ///     or the maximum length of the C string if \a trim is true.
    ///
    /// @param[in] trim
    ///     If \b true, trim \a cstr to it's actual length before adding
    ///     it to the string pool. If \b false then cstr_len is the
    ///     actual length of the C string to add.
    //------------------------------------------------------------------
    void
    SetCStringWithLength (const char *cstr, size_t cstr_len);

    //------------------------------------------------------------------
    /// Set the C string value with the minimum length between
    /// \a fixed_cstr_len and the actual length of the C string. This
    /// can be used for data structures that have a fixed length to
    /// store a C string where the string might not be NULL terminated
    /// if the string takes the entire buffer.
    //------------------------------------------------------------------
    void
    SetTrimmedCStringWithLength (const char *cstr, size_t fixed_cstr_len);

    //------------------------------------------------------------------
    /// Get the memory cost of this object.
    ///
    /// Return the size in bytes that this object takes in memory. This
    /// returns the size in bytes of this object, which does not include
    /// any the shared string values it may refer to.
    ///
    /// @return
    ///     The number of bytes that this object occupies in memory.
    ///
    /// @see ConstString::StaticMemorySize ()
    //------------------------------------------------------------------
    size_t
    MemorySize () const
    {
        return sizeof(ConstString);
    }
    

    //------------------------------------------------------------------
    /// Get the size in bytes of the current global string pool.
    ///
    /// Reports the the size in bytes of all shared C string values,
    /// containers and reference count values as a byte size for the
    /// entire string pool.
    ///
    /// @return
    ///     The number of bytes that the global string pool occupies
    ///     in memory.
    //------------------------------------------------------------------
    static size_t
    StaticMemorySize ();

protected:
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    const char *m_string;
};

//------------------------------------------------------------------
/// Stream the string value \a str to the stream \a s
//------------------------------------------------------------------
Stream& operator << (Stream& s, const ConstString& str);

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_ConstString_h_

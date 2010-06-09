//===-- Language.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Language_h_
#define liblldb_Language_h_

#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Language Language.h "lldb/Core/Language.h"
/// @brief Encapsulates the programming language for an lldb object.
///
/// Languages are represented by an enumeration value.
///
/// The enumeration values used when describing the programming language
/// are the same values as the latest DWARF specification.
//----------------------------------------------------------------------
class Language
{
public:

    //------------------------------------------------------------------
    /// Programming language type.
    ///
    /// These enumerations use the same language enumerations as the
    /// DWARF specification for ease of use and consistency.
    //------------------------------------------------------------------
    typedef enum
    {
        Unknown         = 0x0000,   ///< Unknown or invalid language value.
        C89             = 0x0001,   ///< ISO C:1989.
        C               = 0x0002,   ///< Non-standardized C, such as K&R.
        Ada83           = 0x0003,   ///< ISO Ada:1983.
        C_plus_plus     = 0x0004,   ///< ISO C++:1998.
        Cobol74         = 0x0005,   ///< ISO Cobol:1974.
        Cobol85         = 0x0006,   ///< ISO Cobol:1985.
        Fortran77       = 0x0007,   ///< ISO Fortran 77.
        Fortran90       = 0x0008,   ///< ISO Fortran 90.
        Pascal83        = 0x0009,   ///< ISO Pascal:1983.
        Modula2         = 0x000a,   ///< ISO Modula-2:1996.
        Java            = 0x000b,   ///< Java.
        C99             = 0x000c,   ///< ISO C:1999.
        Ada95           = 0x000d,   ///< ISO Ada:1995.
        Fortran95       = 0x000e,   ///< ISO Fortran 95.
        PLI             = 0x000f,   ///< ANSI PL/I:1976.
        ObjC            = 0x0010,   ///< Objective-C.
        ObjC_plus_plus  = 0x0011,   ///< Objective-C++.
        UPC             = 0x0012,   ///< Unified Parallel C.
        D               = 0x0013,   ///< D.
        Python          = 0x0014    ///< Python.
    } Type;

    //------------------------------------------------------------------
    /// Construct with optional language enumeration.
    //------------------------------------------------------------------
    Language(Language::Type language = Unknown);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual in case this class is subclassed.
    //------------------------------------------------------------------
    virtual
    ~Language();

    //------------------------------------------------------------------
    /// Get the language value as a NULL termianted C string.
    ///
    /// @return
    ///     The C string representation of the language. The returned
    ///     string does not need to be freed as it comes from constant
    ///     strings. NULL can be returned when the language is set to
    ///     a value that doesn't match of of the Language::Type
    ///     enumerations.
    //------------------------------------------------------------------
    const char *
    AsCString (lldb::DescriptionLevel level = lldb::eDescriptionLevelBrief) const;

    void
    Clear();

    void
    GetDescription (Stream *s, lldb::DescriptionLevel level) const;

    //------------------------------------------------------------------
    /// Dump the language value to the stream \a s.
    ///
    /// @param[in] s
    ///     The stream to which to dump the language description.
    //------------------------------------------------------------------
    void
    Dump(Stream *s) const;

    //------------------------------------------------------------------
    /// Get accessor for the language.
    ///
    /// @return
    ///     The enumeration value that describes the programming
    ///     language that an object is associated with.
    //------------------------------------------------------------------
    Language::Type
    GetLanguage() const;

    //------------------------------------------------------------------
    /// Set accessor for the language.
    ///
    /// @param[in] language
    ///     The new enumeration value that describes the programming
    ///     language that an object is associated with.
    //------------------------------------------------------------------
    void
    SetLanguage(Language::Type language);

    //------------------------------------------------------------------
    /// Set accessor for the language.
    ///
    /// @param[in] language_cstr
    ///     The language name as a C string.
    //------------------------------------------------------------------
    bool
    SetLanguageFromCString(const char *language_cstr);


protected:
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    Language::Type m_language; ///< The programming language enumeration value.
                                 ///< The enumeration values are the same as the
                                 ///< latest DWARF specification.
};

//--------------------------------------------------------------
/// Stream the language enumeration as a string object to a
/// Stream.
//--------------------------------------------------------------
Stream& operator << (Stream& s, const Language& language);

} // namespace lldb_private

#endif  // liblldb_Language_h_

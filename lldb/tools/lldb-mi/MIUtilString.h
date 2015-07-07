//===-- MIUtilString.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// Third party headers:
#include <string>
#include <vector>
#include <cinttypes>

// In-house headers:
#include "MIDataTypes.h"

//++ ============================================================================
// Details: MI common code utility class. Used to help handle text.
//          Derived from std::string
// Gotchas: None.
// Authors: Illya Rudkin 02/02/2014.
// Changes: None.
//--
class CMIUtilString : public std::string
{
    // Typedefs:
  public:
    typedef std::vector<CMIUtilString> VecString_t;

    // Static method:
  public:
    static CMIUtilString Format(const CMIUtilString vFormating, ...);
    static CMIUtilString FormatBinary(const MIuint64 vnDecimal);
    static CMIUtilString FormatValist(const CMIUtilString &vrFormating, va_list vArgs);
    static bool IsAllValidAlphaAndNumeric(const char *vpText);
    static bool Compare(const CMIUtilString &vrLhs, const CMIUtilString &vrRhs);
    static CMIUtilString ConvertToPrintableASCII(const char vChar);
    static CMIUtilString ConvertToPrintableASCII(const char16_t vChar16);
    static CMIUtilString ConvertToPrintableASCII(const char32_t vChar32);

    // Methods:
  public:
    /* ctor */ CMIUtilString(void);
    /* ctor */ CMIUtilString(const char *vpData);
    /* ctor */ CMIUtilString(const char *const *vpData);
    //
    bool ExtractNumber(MIint64 &vwrNumber) const;
    CMIUtilString FindAndReplace(const CMIUtilString &vFind, const CMIUtilString &vReplaceWith) const;
    bool IsNumber(void) const;
    bool IsHexadecimalNumber(void) const;
    bool IsQuoted(void) const;
    CMIUtilString RemoveRepeatedCharacters(const char vChar);
    size_t Split(const CMIUtilString &vDelimiter, VecString_t &vwVecSplits) const;
    size_t SplitConsiderQuotes(const CMIUtilString &vDelimiter, VecString_t &vwVecSplits) const;
    size_t SplitLines(VecString_t &vwVecSplits) const;
    CMIUtilString StripCREndOfLine(void) const;
    CMIUtilString StripCRAll(void) const;
    CMIUtilString Trim(void) const;
    CMIUtilString Trim(const char vChar) const;
    size_t FindFirst(const CMIUtilString &vrPattern, size_t vnPos = 0) const;
    size_t FindFirst(const CMIUtilString &vrPattern, bool vbSkipQuotedText, bool &vrwbNotFoundClosedQuote,
                     size_t vnPos = 0) const;
    size_t FindFirstNot(const CMIUtilString &vrPattern, size_t vnPos = 0) const;
    CMIUtilString Escape(bool vbEscapeQuotes = false) const;
    CMIUtilString AddSlashes(void) const;
    CMIUtilString StripSlashes(void) const;
    //
    CMIUtilString &operator=(const char *vpRhs);
    CMIUtilString &operator=(const std::string &vrRhs);

    // Overrideable:
  public:
    /* dtor */ virtual ~CMIUtilString(void);

    // Static method:
  private:
    static CMIUtilString FormatPriv(const CMIUtilString &vrFormat, va_list vArgs);

    // Methods:
  private:
    bool ExtractNumberFromHexadecimal(MIint64 &vwrNumber) const;
    CMIUtilString RemoveRepeatedCharacters(size_t vnPos, const char vChar);
    size_t FindFirstQuote(size_t vnPos) const;
};

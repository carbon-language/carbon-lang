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
    // Typdefs:
  public:
    typedef std::vector<CMIUtilString> VecString_t;

    // Static method:
  public:
    static CMIUtilString Format(const CMIUtilString vFormating, ...);
    static CMIUtilString FormatBinary(const MIuint64 vnDecimal);
    static CMIUtilString FormatValist(const CMIUtilString &vrFormating, va_list vArgs);
    static bool IsAllValidAlphaAndNumeric(const MIchar &vrText);
    static bool Compare(const CMIUtilString &vrLhs, const CMIUtilString &vrRhs);
    static CMIUtilString ConvertToPrintableASCII(const char vChar);
    static CMIUtilString ConvertToPrintableASCII(const char16_t vChar16);
    static CMIUtilString ConvertToPrintableASCII(const char32_t vChar32);

    // Methods:
  public:
    /* ctor */ CMIUtilString(void);
    /* ctor */ CMIUtilString(const MIchar *vpData);
    /* ctor */ CMIUtilString(const MIchar *const *vpData);
    //
    bool ExtractNumber(MIint64 &vwrNumber) const;
    CMIUtilString FindAndReplace(const CMIUtilString &vFind, const CMIUtilString &vReplaceWith) const;
    bool IsNumber(void) const;
    bool IsHexadecimalNumber(void) const;
    bool IsQuoted(void) const;
    CMIUtilString RemoveRepeatedCharacters(const MIchar vChar);
    MIuint Split(const CMIUtilString &vDelimiter, VecString_t &vwVecSplits) const;
    MIuint SplitConsiderQuotes(const CMIUtilString &vDelimiter, VecString_t &vwVecSplits) const;
    MIuint SplitLines(VecString_t &vwVecSplits) const;
    CMIUtilString StripCREndOfLine(void) const;
    CMIUtilString StripCRAll(void) const;
    CMIUtilString Trim(void) const;
    CMIUtilString Trim(const MIchar vChar) const;
    MIuint FindFirst(const CMIUtilString &vrPattern, const MIuint vnPos = 0) const;
    MIuint FindFirst(const CMIUtilString &vrPattern, const bool vbSkipQuotedText, bool &vrwbNotFoundClosedQuote,
                     const MIuint vnPos = 0) const;
    MIuint FindFirstNot(const CMIUtilString &vrPattern, const MIuint vnPos = 0) const;
    CMIUtilString Escape(const bool vbEscapeQuotes = false) const;
    CMIUtilString AddSlashes(void) const;
    CMIUtilString StripSlashes(void) const;
    //
    CMIUtilString &operator=(const MIchar *vpRhs);
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
    CMIUtilString RemoveRepeatedCharacters(const MIint vnPos, const MIchar vChar);
    MIuint FindFirstQuote(const MIuint vnPos) const;
};

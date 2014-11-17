//===-- MIUtilString.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MIUtilString.h
//
// Overview:    CMIUtilString implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// Third party headers
#include <memory>   // std::unique_ptr
#include <stdarg.h> // va_list, va_start, var_end
#include <sstream>  // std::stringstream
#include <string.h> // for strcpy
#include <limits.h> // for ULONG_MAX

// In-house headers:
#include "MIUtilString.h"

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilString constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMIUtilString::CMIUtilString(void)
    : std::string()
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilString constructor.
// Type:    Method.
// Args:    vpData  - Pointer to UTF8 text data.
// Return:  None.
// Throws:  None.
//--
CMIUtilString::CMIUtilString(const MIchar *vpData)
    : std::string(vpData)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilString constructor.
// Type:    Method.
// Args:    vpData  - Pointer to UTF8 text data.
// Return:  None.
// Throws:  None.
//--
CMIUtilString::CMIUtilString(const MIchar *const *vpData)
    : std::string((const char *)vpData)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilString assigment operator.
// Type:    Method.
// Args:    vpRhs   - Pointer to UTF8 text data.
// Return:  CMIUtilString & - *this string.
// Throws:  None.
//--
CMIUtilString &CMIUtilString::operator=(const MIchar *vpRhs)
{
    if (*this == vpRhs)
        return *this;

    if (vpRhs != nullptr)
    {
        assign(vpRhs);
    }

    return *this;
}

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilString assigment operator.
// Type:    Method.
// Args:    vrRhs   - The other string to copy from.
// Return:  CMIUtilString & - *this string.
// Throws:  None.
//--
CMIUtilString &CMIUtilString::operator=(const std::string &vrRhs)
{
    if (*this == vrRhs)
        return *this;

    assign(vrRhs);

    return *this;
}

//++ ------------------------------------------------------------------------------------
// Details: CMIUtilString destructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMIUtilString::~CMIUtilString(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: Perform a snprintf format style on a string data. A new string object is
//          created and returned.
// Type:    Static method.
// Args:    vrFormat      - (R) Format string data instruction.
//          vArgs         - (R) Var list args of any type.
// Return:  CMIUtilString - Number of splits found in the string data.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::FormatPriv(const CMIUtilString &vrFormat, va_list vArgs)
{
    CMIUtilString strResult;
    MIint nFinal = 0;
    MIint n = vrFormat.size();

    // IOR: mysterious crash in this function on some windows builds not able to duplicate
    // but found article which may be related. Crash occurs in vsnprintf() or va_copy()
    // Duplicate vArgs va_list argument pointer to ensure that it can be safely used in
    // a new frame
    // http://julipedia.meroh.net/2011/09/using-vacopy-to-safely-pass-ap.html
    va_list argsDup;
    va_copy(argsDup, vArgs);

    // Create a copy va_list to reset when we spin
    va_list argsCpy;
    va_copy(argsCpy, argsDup);

    if (n == 0)
        return strResult;

    n = n << 4; // Reserve 16 times as much the length of the vrFormat

    std::unique_ptr<char[]> pFormatted;
    while (1)
    {
        pFormatted.reset(new char[n + 1]); // +1 for safety margin
        ::strncpy(&pFormatted[0], vrFormat.c_str(), n);

        //  We need to restore the variable argument list pointer to the start again
        //  before running vsnprintf() more then once
        va_copy(argsDup, argsCpy);

        nFinal = ::vsnprintf(&pFormatted[0], n, vrFormat.c_str(), argsDup);
        if ((nFinal < 0) || (nFinal >= n))
            n += abs(nFinal - n + 1);
        else
            break;
    }

    va_end(argsCpy);
    va_end(argsDup);

    strResult = pFormatted.get();

    return strResult;
}

//++ ------------------------------------------------------------------------------------
// Details: Perform a snprintf format style on a string data. A new string object is
//          created and returned.
// Type:    Static method.
// Args:    vFormat       - (R) Format string data instruction.
//          ...           - (R) Var list args of any type.
// Return:  CMIUtilString - Number of splits found in the string data.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::Format(const CMIUtilString vFormating, ...)
{
    va_list args;
    va_start(args, vFormating);
    CMIUtilString strResult = CMIUtilString::FormatPriv(vFormating, args);
    va_end(args);

    return strResult;
}

//++ ------------------------------------------------------------------------------------
// Details: Perform a snprintf format style on a string data. A new string object is
//          created and returned.
// Type:    Static method.
// Args:    vrFormat      - (R) Format string data instruction.
//          vArgs         - (R) Var list args of any type.
// Return:  CMIUtilString - Number of splits found in the string data.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::FormatValist(const CMIUtilString &vrFormating, va_list vArgs)
{
    return CMIUtilString::FormatPriv(vrFormating, vArgs);
}

//++ ------------------------------------------------------------------------------------
// Details: Splits string into array of strings using delimiter. If multiple delimiter
//          are found in sequence then they are not added to the list of splits.
// Type:    Method.
// Args:    vData       - (R) String data to be split up.
//          vDelimiter  - (R) Delimiter char or text.
//          vwVecSplits - (W) Container of splits found in string data.
// Return:  MIuint - Number of splits found in the string data.
// Throws:  None.
//--
MIuint
CMIUtilString::Split(const CMIUtilString &vDelimiter, VecString_t &vwVecSplits) const
{
    vwVecSplits.clear();

    if (this->empty() || vDelimiter.empty())
        return 0;

    MIint nPos = find(vDelimiter);
    if (nPos == (MIint)std::string::npos)
    {
        vwVecSplits.push_back(*this);
        return 1;
    }
    const MIint strLen(length());
    if (nPos == strLen)
    {
        vwVecSplits.push_back(*this);
        return 1;
    }

    MIuint nAdd1(1);
    if ((nPos > 0) && (substr(0, nPos) != vDelimiter))
    {
        nPos = 0;
        nAdd1 = 0;
    }
    MIint nPos2 = find(vDelimiter, nPos + 1);
    while (nPos2 != (MIint)std::string::npos)
    {
        const MIuint len(nPos2 - nPos - nAdd1);
        const std::string strSection(substr(nPos + nAdd1, len));
        if (strSection != vDelimiter)
            vwVecSplits.push_back(strSection.c_str());
        nPos += len + 1;
        nPos2 = find(vDelimiter, nPos + 1);
        nAdd1 = 0;
    }
    const std::string strSection(substr(nPos, strLen - nPos));
    if ((strSection.length() != 0) && (strSection != vDelimiter))
        vwVecSplits.push_back(strSection.c_str());

    return vwVecSplits.size();
}

//++ ------------------------------------------------------------------------------------
// Details: Splits string into array of strings using delimiter. However the string is
//          also considered for text surrounded by quotes. Text with quotes including the
//          delimiter is treated as a whole. If multiple delimiter are found in sequence
//          then they are not added to the list of splits. Quotes that are embedded in the
//          the string as string formatted quotes are ignored (proceeded by a '\\') i.e.
//          "\"MI GDB local C++.cpp\":88".
// Type:    Method.
// Args:    vData       - (R) String data to be split up.
//          vDelimiter  - (R) Delimiter char or text.
//          vwVecSplits - (W) Container of splits found in string data.
// Return:  MIuint - Number of splits found in the string data.
// Throws:  None.
//--
MIuint
CMIUtilString::SplitConsiderQuotes(const CMIUtilString &vDelimiter, VecString_t &vwVecSplits) const
{
    vwVecSplits.clear();

    if (this->empty() || vDelimiter.empty())
        return 0;

    MIint nPos = find(vDelimiter);
    if (nPos == (MIint)std::string::npos)
    {
        vwVecSplits.push_back(*this);
        return 1;
    }
    const MIint strLen(length());
    if (nPos == strLen)
    {
        vwVecSplits.push_back(*this);
        return 1;
    }

    // Look for more quotes
    bool bHaveQuotes = false;
    const MIchar cBckSlash = '\\';
    const MIchar cQuote = '"';
    MIint nPosQ = find(cQuote);
    MIint nPosQ2 = (MIint)std::string::npos;
    if (nPosQ != (MIint)std::string::npos)
    {
        nPosQ2 = nPosQ + 1;
        while (nPosQ2 < strLen)
        {
            nPosQ2 = find(cQuote, nPosQ2);
            if ((nPosQ2 == (MIint)std::string::npos) || (at(nPosQ2 - 1) != cBckSlash))
                break;
            nPosQ2++;
        }
        bHaveQuotes = (nPosQ2 != (MIint)std::string::npos);
    }

    MIuint nAdd1(1);
    if ((nPos > 0) && (substr(0, nPos) != vDelimiter))
    {
        nPos = 0;
        nAdd1 = 0;
    }
    MIint nPos2 = find(vDelimiter, nPos + 1);
    while (nPos2 != (MIint)std::string::npos)
    {
        if (!bHaveQuotes || (bHaveQuotes && ((nPos2 > nPosQ2) || (nPos2 < nPosQ))))
        {
            // Extract text or quoted text
            const MIuint len(nPos2 - nPos - nAdd1);
            const std::string strSection(substr(nPos + nAdd1, len));
            if (strSection != vDelimiter)
                vwVecSplits.push_back(strSection.c_str());
            nPos += len + 1;
            nPos2 = find(vDelimiter, nPos + 1);
            nAdd1 = 0;

            if (bHaveQuotes && (nPos2 > nPosQ2))
            {
                // Reset, look for more quotes
                bHaveQuotes = false;
                nPosQ = find(cQuote, nPos);
                nPosQ2 = (MIint)std::string::npos;
                if (nPosQ != (MIint)std::string::npos)
                {
                    nPosQ2 = find(cQuote, nPosQ + 1);
                    bHaveQuotes = (nPosQ2 != (MIint)std::string::npos);
                }
            }
        }
        else
        {
            // Skip passed text in quotes
            nPos2 = find(vDelimiter, nPosQ2 + 1);
        }
    }
    const std::string strSection(substr(nPos, strLen - nPos));
    if ((strSection.length() != 0) && (strSection != vDelimiter))
        vwVecSplits.push_back(strSection.c_str());

    return vwVecSplits.size();
}

//++ ------------------------------------------------------------------------------------
// Details: Remove '\n' from the end of string if found. It does not alter
//          *this string.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString - New version of the string.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::StripCREndOfLine(void) const
{
    const MIint nPos = rfind('\n');
    if (nPos == (MIint)std::string::npos)
        return *this;

    const CMIUtilString strNew(substr(0, nPos).c_str());

    return strNew;
}

//++ ------------------------------------------------------------------------------------
// Details: Remove all '\n' from the string and replace with a space. It does not alter
//          *this string.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString - New version of the string.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::StripCRAll(void) const
{
    return FindAndReplace("\n", " ");
}

//++ ------------------------------------------------------------------------------------
// Details: Find and replace all matches of a sub string with another string. It does not
//          alter *this string.
// Type:    Method.
// Args:    vFind         - (R) The string to look for.
//          vReplaceWith  - (R) The string to replace the vFind match.
// Return:  CMIUtilString - New version of the string.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::FindAndReplace(const CMIUtilString &vFind, const CMIUtilString &vReplaceWith) const
{
    if (vFind.empty() || this->empty())
        return *this;

    MIint nPos = find(vFind);
    if (nPos == (MIint)std::string::npos)
        return *this;

    CMIUtilString strNew(*this);
    while (nPos != (MIint)std::string::npos)
    {
        strNew.replace(nPos, vFind.length(), vReplaceWith);
        nPos += vReplaceWith.length();
        nPos = strNew.find(vFind, nPos);
    }

    return strNew;
}

//++ ------------------------------------------------------------------------------------
// Details: Check if *this string is a decimal number.
// Type:    Method.
// Args:    None.
// Return:  bool - True = yes number, false not a number.
// Throws:  None.
//--
bool
CMIUtilString::IsNumber(void) const
{
    if (empty())
        return false;

    if ((at(0) == '-') && (length() == 1))
        return false;

    const MIint nPos = find_first_not_of("-.0123456789");
    if (nPos != (MIint)std::string::npos)
        return false;

    return true;
}

//++ ------------------------------------------------------------------------------------
// Details: Extract the number from the string. The number can be either a hexadecimal or
//          natural number. It cannot contain other non-numeric characters.
// Type:    Method.
// Args:    vwrNumber   - (W) Number exracted from the string.
// Return:  bool - True = yes number, false not a number.
// Throws:  None.
//--
bool
CMIUtilString::ExtractNumber(MIint64 &vwrNumber) const
{
    vwrNumber = 0;

    if (!IsNumber())
    {
        if (ExtractNumberFromHexadecimal(vwrNumber))
            return true;

        return false;
    }

    std::stringstream ss(const_cast<CMIUtilString &>(*this));
    ss >> vwrNumber;

    return true;
}

//++ ------------------------------------------------------------------------------------
// Details: Extract the number from the hexadecimal string..
// Type:    Method.
// Args:    vwrNumber   - (W) Number exracted from the string.
// Return:  bool - True = yes number, false not a number.
// Throws:  None.
//--
bool
CMIUtilString::ExtractNumberFromHexadecimal(MIint64 &vwrNumber) const
{
    vwrNumber = 0;

    const MIint nPos = find_first_not_of("x01234567890ABCDEFabcedf");
    if (nPos != (MIint)std::string::npos)
        return false;

    const MIint64 nNum = ::strtoul(this->c_str(), nullptr, 16);
    if (nNum != LONG_MAX)
    {
        vwrNumber = nNum;
        return true;
    }

    return true;
}

//++ ------------------------------------------------------------------------------------
// Details: Determine if the text is all valid alpha numeric characters. Letters can be
//          either upper or lower case.
// Type:    Static method.
// Args:    vrText  - (R) The text data to examine.
// Return:  bool - True = yes all alpha, false = one or more chars is non alpha.
// Throws:  None.
//--
bool
CMIUtilString::IsAllValidAlphaAndNumeric(const MIchar &vrText)
{
    const MIuint len = ::strlen(&vrText);
    if (len == 0)
        return false;

    MIchar *pPtr = const_cast<MIchar *>(&vrText);
    for (MIuint i = 0; i < len; i++, pPtr++)
    {
        const MIchar c = *pPtr;
        if (::isalnum((int)c) == 0)
            return false;
    }

    return true;
}

//++ ------------------------------------------------------------------------------------
// Details: Check if two strings share equal contents.
// Type:    Method.
// Args:    vrLhs   - (R) String A.
//          vrRhs   - (R) String B.
// Return:  bool - True = yes equal, false - different.
// Throws:  None.
//--
bool
CMIUtilString::Compare(const CMIUtilString &vrLhs, const CMIUtilString &vrRhs)
{
    // Check the sizes match
    if (vrLhs.size() != vrRhs.size())
        return false;

    return (::strncmp(vrLhs.c_str(), vrRhs.c_str(), vrLhs.size()) == 0);
}

//++ ------------------------------------------------------------------------------------
// Details: Remove from either end of *this string the following: " \t\n\v\f\r".
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString - Trimmed string.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::Trim(void) const
{
    CMIUtilString strNew(*this);
    const MIchar *pWhiteSpace = " \t\n\v\f\r";
    const MIint nPos = find_last_not_of(pWhiteSpace);
    if (nPos != (MIint)std::string::npos)
    {
        strNew = substr(0, nPos + 1).c_str();
    }
    const MIint nPos2 = strNew.find_first_not_of(pWhiteSpace);
    if (nPos2 != (MIint)std::string::npos)
    {
        strNew = strNew.substr(nPos2).c_str();
    }

    return strNew;
}

//++ ------------------------------------------------------------------------------------
// Details: Remove from either end of *this string the specified character.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString - Trimmed string.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::Trim(const MIchar vChar) const
{
    CMIUtilString strNew(*this);
    const MIint nLen = strNew.length();
    if (nLen > 1)
    {
        if ((strNew[0] == vChar) && (strNew[nLen - 1] == vChar))
            strNew = strNew.substr(1, nLen - 2).c_str();
    }

    return strNew;
}

//++ ------------------------------------------------------------------------------------
// Details: Do a printf equivalent for printing a number in binary i.e. "b%llB".
// Type:    Static method.
// Args:    vnDecimal   - (R) The number to represent in binary.
// Return:  CMIUtilString - Binary number in text.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::FormatBinary(const MIuint64 vnDecimal)
{
    CMIUtilString strBinaryNumber;

    const MIuint nConstBits = 64;
    MIuint nRem[nConstBits + 1];
    MIint i = 0;
    MIuint nLen = 0;
    MIuint64 nNum = vnDecimal;
    while ((nNum > 0) && (nLen < nConstBits))
    {
        nRem[i++] = nNum % 2;
        nNum = nNum >> 1;
        nLen++;
    }
    MIchar pN[nConstBits + 1];
    MIuint j = 0;
    for (i = nLen; i > 0; --i, j++)
    {
        pN[j] = '0' + nRem[i - 1];
    }
    pN[j] = 0; // String NUL termination

    strBinaryNumber = CMIUtilString::Format("0b%s", &pN[0]);

    return strBinaryNumber;
}

//++ ------------------------------------------------------------------------------------
// Details: Remove from a string doubled up characters so only one set left. Characters
//          are only removed if the previous character is already a same character.
// Type:    Method.
// Args:    vChar   - (R) The character to search for and remove adjacent duplicates.
// Return:  CMIUtilString - New version of the string.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::RemoveRepeatedCharacters(const MIchar vChar)
{
    return RemoveRepeatedCharacters(0, vChar);
}

//++ ------------------------------------------------------------------------------------
// Details: Recursively remove from a string doubled up characters so only one set left.
//          Characters are only removed if the previous character is already a same
//          character.
// Type:    Method.
// Args:    vChar   - (R) The character to search for and remove adjacent duplicates.
//          vnPos   - (R) Character position in the string.
// Return:  CMIUtilString - New version of the string.
// Throws:  None.
//--
CMIUtilString
CMIUtilString::RemoveRepeatedCharacters(const MIint vnPos, const MIchar vChar)
{
    const MIchar cQuote = '"';

    // Look for first quote of two
    MIint nPos = find(cQuote, vnPos);
    if (nPos == (MIint)std::string::npos)
        return *this;

    const MIint nPosNext = nPos + 1;
    if (nPosNext > (MIint)length())
        return *this;

    if (at(nPosNext) == cQuote)
    {
        *this = substr(0, nPos) + substr(nPosNext, length());
        RemoveRepeatedCharacters(nPosNext, vChar);
    }

    return *this;
}

//++ ------------------------------------------------------------------------------------
// Details: Is the text in *this string surrounded by quotes.
// Type:    Method.
// Args:    None.
// Return:  bool - True = Yes string is quoted, false = no quoted.
// Throws:  None.
//--
bool
CMIUtilString::IsQuoted(void) const
{
    const MIchar cQuote = '"';

    if (at(0) != cQuote)
        return false;

    const MIint nLen = length();
    if ((nLen > 0) && (at(nLen - 1) != cQuote))
        return false;

    return true;
}

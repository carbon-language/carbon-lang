//===-- MICmdArgValString.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmdArgValString.cpp
//
// Overview:    CMICmdArgValString implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// In-house headers:
#include "MICmdArgValString.h"
#include "MICmdArgContext.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgValString constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValString::CMICmdArgValString(void)
    : m_bHandleQuotedString(false)
    , m_bAcceptNumbers(false)
    , m_bHandleDirPaths(false)
    , m_bHandleAnything(false)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgValString constructor.
// Type:    Method.
// Args:    vbAnything  - (R) True = Parse a string and accept anything, false = do not accept anything.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValString::CMICmdArgValString(const bool vbAnything)
    : m_bHandleQuotedString(false)
    , m_bAcceptNumbers(false)
    , m_bHandleDirPaths(false)
    , m_bHandleAnything(vbAnything)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgValString constructor.
// Type:    Method.
// Args:    vbHandleQuotes      - (R) True = Parse a string surrounded by quotes spaces are not delimitors, false = only text up to
// next delimiting space character.
//          vbAcceptNumbers     - (R) True = Parse a string and accept as a number if number, false = numbers not recognised
// as string types.
//          vbHandleDirPaths    - (R) True = Parse a string and accept as a file path if a path, false = file paths are not
// recognised as string types.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValString::CMICmdArgValString(const bool vbHandleQuotes, const bool vbAcceptNumbers, const bool vbHandleDirPaths)
    : m_bHandleQuotedString(vbHandleQuotes)
    , m_bAcceptNumbers(vbAcceptNumbers)
    , m_bHandleDirPaths(vbHandleDirPaths)
    , m_bHandleAnything(false)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgValString constructor.
// Type:    Method.
// Args:    vrArgName       - (R) Argument's name to search by.
//          vbMandatory     - (R) True = Yes must be present, false = optional argument.
//          vbHandleByCmd   - (R) True = Command processes *this option, false = not handled.
//          vbHandleQuotes  - (R) True = Parse a string surrounded by quotes spaces are not delimitors, false = only text up to
// next delimiting space character. (Dflt = false)
//          vbAcceptNumbers - (R) True = Parse a string and accept as a number if number, false = numbers not recognised as
// string types. (Dflt = false)
// Return:  None.
// Throws:  None.
//--
CMICmdArgValString::CMICmdArgValString(const CMIUtilString &vrArgName, const bool vbMandatory, const bool vbHandleByCmd,
                                       const bool vbHandleQuotes /* = false */, const bool vbAcceptNumbers /* = false */)
    : CMICmdArgValBaseTemplate(vrArgName, vbMandatory, vbHandleByCmd)
    , m_bHandleQuotedString(vbHandleQuotes)
    , m_bAcceptNumbers(vbAcceptNumbers)
    , m_bHandleDirPaths(false)
    , m_bHandleAnything(false)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgValString destructor.
// Type:    Overridden.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValString::~CMICmdArgValString(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: Parse the command's argument options string and try to extract the value *this
//          argument is looking for.
// Type:    Overridden.
// Args:    vrwArgContext   - (RW) The command's argument options string.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdArgValString::Validate(CMICmdArgContext &vrwArgContext)
{
    if (vrwArgContext.IsEmpty())
        return MIstatus::success;

    if (m_bHandleQuotedString)
        return (ValidateQuotedText(vrwArgContext) || ValidateQuotedTextEmbedded(vrwArgContext));

    return ValidateSingleText(vrwArgContext);
}

//++ ------------------------------------------------------------------------------------
// Details: Parse the command's argument options string and try to extract only the next
//          word delimited by the next space.
// Type:    Method.
// Args:    vrwArgContext   - (RW) The command's argument options string.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdArgValString::ValidateSingleText(CMICmdArgContext &vrwArgContext)
{
    if (vrwArgContext.GetNumberArgsPresent() == 1)
    {
        const CMIUtilString &rArg(vrwArgContext.GetArgsLeftToParse());
        if (IsStringArg(rArg))
        {
            m_bFound = true;
            m_bValid = true;
            m_argValue = rArg;
            vrwArgContext.RemoveArg(rArg);
            return MIstatus::success;
        }
        else
            return MIstatus::failure;
    }

    // More than one option...
    const CMIUtilString::VecString_t vecOptions(vrwArgContext.GetArgs());
    CMIUtilString::VecString_t::const_iterator it = vecOptions.begin();
    while (it != vecOptions.end())
    {
        const CMIUtilString &rArg(*it);
        if (IsStringArg(rArg))
        {
            m_bFound = true;

            if (vrwArgContext.RemoveArg(rArg))
            {
                m_bValid = true;
                m_argValue = rArg;
                return MIstatus::success;
            }
            else
                return MIstatus::failure;
        }

        // Next
        ++it;
    }

    return MIstatus::failure;
}

//++ ------------------------------------------------------------------------------------
// Details: Parse the command's argument options string and try to extract all the words
//          between quotes then delimited by the next space. Can fall through to
//          ValidateSingleText() or ValidateQuotedQuotedTextEmbedded().
// Type:    Method.
// Args:    vrwArgContext   - (RW) The command's argument options string.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdArgValString::ValidateQuotedText(CMICmdArgContext &vrwArgContext)
{
    // CODETAG_QUOTEDTEXT_SIMILAR_CODE
    CMIUtilString strOptions = vrwArgContext.GetArgsLeftToParse();
    const MIchar cQuote = '"';

    // Look for first quote of two
    MIint nPos = strOptions.find(cQuote);
    if (nPos == (MIint)std::string::npos)
        return ValidateSingleText(vrwArgContext);

    // Is one and only quote at end of the string
    const MIint nLen = strOptions.length();
    if (nPos == (MIint)(nLen - 1))
        return MIstatus::failure;

    // Quote must be the first character in the string or be preceeded by a space
    if ((nPos > 0) && (strOptions[nPos - 1] != ' '))
        return MIstatus::failure;

    // Need to find the other quote
    const MIint nPos2 = strOptions.rfind(cQuote);
    if (nPos2 == (MIint)std::string::npos)
        return MIstatus::failure;

    // Is there quotes surrounding string formatting embedded quotes
    if (IsStringArgQuotedQuotedTextEmbedded(strOptions))
        return ValidateQuotedQuotedTextEmbedded(vrwArgContext);

    // Make sure not same back quote, need two quotes
    if (nPos == nPos2)
        return MIstatus::failure;

    // Extract quoted text
    const CMIUtilString strQuotedTxt = strOptions.substr(nPos, nPos2 - nPos + 1).c_str();
    if (vrwArgContext.RemoveArg(strQuotedTxt))
    {
        m_bFound = true;
        m_bValid = true;
        m_argValue = strOptions.substr(nPos + 1, nPos2 - nPos - 1).c_str();
        return MIstatus::success;
    }

    return MIstatus::failure;
}

//++ ------------------------------------------------------------------------------------
// Details: Parse the command's argument options string and try to extract all the words
//          between quotes then delimited by the next space. If there any string format
//          characters '\\' used to embed quotes these are ignored i.e. "\\\"%5d\\\""
//          becomes "%5d". Can fall through to ValidateQuotedText().
// Type:    Method.
// Args:    vrwArgContext   - (RW) The command's argument options string.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdArgValString::ValidateQuotedTextEmbedded(CMICmdArgContext &vrwArgContext)
{
    // CODETAG_QUOTEDTEXT_SIMILAR_CODE
    CMIUtilString strOptions = vrwArgContext.GetArgsLeftToParse();
    const MIchar cBckSlash = '\\';
    const MIint nPos = strOptions.find(cBckSlash);
    if (nPos == (MIint)std::string::npos)
        return ValidateQuotedText(vrwArgContext);

    // Back slash must be the first character in the string or be preceeded by a space
    // or '\\'
    const MIchar cSpace = ' ';
    if ((nPos > 0) && (strOptions[nPos - 1] != cSpace))
        return MIstatus::failure;

    // Need to find the other back slash
    const MIint nPos2 = strOptions.rfind(cBckSlash);
    if (nPos2 == (MIint)std::string::npos)
        return MIstatus::failure;

    // Make sure not same back slash, need two slashs
    if (nPos == nPos2)
        return MIstatus::failure;

    // Look for the two quotes
    const MIint nLen = strOptions.length();
    const MIchar cQuote = '"';
    const MIint nPosQuote1 = nPos + 1;
    const MIint nPosQuote2 = (nPos2 < nLen) ? nPos2 + 1 : nPos2;
    if ((nPosQuote1 != nPosQuote2) && (strOptions[nPosQuote1] != cQuote) && (strOptions[nPosQuote2] != cQuote))
        return MIstatus::failure;

    // Extract quoted text
    const CMIUtilString strQuotedTxt = strOptions.substr(nPos, nPosQuote2 - nPos + 1).c_str();
    if (vrwArgContext.RemoveArg(strQuotedTxt))
    {
        m_bFound = true;
        m_bValid = true;
        m_argValue = strQuotedTxt;
        return MIstatus::success;
    }

    return MIstatus::failure;
}

//++ ------------------------------------------------------------------------------------
// Details: Parse the command's argument options string and try to extract all the words
//          between quotes then delimited by the next space. If there any string format
//          characters '\\' used to embed quotes these are ignored i.e. "\\\"%5d\\\""
//          becomes "%5d".
// Type:    Method.
// Args:    vrwArgContext   - (RW) The command's argument options string.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdArgValString::ValidateQuotedQuotedTextEmbedded(CMICmdArgContext &vrwArgContext)
{
    // CODETAG_QUOTEDTEXT_SIMILAR_CODE
    CMIUtilString strOptions = vrwArgContext.GetArgsLeftToParse();
    const MIint nPos = strOptions.find("\"\\\"");
    if (nPos == (MIint)std::string::npos)
        return MIstatus::failure;

    const MIint nPos2 = strOptions.rfind("\\\"\"");
    if (nPos2 == (MIint)std::string::npos)
        return MIstatus::failure;

    const MIint nLen = strOptions.length();
    if ((nLen > 5) && ((nPos + 2) == (nPos2 - 2)))
        return MIstatus::failure;

    // Quote must be the first character in the string or be preceeded by a space
    // or '\\'
    const MIchar cSpace = ' ';
    if ((nPos > 0) && (strOptions[nPos - 1] != cSpace))
        return MIstatus::failure;

    // Extract quoted text
    const CMIUtilString strQuotedTxt = strOptions.substr(nPos, nPos2 - nPos + 3).c_str();
    if (vrwArgContext.RemoveArg(strQuotedTxt))
    {
        m_bFound = true;
        m_bValid = true;
        m_argValue = strQuotedTxt;
        return MIstatus::success;
    }

    return MIstatus::failure;
}

//++ ------------------------------------------------------------------------------------
// Details: Examine the string and determine if it is a valid string type argument.
// Type:    Method.
// Args:    vrTxt   - (R) Some text.
// Return:  bool    - True = yes valid arg, false = no.
// Throws:  None.
//--
bool
CMICmdArgValString::IsStringArg(const CMIUtilString &vrTxt) const
{
    if (m_bHandleQuotedString)
        return (IsStringArgQuotedText(vrTxt) || IsStringArgQuotedTextEmbedded(vrTxt) || IsStringArgQuotedQuotedTextEmbedded(vrTxt) ||
                IsStringArgSingleText(vrTxt)); // Still test for this as could just be one word still

    return IsStringArgSingleText(vrTxt);
}

//++ ------------------------------------------------------------------------------------
// Details: Examine the string and determine if it is a valid string type argument or
//          option value. If the string looks like a long option, short option, a thread
//          group ID or just a number it is rejected as a string type value. There is an
//          option to allow the string to accept a number as a string type.
// Type:    Method.
// Args:    vrTxt   - (R) Some text.
// Return:  bool    - True = yes valid argument value, false = something else.
// Throws:  None.
//--
bool
CMICmdArgValString::IsStringArgSingleText(const CMIUtilString &vrTxt) const
{
    // Accept anything as string word
    if (m_bHandleAnything)
        return true;

    if (!m_bHandleDirPaths)
    {
        // Look for directory file paths, if found reject
        const bool bHavePosSlash = (vrTxt.find_first_of("/") != std::string::npos);
        const bool bHaveBckSlash = (vrTxt.find_first_of("\\") != std::string::npos);
        if (bHavePosSlash || bHaveBckSlash)
            return false;
    }

    // Look for --someLongOption, if found reject
    if (0 == vrTxt.find("--"))
        return false;

    // Look for -f type short options, if found reject
    if ((0 == vrTxt.find("-")) && (vrTxt.length() == 2))
        return false;

    // Look for thread group i1 i2 i3...., if found reject
    if ((vrTxt.find("i") == 0) && ::isdigit(vrTxt[1]))
        return false;

    // Look for numbers, if found reject
    if (!m_bAcceptNumbers && vrTxt.IsNumber())
        return false;

    return true;
}

//++ ------------------------------------------------------------------------------------
// Details: Examine the string and determine if it is a valid string type argument.
//          Take into account quotes surrounding the text. Note this function falls
//          through to IsStringArgSingleText() should the criteria match fail.
// Type:    Method.
// Args:    vrTxt   - (R) Some text.
// Return:  bool    - True = yes valid arg, false = no.
// Throws:  None.
//--
bool
CMICmdArgValString::IsStringArgQuotedText(const CMIUtilString &vrTxt) const
{
    // CODETAG_QUOTEDTEXT_SIMILAR_CODE
    const MIchar cQuote = '"';
    const MIint nPos = vrTxt.find(cQuote);
    if (nPos == (MIint)std::string::npos)
        return false;

    // Is one and only quote at end of the string
    if (nPos == (MIint)(vrTxt.length() - 1))
        return false;

    // Quote must be the first character in the string or be preceeded by a space
    // Also check for embedded string formating quote
    const MIchar cBckSlash = '\\';
    const MIchar cSpace = ' ';
    if ((nPos > 1) && (vrTxt[nPos - 1] == cBckSlash) && (vrTxt[nPos - 2] != cSpace))
    {
        return false;
    }
    if ((nPos > 0) && (vrTxt[nPos - 1] != cSpace))
        return false;

    // Need to find the other quote
    const MIint nPos2 = vrTxt.rfind(cQuote);
    if (nPos2 == (MIint)std::string::npos)
        return false;

    // Make sure not same quote, need two quotes
    if (nPos == nPos2)
        return MIstatus::failure;

    return true;
}

//++ ------------------------------------------------------------------------------------
// Details: Examine the string and determine if it is a valid string type argument.
//          Take into account quotes surrounding the text. Take into account string format
//          embedded quotes surrounding the text i.e. "\\\"%5d\\\"". Note this function falls
//          through to IsStringArgQuotedText() should the criteria match fail.
// Type:    Method.
// Args:    vrTxt   - (R) Some text.
// Return:  bool    - True = yes valid arg, false = no.
// Throws:  None.
//--
bool
CMICmdArgValString::IsStringArgQuotedTextEmbedded(const CMIUtilString &vrTxt) const
{
    // CODETAG_QUOTEDTEXT_SIMILAR_CODE
    const MIchar cBckSlash = '\\';
    const MIint nPos = vrTxt.find(cBckSlash);
    if (nPos == (MIint)std::string::npos)
        return false;

    // Slash must be the first character in the string or be preceeded by a space
    const MIchar cSpace = ' ';
    if ((nPos > 0) && (vrTxt[nPos - 1] != cSpace))
        return false;

    // Need to find the other matching slash
    const MIint nPos2 = vrTxt.rfind(cBckSlash);
    if (nPos2 == (MIint)std::string::npos)
        return false;

    // Make sure not same back slash, need two slashs
    if (nPos == nPos2)
        return MIstatus::failure;

    return false;
}

//++ ------------------------------------------------------------------------------------
// Details: Examine the string and determine if it is a valid string type argument.
//          Take into account quotes surrounding the text. Take into account string format
//          embedded quotes surrounding the text i.e. "\\\"%5d\\\"". Note this function falls
//          through to IsStringArgQuotedTextEmbedded() should the criteria match fail.
// Type:    Method.
// Args:    vrTxt   - (R) Some text.
// Return:  bool    - True = yes valid arg, false = no.
// Throws:  None.
//--
bool
CMICmdArgValString::IsStringArgQuotedQuotedTextEmbedded(const CMIUtilString &vrTxt) const
{
    const MIint nPos = vrTxt.find("\"\\\"");
    if (nPos == (MIint)std::string::npos)
        return false;

    const MIint nPos2 = vrTxt.rfind("\\\"\"");
    if (nPos2 == (MIint)std::string::npos)
        return false;

    const MIint nLen = vrTxt.length();
    if ((nLen > 5) && ((nPos + 2) == (nPos2 - 2)))
        return false;

    return true;
}

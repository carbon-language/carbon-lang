//===-- MICmnArgContext.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmnArgContext.cpp
//
// Overview:    CMICmdArgContext implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// In-house headers:
#include "MICmdArgContext.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgContext constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdArgContext::CMICmdArgContext(void)
    : m_constCharSpace(' ')
    , m_constStrSpace(" ")
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgContext constructor.
// Type:    Method.
// Args:    vrCmdLineArgsRaw    - (R) The text description of the arguments options.
// Return:  None.
// Throws:  None.
//--
CMICmdArgContext::CMICmdArgContext(const CMIUtilString &vrCmdLineArgsRaw)
    : m_strCmdArgsAndOptions(vrCmdLineArgsRaw)
    , m_constCharSpace(' ')
    , m_constStrSpace(" ")
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgContext destructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdArgContext::~CMICmdArgContext(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve the remainder of the command's argument options left to parse.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString & - Argument options text.
// Throws:  None.
//--
const CMIUtilString &
CMICmdArgContext::GetArgsLeftToParse(void) const
{
    return m_strCmdArgsAndOptions;
}

//++ ------------------------------------------------------------------------------------
// Details: Ask if this arguments string has any arguments.
// Type:    Method.
// Args:    None.
// Return:  bool    - True = Has one or more arguments present, false = no arguments.
// Throws:  None.
//--
bool
CMICmdArgContext::IsEmpty(void) const
{
    return m_strCmdArgsAndOptions.empty();
}

//++ ------------------------------------------------------------------------------------
// Details: Remove the argument from the options text and any space after the argument
//          if applicable.
// Type:    Method.
// Args:    vArg    - (R) The name of the argument.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdArgContext::RemoveArg(const CMIUtilString &vArg)
{
    if (vArg.empty())
        return MIstatus::success;

    const MIuint nLen = vArg.length();
    const MIuint nLenCntxt = m_strCmdArgsAndOptions.length();
    if (nLen > nLenCntxt)
        return MIstatus::failure;

    MIuint nExtraSpace = 0;
    MIint nPos = m_strCmdArgsAndOptions.find(vArg);
    while (1)
    {
        if (nPos == (MIint)std::string::npos)
            return MIstatus::success;

        bool bPass1 = false;
        if (nPos != 0)
        {
            if (m_strCmdArgsAndOptions[nPos - 1] == m_constCharSpace)
                bPass1 = true;
        }
        else
            bPass1 = true;

        const MIuint nEnd = nPos + nLen;

        if (bPass1)
        {
            bool bPass2 = false;
            if (nEnd < nLenCntxt)
            {
                if (m_strCmdArgsAndOptions[nEnd] == m_constCharSpace)
                {
                    bPass2 = true;
                    nExtraSpace = 1;
                }
            }
            else
                bPass2 = true;

            if (bPass2)
                break;
        }

        nPos = m_strCmdArgsAndOptions.find(vArg, nEnd);
    }

    const MIuint nPosEnd = nLen + nExtraSpace;
    m_strCmdArgsAndOptions = m_strCmdArgsAndOptions.replace(nPos, nPosEnd, "").c_str();
    m_strCmdArgsAndOptions = m_strCmdArgsAndOptions.Trim();

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Remove the argument at the Nth word position along in the context string.
//          Any space after the argument is removed if applicable. A search is not
//          performed as there may be more than one vArg with the same 'name' in the
//          context string.
// Type:    Method.
// Args:    vArg        - (R) The name of the argument.
//          nArgIndex   - (R) The word count position to which to remove the vArg word.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdArgContext::RemoveArgAtPos(const CMIUtilString &vArg, const MIuint nArgIndex)
{
    MIuint nWordIndex = 0;
    CMIUtilString strBuildContextUp;
    const CMIUtilString::VecString_t vecWords(GetArgs());
    const bool bSpaceRequired(GetNumberArgsPresent() > 2);

    CMIUtilString::VecString_t::const_iterator it = vecWords.begin();
    const CMIUtilString::VecString_t::const_iterator itEnd = vecWords.end();
    while (it != itEnd)
    {
        const CMIUtilString &rWord(*it);
        if (nWordIndex++ != nArgIndex)
        {
            // Single words
            strBuildContextUp += rWord;
            if (bSpaceRequired)
                strBuildContextUp += m_constStrSpace;
        }
        else
        {
            // If quoted loose quoted text
            if (++it != itEnd)
            {
                CMIUtilString words = rWord;
                while (vArg != words)
                {
                    if (bSpaceRequired)
                        words += m_constStrSpace;
                    words += *it;
                    if (++it == itEnd)
                        break;
                }
                if (it != itEnd)
                    --it;
            }
        }

        // Next
        if (it != itEnd)
            ++it;
    }

    m_strCmdArgsAndOptions = strBuildContextUp;
    m_strCmdArgsAndOptions = m_strCmdArgsAndOptions.Trim();

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve number of arguments or options present in the command's option text.
// Type:    Method.
// Args:    None.
// Return:  MIuint  - 0 to n arguments present.
// Throws:  None.
//--
MIuint
CMICmdArgContext::GetNumberArgsPresent(void) const
{
    CMIUtilString::VecString_t vecOptions;
    return m_strCmdArgsAndOptions.SplitConsiderQuotes(m_constStrSpace, vecOptions);
}

//++ ------------------------------------------------------------------------------------
// Details: Retrieve all the arguments or options remaining in *this context.
// Type:    Method.
// Args:    None.
// Return:  MIUtilString::VecString_t   - List of args remaining.
// Throws:  None.
//--
CMIUtilString::VecString_t
CMICmdArgContext::GetArgs(void) const
{
    CMIUtilString::VecString_t vecOptions;
    m_strCmdArgsAndOptions.SplitConsiderQuotes(m_constStrSpace, vecOptions);
    return vecOptions;
}

//++ ------------------------------------------------------------------------------------
// Details: Copy assignment operator.
// Type:    Method.
// Args:    vOther  - (R) The variable to copy from.
// Return:  CMIUtilString & - this object.
// Throws:  None.
//--
CMICmdArgContext &CMICmdArgContext::operator=(const CMICmdArgContext &vOther)
{
    if (this != &vOther)
    {
        m_strCmdArgsAndOptions = vOther.m_strCmdArgsAndOptions;
    }

    return *this;
}

//===-- MICmdArgValConsume.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmdArgValConsume.cpp
//
// Overview:    CMICmdArgValConsume implementation.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

// In-house headers:
#include "MICmdArgValConsume.h"
#include "MICmdArgContext.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgValConsume constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValConsume::CMICmdArgValConsume(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgValConsume constructor.
// Type:    Method.
// Args:    vrArgName       - (R) Argument's name to search by.
//          vbMandatory     - (R) True = Yes must be present, false = optional argument.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValConsume::CMICmdArgValConsume(const CMIUtilString &vrArgName, const bool vbMandatory)
    : CMICmdArgValBaseTemplate(vrArgName, vbMandatory, true)
{
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdArgValConsume destructor.
// Type:    Overidden.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValConsume::~CMICmdArgValConsume(void)
{
}

//++ ------------------------------------------------------------------------------------
// Details: Parse the command's argument options string and try to extract the value *this
//          argument is looking for.
// Type:    Overridden.
// Args:    vwArgContext    - (R) The command's argument options string.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdArgValConsume::Validate(CMICmdArgContext &vwArgContext)
{
    if (vwArgContext.IsEmpty())
        return MIstatus::success;

    if (vwArgContext.GetNumberArgsPresent() == 1)
    {
        const CMIUtilString &rArg(vwArgContext.GetArgsLeftToParse());
        m_bFound = true;
        m_bValid = true;
        vwArgContext.RemoveArg(rArg);
        return MIstatus::success;
    }

    // In reality there are more than one option,  if so the file option
    // is the last one (don't handle that here - find the best looking one)
    const CMIUtilString::VecString_t vecOptions(vwArgContext.GetArgs());
    CMIUtilString::VecString_t::const_iterator it = vecOptions.begin();
    while (it != vecOptions.end())
    {
        const CMIUtilString &rTxt(*it);
        m_bFound = true;

        if (vwArgContext.RemoveArg(rTxt))
        {
            m_bValid = true;
            return MIstatus::success;
        }
        else
            return MIstatus::success;

        // Next
        ++it;
    }

    return MIstatus::failure;
}

//++ ------------------------------------------------------------------------------------
// Details: Nothing to examine as we just want to consume the argument or option (ignore
//          it).
// Type:    Method.
// Args:    None.
// Return:  bool -  True = yes ok, false = not ok.
// Throws:  None.
//--
bool
CMICmdArgValConsume::IsOk(void) const
{
    return true;
}

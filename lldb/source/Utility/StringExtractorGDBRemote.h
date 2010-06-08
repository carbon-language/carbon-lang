//===-- StringExtractorGDBRemote.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_StringExtractorGDBRemote_h_
#define utility_StringExtractorGDBRemote_h_

// C Includes
// C++ Includes
#include <string>
// Other libraries and framework includes
// Project includes
#include "StringExtractor.h"

class StringExtractorGDBRemote : public StringExtractor
{
public:

    StringExtractorGDBRemote() :
        StringExtractor ()
    {
    }

    StringExtractorGDBRemote(const char *cstr) :
        StringExtractor (cstr)
    {
    }
    StringExtractorGDBRemote(const StringExtractorGDBRemote& rhs) :
        StringExtractor (rhs)
    {
    }

    virtual ~StringExtractorGDBRemote()
    {
    }

    enum Type
    {
        eUnsupported = 0,
        eAck,
        eNack,
        eError,
        eOK,
        eResponse
    };

    StringExtractorGDBRemote::Type
    GetType () const;

    bool
    IsOKPacket() const;

    bool
    IsUnsupportedPacket() const;

    bool
    IsNormalPacket () const;

    bool
    IsErrorPacket() const;

    // Returns zero if the packet isn't a EXX packet where XX are two hex
    // digits. Otherwise the error encoded in XX is returned.
    uint8_t
    GetError();
};

#endif  // utility_StringExtractorGDBRemote_h_

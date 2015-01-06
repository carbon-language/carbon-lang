//===-- UriParser.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_UriParser_h_
#define utility_UriParser_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes

class UriParser
{
public:
    static bool Parse(const char* uri,
        std::string& scheme,
        std::string& hostname,
        int& port,
        std::string& path
        );
};

#endif  // utility_UriParser_h_

//===-- UriParser.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Utility/UriParser.h"

// C Includes
#include <stdlib.h>

// C++ Includes
// Other libraries and framework includes
// Project includes

//----------------------------------------------------------------------
// UriParser::Parse
//----------------------------------------------------------------------
bool
UriParser::Parse(const char* uri,
    std::string& scheme,
    std::string& hostname,
    int& port,
    std::string& path
    )
{
    char scheme_buf[100] = {0};
    char hostname_buf[256] = {0};
    char port_buf[11] = {0}; // 10==strlen(2^32)
    char path_buf[2049] = {'/', 0};
  
    bool ok = false;
         if (4==sscanf(uri, "%99[^:/]://%255[^/:]:%[^/]/%2047s", scheme_buf, hostname_buf, port_buf, path_buf+1)) { ok = true; }
    else if (3==sscanf(uri, "%99[^:/]://%255[^/:]:%[^/]", scheme_buf, hostname_buf, port_buf)) { ok = true; }
    else if (3==sscanf(uri, "%99[^:/]://%255[^/]/%2047s", scheme_buf, hostname_buf, path_buf+1)) { ok = true; }
    else if (2==sscanf(uri, "%99[^:/]://%255[^/]", scheme_buf, hostname_buf)) { ok = true; }

    char* end = port_buf;
    int port_tmp = strtoul(port_buf, &end, 10);
    if (*end != 0)
    {
        // there are invalid characters in port_buf
        return false;
    }

    if (ok)
    {
        scheme.assign(scheme_buf);
        hostname.assign(hostname_buf);
        port = port_tmp;
        path.assign(path_buf);
    }
    return ok;
}


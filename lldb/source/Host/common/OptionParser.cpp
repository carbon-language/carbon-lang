//===-- source/Host/common/OptionParser.cpp ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/OptionParser.h"

#ifdef _MSC_VER
#include "../windows/msvc/getopt.inc"
#else
#ifdef _WIN32
#define _BSD_SOURCE // Required so that getopt.h defines optreset
#endif
#include <getopt.h>
#endif

using namespace lldb_private;

void
OptionParser::Prepare()
{
#ifdef __GLIBC__
    optind = 0;
#else
    optreset = 1;
    optind = 1;
#endif
}

void
OptionParser::EnableError(bool error)
{
    opterr = error ? 1 : 0;
}

int
OptionParser::Parse(int argc, char * const argv [],
        const char *optstring,
        const Option *longopts, int *longindex)
{
    return getopt_long_only(argc, argv, optstring, (const option*)longopts, longindex);
}

char* OptionParser::GetOptionArgument()
{
    return optarg;
}

int OptionParser::GetOptionIndex()
{
    return optind;
}

int OptionParser::GetOptionErrorCause()
{
    return optopt;
}

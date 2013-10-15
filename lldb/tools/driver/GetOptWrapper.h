//===-- GetOptWrapper.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_GetOptWrapper_h_
#define lldb_GetOptWrapper_h_

// from getopt.h
#define no_argument       0
#define required_argument 1
#define optional_argument 2

// defined int unistd.h
extern int   optreset;

// from getopt.h
extern char *optarg;
extern int   optind;
extern int   opterr;
extern int   optopt;

// option structure
struct option
{
    const char *name;
    // has_arg can't be an enum because some compilers complain about
    // type mismatches in all the code that assumes it is an int.
    int  has_arg;
    int *flag;
    int  val;
};

// 
extern int
getopt_long_only
(
    int                  ___argc,
    char *const         *___argv,
    const char          *__shortopts,
    const struct option *__longopts,
    int                 *__longind
);

#endif // lldb_GetOptWrapper_h_
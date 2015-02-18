//===-- lldb-server.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>

    static void
display_usage (const char *progname)
{
    fprintf(stderr, "Usage:\n"
            "  %s g[dbserver] [options]\n"
            "  %s p[latform] [options]\n"
            "Invoke subcommand for additional help", progname, progname);
    exit(0);
}

// Forward declarations of subcommand main methods.
int main_gdbserver (int argc, char *argv[]);
int main_platform (int argc, char *argv[]);

//----------------------------------------------------------------------
// main
//----------------------------------------------------------------------
int
main (int argc, char *argv[])
{
    int option_error = 0;
    const char *progname = argv[0];
    if (argc < 2)
    {
        display_usage(progname);
        exit(option_error);
    }
    else if (argv[1][0] == 'g')
    {
        main_gdbserver(argc, argv);
    }
    else if (argv[1][0] == 'p')
    {
        main_platform(argc, argv);
    }
    else {
        display_usage(progname);
        exit(option_error);
    }
}

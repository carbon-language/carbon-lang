//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>

int
main(int argc, char const *argv[])
{
    // The program writes its output to the "output.txt" file.
    std::ofstream outfile("output.txt");

    for (unsigned i = 0; i < argc; ++i) {
        std::string theArg(argv[i]);
        if (i == 1 && "A" == theArg)
            outfile << "argv[1] matches\n";

        if (i == 2 && "B" == theArg)
            outfile << "argv[2] matches\n";

        if (i == 3 && "C" == theArg)
            outfile << "argv[3] matches\n";
    }

    // For passing environment vars from the debugger to the launched process.
    if (::getenv("MY_ENV_VAR")) {
        std::string MY_ENV_VAR(getenv("MY_ENV_VAR"));
        if ("YES" == MY_ENV_VAR) {
            outfile << "Environment variable 'MY_ENV_VAR' successfully passed.\n";
        }
    }


    // For passing host environment vars to the launched process.
    if (::getenv("MY_HOST_ENV_VAR1")) {
        std::string MY_HOST_ENV_VAR1(getenv("MY_HOST_ENV_VAR1"));
        if ("VAR1" == MY_HOST_ENV_VAR1) {
            outfile << "The host environment variable 'MY_HOST_ENV_VAR1' successfully passed.\n";
        }
    }

    if (::getenv("MY_HOST_ENV_VAR2")) {
        std::string MY_HOST_ENV_VAR2(getenv("MY_HOST_ENV_VAR2"));
        if ("VAR2" == MY_HOST_ENV_VAR2) {
            outfile << "The host environment variable 'MY_HOST_ENV_VAR2' successfully passed.\n";
        }
    }

    std::cout << "This message should go to standard out.\n";

    return 0;
}

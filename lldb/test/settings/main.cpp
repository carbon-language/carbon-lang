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

    if (::getenv("MY_ENV_VAR")) {
        std::string MY_ENV_VAR(getenv("MY_ENV_VAR"));
        if ("YES" == MY_ENV_VAR) {
            outfile << "Environment variable 'MY_ENV_VAR' successfully passed.\n";
        }
    }

    std::cout << "This message should go to standard out.\n";

    return 0;
}

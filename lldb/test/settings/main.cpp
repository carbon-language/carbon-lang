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

int
main(int argc, char const *argv[])
{
    char const *cptr = NULL;
    // The program writes its output to the "output.txt" file.
    std::ofstream outfile("output.txt");

    for (unsigned i = 0, e = sizeof(argv); i < e; ++i) {
        if ((cptr = argv[i]) == NULL)
            break;

        std::string str(cptr);
        if (i == 1 && "A" == str)
            outfile << "argv[1] matches\n";

        if (i == 2 && "B" == str)
            outfile << "argv[2] matches\n";

        if (i == 3 && "C" == str)
            outfile << "argv[3] matches\n";
    }

    if (::getenv("MY_ENV_VAR")) {
        std::string MY_ENV_VAR(getenv("MY_ENV_VAR"));
        if ("YES" == MY_ENV_VAR) {
            outfile << "Environment variable 'MY_ENV_VAR' successfully passed.\n";
        }
    }

    return 0;
}

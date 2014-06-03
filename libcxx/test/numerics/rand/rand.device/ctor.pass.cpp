//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// class random_device;

// explicit random_device(const string& token = "/dev/urandom");

#include <random>
#include <cassert>
#include <unistd.h>

int main()
{
    try
    {
        std::random_device r("wrong file");
        assert(false);
    }
    catch (const std::system_error& e)
    {
    }
    {
        std::random_device r;
    }
    {
        int ec;
        ec = close(STDIN_FILENO);
        assert(!ec);
        ec = close(STDOUT_FILENO);
        assert(!ec);
        ec = close(STDERR_FILENO);
        assert(!ec);
        std::random_device r;
    }
    {
        std::random_device r("/dev/urandom");;
    }
    {
        std::random_device r("/dev/random");;
    }
}

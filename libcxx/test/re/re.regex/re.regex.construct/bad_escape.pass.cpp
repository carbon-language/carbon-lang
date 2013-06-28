//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// template <class ST, class SA>
//    basic_regex(const basic_string<charT, ST, SA>& s);

#include <regex>
#include <cassert>

int main() 
{
    // Correct: Exception thrown for invalid escape char in a character class
    try {
        std::regex char_class_escape("[\\a]");
        assert(false);
    } catch (std::regex_error &ex) {
        assert(ex.code() == std::regex_constants::error_escape);
    }

    // Failure: No exception thrown for invalid escape char in this case.
    try {
        std::regex escape("\\a");
        assert(false);
    } catch (std::regex_error &ex) {
        assert(ex.code() == std::regex_constants::error_escape);
    }
}

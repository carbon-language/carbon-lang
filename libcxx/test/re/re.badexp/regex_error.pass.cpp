// -*- C++ -*-
//===-------------------------- algorithm ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// class regex_error
//     : public runtime_error
// {
// public:
//     explicit regex_error(regex_constants::error_type ecode);
//     regex_constants::error_type code() const;
// };

#include <regex>
#include <cassert>

int main()
{
    {
        std::regex_error e(std::regex_constants::error_collate);
        assert(e.code() == std::regex_constants::error_collate);
        assert(e.what() == std::string("error_collate"));
    }
    {
        std::regex_error e(std::regex_constants::error_ctype);
        assert(e.code() == std::regex_constants::error_ctype);
        assert(e.what() == std::string("error_ctype"));
    }
    {
        std::regex_error e(std::regex_constants::error_escape);
        assert(e.code() == std::regex_constants::error_escape);
        assert(e.what() == std::string("error_escape"));
    }
    {
        std::regex_error e(std::regex_constants::error_backref);
        assert(e.code() == std::regex_constants::error_backref);
        assert(e.what() == std::string("error_backref"));
    }
    {
        std::regex_error e(std::regex_constants::error_brack);
        assert(e.code() == std::regex_constants::error_brack);
        assert(e.what() == std::string("error_brack"));
    }
    {
        std::regex_error e(std::regex_constants::error_paren);
        assert(e.code() == std::regex_constants::error_paren);
        assert(e.what() == std::string("error_paren"));
    }
    {
        std::regex_error e(std::regex_constants::error_brace);
        assert(e.code() == std::regex_constants::error_brace);
        assert(e.what() == std::string("error_brace"));
    }
    {
        std::regex_error e(std::regex_constants::error_badbrace);
        assert(e.code() == std::regex_constants::error_badbrace);
        assert(e.what() == std::string("error_badbrace"));
    }
    {
        std::regex_error e(std::regex_constants::error_range);
        assert(e.code() == std::regex_constants::error_range);
        assert(e.what() == std::string("error_range"));
    }
    {
        std::regex_error e(std::regex_constants::error_space);
        assert(e.code() == std::regex_constants::error_space);
        assert(e.what() == std::string("error_space"));
    }
    {
        std::regex_error e(std::regex_constants::error_badrepeat);
        assert(e.code() == std::regex_constants::error_badrepeat);
        assert(e.what() == std::string("error_badrepeat"));
    }
    {
        std::regex_error e(std::regex_constants::error_complexity);
        assert(e.code() == std::regex_constants::error_complexity);
        assert(e.what() == std::string("error_complexity"));
    }
    {
        std::regex_error e(std::regex_constants::error_stack);
        assert(e.code() == std::regex_constants::error_stack);
        assert(e.what() == std::string("error_stack"));
    }
}

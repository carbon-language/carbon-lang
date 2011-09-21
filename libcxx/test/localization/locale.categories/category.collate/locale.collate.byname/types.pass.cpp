//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT>
// class collate_byname
//     : public collate<charT>
// {
// public:
//     typedef basic_string<charT> string_type;
//     explicit collate_byname(const char*, size_t refs = 0);
//     explicit collate_byname(const string&, size_t refs = 0);
// protected:
//     ~collate_byname();
// };

#include <locale>
#include <string>
#include <cassert>

#include <stdio.h>

int main()
{
    std::locale l("en_US.UTF-8");
    {
        assert(std::has_facet<std::collate_byname<char> >(l));
        assert(&std::use_facet<std::collate<char> >(l)
            == &std::use_facet<std::collate_byname<char> >(l));
    }
    {
        assert(std::has_facet<std::collate_byname<wchar_t> >(l));
        assert(&std::use_facet<std::collate<wchar_t> >(l)
            == &std::use_facet<std::collate_byname<wchar_t> >(l));
    }
}

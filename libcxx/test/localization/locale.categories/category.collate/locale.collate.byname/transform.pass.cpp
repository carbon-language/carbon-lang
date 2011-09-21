//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class collate_byname

// string_type transform(const charT* low, const charT* high) const;

#include <locale>
#include <string>
#include <cassert>

#include <stdio.h>

int main()
{
    // Ensure that the default locale is not C.  If it is, the second tests will fail.
    setenv("LANG", "en_US", 1);
    {
        std::locale l("en_US");
        {
            std::string x("1234");
            const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
            assert(f.transform(x.data(), x.data() + x.size()) != x);
        }
        {
            std::wstring x(L"1234");
            const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);
            assert(f.transform(x.data(), x.data() + x.size()) != x);
        }
    }
    {
        std::locale l("");
        {
            std::string x("1234");
            const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
            assert(f.transform(x.data(), x.data() + x.size()) != x);
        }
        {
            std::wstring x(L"1234");
            const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);
            assert(f.transform(x.data(), x.data() + x.size()) != x);
        }
    }
    {
        std::locale l("C");
        {
            std::string x("1234");
            const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
            assert(f.transform(x.data(), x.data() + x.size()) == x);
        }
        {
            std::wstring x(L"1234");
            const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);
            assert(f.transform(x.data(), x.data() + x.size()) == x);
        }
    }
}

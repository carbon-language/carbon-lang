//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class collate_byname

// int compare(const charT* low1, const charT* high1,
//             const charT* low2, const charT* high2) const;

//  I'm currently unable to confirm that collation based on named locales
//     has any difference from "C" collation.  But I do believe I'm picking
//     up the OS's collation files.

#include <locale>
#include <string>
#include <cassert>

#include <stdio.h>

int main()
{
    {
        std::locale l("en_US");
        {
            const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
            std::string s2("aaaaaaA");
            std::string s3("BaaaaaA");
            assert(f.compare(s2.data(), s2.data() + s2.size(),
                             s3.data(), s3.data() + s3.size()) == 1);
        }
        {
            const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);
            std::wstring s2(L"aaaaaaA");
            std::wstring s3(L"BaaaaaA");
            assert(f.compare(s2.data(), s2.data() + s2.size(),
                             s3.data(), s3.data() + s3.size()) == 1);
        }
    }
    {
        std::locale l("");
        {
            const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
            std::string s2("aaaaaaA");
            std::string s3("BaaaaaA");
            assert(f.compare(s2.data(), s2.data() + s2.size(),
                             s3.data(), s3.data() + s3.size()) == 1);
        }
        {
            const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);
            std::wstring s2(L"aaaaaaA");
            std::wstring s3(L"BaaaaaA");
            assert(f.compare(s2.data(), s2.data() + s2.size(),
                             s3.data(), s3.data() + s3.size()) == 1);
        }
    }
    {
        std::locale l("C");
        {
            const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
            std::string s2("aaaaaaA");
            std::string s3("BaaaaaA");
            assert(f.compare(s2.data(), s2.data() + s2.size(),
                             s3.data(), s3.data() + s3.size()) == 1);
        }
        {
            const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);
            std::wstring s2(L"aaaaaaA");
            std::wstring s3(L"BaaaaaA");
            assert(f.compare(s2.data(), s2.data() + s2.size(),
                             s3.data(), s3.data() + s3.size()) == 1);
        }
    }
}

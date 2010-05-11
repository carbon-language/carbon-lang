//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype_byname;

// const char* widen(const char* low, const char* high, charT* to) const;

// I doubt this test is portable

#include <locale>
#include <string>
#include <vector>
#include <cassert>

int main()
{
    {
        std::locale l("en_US");
        {
            typedef std::ctype<wchar_t> F;
            const F& f = std::use_facet<F>(l);
            std::string in(" A\x07.a1\x85");
            std::vector<wchar_t> v(in.size());
    
            assert(f.widen(&in[0], in.data() + in.size(), v.data()) == in.data() + in.size());
            assert(v[0] == L' ');
            assert(v[1] == L'A');
            assert(v[2] == L'\x07');
            assert(v[3] == L'.');
            assert(v[4] == L'a');
            assert(v[5] == L'1');
            assert(v[6] == wchar_t(-1));
        }
    }
    {
        std::locale l("C");
        {
            typedef std::ctype<wchar_t> F;
            const F& f = std::use_facet<F>(l);
            std::string in(" A\x07.a1\x85");
            std::vector<wchar_t> v(in.size());
    
            assert(f.widen(&in[0], in.data() + in.size(), v.data()) == in.data() + in.size());
            assert(v[0] == L' ');
            assert(v[1] == L'A');
            assert(v[2] == L'\x07');
            assert(v[3] == L'.');
            assert(v[4] == L'a');
            assert(v[5] == L'1');
            assert(v[6] == wchar_t(133));
        }
    }
}

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>;

// char widen(char c) const;

#include <locale>
#include <cassert>

int main()
{
    std::locale l = std::locale::classic();
    {
        typedef std::ctype<char> F;
        const F& f = std::use_facet<F>(l);

        assert(f.widen(' ') == ' ');
        assert(f.widen('A') == 'A');
        assert(f.widen('\x07') == '\x07');
        assert(f.widen('.') == '.');
        assert(f.widen('a') == 'a');
        assert(f.widen('1') == '1');
    }
}

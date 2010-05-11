//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ostream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ostream;

// basic_ostream(basic_ostream&& rhs);

#include <ostream>
#include <cassert>

#ifdef _LIBCPP_MOVE

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    testbuf() {}
};

template <class CharT>
struct test_ostream
    : public std::basic_ostream<CharT>
{
    typedef std::basic_ostream<CharT> base;
    test_ostream(testbuf<CharT>* sb) : base(sb) {}

    test_ostream(test_ostream&& s)
        : base(std::move(s)) {}
};

#endif

int main()
{
#ifdef _LIBCPP_MOVE
    {
        testbuf<char> sb;
        test_ostream<char> os1(&sb);
        test_ostream<char> os(std::move(os1));
        assert(os1.rdbuf() == &sb);
        assert(os.rdbuf() == 0);
        assert(os.tie() == 0);
        assert(os.fill() == ' ');
        assert(os.rdstate() == os.goodbit);
        assert(os.exceptions() == os.goodbit);
        assert(os.flags() == (os.skipws | os.dec));
        assert(os.precision() == 6);
        assert(os.getloc().name() == "C");
    }
    {
        testbuf<wchar_t> sb;
        test_ostream<wchar_t> os1(&sb);
        test_ostream<wchar_t> os(std::move(os1));
        assert(os1.rdbuf() == &sb);
        assert(os.rdbuf() == 0);
        assert(os.tie() == 0);
        assert(os.fill() == L' ');
        assert(os.rdstate() == os.goodbit);
        assert(os.exceptions() == os.goodbit);
        assert(os.flags() == (os.skipws | os.dec));
        assert(os.precision() == 6);
        assert(os.getloc().name() == "C");
    }
#endif
}

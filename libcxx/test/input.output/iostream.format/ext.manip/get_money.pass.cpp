//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iomanip>

// template <class moneyT> T7 get_money(moneyT& mon, bool intl = false);

#include <iomanip>
#include <cassert>

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_string<CharT> string_type;
    typedef std::basic_streambuf<CharT> base;
private:
    string_type str_;
public:

    testbuf() {}
    testbuf(const string_type& str)
        : str_(str)
    {
        base::setg(const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()) + str_.size());
    }
};

int main()
{
    {
        testbuf<char> sb("  -$1,234,567.89");
        std::istream is(&sb);
        is.imbue(std::locale("en_US"));
        long double x = 0;
        is >> std::get_money(x, false);
        assert(x == -123456789);
    }
    {
        testbuf<char> sb("  -USD 1,234,567.89");
        std::istream is(&sb);
        is.imbue(std::locale("en_US"));
        long double x = 0;
        is >> std::get_money(x, true);
        assert(x == -123456789);
    }
    {
        testbuf<wchar_t> sb(L"  -$1,234,567.89");
        std::wistream is(&sb);
        is.imbue(std::locale("en_US"));
        long double x = 0;
        is >> std::get_money(x, false);
        assert(x == -123456789);
    }
    {
        testbuf<wchar_t> sb(L"  -USD 1,234,567.89");
        std::wistream is(&sb);
        is.imbue(std::locale("en_US"));
        long double x = 0;
        is >> std::get_money(x, true);
        assert(x == -123456789);
    }
}

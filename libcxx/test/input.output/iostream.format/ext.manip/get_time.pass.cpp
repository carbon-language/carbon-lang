//===----------------------------------------------------------------------===//
//
// ÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊThe LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iomanip>

// template <class charT> T9 get_time(struct tm* tmb, const charT* fmt);

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
        testbuf<char> sb("  Sat Dec 31 23:55:59 2061");
        std::istream is(&sb);
        is.imbue(std::locale("en_US"));
        std::tm t = {0};
        is >> std::get_time(&t, "%c");
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(t.tm_mday == 31);
        assert(t.tm_mon == 11);
        assert(t.tm_year == 161);
        assert(t.tm_wday == 6);
        assert(is.eof());
        assert(!is.fail());
    }
    {
        testbuf<wchar_t> sb(L"  Sat Dec 31 23:55:59 2061");
        std::wistream is(&sb);
        is.imbue(std::locale("en_US"));
        std::tm t = {0};
        is >> std::get_time(&t, L"%c");
        assert(t.tm_sec == 59);
        assert(t.tm_min == 55);
        assert(t.tm_hour == 23);
        assert(t.tm_mday == 31);
        assert(t.tm_mon == 11);
        assert(t.tm_year == 161);
        assert(t.tm_wday == 6);
        assert(is.eof());
        assert(!is.fail());
    }
}

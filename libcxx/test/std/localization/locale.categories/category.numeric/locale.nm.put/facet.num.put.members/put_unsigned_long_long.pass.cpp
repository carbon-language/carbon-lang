//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class num_put<charT, OutputIterator>

// iter_type put(iter_type s, ios_base& iob, char_type fill, unsigned long long v) const;

#include <locale>
#include <ios>
#include <cassert>
#include <streambuf>
#include "test_macros.h"
#include "test_iterators.h"

typedef std::num_put<char, cpp17_output_iterator<char*> > F;

class my_facet
    : public F
{
public:
    explicit my_facet(std::size_t refs = 0)
        : F(refs) {}
};

class my_numpunct
    : public std::numpunct<char>
{
public:
    my_numpunct() : std::numpunct<char>() {}

protected:
    virtual char_type do_thousands_sep() const {return '_';}
    virtual std::string do_grouping() const {return std::string("\1\2\3");}
};

int main(int, char**)
{
    const my_facet f(1);
    {
        std::ios ios(0);
        unsigned long long v = 0;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0");
    }
    {
        std::ios ios(0);
        unsigned long long v = 1;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "1");
    }
    {
        std::ios ios(0);
        unsigned long long v = static_cast<unsigned long long>(-1);
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "18446744073709551615");
    }
    {
        std::ios ios(0);
        unsigned long long v = static_cast<unsigned long long>(-1000);
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "18446744073709550616");
    }
    {
        std::ios ios(0);
        unsigned long long v = 1000;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "1000");
    }
    {
        std::ios ios(0);
        std::showpos(ios);
        unsigned long long v = 1000;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "1000");
    }
    {
        std::ios ios(0);
        std::oct(ios);
        unsigned long long v = 1000;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "1750");
    }
    {
        std::ios ios(0);
        std::oct(ios);
        std::showbase(ios);
        unsigned long long v = 1000;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "01750");
    }
    {
        std::ios ios(0);
        std::hex(ios);
        unsigned long long v = 1000;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "3e8");
    }
    {
        std::ios ios(0);
        std::hex(ios);
        std::showbase(ios);
        unsigned long long v = 1000;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0x3e8");
    }
    {
        std::ios ios(0);
        std::hex(ios);
        std::showbase(ios);
        std::uppercase(ios);
        unsigned long long v = 1000;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0X3E8");
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::hex(ios);
        std::showbase(ios);
        std::uppercase(ios);
        unsigned long long v = 1000;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0X3E_8");
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::hex(ios);
        std::showbase(ios);
        unsigned long long v = 2147483647;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0x7f_fff_ff_f");
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::oct(ios);
        unsigned long long v = 0123467;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "123_46_7");
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::oct(ios);
        std::showbase(ios);
        unsigned long long v = 0123467;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0_123_46_7");
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::oct(ios);
        std::showbase(ios);
        std::right(ios);
        ios.width(15);
        unsigned long long v = 0123467;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "*****0_123_46_7");
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::oct(ios);
        std::showbase(ios);
        std::left(ios);
        ios.width(15);
        unsigned long long v = 0123467;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0_123_46_7*****");
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::oct(ios);
        std::showbase(ios);
        std::internal(ios);
        ios.width(15);
        unsigned long long v = 0123467;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "*****0_123_46_7");
        assert(ios.width() == 0);
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::hex(ios);
        std::showbase(ios);
        std::right(ios);
        ios.width(15);
        unsigned long long v = 2147483647;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "**0x7f_fff_ff_f");
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::hex(ios);
        std::showbase(ios);
        std::left(ios);
        ios.width(15);
        unsigned long long v = 2147483647;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0x7f_fff_ff_f**");
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::hex(ios);
        std::showbase(ios);
        std::internal(ios);
        ios.width(15);
        unsigned long long v = 2147483647;
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "0x**7f_fff_ff_f");
        assert(ios.width() == 0);
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::showpos(ios);
        unsigned long long v = 1000;
        std::right(ios);
        ios.width(10);
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "****1_00_0");
        assert(ios.width() == 0);
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::showpos(ios);
        unsigned long long v = 1000;
        std::left(ios);
        ios.width(10);
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "1_00_0****");
        assert(ios.width() == 0);
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        std::showpos(ios);
        unsigned long long v = 1000;
        std::internal(ios);
        ios.width(10);
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "****1_00_0");
        assert(ios.width() == 0);
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        unsigned long long v = static_cast<unsigned long long>(-1000);
        std::right(ios);
        std::showpos(ios);
        ios.width(10);
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "18_446_744_073_709_550_61_6");
        assert(ios.width() == 0);
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        unsigned long long v = static_cast<unsigned long long>(-1000);
        std::left(ios);
        ios.width(10);
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "18_446_744_073_709_550_61_6");
        assert(ios.width() == 0);
    }
    {
        std::ios ios(0);
        ios.imbue(std::locale(std::locale::classic(), new my_numpunct));
        unsigned long long v = static_cast<unsigned long long>(-1000);
        std::internal(ios);
        ios.width(10);
        char str[50];
        cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
        std::string ex(str, iter.base());
        assert(ex == "18_446_744_073_709_550_61_6");
        assert(ios.width() == 0);
    }

  return 0;
}

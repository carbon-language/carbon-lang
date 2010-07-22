//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class BidirectionalIterator, class Allocator, class charT, class traits>
//     bool
//     regex_search(BidirectionalIterator first, BidirectionalIterator last,
//                  match_results<BidirectionalIterator, Allocator>& m,
//                  const basic_regex<charT, traits>& e,
//                  regex_constants::match_flag_type flags = regex_constants::match_default);

#include <regex>
#include <cassert>

#include "../../iterators.h"

int main()
{
    {
        std::cmatch m;
        const char s[] = "a";
        assert(std::regex_search(s, m, std::regex("a", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.empty());
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+1);
        assert(m.length(0) == 1);
        assert(m.position(0) == 0);
        assert(m.str(0) == "a");
    }
    {
        std::cmatch m;
        const char s[] = "ab";
        assert(std::regex_search(s, m, std::regex("ab", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+2);
        assert(m.length(0) == 2);
        assert(m.position(0) == 0);
        assert(m.str(0) == "ab");
    }
    {
        std::cmatch m;
        const char s[] = "ab";
        assert(!std::regex_search(s, m, std::regex("ba", std::regex_constants::extended)));
        assert(m.size() == 0);
        assert(m.empty());
    }
    {
        std::cmatch m;
        const char s[] = "aab";
        assert(std::regex_search(s, m, std::regex("ab", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+3);
        assert(m.length(0) == 2);
        assert(m.position(0) == 1);
        assert(m.str(0) == "ab");
    }
    {
        std::cmatch m;
        const char s[] = "aab";
        assert(!std::regex_search(s, m, std::regex("ab", std::regex_constants::extended),
                                            std::regex_constants::match_continuous));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abcd";
        assert(std::regex_search(s, m, std::regex("bc", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+4);
        assert(m.length(0) == 2);
        assert(m.position(0) == 1);
        assert(m.str(0) == "bc");
    }
    {
        std::cmatch m;
        const char s[] = "abbc";
        assert(std::regex_search(s, m, std::regex("ab*c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+4);
        assert(m.length(0) == 4);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "ababc";
        assert(std::regex_search(s, m, std::regex("(ab)*c", std::regex_constants::extended)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+5);
        assert(m.length(0) == 5);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
        assert(m.length(1) == 2);
        assert(m.position(1) == 2);
        assert(m.str(1) == "ab");
    }
    {
        std::cmatch m;
        const char s[] = "abcdefghijk";
        assert(std::regex_search(s, m, std::regex("cd((e)fg)hi",
                                 std::regex_constants::extended)));
        assert(m.size() == 3);
        assert(m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+std::regex_traits<char>::length(s));
        assert(m.length(0) == 7);
        assert(m.position(0) == 2);
        assert(m.str(0) == "cdefghi");
        assert(m.length(1) == 3);
        assert(m.position(1) == 4);
        assert(m.str(1) == "efg");
        assert(m.length(2) == 1);
        assert(m.position(2) == 4);
        assert(m.str(2) == "e");
    }
    {
        std::cmatch m;
        const char s[] = "abc";
        assert(std::regex_search(s, m, std::regex("^abc", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+3);
        assert(m.length(0) == 3);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "abcd";
        assert(std::regex_search(s, m, std::regex("^abc", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+4);
        assert(m.length(0) == 3);
        assert(m.position(0) == 0);
        assert(m.str(0) == "abc");
    }
    {
        std::cmatch m;
        const char s[] = "aabc";
        assert(!std::regex_search(s, m, std::regex("^abc", std::regex_constants::extended)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abc";
        assert(std::regex_search(s, m, std::regex("abc$", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+3);
        assert(m.length(0) == 3);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "efabc";
        assert(std::regex_search(s, m, std::regex("abc$", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+5);
        assert(m.length(0) == 3);
        assert(m.position(0) == 2);
        assert(m.str(0) == s+2);
    }
    {
        std::cmatch m;
        const char s[] = "efabcg";
        assert(!std::regex_search(s, m, std::regex("abc$", std::regex_constants::extended)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abc";
        assert(std::regex_search(s, m, std::regex("a.c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+3);
        assert(m.length(0) == 3);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "acc";
        assert(std::regex_search(s, m, std::regex("a.c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+3);
        assert(m.length(0) == 3);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "acc";
        assert(std::regex_search(s, m, std::regex("a.c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+3);
        assert(m.length(0) == 3);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "abcdef";
        assert(std::regex_search(s, m, std::regex("(.*).*", std::regex_constants::extended)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+6);
        assert(m.length(0) == 6);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
        assert(m.length(1) == 6);
        assert(m.position(1) == 0);
        assert(m.str(1) == s);
    }
    {
        std::cmatch m;
        const char s[] = "bc";
        assert(std::regex_search(s, m, std::regex("(a*)*", std::regex_constants::extended)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+2);
        assert(m.length(0) == 0);
        assert(m.position(0) == 0);
        assert(m.str(0) == "");
        assert(m.length(1) == 0);
        assert(m.position(1) == 0);
        assert(m.str(1) == "");
    }
    {
        std::cmatch m;
        const char s[] = "abbc";
        assert(!std::regex_search(s, m, std::regex("ab{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abbbc";
        assert(std::regex_search(s, m, std::regex("ab{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "abbbbc";
        assert(std::regex_search(s, m, std::regex("ab{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "abbbbbc";
        assert(std::regex_search(s, m, std::regex("ab{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "adefc";
        assert(!std::regex_search(s, m, std::regex("ab{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abbbbbbc";
        assert(!std::regex_search(s, m, std::regex("ab{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "adec";
        assert(!std::regex_search(s, m, std::regex("a.{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "adefc";
        assert(std::regex_search(s, m, std::regex("a.{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "adefgc";
        assert(std::regex_search(s, m, std::regex("a.{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "adefghc";
        assert(std::regex_search(s, m, std::regex("a.{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "adefghic";
        assert(!std::regex_search(s, m, std::regex("a.{3,5}c", std::regex_constants::extended)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "tournament";
        assert(std::regex_search(s, m, std::regex("tour|to|tournament",
                                              std::regex_constants::extended)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "tournamenttotour";
        assert(std::regex_search(s, m, std::regex("(tour|to|tournament)+",
               std::regex_constants::extended | std::regex_constants::nosubs)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "ttotour";
        assert(std::regex_search(s, m, std::regex("(tour|to|t)+",
                                              std::regex_constants::extended)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
        assert(m.length(1) == 4);
        assert(m.position(1) == 3);
        assert(m.str(1) == "tour");
    }
}

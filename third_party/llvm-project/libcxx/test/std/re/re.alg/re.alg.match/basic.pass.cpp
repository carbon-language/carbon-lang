//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <regex>

// template <class BidirectionalIterator, class Allocator, class charT, class traits>
//     bool
//     regex_match(BidirectionalIterator first, BidirectionalIterator last,
//                  match_results<BidirectionalIterator, Allocator>& m,
//                  const basic_regex<charT, traits>& e,
//                  regex_constants::match_flag_type flags = regex_constants::match_default);

#include <regex>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

int main(int, char**)
{
    {
        std::cmatch m;
        assert(!std::regex_match("a", m, std::regex()));
        assert(m.size() == 0);
        assert(m.empty());
    }
    {
        std::cmatch m;
        const char s[] = "a";
        assert(std::regex_match(s, m, std::regex("a", std::regex_constants::basic)));
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
        assert(std::regex_match(s, m, std::regex("ab", std::regex_constants::basic)));
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
        assert(!std::regex_match(s, m, std::regex("ba", std::regex_constants::basic)));
        assert(m.size() == 0);
        assert(m.empty());
    }
    {
        std::cmatch m;
        const char s[] = "aab";
        assert(!std::regex_match(s, m, std::regex("ab", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "aab";
        assert(!std::regex_match(s, m, std::regex("ab", std::regex_constants::basic),
                                            std::regex_constants::match_continuous));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abcd";
        assert(!std::regex_match(s, m, std::regex("bc", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abbc";
        assert(std::regex_match(s, m, std::regex("ab*c", std::regex_constants::basic)));
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
        assert(std::regex_match(s, m, std::regex("\\(ab\\)*c", std::regex_constants::basic)));
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
        assert(!std::regex_match(s, m, std::regex("cd\\(\\(e\\)fg\\)hi",
                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abc";
        assert(std::regex_match(s, m, std::regex("^abc", std::regex_constants::basic)));
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
        assert(!std::regex_match(s, m, std::regex("^abc", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "aabc";
        assert(!std::regex_match(s, m, std::regex("^abc", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abc";
        assert(std::regex_match(s, m, std::regex("abc$", std::regex_constants::basic)));
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
        assert(!std::regex_match(s, m, std::regex("abc$", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "efabcg";
        assert(!std::regex_match(s, m, std::regex("abc$", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abc";
        assert(std::regex_match(s, m, std::regex("a.c", std::regex_constants::basic)));
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
        assert(std::regex_match(s, m, std::regex("a.c", std::regex_constants::basic)));
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
        assert(std::regex_match(s, m, std::regex("a.c", std::regex_constants::basic)));
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
        assert(std::regex_match(s, m, std::regex("\\(.*\\).*", std::regex_constants::basic)));
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
        assert(!std::regex_match(s, m, std::regex("\\(a*\\)*", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abbc";
        assert(!std::regex_match(s, m, std::regex("ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abbbc";
        assert(std::regex_match(s, m, std::regex("ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == sizeof(s)-1);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "abbbbc";
        assert(std::regex_match(s, m, std::regex("ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == sizeof(s)-1);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "abbbbbc";
        assert(std::regex_match(s, m, std::regex("ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == sizeof(s)-1);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "adefc";
        assert(!std::regex_match(s, m, std::regex("ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abbbbbbc";
        assert(!std::regex_match(s, m, std::regex("ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "adec";
        assert(!std::regex_match(s, m, std::regex("a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "adefc";
        assert(std::regex_match(s, m, std::regex("a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == sizeof(s)-1);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "adefgc";
        assert(std::regex_match(s, m, std::regex("a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == sizeof(s)-1);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "adefghc";
        assert(std::regex_match(s, m, std::regex("a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == sizeof(s)-1);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "adefghic";
        assert(!std::regex_match(s, m, std::regex("a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "-ab,ab-";
        assert(std::regex_match(s, m, std::regex("-\\(.*\\),\\1-", std::regex_constants::basic)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
        assert(m.length(1) == 2);
        assert(m.position(1) == 1);
        assert(m.str(1) == "ab");
    }
    {
        std::cmatch m;
        const char s[] = "ababbabb";
        assert(std::regex_match(s, m, std::regex("^\\(ab*\\)*\\1$", std::regex_constants::basic)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
        assert(m.length(1) == 3);
        assert(m.position(1) == 2);
        assert(m.str(1) == "abb");
    }
    {
        std::cmatch m;
        const char s[] = "ababbab";
        assert(!std::regex_match(s, m, std::regex("^\\(ab*\\)*\\1$", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "aBAbbAbB";
        assert(std::regex_match(s, m, std::regex("^\\(Ab*\\)*\\1$",
                   std::regex_constants::basic | std::regex_constants::icase)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
        assert(m.length(1) == 3);
        assert(m.position(1) == 2);
        assert(m.str(1) == "Abb");
    }
    {
        std::cmatch m;
        const char s[] = "aBAbbAbB";
        assert(!std::regex_match(s, m, std::regex("^\\(Ab*\\)*\\1$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "a";
        assert(std::regex_match(s, m, std::regex("^[a]$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == 1);
        assert(m.position(0) == 0);
        assert(m.str(0) == "a");
    }
    {
        std::cmatch m;
        const char s[] = "a";
        assert(std::regex_match(s, m, std::regex("^[ab]$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == 1);
        assert(m.position(0) == 0);
        assert(m.str(0) == "a");
    }
    {
        std::cmatch m;
        const char s[] = "c";
        assert(std::regex_match(s, m, std::regex("^[a-f]$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == 1);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "g";
        assert(!std::regex_match(s, m, std::regex("^[a-f]$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "Iraqi";
        assert(!std::regex_match(s, m, std::regex("q[^u]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "Iraq";
        assert(!std::regex_match(s, m, std::regex("q[^u]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "AmB";
        assert(std::regex_match(s, m, std::regex("A[[:lower:]]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "AMB";
        assert(!std::regex_match(s, m, std::regex("A[[:lower:]]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "AMB";
        assert(std::regex_match(s, m, std::regex("A[^[:lower:]]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "AmB";
        assert(!std::regex_match(s, m, std::regex("A[^[:lower:]]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "A5B";
        assert(!std::regex_match(s, m, std::regex("A[^[:lower:]0-9]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "A?B";
        assert(std::regex_match(s, m, std::regex("A[^[:lower:]0-9]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "-";
        assert(std::regex_match(s, m, std::regex("[a[.hyphen.]z]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "z";
        assert(std::regex_match(s, m, std::regex("[a[.hyphen.]z]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<char>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::cmatch m;
        const char s[] = "m";
        assert(!std::regex_match(s, m, std::regex("[a[.hyphen.]z]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "01a45cef9";
        assert(!std::regex_match(s, m, std::regex("[ace1-9]*",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "01a45cef9";
        assert(!std::regex_match(s, m, std::regex("[ace1-9]\\{1,\\}",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        const char r[] = "^[-+]\\{0,1\\}[0-9]\\{1,\\}[CF]$";
        std::ptrdiff_t sr = std::char_traits<char>::length(r);
        typedef forward_iterator<const char*> FI;
        typedef bidirectional_iterator<const char*> BI;
        std::regex regex(FI(r), FI(r+sr), std::regex_constants::basic);
        std::match_results<BI> m;
        const char s[] = "-40C";
        std::ptrdiff_t ss = std::char_traits<char>::length(s);
        assert(std::regex_match(BI(s), BI(s+ss), m, regex));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == BI(s));
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == 4);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wcmatch m;
        assert(!std::regex_match(L"a", m, std::wregex()));
        assert(m.size() == 0);
        assert(m.empty());
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"a";
        assert(std::regex_match(s, m, std::wregex(L"a", std::regex_constants::basic)));
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
        assert(m.str(0) == L"a");
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"ab";
        assert(std::regex_match(s, m, std::wregex(L"ab", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s+2);
        assert(m.length(0) == 2);
        assert(m.position(0) == 0);
        assert(m.str(0) == L"ab");
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"ab";
        assert(!std::regex_match(s, m, std::wregex(L"ba", std::regex_constants::basic)));
        assert(m.size() == 0);
        assert(m.empty());
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"aab";
        assert(!std::regex_match(s, m, std::wregex(L"ab", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"aab";
        assert(!std::regex_match(s, m, std::wregex(L"ab", std::regex_constants::basic),
                                            std::regex_constants::match_continuous));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abcd";
        assert(!std::regex_match(s, m, std::wregex(L"bc", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abbc";
        assert(std::regex_match(s, m, std::wregex(L"ab*c", std::regex_constants::basic)));
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
        std::wcmatch m;
        const wchar_t s[] = L"ababc";
        assert(std::regex_match(s, m, std::wregex(L"\\(ab\\)*c", std::regex_constants::basic)));
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
        assert(m.str(1) == L"ab");
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abcdefghijk";
        assert(!std::regex_match(s, m, std::wregex(L"cd\\(\\(e\\)fg\\)hi",
                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abc";
        assert(std::regex_match(s, m, std::wregex(L"^abc", std::regex_constants::basic)));
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
        std::wcmatch m;
        const wchar_t s[] = L"abcd";
        assert(!std::regex_match(s, m, std::wregex(L"^abc", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"aabc";
        assert(!std::regex_match(s, m, std::wregex(L"^abc", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abc";
        assert(std::regex_match(s, m, std::wregex(L"abc$", std::regex_constants::basic)));
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
        std::wcmatch m;
        const wchar_t s[] = L"efabc";
        assert(!std::regex_match(s, m, std::wregex(L"abc$", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"efabcg";
        assert(!std::regex_match(s, m, std::wregex(L"abc$", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abc";
        assert(std::regex_match(s, m, std::wregex(L"a.c", std::regex_constants::basic)));
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
        std::wcmatch m;
        const wchar_t s[] = L"acc";
        assert(std::regex_match(s, m, std::wregex(L"a.c", std::regex_constants::basic)));
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
        std::wcmatch m;
        const wchar_t s[] = L"acc";
        assert(std::regex_match(s, m, std::wregex(L"a.c", std::regex_constants::basic)));
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
        std::wcmatch m;
        const wchar_t s[] = L"abcdef";
        assert(std::regex_match(s, m, std::wregex(L"\\(.*\\).*", std::regex_constants::basic)));
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
        std::wcmatch m;
        const wchar_t s[] = L"bc";
        assert(!std::regex_match(s, m, std::wregex(L"\\(a*\\)*", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abbc";
        assert(!std::regex_match(s, m, std::wregex(L"ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abbbc";
        assert(std::regex_match(s, m, std::wregex(L"ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abbbbc";
        assert(std::regex_match(s, m, std::wregex(L"ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abbbbbc";
        assert(std::regex_match(s, m, std::wregex(L"ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"adefc";
        assert(!std::regex_match(s, m, std::wregex(L"ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"abbbbbbc";
        assert(!std::regex_match(s, m, std::wregex(L"ab\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"adec";
        assert(!std::regex_match(s, m, std::wregex(L"a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"adefc";
        assert(std::regex_match(s, m, std::wregex(L"a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"adefgc";
        assert(std::regex_match(s, m, std::wregex(L"a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"adefghc";
        assert(std::regex_match(s, m, std::wregex(L"a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"adefghic";
        assert(!std::regex_match(s, m, std::wregex(L"a.\\{3,5\\}c", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"-ab,ab-";
        assert(std::regex_match(s, m, std::wregex(L"-\\(.*\\),\\1-", std::regex_constants::basic)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
        assert(m.length(1) == 2);
        assert(m.position(1) == 1);
        assert(m.str(1) == L"ab");
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"ababbabb";
        assert(std::regex_match(s, m, std::wregex(L"^\\(ab*\\)*\\1$", std::regex_constants::basic)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
        assert(m.length(1) == 3);
        assert(m.position(1) == 2);
        assert(m.str(1) == L"abb");
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"ababbab";
        assert(!std::regex_match(s, m, std::wregex(L"^\\(ab*\\)*\\1$", std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"aBAbbAbB";
        assert(std::regex_match(s, m, std::wregex(L"^\\(Ab*\\)*\\1$",
                   std::regex_constants::basic | std::regex_constants::icase)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
        assert(m.length(1) == 3);
        assert(m.position(1) == 2);
        assert(m.str(1) == L"Abb");
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"aBAbbAbB";
        assert(!std::regex_match(s, m, std::wregex(L"^\\(Ab*\\)*\\1$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"a";
        assert(std::regex_match(s, m, std::wregex(L"^[a]$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == 1);
        assert(m.position(0) == 0);
        assert(m.str(0) == L"a");
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"a";
        assert(std::regex_match(s, m, std::wregex(L"^[ab]$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == 1);
        assert(m.position(0) == 0);
        assert(m.str(0) == L"a");
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"c";
        assert(std::regex_match(s, m, std::wregex(L"^[a-f]$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == 1);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"g";
        assert(!std::regex_match(s, m, std::wregex(L"^[a-f]$",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"Iraqi";
        assert(!std::regex_match(s, m, std::wregex(L"q[^u]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"Iraq";
        assert(!std::regex_match(s, m, std::wregex(L"q[^u]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"AmB";
        assert(std::regex_match(s, m, std::wregex(L"A[[:lower:]]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"AMB";
        assert(!std::regex_match(s, m, std::wregex(L"A[[:lower:]]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"AMB";
        assert(std::regex_match(s, m, std::wregex(L"A[^[:lower:]]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"AmB";
        assert(!std::regex_match(s, m, std::wregex(L"A[^[:lower:]]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"A5B";
        assert(!std::regex_match(s, m, std::wregex(L"A[^[:lower:]0-9]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"A?B";
        assert(std::regex_match(s, m, std::wregex(L"A[^[:lower:]0-9]B",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"-";
        assert(std::regex_match(s, m, std::wregex(L"[a[.hyphen.]z]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"z";
        assert(std::regex_match(s, m, std::wregex(L"[a[.hyphen.]z]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) >= 0 && static_cast<size_t>(m.length(0)) == std::char_traits<wchar_t>::length(s));
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"m";
        assert(!std::regex_match(s, m, std::wregex(L"[a[.hyphen.]z]",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"01a45cef9";
        assert(!std::regex_match(s, m, std::wregex(L"[ace1-9]*",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        std::wcmatch m;
        const wchar_t s[] = L"01a45cef9";
        assert(!std::regex_match(s, m, std::wregex(L"[ace1-9]\\{1,\\}",
                                                 std::regex_constants::basic)));
        assert(m.size() == 0);
    }
    {
        const wchar_t r[] = L"^[-+]\\{0,1\\}[0-9]\\{1,\\}[CF]$";
        std::ptrdiff_t sr = std::char_traits<wchar_t>::length(r);
        typedef forward_iterator<const wchar_t*> FI;
        typedef bidirectional_iterator<const wchar_t*> BI;
        std::wregex regex(FI(r), FI(r+sr), std::regex_constants::basic);
        std::match_results<BI> m;
        const wchar_t s[] = L"-40C";
        std::ptrdiff_t ss = std::char_traits<wchar_t>::length(s);
        assert(std::regex_match(BI(s), BI(s+ss), m, regex));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == BI(s));
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == m[0].second);
        assert(m.length(0) == 4);
        assert(m.position(0) == 0);
        assert(m.str(0) == s);
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

    { // LWG 2273
        std::regex re("Foo|FooBar");
        std::cmatch m;
        {
            assert(std::regex_match("FooBar", m, re));
            assert(m.size() == 1);
            assert(m[0] == "FooBar");
        }
        {
            assert(std::regex_match("Foo", m, re));
            assert(m.size() == 1);
            assert(m[0] == "Foo");
        }
        {
            assert(!std::regex_match("FooBarBaz", m, re));
            assert(m.size() == 0);
            assert(m.empty());
        }
        {
            assert(!std::regex_match("FooBa", m, re));
            assert(m.size() == 0);
            assert(m.empty());
        }
    }

  return 0;
}

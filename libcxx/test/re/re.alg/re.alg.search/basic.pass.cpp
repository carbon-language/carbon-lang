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

int main()
{
    {
        std::cmatch m;
        assert(!std::regex_search("a", m, std::regex()));
        assert(m.size() == 0);
        assert(m.empty());
    }
    {
        std::cmatch m;
        const char s[] = "a";
        assert(std::regex_search(s, m, std::regex("a", std::regex_constants::basic)));
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
        assert(std::regex_search(s, m, std::regex("ab", std::regex_constants::basic)));
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
        assert(!std::regex_search(s, m, std::regex("ba", std::regex_constants::basic)));
        assert(m.size() == 0);
        assert(m.empty());
    }
    {
        std::cmatch m;
        const char s[] = "aab";
        assert(std::regex_search(s, m, std::regex("ab", std::regex_constants::basic)));
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
        assert(!std::regex_search(s, m, std::regex("ab", std::regex_constants::basic),
                                            std::regex_constants::match_continuous));
        assert(m.size() == 0);
    }
    {
        std::cmatch m;
        const char s[] = "abcd";
        assert(std::regex_search(s, m, std::regex("bc", std::regex_constants::basic)));
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
        assert(std::regex_search(s, m, std::regex("ab*c", std::regex_constants::basic)));
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
//     {
//         std::cmatch m;
//         const char s[] = "abcdefghijk";
//         assert(std::regex_search(s, m, std::regex("cd\\(\\(e\\)fg\\)hi",
//                                  std::regex_constants::basic)));
//         assert(m.size() == 3);
//         assert(m.prefix().matched);
//         assert(m.prefix().first == s);
//         assert(m.prefix().second == m[0].first);
//         assert(m.suffix().matched);
//         assert(m.suffix().first == m[0].second);
//         assert(m.suffix().second == s+std::regex_traits<char>::length(s));
//         assert(m.length(0) == 7);
//         assert(m.position(0) == 2);
//         assert(m.str(0) == "cdefghi");
//         assert(m.length(1) == 3);
//         assert(m.position(1) == 4);
//         assert(m.str(1) == "efg");
//         assert(m.length(2) == 1);
//         assert(m.position(2) == 4);
//         assert(m.str(2) == "e");
//     }
}

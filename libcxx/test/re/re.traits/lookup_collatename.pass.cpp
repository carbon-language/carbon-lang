//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT> struct regex_traits;

// template <class ForwardIterator>
//   string_type
//   lookup_collatename(ForwardIterator first, ForwardIterator last) const;

#include <regex>
#include <iterator>
#include <cassert>
#include "iterators.h"

template <class char_type>
void
test(const char_type* A, const char_type* expected)
{
    std::regex_traits<char_type> t;
    typedef forward_iterator<const char_type*> F;
    assert(t.lookup_collatename(F(A), F(A + t.length(A))) == expected);
}

int main()
{
    test("NUL", "\x00");
    test("alert", "\x07");
    test("backspace", "\x08");
    test("tab", "\x09");
    test("carriage-return", "\x0D");
    test("newline", "\x0A");
    test("vertical-tab", "\x0B");
    test("form-feed", "\x0C");
    test("space", " ");
    test("exclamation-mark", "!");
    test("quotation-mark", "\"");
    test("number-sign", "#");
    test("dollar-sign", "$");
    test("percent-sign", "%");
    test("ampersand", "&");
    test("apostrophe", "\'");
    test("left-parenthesis", "(");
    test("right-parenthesis", ")");
    test("asterisk", "*");
    test("plus-sign", "+");
    test("comma", ",");
    test("hyphen-minus", "-");
    test("hyphen", "-");
    test("full-stop", ".");
    test("period", ".");
    test("slash", "/");
    test("solidus", "/");
    test("zero", "0");
    test("one", "1");
    test("two", "2");
    test("three", "3");
    test("four", "4");
    test("five", "5");
    test("six", "6");
    test("seven", "7");
    test("eight", "8");
    test("nine", "9");
    test("colon", ":");
    test("semicolon", ";");
    test("less-than-sign", "<");
    test("equals-sign", "=");
    test("greater-than-sign", ">");
    test("question-mark", "?");
    test("commercial-at", "@");
    for (char c = 'A'; c <= 'Z'; ++c)
    {
        const char a[2] = {c};
        test(a, a);
    }
    test("left-square-bracket", "[");
    test("backslash", "\\");
    test("reverse-solidus", "\\");
    test("right-square-bracket", "]");
    test("circumflex-accent", "^");
    test("circumflex", "^");
    test("low-line", "_");
    test("underscore", "_");
    test("grave-accent", "`");
    for (char c = 'a'; c <= 'z'; ++c)
    {
        const char a[2] = {c};
        test(a, a);
    }
    test("left-brace", "{");
    test("left-curly-bracket", "{");
    test("vertical-line", "|");
    test("right-brace", "}");
    test("right-curly-bracket", "}");
    test("tilde", "~");

    test("tild", "");
    test("ch", "");
    std::locale::global(std::locale("cs_CZ.ISO8859-2"));
    test("ch", "ch");
    std::locale::global(std::locale("C"));

    test(L"NUL", L"\x00");
    test(L"alert", L"\x07");
    test(L"backspace", L"\x08");
    test(L"tab", L"\x09");
    test(L"carriage-return", L"\x0D");
    test(L"newline", L"\x0A");
    test(L"vertical-tab", L"\x0B");
    test(L"form-feed", L"\x0C");
    test(L"space", L" ");
    test(L"exclamation-mark", L"!");
    test(L"quotation-mark", L"\"");
    test(L"number-sign", L"#");
    test(L"dollar-sign", L"$");
    test(L"percent-sign", L"%");
    test(L"ampersand", L"&");
    test(L"apostrophe", L"\'");
    test(L"left-parenthesis", L"(");
    test(L"right-parenthesis", L")");
    test(L"asterisk", L"*");
    test(L"plus-sign", L"+");
    test(L"comma", L",");
    test(L"hyphen-minus", L"-");
    test(L"hyphen", L"-");
    test(L"full-stop", L".");
    test(L"period", L".");
    test(L"slash", L"/");
    test(L"solidus", L"/");
    test(L"zero", L"0");
    test(L"one", L"1");
    test(L"two", L"2");
    test(L"three", L"3");
    test(L"four", L"4");
    test(L"five", L"5");
    test(L"six", L"6");
    test(L"seven", L"7");
    test(L"eight", L"8");
    test(L"nine", L"9");
    test(L"colon", L":");
    test(L"semicolon", L";");
    test(L"less-than-sign", L"<");
    test(L"equals-sign", L"=");
    test(L"greater-than-sign", L">");
    test(L"question-mark", L"?");
    test(L"commercial-at", L"@");
    for (wchar_t c = L'A'; c <= L'Z'; ++c)
    {
        const wchar_t a[2] = {c};
        test(a, a);
    }
    test(L"left-square-bracket", L"[");
    test(L"backslash", L"\\");
    test(L"reverse-solidus", L"\\");
    test(L"right-square-bracket", L"]");
    test(L"circumflex-accent", L"^");
    test(L"circumflex", L"^");
    test(L"low-line", L"_");
    test(L"underscore", L"_");
    test(L"grave-accent", L"`");
    for (wchar_t c = L'a'; c <= L'z'; ++c)
    {
        const wchar_t a[2] = {c};
        test(a, a);
    }
    test(L"left-brace", L"{");
    test(L"left-curly-bracket", L"{");
    test(L"vertical-line", L"|");
    test(L"right-brace", L"}");
    test(L"right-curly-bracket", L"}");
    test(L"tilde", L"~");

    test(L"tild", L"");
    test(L"ch", L"");
    std::locale::global(std::locale("cs_CZ.ISO8859-2"));
    test(L"ch", L"ch");
    std::locale::global(std::locale("C"));
}

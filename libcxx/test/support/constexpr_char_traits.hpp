// -*- C++ -*-
//===-------_------------ constexpr_char_traits ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _CONSTEXPR_CHAR_TRAITS
#define _CONSTEXPR_CHAR_TRAITS

#include <__config>
#include <string>


template <class _CharT>
struct constexpr_char_traits
{
    typedef _CharT    char_type;
    typedef int       int_type;
    typedef std::streamoff off_type;
    typedef std::streampos pos_type;
    typedef std::mbstate_t state_type;

    static _LIBCPP_CONSTEXPR_AFTER_CXX11 void assign(char_type& __c1, const char_type& __c2) _NOEXCEPT
        {__c1 = __c2;}

    static _LIBCPP_CONSTEXPR bool eq(char_type __c1, char_type __c2) _NOEXCEPT
        {return __c1 == __c2;}

    static _LIBCPP_CONSTEXPR  bool lt(char_type __c1, char_type __c2) _NOEXCEPT
        {return __c1 < __c2;}

    static _LIBCPP_CONSTEXPR_AFTER_CXX11 int              compare(const char_type* __s1, const char_type* __s2, size_t __n);
    static _LIBCPP_CONSTEXPR_AFTER_CXX11 size_t           length(const char_type* __s);
    static _LIBCPP_CONSTEXPR_AFTER_CXX11 const char_type* find(const char_type* __s, size_t __n, const char_type& __a);
    static _LIBCPP_CONSTEXPR_AFTER_CXX11 char_type*       move(char_type* __s1, const char_type* __s2, size_t __n);
    static _LIBCPP_CONSTEXPR_AFTER_CXX11 char_type*       copy(char_type* __s1, const char_type* __s2, size_t __n);
    static _LIBCPP_CONSTEXPR_AFTER_CXX11 char_type*       assign(char_type* __s, size_t __n, char_type __a);

    static _LIBCPP_CONSTEXPR int_type  not_eof(int_type __c) _NOEXCEPT
        {return eq_int_type(__c, eof()) ? ~eof() : __c;}

    static _LIBCPP_CONSTEXPR char_type to_char_type(int_type __c) _NOEXCEPT
        {return char_type(__c);}

    static _LIBCPP_CONSTEXPR int_type  to_int_type(char_type __c) _NOEXCEPT
        {return int_type(__c);}

    static _LIBCPP_CONSTEXPR bool      eq_int_type(int_type __c1, int_type __c2) _NOEXCEPT
        {return __c1 == __c2;}

    static _LIBCPP_CONSTEXPR int_type  eof() _NOEXCEPT
        {return int_type(EOF);}
};


template <class _CharT>
_LIBCPP_CONSTEXPR_AFTER_CXX11 int
constexpr_char_traits<_CharT>::compare(const char_type* __s1, const char_type* __s2, size_t __n)
{
    for (; __n; --__n, ++__s1, ++__s2)
    {
        if (lt(*__s1, *__s2))
            return -1;
        if (lt(*__s2, *__s1))
            return 1;
    }
    return 0;
}

template <class _CharT>
_LIBCPP_CONSTEXPR_AFTER_CXX11 size_t
constexpr_char_traits<_CharT>::length(const char_type* __s)
{
    size_t __len = 0;
    for (; !eq(*__s, char_type(0)); ++__s)
        ++__len;
    return __len;
}

template <class _CharT>
_LIBCPP_CONSTEXPR_AFTER_CXX11 const _CharT*
constexpr_char_traits<_CharT>::find(const char_type* __s, size_t __n, const char_type& __a)
{
    for (; __n; --__n)
    {
        if (eq(*__s, __a))
            return __s;
        ++__s;
    }
    return 0;
}

template <class _CharT>
_LIBCPP_CONSTEXPR_AFTER_CXX11 _CharT*
constexpr_char_traits<_CharT>::move(char_type* __s1, const char_type* __s2, size_t __n)
{
    char_type* __r = __s1;
    if (__s1 < __s2)
    {
        for (; __n; --__n, ++__s1, ++__s2)
            assign(*__s1, *__s2);
    }
    else if (__s2 < __s1)
    {
        __s1 += __n;
        __s2 += __n;
        for (; __n; --__n)
            assign(*--__s1, *--__s2);
    }
    return __r;
}

template <class _CharT>
_LIBCPP_CONSTEXPR_AFTER_CXX11 _CharT*
constexpr_char_traits<_CharT>::copy(char_type* __s1, const char_type* __s2, size_t __n)
{
    _LIBCPP_ASSERT(__s2 < __s1 || __s2 >= __s1+__n, "char_traits::copy overlapped range");
    char_type* __r = __s1;
    for (; __n; --__n, ++__s1, ++__s2)
        assign(*__s1, *__s2);
    return __r;
}

template <class _CharT>
_LIBCPP_CONSTEXPR_AFTER_CXX11 _CharT*
constexpr_char_traits<_CharT>::assign(char_type* __s, size_t __n, char_type __a)
{
    char_type* __r = __s;
    for (; __n; --__n, ++__s)
        assign(*__s, __a);
    return __r;
}

#endif // _CONSTEXPR_CHAR_TRAITS

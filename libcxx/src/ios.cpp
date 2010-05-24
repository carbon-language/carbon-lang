//===-------------------------- ios.cpp -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ios"
#include "streambuf"
#include "istream"
#include "string"
#include "__locale"
#include "algorithm"
#include "memory"
#include "new"
#include "limits"
#include <stdlib.h>

_LIBCPP_BEGIN_NAMESPACE_STD

template class __basic_string_common<true>;

template class basic_string<char>;
template class basic_string<wchar_t>;

template enable_if<__is_forward_iterator<char const*>::value, void>::type
    basic_string<char, char_traits<char>, allocator<char> >
    ::__init<char const*>(char const*, char const*);

template enable_if<__is_forward_iterator<wchar_t const*>::value, void>::type
    basic_string<wchar_t, char_traits<wchar_t>, allocator<wchar_t> >
    ::__init<wchar_t const*>(wchar_t const*, wchar_t const*);

template
    enable_if<__is_forward_iterator<char*>::value,
    basic_string<char, char_traits<char>, allocator<char> >&>::type
    basic_string<char, char_traits<char>, allocator<char> >::
    append<char*>(char*, char*);

template
    enable_if<__is_forward_iterator<wchar_t*>::value,
    basic_string<wchar_t, char_traits<wchar_t>, allocator<wchar_t> >&>::type
    basic_string<wchar_t, char_traits<wchar_t>, allocator<wchar_t> >::
    append<wchar_t*>(wchar_t*, wchar_t*);

template
    enable_if<__is_forward_iterator<char const*>::value,
    string::iterator>::type
    string::
    insert<char const*>(string::const_iterator, char const*, char const*);

template
    enable_if<__is_forward_iterator<wchar_t const*>::value,
    wstring::iterator>::type
    wstring::
    insert<wchar_t const*>(wstring::const_iterator, wchar_t const*, wchar_t const*);

template
    enable_if<__is_input_iterator<char const*>::value, string&>::type
    string::
    replace<char const*>(string::iterator, string::iterator, char const*, char const*);

template
    enable_if<__is_input_iterator<wchar_t const*>::value, wstring&>::type
    wstring::
    replace<wchar_t const*>(wstring::iterator, wstring::iterator, wchar_t const*, wchar_t const*);

template
    enable_if<__is_forward_iterator<wchar_t*>::value, wstring&>::type
    wstring::assign<wchar_t*>(wchar_t*, wchar_t*);

template
    string
    operator+<char, char_traits<char>, allocator<char> >(char const*, string const&);

template class basic_ios<char>;
template class basic_ios<wchar_t>;

template class basic_streambuf<char>;
template class basic_streambuf<wchar_t>;

template class basic_istream<char>;
template class basic_istream<wchar_t>;

template class basic_ostream<char>;
template class basic_ostream<wchar_t>;

template class basic_iostream<char>;

class _LIBCPP_HIDDEN __iostream_category
    : public __do_message
{
public:
    virtual const char* name() const;
    virtual string message(int ev) const;
};

const char*
__iostream_category::name() const
{
    return "iostream";
}

string
__iostream_category::message(int ev) const
{
    if (ev != static_cast<int>(io_errc::stream)
#ifdef ELAST
        && ev <= ELAST
#endif
        )
        return __do_message::message(ev);
    return string("unspecified iostream_category error");
}

const error_category&
iostream_category()
{
    static __iostream_category s;
    return s;
}

// ios_base::failure

ios_base::failure::failure(const string& msg, const error_code& ec)
    : system_error(ec, msg)
{
}

ios_base::failure::failure(const char* msg, const error_code& ec)
    : system_error(ec, msg)
{
}

ios_base::failure::~failure() throw()
{
}

// ios_base locale

const ios_base::fmtflags ios_base::boolalpha;
const ios_base::fmtflags ios_base::dec;
const ios_base::fmtflags ios_base::fixed;
const ios_base::fmtflags ios_base::hex;
const ios_base::fmtflags ios_base::internal;
const ios_base::fmtflags ios_base::left;
const ios_base::fmtflags ios_base::oct;
const ios_base::fmtflags ios_base::right;
const ios_base::fmtflags ios_base::scientific;
const ios_base::fmtflags ios_base::showbase;
const ios_base::fmtflags ios_base::showpoint;
const ios_base::fmtflags ios_base::showpos;
const ios_base::fmtflags ios_base::skipws;
const ios_base::fmtflags ios_base::unitbuf;
const ios_base::fmtflags ios_base::uppercase;
const ios_base::fmtflags ios_base::adjustfield;
const ios_base::fmtflags ios_base::basefield;
const ios_base::fmtflags ios_base::floatfield;

const ios_base::iostate ios_base::badbit;
const ios_base::iostate ios_base::eofbit;
const ios_base::iostate ios_base::failbit;
const ios_base::iostate ios_base::goodbit;

const ios_base::openmode ios_base::app;
const ios_base::openmode ios_base::ate;
const ios_base::openmode ios_base::binary;
const ios_base::openmode ios_base::in;
const ios_base::openmode ios_base::out;
const ios_base::openmode ios_base::trunc;

void
ios_base::__call_callbacks(event ev)
{
    for (size_t i = __event_size_; i;)
    {
        --i;
        __fn_[i](ev, *this, __index_[i]);
    }
}

// locale

locale
ios_base::imbue(const locale& newloc)
{
    static_assert(sizeof(locale) == sizeof(__loc_), "");
    locale& loc_storage = *(locale*)&__loc_;
    locale oldloc = loc_storage;
    loc_storage = newloc;
    __call_callbacks(imbue_event);
    return oldloc;
}

locale
ios_base::getloc() const
{
    const locale& loc_storage = *(locale*)&__loc_;
    return loc_storage;
}

// xalloc

int ios_base::__xindex_ = 0;

int
ios_base::xalloc()
{
    return __xindex_++;
}

long&
ios_base::iword(int index)
{
    size_t req_size = static_cast<size_t>(index)+1;
    if (req_size > __iarray_cap_)
    {
        size_t newcap;
        const size_t mx = std::numeric_limits<size_t>::max();
        if (req_size < mx/2)
            newcap = max(2 * __iarray_cap_, req_size);
        else
            newcap = mx;
        long* iarray = (long*)realloc(__iarray_, newcap * sizeof(long));
        if (iarray == 0)
        {
            setstate(badbit);
            static long error;
            error = 0;
            return error;
        }
        __iarray_ = iarray;
        for (long* p = __iarray_ + __iarray_size_; __iarray_cap_ < newcap; ++__iarray_cap_, ++p)
            *p = 0;
    }
    __iarray_size_ = max<size_t>(__iarray_size_, req_size);
    return __iarray_[index];
}

void*&
ios_base::pword(int index)
{
    size_t req_size = static_cast<size_t>(index)+1;
    if (req_size > __parray_cap_)
    {
        size_t newcap;
        const size_t mx = std::numeric_limits<size_t>::max();
        if (req_size < mx/2)
            newcap = max(2 * __parray_cap_, req_size);
        else
            newcap = mx;
        void** parray = (void**)realloc(__parray_, newcap * sizeof(void*));
        if (parray == 0)
        {
            setstate(badbit);
            static void* error;
            error = 0;
            return error;
        }
        __parray_ = parray;
        for (void** p = __parray_ + __parray_size_; __parray_cap_ < newcap; ++__parray_cap_, ++p)
            *p = 0;
    }
    __parray_size_ = max<size_t>(__parray_size_, req_size);
    return __parray_[index];
}

// register_callback

void
ios_base::register_callback(event_callback fn, int index)
{
    size_t req_size = __event_size_ + 1;
    if (req_size > __event_cap_)
    {
        size_t newcap;
        const size_t mx = std::numeric_limits<size_t>::max();
        if (req_size < mx/2)
            newcap = max(2 * __event_cap_, req_size);
        else
            newcap = mx;
        event_callback* fns = (event_callback*)realloc(__fn_, newcap * sizeof(event_callback));
        if (fns == 0)
            setstate(badbit);
        __fn_ = fns;
        int* indxs = (int*)realloc(__index_, newcap * sizeof(int));
        if (indxs == 0)
            setstate(badbit);
        __index_ = indxs;
    }
    __fn_[__event_size_] = fn;
    __index_[__event_size_] = index;
    ++__event_size_;
}

ios_base::~ios_base()
{
    __call_callbacks(erase_event);
    locale& loc_storage = *(locale*)&__loc_;
    loc_storage.~locale();
    free(__fn_);
    free(__index_);
    free(__iarray_);
    free(__parray_);
}

// iostate

void
ios_base::clear(iostate state)
{
    if (__rdbuf_)
        __rdstate_ = state;
    else
        __rdstate_ = state | badbit;
    if (((state | (__rdbuf_ ? goodbit : badbit)) & __exceptions_) != 0)
        throw failure("ios_base::clear");
}

// init

void
ios_base::init(void* sb)
{
    __rdbuf_ = sb;
    __rdstate_ = __rdbuf_ ? goodbit : badbit;
    __exceptions_ = goodbit;
    __fmtflags_ = skipws | dec;
    __width_ = 0;
    __precision_ = 6;
    __fn_ = 0;
    __index_ = 0;
    __event_size_ = 0;
    __event_cap_ = 0;
    __iarray_ = 0;
    __iarray_size_ = 0;
    __iarray_cap_ = 0;
    __parray_ = 0;
    __parray_size_ = 0;
    __parray_cap_ = 0;
    ::new(&__loc_) locale;
}

void
ios_base::copyfmt(const ios_base& rhs)
{
    // If we can't acquire the needed resources, throw bad_alloc (can't set badbit)
    // Don't alter *this until all needed resources are aquired
    unique_ptr<event_callback, void (*)(void*)> new_callbacks(0, free);
    unique_ptr<int, void (*)(void*)> new_ints(0, free);
    unique_ptr<long, void (*)(void*)> new_longs(0, free);
    unique_ptr<void*, void (*)(void*)> new_pointers(0, free);
    if (__event_cap_ < rhs.__event_size_)
    {
        new_callbacks.reset((event_callback*)malloc(sizeof(event_callback) * rhs.__event_size_));
        if (!new_callbacks)
            throw bad_alloc();
        new_ints.reset((int*)malloc(sizeof(int) * rhs.__event_size_));
        if (!new_ints)
            throw bad_alloc();
    }
    if (__iarray_cap_ < rhs.__iarray_size_)
    {
        new_longs.reset((long*)malloc(sizeof(long) * rhs.__iarray_size_));
        if (!new_longs)
            throw bad_alloc();
    }
    if (__parray_cap_ < rhs.__parray_size_)
    {
        new_pointers.reset((void**)malloc(sizeof(void*) * rhs.__parray_size_));
        if (!new_pointers)
            throw bad_alloc();
    }
    // Got everything we need.  Copy everything but __rdstate_, __rdbuf_ and __exceptions_
    __fmtflags_ = rhs.__fmtflags_;
    __precision_ = rhs.__precision_;
    __width_ = rhs.__width_;
    locale& lhs_loc = *(locale*)&__loc_;
    locale& rhs_loc = *(locale*)&rhs.__loc_;
    lhs_loc = rhs_loc;
    if (__event_cap_ < rhs.__event_size_)
    {
        free(__fn_);
        __fn_ = new_callbacks.release();
        free(__index_);
        __index_ = new_ints.release();
        __event_cap_ = rhs.__event_size_;
    }
    for (__event_size_ = 0; __event_size_ < rhs.__event_size_; ++__event_size_)
    {
        __fn_[__event_size_] = rhs.__fn_[__event_size_];
        __index_[__event_size_] = rhs.__index_[__event_size_];
    }
    if (__iarray_cap_ < rhs.__iarray_size_)
    {
        free(__iarray_);
        __iarray_ = new_longs.release();
        __iarray_cap_ = rhs.__iarray_size_;
    }
    for (__iarray_size_ = 0; __iarray_size_ < rhs.__iarray_size_; ++__iarray_size_)
        __iarray_[__iarray_size_] = rhs.__iarray_[__iarray_size_];
    if (__parray_cap_ < rhs.__parray_size_)
    {
        free(__parray_);
        __parray_ = new_pointers.release();
        __parray_cap_ = rhs.__parray_size_;
    }
    for (__parray_size_ = 0; __parray_size_ < rhs.__parray_size_; ++__parray_size_)
        __parray_[__parray_size_] = rhs.__parray_[__parray_size_];
}

void
ios_base::move(ios_base& rhs)
{
    // *this is uninitialized
    __fmtflags_ = rhs.__fmtflags_;
    __precision_ = rhs.__precision_;
    __width_ = rhs.__width_;
    __rdstate_ = rhs.__rdstate_;
    __exceptions_ = rhs.__exceptions_;
    __rdbuf_ = 0;
    locale& rhs_loc = *(locale*)&rhs.__loc_;
    ::new(&__loc_) locale(rhs_loc);
    __fn_ = rhs.__fn_;
    rhs.__fn_ = 0;
    __index_ = rhs.__index_;
    rhs.__index_ = 0;
    __event_size_ = rhs.__event_size_;
    rhs.__event_size_ = 0;
    __event_cap_ = rhs.__event_cap_;
    rhs.__event_cap_ = 0;
    __iarray_ = rhs.__iarray_;
    rhs.__iarray_ = 0;
    __iarray_size_ = rhs.__iarray_size_;
    rhs.__iarray_size_ = 0;
    __iarray_cap_ = rhs.__iarray_cap_;
    rhs.__iarray_cap_ = 0;
    __parray_ = rhs.__parray_;
    rhs.__parray_ = 0;
    __parray_size_ = rhs.__parray_size_;
    rhs.__parray_size_ = 0;
    __parray_cap_ = rhs.__parray_cap_;
    rhs.__parray_cap_ = 0;
}

void
ios_base::swap(ios_base& rhs)
{
    _STD::swap(__fmtflags_, rhs.__fmtflags_);
    _STD::swap(__precision_, rhs.__precision_);
    _STD::swap(__width_, rhs.__width_);
    _STD::swap(__rdstate_, rhs.__rdstate_);
    _STD::swap(__exceptions_, rhs.__exceptions_);
    locale& lhs_loc = *(locale*)&__loc_;
    locale& rhs_loc = *(locale*)&rhs.__loc_;
    _STD::swap(lhs_loc, rhs_loc);
    _STD::swap(__fn_, rhs.__fn_);
    _STD::swap(__index_, rhs.__index_);
    _STD::swap(__event_size_, rhs.__event_size_);
    _STD::swap(__event_cap_, rhs.__event_cap_);
    _STD::swap(__iarray_, rhs.__iarray_);
    _STD::swap(__iarray_size_, rhs.__iarray_size_);
    _STD::swap(__iarray_cap_, rhs.__iarray_cap_);
    _STD::swap(__parray_, rhs.__parray_);
    _STD::swap(__parray_size_, rhs.__parray_size_);
    _STD::swap(__parray_cap_, rhs.__parray_cap_);
}

void
ios_base::__set_badbit_and_consider_rethrow()
{
    __rdstate_ |= badbit;
    if (__exceptions_ & badbit)
        throw;
}

void
ios_base::__set_failbit_and_consider_rethrow()
{
    __rdstate_ |= failbit;
    if (__exceptions_ & failbit)
        throw;
}

bool
ios_base::sync_with_stdio(bool sync)
{
    static bool previous_state = true;
    bool r = previous_state;
    previous_state = sync;
    return r;
}

_LIBCPP_END_NAMESPACE_STD

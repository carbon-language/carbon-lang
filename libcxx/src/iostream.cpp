//===------------------------ iostream.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "__std_stream"
#include "string"
#include "new"

_LIBCPP_BEGIN_NAMESPACE_STD

alignas (__stdinbuf<char> ) static char __cin [sizeof(__stdinbuf <char>)];
alignas (__stdoutbuf<char>) static char __cout[sizeof(__stdoutbuf<char>)];
alignas (__stdoutbuf<char>) static char __cerr[sizeof(__stdoutbuf<char>)];
alignas (__stdinbuf<wchar_t> ) static char __wcin [sizeof(__stdinbuf <wchar_t>)];
alignas (__stdoutbuf<wchar_t>) static char __wcout[sizeof(__stdoutbuf<wchar_t>)];
alignas (__stdoutbuf<wchar_t>) static char __wcerr[sizeof(__stdoutbuf<wchar_t>)];

alignas (istream) char cin [sizeof(istream)];
alignas (ostream) char cout[sizeof(ostream)];
alignas (ostream) char cerr[sizeof(ostream)];
alignas (ostream) char clog[sizeof(ostream)];
alignas (wistream) char wcin [sizeof(wistream)];
alignas (wostream) char wcout[sizeof(wostream)];
alignas (wostream) char wcerr[sizeof(wostream)];
alignas (wostream) char wclog[sizeof(wostream)];

ios_base::Init __start_std_streams;

ios_base::Init::Init()
{
    istream* cin_ptr  = ::new(cin)  istream(::new(__cin)  __stdinbuf <char>(stdin) );
    ostream* cout_ptr = ::new(cout) ostream(::new(__cout) __stdoutbuf<char>(stdout));
    ostream* cerr_ptr = ::new(cerr) ostream(::new(__cerr) __stdoutbuf<char>(stderr));
                        ::new(clog) ostream(cerr_ptr->rdbuf());
    cin_ptr->tie(cout_ptr);
    _VSTD::unitbuf(*cerr_ptr);
    cerr_ptr->tie(cout_ptr);

    wistream* wcin_ptr  = ::new(wcin)  wistream(::new(__wcin)  __stdinbuf <wchar_t>(stdin) );
    wostream* wcout_ptr = ::new(wcout) wostream(::new(__wcout) __stdoutbuf<wchar_t>(stdout));
    wostream* wcerr_ptr = ::new(wcerr) wostream(::new(__wcerr) __stdoutbuf<wchar_t>(stderr));
                          ::new(wclog) wostream(wcerr_ptr->rdbuf());
    wcin_ptr->tie(wcout_ptr);
    _VSTD::unitbuf(*wcerr_ptr);
    wcerr_ptr->tie(wcout_ptr);
}

ios_base::Init::~Init()
{
    ostream* cout_ptr = (ostream*)cout;
    ostream* clog_ptr = (ostream*)clog;
    cout_ptr->flush();
    clog_ptr->flush();

    wostream* wcout_ptr = (wostream*)wcout;
    wostream* wclog_ptr = (wostream*)wclog;
    wcout_ptr->flush();
    wclog_ptr->flush();
}

_LIBCPP_END_NAMESPACE_STD

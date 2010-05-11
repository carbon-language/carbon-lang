//===---------------------- system_error.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "system_error"
#include "string"
#include "cstring"

_LIBCPP_BEGIN_NAMESPACE_STD

// class error_category

error_category::error_category()
{
}

error_category::~error_category()
{
}

error_condition
error_category::default_error_condition(int ev) const
{
    return error_condition(ev, *this);
}

bool
error_category::equivalent(int code, const error_condition& condition) const
{
    return default_error_condition(code) == condition;
}

bool
error_category::equivalent(const error_code& code, int condition) const
{
    return *this == code.category() && code.value() == condition;
}

string
__do_message::message(int ev) const
{
    return string(strerror(ev));
}

class _LIBCPP_HIDDEN __generic_error_category
    : public __do_message
{
public:
    virtual const char* name() const;
    virtual string message(int ev) const;
};

const char*
__generic_error_category::name() const
{
    return "generic";
}

string
__generic_error_category::message(int ev) const
{
    if (ev <= ELAST)
        return __do_message::message(ev);
    return string("unspecified generic_category error");
}

const error_category&
generic_category()
{
    static __generic_error_category s;
    return s;
}

class _LIBCPP_HIDDEN __system_error_category
    : public __do_message
{
public:
    virtual const char* name() const;
    virtual string message(int ev) const;
    virtual error_condition default_error_condition(int ev) const;
};

const char*
__system_error_category::name() const
{
    return "system";
}

string
__system_error_category::message(int ev) const
{
    if (ev <= ELAST)
        return __do_message::message(ev);
    return string("unspecified system_category error");
}

error_condition
__system_error_category::default_error_condition(int ev) const
{
    if (ev <= ELAST)
        return error_condition(ev, generic_category());
    return error_condition(ev, system_category());
}

const error_category&
system_category()
{
    static __system_error_category s;
    return s;
}

// error_condition

string
error_condition::message() const
{
    return __cat_->message(__val_);
}

// error_code

string
error_code::message() const
{
    return __cat_->message(__val_);
}

// system_error

string
system_error::__init(const error_code& ec, string what_arg)
{
    if (ec)
    {
        if (!what_arg.empty())
            what_arg += ": ";
        what_arg += ec.message();
    }
    return _STD::move(what_arg);
}

system_error::system_error(error_code ec, const string& what_arg)
    : runtime_error(__init(ec, what_arg)),
      __ec_(ec)
{
}

system_error::system_error(error_code ec, const char* what_arg)
    : runtime_error(__init(ec, what_arg)),
      __ec_(ec)
{
}

system_error::system_error(error_code ec)
    : runtime_error(__init(ec, "")),
      __ec_(ec)
{
}

system_error::system_error(int ev, const error_category& ecat, const string& what_arg)
    : runtime_error(__init(error_code(ev, ecat), what_arg)),
      __ec_(error_code(ev, ecat))
{
}

system_error::system_error(int ev, const error_category& ecat, const char* what_arg)
    : runtime_error(__init(error_code(ev, ecat), what_arg)),
      __ec_(error_code(ev, ecat))
{
}

system_error::system_error(int ev, const error_category& ecat)
    : runtime_error(__init(error_code(ev, ecat), "")),
      __ec_(error_code(ev, ecat))
{
}

system_error::~system_error() throw()
{
}

void
__throw_system_error(int ev, const char* what_arg)
{
    throw system_error(error_code(ev, system_category()), what_arg);
}

_LIBCPP_END_NAMESPACE_STD

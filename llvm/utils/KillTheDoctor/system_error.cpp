//===---------------------- system_error.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This was lifted from libc++ and modified for C++03.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "system_error.h"
#include <string>
#include <cstring>

namespace llvm {

// class error_category

error_category::error_category() {
}

error_category::~error_category() {
}

error_condition
error_category::default_error_condition(int ev) const {
  return error_condition(ev, *this);
}

bool
error_category::equivalent(int code, const error_condition& condition) const {
  return default_error_condition(code) == condition;
}

bool
error_category::equivalent(const error_code& code, int condition) const {
  return *this == code.category() && code.value() == condition;
}

std::string
_do_message::message(int ev) const {
  return std::string(std::strerror(ev));
}

class _generic_error_category : public _do_message {
public:
  virtual const char* name() const;
  virtual std::string message(int ev) const;
};

const char*
_generic_error_category::name() const {
  return "generic";
}

std::string
_generic_error_category::message(int ev) const {
#ifdef ELAST
  if (ev > ELAST)
    return std::string("unspecified generic_category error");
#endif  // ELAST
  return _do_message::message(ev);
}

const error_category&
generic_category() {
  static _generic_error_category s;
  return s;
}

class _system_error_category : public _do_message {
public:
  virtual const char* name() const;
  virtual std::string message(int ev) const;
  virtual error_condition default_error_condition(int ev) const;
};

const char*
_system_error_category::name() const {
  return "system";
}

// std::string _system_error_category::message(int ev) const {
// Is in Platform/system_error.inc

// error_condition _system_error_category::default_error_condition(int ev) const
// Is in Platform/system_error.inc

const error_category&
system_category() {
  static _system_error_category s;
  return s;
}

// error_condition

std::string
error_condition::message() const {
  return _cat_->message(_val_);
}

// error_code

std::string
error_code::message() const {
  return _cat_->message(_val_);
}

// system_error

std::string
system_error::_init(const error_code& ec, std::string what_arg) {
  if (ec)
  {
    if (!what_arg.empty())
      what_arg += ": ";
    what_arg += ec.message();
  }
  return what_arg;
}

system_error::system_error(error_code ec, const std::string& what_arg)
  : runtime_error(_init(ec, what_arg)), _ec_(ec) {
}

system_error::system_error(error_code ec, const char* what_arg)
  : runtime_error(_init(ec, what_arg)), _ec_(ec) {
}

system_error::system_error(error_code ec)
  : runtime_error(_init(ec, "")), _ec_(ec) {
}

system_error::system_error(int ev, const error_category& ecat,
                           const std::string& what_arg)
  : runtime_error(_init(error_code(ev, ecat), what_arg))
  , _ec_(error_code(ev, ecat)) {
}

system_error::system_error(int ev, const error_category& ecat,
                           const char* what_arg)
  : runtime_error(_init(error_code(ev, ecat), what_arg))
  , _ec_(error_code(ev, ecat)) {
}

system_error::system_error(int ev, const error_category& ecat)
  : runtime_error(_init(error_code(ev, ecat), "")), _ec_(error_code(ev, ecat)) {
}

system_error::~system_error() throw() {
}

void
_throw_system_error(int ev, const char* what_arg) {
  throw system_error(error_code(ev, system_category()), what_arg);
}

} // end namespace llvm

#ifdef LLVM_ON_WIN32
#include <Windows.h>
#include <WinError.h>

namespace llvm {

std::string
_system_error_category::message(int ev) const {
  LPVOID lpMsgBuf = 0;
  DWORD retval = ::FormatMessageA(
    FORMAT_MESSAGE_ALLOCATE_BUFFER |
    FORMAT_MESSAGE_FROM_SYSTEM |
    FORMAT_MESSAGE_IGNORE_INSERTS,
    NULL,
    ev,
    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
    (LPSTR) &lpMsgBuf,
    0,
    NULL);
  if (retval == 0) {
    ::LocalFree(lpMsgBuf);
    return std::string("Unknown error");
  }

  std::string str( static_cast<LPCSTR>(lpMsgBuf) );
  ::LocalFree(lpMsgBuf);

  while (str.size()
     && (str[str.size()-1] == '\n' || str[str.size()-1] == '\r'))
    str.erase( str.size()-1 );
  if (str.size() && str[str.size()-1] == '.')
    str.erase( str.size()-1 );
  return str;
}

error_condition
_system_error_category::default_error_condition(int ev) const {
  switch (ev)
  {
  case 0: return make_error_condition(errc::success);
  // Windows system -> posix_errno decode table  ---------------------------//
  // see WinError.h comments for descriptions of errors
  case ERROR_ACCESS_DENIED: return make_error_condition(errc::permission_denied);
  case ERROR_ALREADY_EXISTS: return make_error_condition(errc::file_exists);
  case ERROR_BAD_UNIT: return make_error_condition(errc::no_such_device);
  case ERROR_BUFFER_OVERFLOW: return make_error_condition(errc::filename_too_long);
  case ERROR_BUSY: return make_error_condition(errc::device_or_resource_busy);
  case ERROR_BUSY_DRIVE: return make_error_condition(errc::device_or_resource_busy);
  case ERROR_CANNOT_MAKE: return make_error_condition(errc::permission_denied);
  case ERROR_CANTOPEN: return make_error_condition(errc::io_error);
  case ERROR_CANTREAD: return make_error_condition(errc::io_error);
  case ERROR_CANTWRITE: return make_error_condition(errc::io_error);
  case ERROR_CURRENT_DIRECTORY: return make_error_condition(errc::permission_denied);
  case ERROR_DEV_NOT_EXIST: return make_error_condition(errc::no_such_device);
  case ERROR_DEVICE_IN_USE: return make_error_condition(errc::device_or_resource_busy);
  case ERROR_DIR_NOT_EMPTY: return make_error_condition(errc::directory_not_empty);
  case ERROR_DIRECTORY: return make_error_condition(errc::invalid_argument);
  case ERROR_DISK_FULL: return make_error_condition(errc::no_space_on_device);
  case ERROR_FILE_EXISTS: return make_error_condition(errc::file_exists);
  case ERROR_FILE_NOT_FOUND: return make_error_condition(errc::no_such_file_or_directory);
  case ERROR_HANDLE_DISK_FULL: return make_error_condition(errc::no_space_on_device);
  case ERROR_INVALID_ACCESS: return make_error_condition(errc::permission_denied);
  case ERROR_INVALID_DRIVE: return make_error_condition(errc::no_such_device);
  case ERROR_INVALID_FUNCTION: return make_error_condition(errc::function_not_supported);
  case ERROR_INVALID_HANDLE: return make_error_condition(errc::invalid_argument);
  case ERROR_INVALID_NAME: return make_error_condition(errc::invalid_argument);
  case ERROR_LOCK_VIOLATION: return make_error_condition(errc::no_lock_available);
  case ERROR_LOCKED: return make_error_condition(errc::no_lock_available);
  case ERROR_NEGATIVE_SEEK: return make_error_condition(errc::invalid_argument);
  case ERROR_NOACCESS: return make_error_condition(errc::permission_denied);
  case ERROR_NOT_ENOUGH_MEMORY: return make_error_condition(errc::not_enough_memory);
  case ERROR_NOT_READY: return make_error_condition(errc::resource_unavailable_try_again);
  case ERROR_NOT_SAME_DEVICE: return make_error_condition(errc::cross_device_link);
  case ERROR_OPEN_FAILED: return make_error_condition(errc::io_error);
  case ERROR_OPEN_FILES: return make_error_condition(errc::device_or_resource_busy);
  case ERROR_OPERATION_ABORTED: return make_error_condition(errc::operation_canceled);
  case ERROR_OUTOFMEMORY: return make_error_condition(errc::not_enough_memory);
  case ERROR_PATH_NOT_FOUND: return make_error_condition(errc::no_such_file_or_directory);
  case ERROR_READ_FAULT: return make_error_condition(errc::io_error);
  case ERROR_RETRY: return make_error_condition(errc::resource_unavailable_try_again);
  case ERROR_SEEK: return make_error_condition(errc::io_error);
  case ERROR_SHARING_VIOLATION: return make_error_condition(errc::permission_denied);
  case ERROR_TOO_MANY_OPEN_FILES: return make_error_condition(errc::too_many_files_open);
  case ERROR_WRITE_FAULT: return make_error_condition(errc::io_error);
  case ERROR_WRITE_PROTECT: return make_error_condition(errc::permission_denied);
  case ERROR_SEM_TIMEOUT: return make_error_condition(errc::timed_out);
  case WSAEACCES: return make_error_condition(errc::permission_denied);
  case WSAEADDRINUSE: return make_error_condition(errc::address_in_use);
  case WSAEADDRNOTAVAIL: return make_error_condition(errc::address_not_available);
  case WSAEAFNOSUPPORT: return make_error_condition(errc::address_family_not_supported);
  case WSAEALREADY: return make_error_condition(errc::connection_already_in_progress);
  case WSAEBADF: return make_error_condition(errc::bad_file_descriptor);
  case WSAECONNABORTED: return make_error_condition(errc::connection_aborted);
  case WSAECONNREFUSED: return make_error_condition(errc::connection_refused);
  case WSAECONNRESET: return make_error_condition(errc::connection_reset);
  case WSAEDESTADDRREQ: return make_error_condition(errc::destination_address_required);
  case WSAEFAULT: return make_error_condition(errc::bad_address);
  case WSAEHOSTUNREACH: return make_error_condition(errc::host_unreachable);
  case WSAEINPROGRESS: return make_error_condition(errc::operation_in_progress);
  case WSAEINTR: return make_error_condition(errc::interrupted);
  case WSAEINVAL: return make_error_condition(errc::invalid_argument);
  case WSAEISCONN: return make_error_condition(errc::already_connected);
  case WSAEMFILE: return make_error_condition(errc::too_many_files_open);
  case WSAEMSGSIZE: return make_error_condition(errc::message_size);
  case WSAENAMETOOLONG: return make_error_condition(errc::filename_too_long);
  case WSAENETDOWN: return make_error_condition(errc::network_down);
  case WSAENETRESET: return make_error_condition(errc::network_reset);
  case WSAENETUNREACH: return make_error_condition(errc::network_unreachable);
  case WSAENOBUFS: return make_error_condition(errc::no_buffer_space);
  case WSAENOPROTOOPT: return make_error_condition(errc::no_protocol_option);
  case WSAENOTCONN: return make_error_condition(errc::not_connected);
  case WSAENOTSOCK: return make_error_condition(errc::not_a_socket);
  case WSAEOPNOTSUPP: return make_error_condition(errc::operation_not_supported);
  case WSAEPROTONOSUPPORT: return make_error_condition(errc::protocol_not_supported);
  case WSAEPROTOTYPE: return make_error_condition(errc::wrong_protocol_type);
  case WSAETIMEDOUT: return make_error_condition(errc::timed_out);
  case WSAEWOULDBLOCK: return make_error_condition(errc::operation_would_block);
  default: return error_condition(ev, system_category());
  }
}

} // end namespace llvm

#endif // LLVM_ON_WIN32

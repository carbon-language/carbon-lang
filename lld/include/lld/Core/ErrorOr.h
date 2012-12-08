//===- lld/Core/ErrorOr.h - Error Smart Pointer ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Provides ErrorOr<T> smart pointer.
///
/// This should be moved to LLVMSupport when someone has time to make it c++03.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_ERROR_OR_H
#define LLD_CORE_ERROR_OR_H

#include "llvm/Support/AlignOf.h"
#include "llvm/Support/system_error.h"

#include <cassert>
#include <type_traits>

namespace lld {
template<class T>
class ErrorOrBase {
  static const bool isRef = std::is_reference<T>::value;
  typedef std::reference_wrapper<typename std::remove_reference<T>::type> wrap;

public:
  typedef typename
    std::conditional< isRef
                    , wrap
                    , T
                    >::type storage_type;

private:
  typedef T &reference;
  typedef typename std::remove_reference<T>::type *pointer;

  ErrorOrBase(const ErrorOrBase&) LLVM_DELETED_FUNCTION;
  ErrorOrBase &operator =(const ErrorOrBase&) LLVM_DELETED_FUNCTION;
  ErrorOrBase(ErrorOrBase &&other) LLVM_DELETED_FUNCTION;
  ErrorOrBase &operator =(ErrorOrBase &&other) LLVM_DELETED_FUNCTION;

public:
  ErrorOrBase() : _error(llvm::make_error_code(llvm::errc::invalid_argument)) {}

  ErrorOrBase(llvm::error_code ec) {
    if (!_error)
      get()->~storage_type();
    _error = ec;
  }

  ErrorOrBase(T t) : _error(llvm::error_code::success()) {
    new (get()) storage_type(t);
  }

  ~ErrorOrBase() {
    if (!_error)
      get()->~storage_type();
  }

  /// \brief Return false if there is an error.
  operator bool() {
    return !_error;
  }

  operator llvm::error_code() {
    return _error;
  }

  operator reference() {
    return *get();
  }

  pointer operator ->() {
    return toPointer(get());
  }

  reference operator *() {
    return *get();
  }

private:
  pointer toPointer(pointer t) {
    return t;
  }

  pointer toPointer(wrap *t) {
    return &t->get();
  }

protected:
  storage_type *get() {
    assert(!_error && "T not valid!");
    return reinterpret_cast<storage_type*>(_t.buffer);
  }

  llvm::error_code _error;
  llvm::AlignedCharArrayUnion<storage_type> _t;
};

/// \brief Represents either an error or a value T.
///
/// ErrorOr<T> is a pointer-like class that represents the result of an
/// operation. The result is either an error, or a value of type T. This is
/// designed to emulate the usage of returning a pointer where nullptr indicates
/// failure. However instead of just knowing that the operation failed, we also
/// have an error_code that describes why it failed.
///
/// It is used like the following.
/// \code
///   ErrorOr<Buffer> getBuffer();
///   void handleError(error_code ec);
///
///   auto buffer = getBuffer();
///   if (!buffer)
///     handleError(buffer);
///   buffer->write("adena");
/// \endcode
///
/// An implicit conversion to bool provides a way to check if there was an
/// error. The unary * and -> operators provide pointer like access to the
/// value. Accessing the value when there is an error has undefined behavior.
///
/// When T is a reference type the behaivor is slightly different. The reference
/// is held in a std::reference_wrapper<std::remove_reference<T>::type>, and
/// there is special handling to make operator -> work as if T was not a
/// reference.
///
/// T cannot be a rvalue reference.
template<class T,
  bool isMoveable =
    std::is_move_constructible<typename ErrorOrBase<T>::storage_type>::value>
class ErrorOr;

template<class T>
class ErrorOr<T, true> : public ErrorOrBase<T> {
  ErrorOr(const ErrorOr &other) LLVM_DELETED_FUNCTION;
  ErrorOr &operator =(const ErrorOr &other) LLVM_DELETED_FUNCTION;
public:
  ErrorOr(llvm::error_code ec) : ErrorOrBase<T>(ec) {}
  ErrorOr(T t) : ErrorOrBase<T>(t) {}
  ErrorOr(ErrorOr &&other) : ErrorOrBase<T>() {
    // Get the other value.
    if (!other._error)
      new (this->get())
        typename ErrorOrBase<T>::storage_type(std::move(*other.get()));

    // Get the other error.
    this->_error = other._error;

    // Make sure other doesn't try to delete its storage.
    other._error = llvm::make_error_code(llvm::errc::invalid_argument);
  }

  ErrorOr &operator =(ErrorOr &&other) {
    // Delete any existing value.
    if (!this->_error)
      this->get()->~storage_type();

    // Get the other value.
    if (!other._error)
      new (this->get())
        typename ErrorOrBase<T>::storage_type(std::move(*other.get()));

    // Get the other error.
    this->_error = other._error;

    // Make sure other doesn't try to delete its storage.
    other._error = llvm::make_error_code(llvm::errc::invalid_argument);
  }
};

template<class T>
class ErrorOr<T, false> : public ErrorOrBase<T> {
  static_assert(std::is_copy_constructible<T>::value,
                "T must be copy or move constructible!");

  ErrorOr(ErrorOr &&other) LLVM_DELETED_FUNCTION;
  ErrorOr &operator =(ErrorOr &&other) LLVM_DELETED_FUNCTION;
public:
  ErrorOr(llvm::error_code ec) : ErrorOrBase<T>(ec) {}
  ErrorOr(T t) : ErrorOrBase<T>(t) {}
  ErrorOr(const ErrorOr &other) : ErrorOrBase<T>() {
    // Get the other value.
    if (!other._error)
      new (this->get()) typename ErrorOrBase<T>::storage_type(*other.get());

    // Get the other error.
    this->_error = other._error;
  }

  ErrorOr &operator =(const ErrorOr &other) {
    // Delete any existing value.
    if (!this->_error)
      this->get()->~storage_type();

    // Get the other value.
    if (!other._error)
      new (this->get()) typename ErrorOrBase<T>::storage_type(*other.get());

    // Get the other error.
    this->_error = other._error;
  }
};
}

#endif

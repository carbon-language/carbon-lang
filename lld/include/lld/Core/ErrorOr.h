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

#include <atomic>
#include <cassert>
#include <type_traits>

namespace lld {
struct ErrorHolderBase {
  llvm::error_code Error;
  std::atomic<uint16_t> RefCount;
  bool HasUserData;

  ErrorHolderBase() : RefCount(1) {}

  void aquire() {
    ++RefCount;
  }

  void release() {
    if (RefCount.fetch_sub(1) == 1)
      delete this;
  }

protected:
  virtual ~ErrorHolderBase() {}
};

template<class T>
struct ErrorHolder : ErrorHolderBase {
  ErrorHolder(T &&UD) : UserData(UD) {}
  T UserData;
};

template<class Tp> struct ErrorOrUserDataTraits : std::false_type {};

template<class T, class V>
typename std::enable_if< std::is_constructible<T, V>::value
                       , typename std::remove_reference<V>::type>::type &&
 moveIfMoveConstructible(V &t) {
  return std::move(t);
}

template<class T, class V>
typename std::enable_if< !std::is_constructible<T, V>::value
                       , typename std::remove_reference<V>::type>::type &
moveIfMoveConstructible(V &t) {
  return t;
}

/// \brief Represents either an error or a value T.
///
/// ErrorOr<T> is a pointer-like class that represents the result of an
/// operation. The result is either an error, or a value of type T. This is
/// designed to emulate the usage of returning a pointer where nullptr indicates
/// failure. However instead of just knowing that the operation failed, we also
/// have an error_code and optional user data that describes why it failed.
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
/// ErrorOr<T> also supports user defined data for specific error_codes. To use
/// this feature you must first add a template specialization of
/// ErrorOrUserDataTraits derived from std::true_type for your type in the lld
/// namespace. This specialization must have a static error_code error()
/// function that returns the error_code this data is used with.
///
/// getError<UserData>() may be called to get either the stored user data, or
/// a default constructed UserData if none was stored.
///
/// Example:
/// \code
///   struct InvalidArgError {
///     InvalidArgError() {}
///     InvalidArgError(std::string S) : ArgName(S) {}
///     std::string ArgName;
///   };
///
///  namespace lld {
///  template<>
///   struct ErrorOrUserDataTraits<InvalidArgError> : std::true_type {
///     static error_code error() {
///       return make_error_code(errc::invalid_argument);
///     }
///   };
///   } // end namespace lld
///
///   using namespace lld;
///
///   ErrorOr<int> foo() {
///     return InvalidArgError("adena");
///   }
///
///   int main() {
///     auto a = foo();
///     if (!a && error_code(a) == errc::invalid_argument)
///       llvm::errs() << a.getError<InvalidArgError>().ArgName << "\n";
///   }
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
template<class T>
class ErrorOr {
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

public:
  ErrorOr() : IsValid(false) {}

  ErrorOr(llvm::error_code ec) : HasError(true), IsValid(true) {
    Error = new ErrorHolderBase;
    Error->Error = ec;
    Error->HasUserData = false;
  }

  template<class UserDataT>
  ErrorOr(UserDataT UD, typename
          std::enable_if<ErrorOrUserDataTraits<UserDataT>::value>::type* = 0)
    : HasError(true), IsValid(true) {
    Error = new ErrorHolder<UserDataT>(std::move(UD));
    Error->Error = ErrorOrUserDataTraits<UserDataT>::error();
    Error->HasUserData = true;
  }

  ErrorOr(T t) : HasError(false), IsValid(true) {
    new (get()) storage_type(moveIfMoveConstructible<storage_type>(t));
  }

  ErrorOr(const ErrorOr &other) : IsValid(false) {
    // Construct an invalid ErrorOr if other is invalid.
    if (!other.IsValid)
      return;
    if (!other.HasError) {
      // Get the other value.
      new (get()) storage_type(*other.get());
      HasError = false;
    } else {
      // Get other's error.
      Error = other.Error;
      HasError = true;
      Error->aquire();
    }

    IsValid = true;
  }

  ErrorOr &operator =(const ErrorOr &other) {
    if (this == &other)
      return;

    this->~ErrorOr();
    new (this) ErrorOr(other);

    return *this;
  }

  ErrorOr(ErrorOr &&other) : IsValid(false) {
    // Construct an invalid ErrorOr if other is invalid.
    if (!other.IsValid)
      return;
    if (!other.HasError) {
      // Get the other value.
      new (get()) storage_type(std::move(*other.get()));
      HasError = false;
      // Tell other not to do any destruction.
      other.IsValid = false;
    } else {
      // Get other's error.
      Error = other.Error;
      HasError = true;
      // Tell other not to do any destruction.
      other.IsValid = false;
    }

    IsValid = true;
  }

  ErrorOr &operator =(ErrorOr &&other) {
    if (this == &other)
      return *this;

    this->~ErrorOr();
    new (this) ErrorOr(other);

    return *this;
  }

  ~ErrorOr() {
    if (!IsValid)
      return;
    if (HasError)
      Error->release();
    else
      get()->~storage_type();
  }

  template<class ET>
  ET getError() const {
    assert(IsValid && "Cannot get the error of a default constructed ErrorOr!");
    assert(HasError && "Cannot get an error if none exists!");
    assert(ErrorOrUserDataTraits<ET>::error() == Error->Error &&
           "Incorrect user error data type for error!");
    if (!Error->HasUserData)
      return ET();
    return reinterpret_cast<const ErrorHolder<ET>*>(Error)->UserData;
  }

  typedef void (*unspecified_bool_type)();
  static void unspecified_bool_true() {}

  /// \brief Return false if there is an error.
  operator unspecified_bool_type() const {
    assert(IsValid && "Can't do anything on a default constructed ErrorOr!");
    return HasError ? 0 : unspecified_bool_true;
  }

  operator llvm::error_code() const {
    assert(IsValid && "Can't do anything on a default constructed ErrorOr!");
    return HasError ? Error->Error : llvm::error_code::success();
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
    assert(IsValid && "Can't do anything on a default constructed ErrorOr!");
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<storage_type*>(_t.buffer);
  }

  const storage_type *get() const {
    assert(IsValid && "Can't do anything on a default constructed ErrorOr!");
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<const storage_type*>(_t.buffer);
  }

  union {
    llvm::AlignedCharArrayUnion<storage_type> _t;
    ErrorHolderBase *Error;
  };
  bool HasError : 1;
  bool IsValid : 1;
};

template<class T, class E>
typename std::enable_if<llvm::is_error_code_enum<E>::value ||
                        llvm::is_error_condition_enum<E>::value, bool>::type
operator ==(ErrorOr<T> &Err, E Code) {
  return error_code(Err) == Code;
}
} // end namespace lld

#endif

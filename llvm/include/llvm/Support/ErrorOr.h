//===- llvm/Support/ErrorOr.h - Error Smart Pointer -----------------------===//
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
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ERROR_OR_H
#define LLVM_SUPPORT_ERROR_OR_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/type_traits.h"

#include <cassert>
#if LLVM_HAS_CXX11_TYPETRAITS
#include <type_traits>
#endif

namespace llvm {
struct ErrorHolderBase {
  error_code Error;
  uint16_t RefCount;
  bool HasUserData;

  ErrorHolderBase() : RefCount(1) {}

  void aquire() {
    ++RefCount;
  }

  void release() {
    if (--RefCount == 0)
      delete this;
  }

protected:
  virtual ~ErrorHolderBase() {}
};

template<class T>
struct ErrorHolder : ErrorHolderBase {
#if LLVM_HAS_RVALUE_REFERENCES
  ErrorHolder(T &&UD) : UserData(llvm_move(UD)) {}
#else
  ErrorHolder(T &UD) : UserData(UD) {}
#endif
  T UserData;
};

template<class Tp> struct ErrorOrUserDataTraits : llvm::false_type {};

#if LLVM_HAS_CXX11_TYPETRAITS && LLVM_HAS_RVALUE_REFERENCES
template<class T, class V>
typename std::enable_if< std::is_constructible<T, V>::value
                       , typename std::remove_reference<V>::type>::type &&
 moveIfMoveConstructible(V &Val) {
  return std::move(Val);
}

template<class T, class V>
typename std::enable_if< !std::is_constructible<T, V>::value
                       , typename std::remove_reference<V>::type>::type &
moveIfMoveConstructible(V &Val) {
  return Val;
}
#else
template<class T, class V>
V &moveIfMoveConstructible(V &Val) {
  return Val;
}
#endif

/// \brief Stores a reference that can be changed.
template <typename T>
class ReferenceStorage {
  T *Storage;

public:
  ReferenceStorage(T &Ref) : Storage(&Ref) {}

  operator T &() const { return *Storage; }
  T &get() const { return *Storage; }
};

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
///   namespace llvm {
///   template<>
///   struct ErrorOrUserDataTraits<InvalidArgError> : std::true_type {
///     static error_code error() {
///       return make_error_code(errc::invalid_argument);
///     }
///   };
///   } // end namespace llvm
///
///   using namespace llvm;
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
  template <class OtherT> friend class ErrorOr;
  static const bool isRef = is_reference<T>::value;
  typedef ReferenceStorage<typename remove_reference<T>::type> wrap;

public:
  typedef typename
    conditional< isRef
               , wrap
               , T
               >::type storage_type;

private:
  typedef typename remove_reference<T>::type &reference;
  typedef typename remove_reference<T>::type *pointer;

public:
  ErrorOr() : IsValid(false) {}

  template <class E>
  ErrorOr(E ErrorCode, typename enable_if_c<is_error_code_enum<E>::value ||
                                            is_error_condition_enum<E>::value,
                                            void *>::type = 0)
      : HasError(true), IsValid(true) {
    Error = new ErrorHolderBase;
    Error->Error = make_error_code(ErrorCode);
    Error->HasUserData = false;
  }

  ErrorOr(llvm::error_code EC) : HasError(true), IsValid(true) {
    Error = new ErrorHolderBase;
    Error->Error = EC;
    Error->HasUserData = false;
  }

  template<class UserDataT>
  ErrorOr(UserDataT UD, typename
          enable_if_c<ErrorOrUserDataTraits<UserDataT>::value>::type* = 0)
    : HasError(true), IsValid(true) {
    Error = new ErrorHolder<UserDataT>(llvm_move(UD));
    Error->Error = ErrorOrUserDataTraits<UserDataT>::error();
    Error->HasUserData = true;
  }

  ErrorOr(T Val) : HasError(false), IsValid(true) {
    new (get()) storage_type(moveIfMoveConstructible<storage_type>(Val));
  }

  ErrorOr(const ErrorOr &Other) : IsValid(false) {
    copyConstruct(Other);
  }

  template <class OtherT>
  ErrorOr(const ErrorOr<OtherT> &Other) : IsValid(false) {
    copyConstruct(Other);
  }

  ErrorOr &operator =(const ErrorOr &Other) {
    copyAssign(Other);
    return *this;
  }

  template <class OtherT>
  ErrorOr &operator =(const ErrorOr<OtherT> &Other) {
    copyAssign(Other);
    return *this;
  }

#if LLVM_HAS_RVALUE_REFERENCES
  ErrorOr(ErrorOr &&Other) : IsValid(false) {
    moveConstruct(std::move(Other));
  }

  template <class OtherT>
  ErrorOr(ErrorOr<OtherT> &&Other) : IsValid(false) {
    moveConstruct(std::move(Other));
  }

  ErrorOr &operator =(ErrorOr &&Other) {
    moveAssign(std::move(Other));
    return *this;
  }

  template <class OtherT>
  ErrorOr &operator =(ErrorOr<OtherT> &&Other) {
    moveAssign(std::move(Other));
    return *this;
  }
#endif

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
  template <class OtherT>
  void copyConstruct(const ErrorOr<OtherT> &Other) {
    // Construct an invalid ErrorOr if other is invalid.
    if (!Other.IsValid)
      return;
    IsValid = true;
    if (!Other.HasError) {
      // Get the other value.
      HasError = false;
      new (get()) storage_type(*Other.get());
    } else {
      // Get other's error.
      Error = Other.Error;
      HasError = true;
      Error->aquire();
    }
  }

  template <class T1>
  static bool compareThisIfSameType(const T1 &a, const T1 &b) {
    return &a == &b;
  }

  template <class T1, class T2>
  static bool compareThisIfSameType(const T1 &a, const T2 &b) {
    return false;
  }

  template <class OtherT>
  void copyAssign(const ErrorOr<OtherT> &Other) {
    if (compareThisIfSameType(*this, Other))
      return;

    this->~ErrorOr();
    new (this) ErrorOr(Other);
  }

#if LLVM_HAS_RVALUE_REFERENCES
  template <class OtherT>
  void moveConstruct(ErrorOr<OtherT> &&Other) {
    // Construct an invalid ErrorOr if other is invalid.
    if (!Other.IsValid)
      return;
    IsValid = true;
    if (!Other.HasError) {
      // Get the other value.
      HasError = false;
      new (get()) storage_type(std::move(*Other.get()));
      // Tell other not to do any destruction.
      Other.IsValid = false;
    } else {
      // Get other's error.
      Error = Other.Error;
      HasError = true;
      // Tell other not to do any destruction.
      Other.IsValid = false;
    }
  }

  template <class OtherT>
  void moveAssign(ErrorOr<OtherT> &&Other) {
    if (compareThisIfSameType(*this, Other))
      return;

    this->~ErrorOr();
    new (this) ErrorOr(std::move(Other));
  }
#endif

  pointer toPointer(pointer Val) {
    return Val;
  }

  pointer toPointer(wrap *Val) {
    return &Val->get();
  }

  storage_type *get() {
    assert(IsValid && "Can't do anything on a default constructed ErrorOr!");
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<storage_type*>(TStorage.buffer);
  }

  const storage_type *get() const {
    assert(IsValid && "Can't do anything on a default constructed ErrorOr!");
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<const storage_type*>(TStorage.buffer);
  }

  union {
    AlignedCharArrayUnion<storage_type> TStorage;
    ErrorHolderBase *Error;
  };
  bool HasError : 1;
  bool IsValid : 1;
};

// ErrorOr specialization for void.
template <>
class ErrorOr<void> {
public:
  ErrorOr() : Error(0, 0) {}

  template <class E>
  ErrorOr(E ErrorCode, typename enable_if_c<is_error_code_enum<E>::value ||
                                            is_error_condition_enum<E>::value,
                                            void *> ::type = 0)
      : Error(0, 0) {
    error_code EC = make_error_code(ErrorCode);
    if (EC == errc::success) {
      Error.setInt(1);
      return;
    }
    ErrorHolderBase *EHB = new ErrorHolderBase;
    EHB->Error = EC;
    EHB->HasUserData = false;
    Error.setPointer(EHB);
  }

  ErrorOr(llvm::error_code EC) : Error(0, 0) {
    if (EC == errc::success) {
      Error.setInt(1);
      return;
    }
    ErrorHolderBase *E = new ErrorHolderBase;
    E->Error = EC;
    E->HasUserData = false;
    Error.setPointer(E);
  }

  template<class UserDataT>
  ErrorOr(UserDataT UD, typename
          enable_if_c<ErrorOrUserDataTraits<UserDataT>::value>::type* = 0)
      : Error(0, 0) {
    ErrorHolderBase *E = new ErrorHolder<UserDataT>(llvm_move(UD));
    E->Error = ErrorOrUserDataTraits<UserDataT>::error();
    E->HasUserData = true;
    Error.setPointer(E);
  }

  ErrorOr(const ErrorOr &Other) : Error(0, 0) {
    Error = Other.Error;
    if (Other.Error.getPointer()->Error) {
      Error.getPointer()->aquire();
    }
  }

  ErrorOr &operator =(const ErrorOr &Other) {
    if (this == &Other)
      return *this;

    this->~ErrorOr();
    new (this) ErrorOr(Other);

    return *this;
  }

#if LLVM_HAS_RVALUE_REFERENCES
  ErrorOr(ErrorOr &&Other) : Error(0) {
    // Get other's error.
    Error = Other.Error;
    // Tell other not to do any destruction.
    Other.Error.setPointer(0);
  }

  ErrorOr &operator =(ErrorOr &&Other) {
    if (this == &Other)
      return *this;

    this->~ErrorOr();
    new (this) ErrorOr(std::move(Other));

    return *this;
  }
#endif

  ~ErrorOr() {
    if (Error.getPointer())
      Error.getPointer()->release();
  }

  template<class ET>
  ET getError() const {
    assert(ErrorOrUserDataTraits<ET>::error() == *this &&
           "Incorrect user error data type for error!");
    if (!Error.getPointer()->HasUserData)
      return ET();
    return reinterpret_cast<const ErrorHolder<ET> *>(
        Error.getPointer())->UserData;
  }

  typedef void (*unspecified_bool_type)();
  static void unspecified_bool_true() {}

  /// \brief Return false if there is an error.
  operator unspecified_bool_type() const {
    return Error.getInt() ? unspecified_bool_true : 0;
  }

  operator llvm::error_code() const {
    return Error.getInt() ? make_error_code(errc::success)
                          : Error.getPointer()->Error;
  }

private:
  // If the bit is 1, the error is success.
  llvm::PointerIntPair<ErrorHolderBase *, 1> Error;
};

template<class T, class E>
typename enable_if_c<is_error_code_enum<E>::value ||
                     is_error_condition_enum<E>::value, bool>::type
operator ==(ErrorOr<T> &Err, E Code) {
  return error_code(Err) == Code;
}
} // end namespace llvm

#endif

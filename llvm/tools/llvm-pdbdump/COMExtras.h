//===- COMExtras.h - Helper files for COM operations -------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PDBDUMP_COMEXTRAS_H
#define LLVM_TOOLS_LLVM_PDBDUMP_COMEXTRAS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ConvertUTF.h"

#include <tuple>

namespace llvm {

template <typename F> struct function_traits;

#if LLVM_HAS_VARIADIC_TEMPLATES
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> : public function_traits<R(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (__stdcall C::*)(Args...)> {
  using args_tuple = std::tuple<Args...>;
};
#else

// For the sake of COM, we only need a 3 argument version and a 5 argument
// version. We could provide 1, 2, 4, and other length of argument lists if
// this were intended to be more generic.  Alternatively, this will "just work"
// if VS2012 support is dropped and we can use the variadic template case
// exclusively.
template <typename C, typename R, typename A1, typename A2, typename A3>
struct function_traits<R (__stdcall C::*)(A1, A2, A3)> {
  typedef std::tuple<A1, A2, A3> args_tuple;
};

template <typename C, typename R, typename A1, typename A2, typename A3,
          typename A4, typename A5>
struct function_traits<R (__stdcall C::*)(A1, A2, A3, A4, A5)> {
  typedef std::tuple<A1, A2, A3, A4, A5> args_tuple;
};
#endif

template <class FuncTraits, std::size_t arg> struct function_arg {
  // Writing function_arg as a separate class that accesses the tuple from
  // function_traits is necessary due to what appears to be a bug in MSVC.
  // If you write a nested class inside function_traits like this:
  // template<std::size_t ArgIndex>
  // struct Argument
  // {
  //   typedef typename
  //     std::tuple_element<ArgIndex, std::tuple<Args...>>::type type;
  // };
  // MSVC encounters a parsing error.
  typedef
      typename std::tuple_element<arg, typename FuncTraits::args_tuple>::type
          type;
};

template <class T> struct remove_double_pointer {};
template <class T> struct remove_double_pointer<T **> { typedef T type; };

namespace sys {
namespace windows {

/// A helper class for allowing the use of COM enumerators in range-based
/// for loops.
///
/// A common idiom in the COM world is to have an enumerator interface, say
/// IMyEnumerator.  It's responsible for enumerating over some child data type,
/// say IChildType.  You do the enumeration by calling IMyEnumerator::Next()
/// one of whose arguments will be an IChildType**.  Eventually Next() fails,
/// indicating that there are no more items.
///
/// com_iterator represents a single point-in-time of this iteration.  It is
/// used by ComEnumerator to support iterating in this fashion via range-based
/// for loops and other common C++ paradigms.
template <class EnumeratorType, std::size_t ArgIndex> class com_iterator {
  using FunctionTraits = function_traits<decltype(&EnumeratorType::Next)>;
  typedef typename function_arg<FunctionTraits, ArgIndex>::type FuncArgType;
  // FuncArgType is now something like ISomeCOMInterface **.  Remove both
  // pointers, so we can make a CComPtr<T> out of it.
  typedef typename remove_double_pointer<FuncArgType>::type EnumDataType;

  CComPtr<EnumeratorType> EnumeratorObject;
  CComPtr<EnumDataType> CurrentItem;

public:
  typedef CComPtr<EnumDataType> value_type;
  typedef std::ptrdiff_t difference_type;
  typedef value_type *pointer_type;
  typedef value_type &reference_type;
  typedef std::forward_iterator_tag iterator_category;

  explicit com_iterator(CComPtr<EnumeratorType> Enumerator,
                        CComPtr<EnumDataType> Current)
      : EnumeratorObject(Enumerator), CurrentItem(Current) {}
  com_iterator() {}

  com_iterator &operator++() {
    // EnumeratorObject->Next() expects CurrentItem to be NULL.
    CurrentItem.Release();
    ULONG Count = 0;
    HRESULT hr = EnumeratorObject->Next(1, &CurrentItem, &Count);
    if (FAILED(hr) || Count == 0)
      *this = com_iterator();

    return *this;
  }

  value_type operator*() { return CurrentItem; }

  bool operator==(const com_iterator &other) const {
    return (EnumeratorObject == other.EnumeratorObject) &&
           (CurrentItem == other.CurrentItem);
  }

  bool operator!=(const com_iterator &other) const { return !(*this == other); }

  com_iterator &operator=(const com_iterator &other) {
    EnumeratorObject = other.EnumeratorObject;
    CurrentItem = other.CurrentItem;
    return *this;
  }
};

/// ComEnumerator implements the interfaced required for C++ to allow its use
/// in range-based for loops.  In particular, a begin() and end() method.
/// These methods simply construct and return an appropriate ComIterator
/// instance.
template <class EnumeratorType, std::size_t ArgIndex> class com_enumerator {
  typedef function_traits<decltype(&EnumeratorType::Next)> FunctionTraits;
  typedef typename function_arg<FunctionTraits, ArgIndex>::type FuncArgType;
  typedef typename remove_double_pointer<FuncArgType>::type EnumDataType;

  CComPtr<EnumeratorType> EnumeratorObject;

public:
  com_enumerator(CComPtr<EnumeratorType> Enumerator)
      : EnumeratorObject(Enumerator) {}

  com_iterator<EnumeratorType, ArgIndex> begin() {
    if (!EnumeratorObject)
      return end();

    EnumeratorObject->Reset();
    ULONG Count = 0;
    CComPtr<EnumDataType> FirstItem;
    HRESULT hr = EnumeratorObject->Next(1, &FirstItem, &Count);
    return (FAILED(hr) || Count == 0) ? end()
                                      : com_iterator<EnumeratorType, ArgIndex>(
                                            EnumeratorObject, FirstItem);
  }

  com_iterator<EnumeratorType, ArgIndex> end() {
    return com_iterator<EnumeratorType, ArgIndex>();
  }
};

/// A helper class for allowing the use of COM record enumerators in range-
/// based for loops.
///
/// A record enumerator is almost the same as a regular enumerator, except
/// that it returns raw byte-data instead of interfaces to other COM objects.
/// As a result, the enumerator's Next() method has a slightly different
/// signature, and an iterator dereferences to an ArrayRef instead of a
/// CComPtr.
template <class EnumeratorType> class com_data_record_iterator {
public:
  typedef llvm::ArrayRef<uint8_t> value_type;
  typedef std::ptrdiff_t difference_type;
  typedef value_type *pointer_type;
  typedef value_type &reference_type;
  typedef std::forward_iterator_tag iterator_category;

  explicit com_data_record_iterator(CComPtr<EnumeratorType> enumerator)
      : Enumerator(enumerator), CurrentRecord(0) {
    // Make sure we start at the beginning.  If there are no records,
    // immediately set ourselves equal to end().
    if (enumerator)
      enumerator->Reset();

    if (!ReadNextRecord())
      *this = com_data_record_iterator();
  }
  com_data_record_iterator() {}

  com_data_record_iterator &operator++() {
    ++CurrentRecord;
    // If we can't read any more records, either because there are no more
    // or because we encountered an error, we should compare equal to end.
    if (!ReadNextRecord())
      *this = com_data_record_iterator();
    return *this;
  }

  value_type operator*() {
    return llvm::ArrayRef<uint8_t>(RecordData.begin(), RecordData.end());
  }

  bool operator==(const com_data_record_iterator &other) const {
    return (Enumerator == other.Enumerator) &&
           (CurrentRecord == other.CurrentRecord);
  }

  bool operator!=(const com_data_record_iterator &other) const {
    return !(*this == other);
  }

private:
  bool ReadNextRecord() {
    RecordData.clear();
    ULONG Count = 0;
    DWORD RequiredBufferSize;
    HRESULT hr = Enumerator->Next(1, 0, &RequiredBufferSize, nullptr, &Count);
    if (hr == S_OK) {
      RecordData.resize(RequiredBufferSize);
      DWORD BytesRead = 0;
      hr = Enumerator->Next(1, RequiredBufferSize, &BytesRead,
                            RecordData.data(), &Count);
    }
    if (hr != S_OK)
      RecordData.clear();
    return (hr == S_OK);
  }

  CComPtr<EnumeratorType> Enumerator;
  uint32_t CurrentRecord;
  llvm::SmallVector<uint8_t, 32> RecordData;
};

/// Similar to ComEnumerator, com_data_record_enumerator implements the range
/// interface for ComDataRecordIterators.
template <class EnumeratorType> class com_data_record_enumerator {
public:
  com_data_record_enumerator(CComPtr<EnumeratorType> enumerator)
      : Enumerator(enumerator) {}

  com_data_record_iterator<EnumeratorType> begin() {
    return com_data_record_iterator<EnumeratorType>(Enumerator);
  }

  com_data_record_iterator<EnumeratorType> end() {
    LONG NumElts = 0;
    HRESULT hr = Enumerator->get_Count(&NumElts);
    return (FAILED(hr)) ? com_data_record_iterator<EnumeratorType>(Enumerator)
                        : com_data_record_iterator<EnumeratorType>();
  }

private:
  CComPtr<EnumeratorType> Enumerator;
};

/// com_enumerator is a simple helper function to allow the enumerator
/// class's type to be inferred automatically.
/// This allows you to write simply:
///   for (auto item : com_enumerator(MyEnumerator)) {
///   }
template <class EnumeratorType>
com_enumerator<EnumeratorType, 1>
make_com_enumerator(CComPtr<EnumeratorType> Enumerator) {
  return com_enumerator<EnumeratorType, 1>(Enumerator);
}

/// com_data_record_enumerator is a simple helper function to allow the
/// enumerator class's type to be inferred automatically.
/// This allows you to write simply:
///   for (auto item : com_data_record_enumerator(MyEnumerator)) {
///   }
//=============================================================================
template <class EnumeratorType>
com_data_record_enumerator<EnumeratorType>
make_com_data_record_enumerator(CComPtr<EnumeratorType> Enumerator) {
  return com_data_record_enumerator<EnumeratorType>(Enumerator);
}

inline bool BSTRToUTF8(BSTR String16, std::string &String8) {
  UINT ByteLength = ::SysStringByteLen(String16);
  char *Bytes = reinterpret_cast<char *>(String16);
  String8.clear();
  return llvm::convertUTF16ToUTF8String(ArrayRef<char>(Bytes, ByteLength),
                                        String8);
}

} // namespace windows
} // namespace sys
} // namespace llvm

#endif

//===--- StringRef.h - Constant String Reference Wrapper --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGREF_H
#define LLVM_ADT_STRINGREF_H

#include <cstring>
#include <string>

namespace llvm {
  /// StringRef - Represent a constant reference to a string, i.e. a character
  /// array and a length, which need not be null terminated.
  ///
  /// This class does not own the string data, it is expected to be used in
  /// situations where the character data resides in some other buffer, whose
  /// lifetime extends past that of the StringRef. For this reason, it is not in
  /// general safe to store a StringRef.
  class StringRef {
  public:
    typedef const char *iterator;

  private:
    /// The start of the string, in an external buffer.
    const char *Data;

    /// The length of the string.
    unsigned Length;

  public:
    /// @name Constructors
    /// @{

    /// Construct an empty string ref.
    /*implicit*/ StringRef() : Data(0), Length(0) {}

    /// Construct a string ref from a cstring.
    /*implicit*/ StringRef(const char *Str) 
      : Data(Str), Length(::strlen(Str)) {}
 
    /// Construct a string ref from a pointer and length.
    /*implicit*/ StringRef(const char *_Data, unsigned _Length)
      : Data(_Data), Length(_Length) {}

    /// Construct a string ref from an std::string.
    /*implicit*/ StringRef(const std::string &Str) 
      : Data(Str.c_str()), Length(Str.length()) {}

    /// @}
    /// @name Iterators
    /// @{

    iterator begin() const { return Data; }

    iterator end() const { return Data + Length; }

    /// @}
    /// @name String Operations
    /// @{

    /// data - Get a pointer to the start of the string (which may not be null
    /// terminated).
    const char *data() const { return Data; }

    /// empty - Check if the string is empty.
    bool empty() const { return Length == 0; }

    /// size - Get the string size.
    unsigned size() const { return Length; }

    /// compare - Compare two strings; the result is -1, 0, or 1 if this string
    /// is lexicographically less than, equal to, or greater than the \arg RHS.
    int compare(const StringRef &RHS) const {
      // Check the prefix for a mismatch.
      if (int Res = memcmp(Data, RHS.Data, std::min(Length, RHS.Length)))
        return Res < 0 ? -1 : 1;

      // Otherwise the prefixes match, so we only need to check the lengths.
      if (Length == RHS.Length)
        return 0;
      return Length < RHS.Length ? -1 : 1;
    }

    /// str - Get the contents as an std::string.
    std::string str() const { return std::string(Data, Length); }

    /// @}
    /// @name Operator Overloads
    /// @{

    bool operator==(const StringRef &RHS) const { 
      return Length == RHS.Length && memcmp(Data, RHS.Data, Length) == 0; 
    }

    bool operator!=(const StringRef &RHS) const { return !(*this == RHS); }

    bool operator<(const StringRef &RHS) const { return compare(RHS) == -1; }

    bool operator<=(const StringRef &RHS) const { return compare(RHS) != 1; }

    bool operator>(const StringRef &RHS) const { return compare(RHS) == 1; }

    bool operator>=(const StringRef &RHS) const { return compare(RHS) != -1; }

    char operator[](size_t Index) const { 
      assert(Index < Length && "Invalid index!");
      return Data[Index]; 
    }

    /// @}
    /// @name Type Conversions
    /// @{

    operator std::string() const {
      return str();
    }

    /// @}
  };
}

#endif

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

#include <algorithm>
#include <cassert>
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
    static const size_t npos = ~size_t(0);

  private:
    /// The start of the string, in an external buffer.
    const char *Data;

    /// The length of the string.
    size_t Length;

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
    size_t size() const { return Length; }
    
    char back() const {
      assert(!empty());
      return Data[Length-1];
    }

    /// equals - Check for string equality, this is more efficient than
    /// compare() in when the relative ordering of inequal strings isn't needed.
    bool equals(const StringRef &RHS) const {
      return (Length == RHS.Length && 
              memcmp(Data, RHS.Data, RHS.Length) == 0);
    }

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
    /// @name Utility Functions
    /// @{

    /// substr - Return a reference to the substring from [Start, Start + N).
    ///
    /// \param Start - The index of the starting character in the substring; if
    /// the index is npos or greater than the length of the string then the
    /// empty substring will be returned.
    ///
    /// \param N - The number of characters to included in the substring. If N
    /// exceeds the number of characters remaining in the string, the string
    /// suffix (starting with \arg Start) will be returned.
    StringRef substr(size_t Start, size_t N = npos) const {
      Start = std::min(Start, Length);
      return StringRef(Data + Start, std::min(N, Length - Start));
    }

    /// slice - Return a reference to the substring from [Start, End).
    ///
    /// \param Start - The index of the starting character in the substring; if
    /// the index is npos or greater than the length of the string then the
    /// empty substring will be returned.
    ///
    /// \param End - The index following the last character to include in the
    /// substring. If this is npos, or less than \arg Start, or exceeds the
    /// number of characters remaining in the string, the string suffix
    /// (starting with \arg Start) will be returned.
    StringRef slice(size_t Start, size_t End) const {
      Start = std::min(Start, Length);
      End = std::min(std::max(Start, End), Length);
      return StringRef(Data + Start, End - Start);
    }

    /// split - Split into two substrings around the first occurence of a
    /// separator character.
    ///
    /// If \arg Separator is in the string, then the result is a pair (LHS, RHS)
    /// such that (*this == LHS + Separator + RHS) is true and RHS is
    /// maximal. If \arg Separator is not in the string, then the result is a
    /// pair (LHS, RHS) where (*this == LHS) and (RHS == "").
    ///
    /// \param Separator - The character to split on.
    /// \return - The split substrings.
    std::pair<StringRef, StringRef> split(char Separator) const {
      iterator it = std::find(begin(), end(), Separator);
      if (it == end())
        return std::make_pair(*this, StringRef());
      return std::make_pair(StringRef(begin(), it - begin()),
                            StringRef(it + 1, end() - (it + 1)));
    }

    /// startswith - Check if this string starts with the given \arg Prefix.
    bool startswith(const StringRef &Prefix) const { 
      return substr(0, Prefix.Length).equals(Prefix);
    }

    /// endswith - Check if this string ends with the given \arg Suffix.
    bool endswith(const StringRef &Suffix) const {
      return slice(size() - Suffix.Length, size()).equals(Suffix);
    }

    /// @}
  };

  /// @name StringRef Comparison Operators
  /// @{

  inline bool operator==(const StringRef &LHS, const StringRef &RHS) {
    return LHS.equals(RHS);
  }

  inline bool operator!=(const StringRef &LHS, const StringRef &RHS) { 
    return !(LHS == RHS);
  }
  
  inline bool operator<(const StringRef &LHS, const StringRef &RHS) {
    return LHS.compare(RHS) == -1; 
  }

  inline bool operator<=(const StringRef &LHS, const StringRef &RHS) {
    return LHS.compare(RHS) != 1; 
  }

  inline bool operator>(const StringRef &LHS, const StringRef &RHS) {
    return LHS.compare(RHS) == 1; 
  }

  inline bool operator>=(const StringRef &LHS, const StringRef &RHS) {
    return LHS.compare(RHS) != -1; 
  }

  /// @}

}

#endif

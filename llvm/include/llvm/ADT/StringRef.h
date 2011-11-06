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

#include <cassert>
#include <cstring>
#include <utility>
#include <string>

namespace llvm {
  template<typename T>
  class SmallVectorImpl;
  class APInt;

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
    typedef const char *const_iterator;
    static const size_t npos = ~size_t(0);
    typedef size_t size_type;

  private:
    /// The start of the string, in an external buffer.
    const char *Data;

    /// The length of the string.
    size_t Length;

    // Workaround PR5482: nearly all gcc 4.x miscompile StringRef and std::min()
    // Changing the arg of min to be an integer, instead of a reference to an
    // integer works around this bug.
    static size_t min(size_t a, size_t b) { return a < b ? a : b; }
    static size_t max(size_t a, size_t b) { return a > b ? a : b; }
    
    // Workaround memcmp issue with null pointers (undefined behavior)
    // by providing a specialized version
    static int compareMemory(const char *Lhs, const char *Rhs, size_t Length) {
      if (Length == 0) { return 0; }
      return ::memcmp(Lhs,Rhs,Length);
    }
    
  public:
    /// @name Constructors
    /// @{

    /// Construct an empty string ref.
    /*implicit*/ StringRef() : Data(0), Length(0) {}

    /// Construct a string ref from a cstring.
    /*implicit*/ StringRef(const char *Str)
      : Data(Str) {
        assert(Str && "StringRef cannot be built from a NULL argument");
        Length = ::strlen(Str); // invoking strlen(NULL) is undefined behavior
      }

    /// Construct a string ref from a pointer and length.
    /*implicit*/ StringRef(const char *data, size_t length)
      : Data(data), Length(length) {
        assert((data || length == 0) &&
        "StringRef cannot be built from a NULL argument with non-null length");
      }

    /// Construct a string ref from an std::string.
    /*implicit*/ StringRef(const std::string &Str)
      : Data(Str.data()), Length(Str.length()) {}

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

    /// front - Get the first character in the string.
    char front() const {
      assert(!empty());
      return Data[0];
    }

    /// back - Get the last character in the string.
    char back() const {
      assert(!empty());
      return Data[Length-1];
    }

    /// equals - Check for string equality, this is more efficient than
    /// compare() when the relative ordering of inequal strings isn't needed.
    bool equals(StringRef RHS) const {
      return (Length == RHS.Length &&
              compareMemory(Data, RHS.Data, RHS.Length) == 0);
    }

    /// equals_lower - Check for string equality, ignoring case.
    bool equals_lower(StringRef RHS) const {
      return Length == RHS.Length && compare_lower(RHS) == 0;
    }

    /// compare - Compare two strings; the result is -1, 0, or 1 if this string
    /// is lexicographically less than, equal to, or greater than the \arg RHS.
    int compare(StringRef RHS) const {
      // Check the prefix for a mismatch.
      if (int Res = compareMemory(Data, RHS.Data, min(Length, RHS.Length)))
        return Res < 0 ? -1 : 1;

      // Otherwise the prefixes match, so we only need to check the lengths.
      if (Length == RHS.Length)
        return 0;
      return Length < RHS.Length ? -1 : 1;
    }

    /// compare_lower - Compare two strings, ignoring case.
    int compare_lower(StringRef RHS) const;

    /// compare_numeric - Compare two strings, treating sequences of digits as
    /// numbers.
    int compare_numeric(StringRef RHS) const;

    /// \brief Determine the edit distance between this string and another
    /// string.
    ///
    /// \param Other the string to compare this string against.
    ///
    /// \param AllowReplacements whether to allow character
    /// replacements (change one character into another) as a single
    /// operation, rather than as two operations (an insertion and a
    /// removal).
    ///
    /// \param MaxEditDistance If non-zero, the maximum edit distance that
    /// this routine is allowed to compute. If the edit distance will exceed
    /// that maximum, returns \c MaxEditDistance+1.
    ///
    /// \returns the minimum number of character insertions, removals,
    /// or (if \p AllowReplacements is \c true) replacements needed to
    /// transform one of the given strings into the other. If zero,
    /// the strings are identical.
    unsigned edit_distance(StringRef Other, bool AllowReplacements = true,
                           unsigned MaxEditDistance = 0);

    /// str - Get the contents as an std::string.
    std::string str() const {
      if (Data == 0) return std::string();
      return std::string(Data, Length);
    }

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
    /// @name String Predicates
    /// @{

    /// startswith - Check if this string starts with the given \arg Prefix.
    bool startswith(StringRef Prefix) const {
      return Length >= Prefix.Length &&
             compareMemory(Data, Prefix.Data, Prefix.Length) == 0;
    }

    /// endswith - Check if this string ends with the given \arg Suffix.
    bool endswith(StringRef Suffix) const {
      return Length >= Suffix.Length &&
        compareMemory(end() - Suffix.Length, Suffix.Data, Suffix.Length) == 0;
    }

    /// @}
    /// @name String Searching
    /// @{

    /// find - Search for the first character \arg C in the string.
    ///
    /// \return - The index of the first occurrence of \arg C, or npos if not
    /// found.
    size_t find(char C, size_t From = 0) const {
      for (size_t i = min(From, Length), e = Length; i != e; ++i)
        if (Data[i] == C)
          return i;
      return npos;
    }

    /// find - Search for the first string \arg Str in the string.
    ///
    /// \return - The index of the first occurrence of \arg Str, or npos if not
    /// found.
    size_t find(StringRef Str, size_t From = 0) const;

    /// rfind - Search for the last character \arg C in the string.
    ///
    /// \return - The index of the last occurrence of \arg C, or npos if not
    /// found.
    size_t rfind(char C, size_t From = npos) const {
      From = min(From, Length);
      size_t i = From;
      while (i != 0) {
        --i;
        if (Data[i] == C)
          return i;
      }
      return npos;
    }

    /// rfind - Search for the last string \arg Str in the string.
    ///
    /// \return - The index of the last occurrence of \arg Str, or npos if not
    /// found.
    size_t rfind(StringRef Str) const;

    /// find_first_of - Find the first character in the string that is \arg C,
    /// or npos if not found. Same as find.
    size_type find_first_of(char C, size_t From = 0) const {
      return find(C, From);
    }

    /// find_first_of - Find the first character in the string that is in \arg
    /// Chars, or npos if not found.
    ///
    /// Note: O(size() + Chars.size())
    size_type find_first_of(StringRef Chars, size_t From = 0) const;

    /// find_first_not_of - Find the first character in the string that is not
    /// \arg C or npos if not found.
    size_type find_first_not_of(char C, size_t From = 0) const;

    /// find_first_not_of - Find the first character in the string that is not
    /// in the string \arg Chars, or npos if not found.
    ///
    /// Note: O(size() + Chars.size())
    size_type find_first_not_of(StringRef Chars, size_t From = 0) const;

    /// find_last_of - Find the last character in the string that is \arg C, or
    /// npos if not found.
    size_type find_last_of(char C, size_t From = npos) const {
      return rfind(C, From);
    }

    /// find_last_of - Find the last character in the string that is in \arg C,
    /// or npos if not found.
    ///
    /// Note: O(size() + Chars.size())
    size_type find_last_of(StringRef Chars, size_t From = npos) const;

    /// @}
    /// @name Helpful Algorithms
    /// @{

    /// count - Return the number of occurrences of \arg C in the string.
    size_t count(char C) const {
      size_t Count = 0;
      for (size_t i = 0, e = Length; i != e; ++i)
        if (Data[i] == C)
          ++Count;
      return Count;
    }

    /// count - Return the number of non-overlapped occurrences of \arg Str in
    /// the string.
    size_t count(StringRef Str) const;

    /// getAsInteger - Parse the current string as an integer of the specified
    /// radix.  If Radix is specified as zero, this does radix autosensing using
    /// extended C rules: 0 is octal, 0x is hex, 0b is binary.
    ///
    /// If the string is invalid or if only a subset of the string is valid,
    /// this returns true to signify the error.  The string is considered
    /// erroneous if empty.
    ///
    bool getAsInteger(unsigned Radix, long long &Result) const;
    bool getAsInteger(unsigned Radix, unsigned long long &Result) const;
    bool getAsInteger(unsigned Radix, int &Result) const;
    bool getAsInteger(unsigned Radix, unsigned &Result) const;

    // TODO: Provide overloads for int/unsigned that check for overflow.

    /// getAsInteger - Parse the current string as an integer of the
    /// specified radix, or of an autosensed radix if the radix given
    /// is 0.  The current value in Result is discarded, and the
    /// storage is changed to be wide enough to store the parsed
    /// integer.
    ///
    /// Returns true if the string does not solely consist of a valid
    /// non-empty number in the appropriate base.
    ///
    /// APInt::fromString is superficially similar but assumes the
    /// string is well-formed in the given radix.
    bool getAsInteger(unsigned Radix, APInt &Result) const;

    /// @}
    /// @name String Operations
    /// @{

    // lower - Convert the given ASCII string to lowercase.
    std::string lower() const;

    /// upper - Convert the given ASCII string to uppercase.
    std::string upper() const;

    /// @}
    /// @name Substring Operations
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
      Start = min(Start, Length);
      return StringRef(Data + Start, min(N, Length - Start));
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
      Start = min(Start, Length);
      End = min(max(Start, End), Length);
      return StringRef(Data + Start, End - Start);
    }

    /// split - Split into two substrings around the first occurrence of a
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
      size_t Idx = find(Separator);
      if (Idx == npos)
        return std::make_pair(*this, StringRef());
      return std::make_pair(slice(0, Idx), slice(Idx+1, npos));
    }

    /// split - Split into two substrings around the first occurrence of a
    /// separator string.
    ///
    /// If \arg Separator is in the string, then the result is a pair (LHS, RHS)
    /// such that (*this == LHS + Separator + RHS) is true and RHS is
    /// maximal. If \arg Separator is not in the string, then the result is a
    /// pair (LHS, RHS) where (*this == LHS) and (RHS == "").
    ///
    /// \param Separator - The string to split on.
    /// \return - The split substrings.
    std::pair<StringRef, StringRef> split(StringRef Separator) const {
      size_t Idx = find(Separator);
      if (Idx == npos)
        return std::make_pair(*this, StringRef());
      return std::make_pair(slice(0, Idx), slice(Idx + Separator.size(), npos));
    }

    /// split - Split into substrings around the occurrences of a separator
    /// string.
    ///
    /// Each substring is stored in \arg A. If \arg MaxSplit is >= 0, at most
    /// \arg MaxSplit splits are done and consequently <= \arg MaxSplit
    /// elements are added to A.
    /// If \arg KeepEmpty is false, empty strings are not added to \arg A. They
    /// still count when considering \arg MaxSplit
    /// An useful invariant is that
    /// Separator.join(A) == *this if MaxSplit == -1 and KeepEmpty == true
    ///
    /// \param A - Where to put the substrings.
    /// \param Separator - The string to split on.
    /// \param MaxSplit - The maximum number of times the string is split.
    /// \param KeepEmpty - True if empty substring should be added.
    void split(SmallVectorImpl<StringRef> &A,
               StringRef Separator, int MaxSplit = -1,
               bool KeepEmpty = true) const;

    /// rsplit - Split into two substrings around the last occurrence of a
    /// separator character.
    ///
    /// If \arg Separator is in the string, then the result is a pair (LHS, RHS)
    /// such that (*this == LHS + Separator + RHS) is true and RHS is
    /// minimal. If \arg Separator is not in the string, then the result is a
    /// pair (LHS, RHS) where (*this == LHS) and (RHS == "").
    ///
    /// \param Separator - The character to split on.
    /// \return - The split substrings.
    std::pair<StringRef, StringRef> rsplit(char Separator) const {
      size_t Idx = rfind(Separator);
      if (Idx == npos)
        return std::make_pair(*this, StringRef());
      return std::make_pair(slice(0, Idx), slice(Idx+1, npos));
    }

    /// @}
  };

  /// @name StringRef Comparison Operators
  /// @{

  inline bool operator==(StringRef LHS, StringRef RHS) {
    return LHS.equals(RHS);
  }

  inline bool operator!=(StringRef LHS, StringRef RHS) {
    return !(LHS == RHS);
  }

  inline bool operator<(StringRef LHS, StringRef RHS) {
    return LHS.compare(RHS) == -1;
  }

  inline bool operator<=(StringRef LHS, StringRef RHS) {
    return LHS.compare(RHS) != 1;
  }

  inline bool operator>(StringRef LHS, StringRef RHS) {
    return LHS.compare(RHS) == 1;
  }

  inline bool operator>=(StringRef LHS, StringRef RHS) {
    return LHS.compare(RHS) != -1;
  }

  inline std::string &operator+=(std::string &buffer, llvm::StringRef string) {
    return buffer.append(string.data(), string.size());
  }

  /// @}

  // StringRefs can be treated like a POD type.
  template <typename T> struct isPodLike;
  template <> struct isPodLike<StringRef> { static const bool value = true; };

}

#endif

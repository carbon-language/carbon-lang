//===-- StringRef.cpp - Lightweight String References ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/OwningPtr.h"
#include <bitset>

using namespace llvm;

// MSVC emits references to this into the translation units which reference it.
#ifndef _MSC_VER
const size_t StringRef::npos;
#endif

static char ascii_tolower(char x) {
  if (x >= 'A' && x <= 'Z')
    return x - 'A' + 'a';
  return x;
}

static char ascii_toupper(char x) {
  if (x >= 'a' && x <= 'z')
    return x - 'a' + 'A';
  return x;
}

static bool ascii_isdigit(char x) {
  return x >= '0' && x <= '9';
}

/// compare_lower - Compare strings, ignoring case.
int StringRef::compare_lower(StringRef RHS) const {
  for (size_t I = 0, E = min(Length, RHS.Length); I != E; ++I) {
    unsigned char LHC = ascii_tolower(Data[I]);
    unsigned char RHC = ascii_tolower(RHS.Data[I]);
    if (LHC != RHC)
      return LHC < RHC ? -1 : 1;
  }

  if (Length == RHS.Length)
    return 0;
  return Length < RHS.Length ? -1 : 1;
}

/// compare_numeric - Compare strings, handle embedded numbers.
int StringRef::compare_numeric(StringRef RHS) const {
  for (size_t I = 0, E = min(Length, RHS.Length); I != E; ++I) {
    // Check for sequences of digits.
    if (ascii_isdigit(Data[I]) && ascii_isdigit(RHS.Data[I])) {
      // The longer sequence of numbers is considered larger.
      // This doesn't really handle prefixed zeros well.
      size_t J;
      for (J = I + 1; J != E + 1; ++J) {
        bool ld = J < Length && ascii_isdigit(Data[J]);
        bool rd = J < RHS.Length && ascii_isdigit(RHS.Data[J]);
        if (ld != rd)
          return rd ? -1 : 1;
        if (!rd)
          break;
      }
      // The two number sequences have the same length (J-I), just memcmp them.
      if (int Res = compareMemory(Data + I, RHS.Data + I, J - I))
        return Res < 0 ? -1 : 1;
      // Identical number sequences, continue search after the numbers.
      I = J - 1;
      continue;
    }
    if (Data[I] != RHS.Data[I])
      return (unsigned char)Data[I] < (unsigned char)RHS.Data[I] ? -1 : 1;
  }
  if (Length == RHS.Length)
    return 0;
  return Length < RHS.Length ? -1 : 1;
}

// Compute the edit distance between the two given strings.
unsigned StringRef::edit_distance(llvm::StringRef Other,
                                  bool AllowReplacements,
                                  unsigned MaxEditDistance) {
  // The algorithm implemented below is the "classic"
  // dynamic-programming algorithm for computing the Levenshtein
  // distance, which is described here:
  //
  //   http://en.wikipedia.org/wiki/Levenshtein_distance
  //
  // Although the algorithm is typically described using an m x n
  // array, only two rows are used at a time, so this implemenation
  // just keeps two separate vectors for those two rows.
  size_type m = size();
  size_type n = Other.size();

  const unsigned SmallBufferSize = 64;
  unsigned SmallBuffer[SmallBufferSize];
  llvm::OwningArrayPtr<unsigned> Allocated;
  unsigned *previous = SmallBuffer;
  if (2*(n + 1) > SmallBufferSize) {
    previous = new unsigned [2*(n+1)];
    Allocated.reset(previous);
  }
  unsigned *current = previous + (n + 1);

  for (unsigned i = 0; i <= n; ++i)
    previous[i] = i;

  for (size_type y = 1; y <= m; ++y) {
    current[0] = y;
    unsigned BestThisRow = current[0];

    for (size_type x = 1; x <= n; ++x) {
      if (AllowReplacements) {
        current[x] = min(previous[x-1] + ((*this)[y-1] == Other[x-1]? 0u:1u),
                         min(current[x-1], previous[x])+1);
      }
      else {
        if ((*this)[y-1] == Other[x-1]) current[x] = previous[x-1];
        else current[x] = min(current[x-1], previous[x]) + 1;
      }
      BestThisRow = min(BestThisRow, current[x]);
    }

    if (MaxEditDistance && BestThisRow > MaxEditDistance)
      return MaxEditDistance + 1;

    unsigned *tmp = current;
    current = previous;
    previous = tmp;
  }

  unsigned Result = previous[n];
  return Result;
}

//===----------------------------------------------------------------------===//
// String Operations
//===----------------------------------------------------------------------===//

std::string StringRef::lower() const {
  std::string Result(size(), char());
  for (size_type i = 0, e = size(); i != e; ++i) {
    Result[i] = ascii_tolower(Data[i]);
  }
  return Result;
}

std::string StringRef::upper() const {
  std::string Result(size(), char());
  for (size_type i = 0, e = size(); i != e; ++i) {
    Result[i] = ascii_tolower(Data[i]);
  }
  return Result;
}

//===----------------------------------------------------------------------===//
// String Searching
//===----------------------------------------------------------------------===//


/// find - Search for the first string \arg Str in the string.
///
/// \return - The index of the first occurrence of \arg Str, or npos if not
/// found.
size_t StringRef::find(StringRef Str, size_t From) const {
  size_t N = Str.size();
  if (N > Length)
    return npos;

  // For short haystacks or unsupported needles fall back to the naive algorithm
  if (Length < 16 || N > 255 || N == 0) {
    for (size_t e = Length - N + 1, i = min(From, e); i != e; ++i)
      if (substr(i, N).equals(Str))
        return i;
    return npos;
  }

  if (From >= Length)
    return npos;

  // Build the bad char heuristic table, with uint8_t to reduce cache thrashing.
  uint8_t BadCharSkip[256];
  std::memset(BadCharSkip, N, 256);
  for (unsigned i = 0; i != N-1; ++i)
    BadCharSkip[(uint8_t)Str[i]] = N-1-i;

  unsigned Len = Length-From, Pos = From;
  while (Len >= N) {
    if (substr(Pos, N).equals(Str)) // See if this is the correct substring.
      return Pos;

    // Otherwise skip the appropriate number of bytes.
    uint8_t Skip = BadCharSkip[(uint8_t)(*this)[Pos+N-1]];
    Len -= Skip;
    Pos += Skip;
  }

  return npos;
}

/// rfind - Search for the last string \arg Str in the string.
///
/// \return - The index of the last occurrence of \arg Str, or npos if not
/// found.
size_t StringRef::rfind(StringRef Str) const {
  size_t N = Str.size();
  if (N > Length)
    return npos;
  for (size_t i = Length - N + 1, e = 0; i != e;) {
    --i;
    if (substr(i, N).equals(Str))
      return i;
  }
  return npos;
}

/// find_first_of - Find the first character in the string that is in \arg
/// Chars, or npos if not found.
///
/// Note: O(size() + Chars.size())
StringRef::size_type StringRef::find_first_of(StringRef Chars,
                                              size_t From) const {
  std::bitset<1 << CHAR_BIT> CharBits;
  for (size_type i = 0; i != Chars.size(); ++i)
    CharBits.set((unsigned char)Chars[i]);

  for (size_type i = min(From, Length), e = Length; i != e; ++i)
    if (CharBits.test((unsigned char)Data[i]))
      return i;
  return npos;
}

/// find_first_not_of - Find the first character in the string that is not
/// \arg C or npos if not found.
StringRef::size_type StringRef::find_first_not_of(char C, size_t From) const {
  for (size_type i = min(From, Length), e = Length; i != e; ++i)
    if (Data[i] != C)
      return i;
  return npos;
}

/// find_first_not_of - Find the first character in the string that is not
/// in the string \arg Chars, or npos if not found.
///
/// Note: O(size() + Chars.size())
StringRef::size_type StringRef::find_first_not_of(StringRef Chars,
                                                  size_t From) const {
  std::bitset<1 << CHAR_BIT> CharBits;
  for (size_type i = 0; i != Chars.size(); ++i)
    CharBits.set((unsigned char)Chars[i]);

  for (size_type i = min(From, Length), e = Length; i != e; ++i)
    if (!CharBits.test((unsigned char)Data[i]))
      return i;
  return npos;
}

/// find_last_of - Find the last character in the string that is in \arg C,
/// or npos if not found.
///
/// Note: O(size() + Chars.size())
StringRef::size_type StringRef::find_last_of(StringRef Chars,
                                             size_t From) const {
  std::bitset<1 << CHAR_BIT> CharBits;
  for (size_type i = 0; i != Chars.size(); ++i)
    CharBits.set((unsigned char)Chars[i]);

  for (size_type i = min(From, Length) - 1, e = -1; i != e; --i)
    if (CharBits.test((unsigned char)Data[i]))
      return i;
  return npos;
}

//===----------------------------------------------------------------------===//
// Helpful Algorithms
//===----------------------------------------------------------------------===//

/// count - Return the number of non-overlapped occurrences of \arg Str in
/// the string.
size_t StringRef::count(StringRef Str) const {
  size_t Count = 0;
  size_t N = Str.size();
  if (N > Length)
    return 0;
  for (size_t i = 0, e = Length - N + 1; i != e; ++i)
    if (substr(i, N).equals(Str))
      ++Count;
  return Count;
}

static unsigned GetAutoSenseRadix(StringRef &Str) {
  if (Str.startswith("0x")) {
    Str = Str.substr(2);
    return 16;
  } else if (Str.startswith("0b")) {
    Str = Str.substr(2);
    return 2;
  } else if (Str.startswith("0")) {
    return 8;
  } else {
    return 10;
  }
}


/// GetAsUnsignedInteger - Workhorse method that converts a integer character
/// sequence of radix up to 36 to an unsigned long long value.
static bool GetAsUnsignedInteger(StringRef Str, unsigned Radix,
                                 unsigned long long &Result) {
  // Autosense radix if not specified.
  if (Radix == 0)
    Radix = GetAutoSenseRadix(Str);

  // Empty strings (after the radix autosense) are invalid.
  if (Str.empty()) return true;

  // Parse all the bytes of the string given this radix.  Watch for overflow.
  Result = 0;
  while (!Str.empty()) {
    unsigned CharVal;
    if (Str[0] >= '0' && Str[0] <= '9')
      CharVal = Str[0]-'0';
    else if (Str[0] >= 'a' && Str[0] <= 'z')
      CharVal = Str[0]-'a'+10;
    else if (Str[0] >= 'A' && Str[0] <= 'Z')
      CharVal = Str[0]-'A'+10;
    else
      return true;

    // If the parsed value is larger than the integer radix, the string is
    // invalid.
    if (CharVal >= Radix)
      return true;

    // Add in this character.
    unsigned long long PrevResult = Result;
    Result = Result*Radix+CharVal;

    // Check for overflow.
    if (Result < PrevResult)
      return true;

    Str = Str.substr(1);
  }

  return false;
}

bool StringRef::getAsInteger(unsigned Radix, unsigned long long &Result) const {
  return GetAsUnsignedInteger(*this, Radix, Result);
}


bool StringRef::getAsInteger(unsigned Radix, long long &Result) const {
  unsigned long long ULLVal;

  // Handle positive strings first.
  if (empty() || front() != '-') {
    if (GetAsUnsignedInteger(*this, Radix, ULLVal) ||
        // Check for value so large it overflows a signed value.
        (long long)ULLVal < 0)
      return true;
    Result = ULLVal;
    return false;
  }

  // Get the positive part of the value.
  if (GetAsUnsignedInteger(substr(1), Radix, ULLVal) ||
      // Reject values so large they'd overflow as negative signed, but allow
      // "-0".  This negates the unsigned so that the negative isn't undefined
      // on signed overflow.
      (long long)-ULLVal > 0)
    return true;

  Result = -ULLVal;
  return false;
}

bool StringRef::getAsInteger(unsigned Radix, int &Result) const {
  long long Val;
  if (getAsInteger(Radix, Val) ||
      (int)Val != Val)
    return true;
  Result = Val;
  return false;
}

bool StringRef::getAsInteger(unsigned Radix, unsigned &Result) const {
  unsigned long long Val;
  if (getAsInteger(Radix, Val) ||
      (unsigned)Val != Val)
    return true;
  Result = Val;
  return false;
}

bool StringRef::getAsInteger(unsigned Radix, APInt &Result) const {
  StringRef Str = *this;

  // Autosense radix if not specified.
  if (Radix == 0)
    Radix = GetAutoSenseRadix(Str);

  assert(Radix > 1 && Radix <= 36);

  // Empty strings (after the radix autosense) are invalid.
  if (Str.empty()) return true;

  // Skip leading zeroes.  This can be a significant improvement if
  // it means we don't need > 64 bits.
  while (!Str.empty() && Str.front() == '0')
    Str = Str.substr(1);

  // If it was nothing but zeroes....
  if (Str.empty()) {
    Result = APInt(64, 0);
    return false;
  }

  // (Over-)estimate the required number of bits.
  unsigned Log2Radix = 0;
  while ((1U << Log2Radix) < Radix) Log2Radix++;
  bool IsPowerOf2Radix = ((1U << Log2Radix) == Radix);

  unsigned BitWidth = Log2Radix * Str.size();
  if (BitWidth < Result.getBitWidth())
    BitWidth = Result.getBitWidth(); // don't shrink the result
  else
    Result = Result.zext(BitWidth);

  APInt RadixAP, CharAP; // unused unless !IsPowerOf2Radix
  if (!IsPowerOf2Radix) {
    // These must have the same bit-width as Result.
    RadixAP = APInt(BitWidth, Radix);
    CharAP = APInt(BitWidth, 0);
  }

  // Parse all the bytes of the string given this radix.
  Result = 0;
  while (!Str.empty()) {
    unsigned CharVal;
    if (Str[0] >= '0' && Str[0] <= '9')
      CharVal = Str[0]-'0';
    else if (Str[0] >= 'a' && Str[0] <= 'z')
      CharVal = Str[0]-'a'+10;
    else if (Str[0] >= 'A' && Str[0] <= 'Z')
      CharVal = Str[0]-'A'+10;
    else
      return true;

    // If the parsed value is larger than the integer radix, the string is
    // invalid.
    if (CharVal >= Radix)
      return true;

    // Add in this character.
    if (IsPowerOf2Radix) {
      Result <<= Log2Radix;
      Result |= CharVal;
    } else {
      Result *= RadixAP;
      CharAP = CharVal;
      Result += CharAP;
    }

    Str = Str.substr(1);
  }

  return false;
}

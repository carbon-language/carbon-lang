//===--- LiteralSupport.cpp - Code to parse and process literals ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the NumericLiteralParser, CharLiteralParser, and
// StringLiteralParser interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;

/// HexDigitValue - Return the value of the specified hex digit, or -1 if it's
/// not valid.
static int HexDigitValue(char C) {
  if (C >= '0' && C <= '9') return C-'0';
  if (C >= 'a' && C <= 'f') return C-'a'+10;
  if (C >= 'A' && C <= 'F') return C-'A'+10;
  return -1;
}

/// ProcessCharEscape - Parse a standard C escape sequence, which can occur in
/// either a character or a string literal.
static unsigned ProcessCharEscape(const char *&ThisTokBuf,
                                  const char *ThisTokEnd, bool &HadError,
                                  SourceLocation Loc, bool IsWide,
                                  Preprocessor &PP) {
  // Skip the '\' char.
  ++ThisTokBuf;

  // We know that this character can't be off the end of the buffer, because
  // that would have been \", which would not have been the end of string.
  unsigned ResultChar = *ThisTokBuf++;
  switch (ResultChar) {
  // These map to themselves.
  case '\\': case '\'': case '"': case '?': break;

    // These have fixed mappings.
  case 'a':
    // TODO: K&R: the meaning of '\\a' is different in traditional C
    ResultChar = 7;
    break;
  case 'b':
    ResultChar = 8;
    break;
  case 'e':
    PP.Diag(Loc, diag::ext_nonstandard_escape) << "e";
    ResultChar = 27;
    break;
  case 'E':
    PP.Diag(Loc, diag::ext_nonstandard_escape) << "E";
    ResultChar = 27;
    break;
  case 'f':
    ResultChar = 12;
    break;
  case 'n':
    ResultChar = 10;
    break;
  case 'r':
    ResultChar = 13;
    break;
  case 't':
    ResultChar = 9;
    break;
  case 'v':
    ResultChar = 11;
    break;
  case 'x': { // Hex escape.
    ResultChar = 0;
    if (ThisTokBuf == ThisTokEnd || !isxdigit(*ThisTokBuf)) {
      PP.Diag(Loc, diag::err_hex_escape_no_digits);
      HadError = 1;
      break;
    }

    // Hex escapes are a maximal series of hex digits.
    bool Overflow = false;
    for (; ThisTokBuf != ThisTokEnd; ++ThisTokBuf) {
      int CharVal = HexDigitValue(ThisTokBuf[0]);
      if (CharVal == -1) break;
      // About to shift out a digit?
      Overflow |= (ResultChar & 0xF0000000) ? true : false;
      ResultChar <<= 4;
      ResultChar |= CharVal;
    }

    // See if any bits will be truncated when evaluated as a character.
    unsigned CharWidth = IsWide
                       ? PP.getTargetInfo().getWCharWidth()
                       : PP.getTargetInfo().getCharWidth();

    if (CharWidth != 32 && (ResultChar >> CharWidth) != 0) {
      Overflow = true;
      ResultChar &= ~0U >> (32-CharWidth);
    }

    // Check for overflow.
    if (Overflow)   // Too many digits to fit in
      PP.Diag(Loc, diag::warn_hex_escape_too_large);
    break;
  }
  case '0': case '1': case '2': case '3':
  case '4': case '5': case '6': case '7': {
    // Octal escapes.
    --ThisTokBuf;
    ResultChar = 0;

    // Octal escapes are a series of octal digits with maximum length 3.
    // "\0123" is a two digit sequence equal to "\012" "3".
    unsigned NumDigits = 0;
    do {
      ResultChar <<= 3;
      ResultChar |= *ThisTokBuf++ - '0';
      ++NumDigits;
    } while (ThisTokBuf != ThisTokEnd && NumDigits < 3 &&
             ThisTokBuf[0] >= '0' && ThisTokBuf[0] <= '7');

    // Check for overflow.  Reject '\777', but not L'\777'.
    unsigned CharWidth = IsWide
                       ? PP.getTargetInfo().getWCharWidth()
                       : PP.getTargetInfo().getCharWidth();

    if (CharWidth != 32 && (ResultChar >> CharWidth) != 0) {
      PP.Diag(Loc, diag::warn_octal_escape_too_large);
      ResultChar &= ~0U >> (32-CharWidth);
    }
    break;
  }

    // Otherwise, these are not valid escapes.
  case '(': case '{': case '[': case '%':
    // GCC accepts these as extensions.  We warn about them as such though.
    PP.Diag(Loc, diag::ext_nonstandard_escape)
      << std::string()+(char)ResultChar;
    break;
  default:
    if (isgraph(ThisTokBuf[0]))
      PP.Diag(Loc, diag::ext_unknown_escape) << std::string()+(char)ResultChar;
    else
      PP.Diag(Loc, diag::ext_unknown_escape) << "x"+llvm::utohexstr(ResultChar);
    break;
  }

  return ResultChar;
}

/// ProcessUCNEscape - Read the Universal Character Name, check constraints and
/// convert the UTF32 to UTF8. This is a subroutine of StringLiteralParser.
/// When we decide to implement UCN's for character constants and identifiers,
/// we will likely rework our support for UCN's.
static void ProcessUCNEscape(const char *&ThisTokBuf, const char *ThisTokEnd,
                             char *&ResultBuf, bool &HadError,
                             SourceLocation Loc, bool IsWide, Preprocessor &PP)
{
  // FIXME: Add a warning - UCN's are only valid in C++ & C99.
  // FIXME: Handle wide strings.

  // Save the beginning of the string (for error diagnostics).
  const char *ThisTokBegin = ThisTokBuf;

  // Skip the '\u' char's.
  ThisTokBuf += 2;

  if (ThisTokBuf == ThisTokEnd || !isxdigit(*ThisTokBuf)) {
    PP.Diag(Loc, diag::err_ucn_escape_no_digits);
    HadError = 1;
    return;
  }
  typedef uint32_t UTF32;

  UTF32 UcnVal = 0;
  unsigned short UcnLen = (ThisTokBuf[-1] == 'u' ? 4 : 8);
  for (; ThisTokBuf != ThisTokEnd && UcnLen; ++ThisTokBuf, UcnLen--) {
    int CharVal = HexDigitValue(ThisTokBuf[0]);
    if (CharVal == -1) break;
    UcnVal <<= 4;
    UcnVal |= CharVal;
  }
  // If we didn't consume the proper number of digits, there is a problem.
  if (UcnLen) {
    PP.Diag(PP.AdvanceToTokenCharacter(Loc, ThisTokBuf-ThisTokBegin),
            diag::err_ucn_escape_incomplete);
    HadError = 1;
    return;
  }
  // Check UCN constraints (C99 6.4.3p2).
  if ((UcnVal < 0xa0 &&
      (UcnVal != 0x24 && UcnVal != 0x40 && UcnVal != 0x60 )) // $, @, `
      || (UcnVal >= 0xD800 && UcnVal <= 0xDFFF)
      || (UcnVal > 0x10FFFF)) /* the maximum legal UTF32 value */ {
    PP.Diag(Loc, diag::err_ucn_escape_invalid);
    HadError = 1;
    return;
  }
  // Now that we've parsed/checked the UCN, we convert from UTF32->UTF8.
  // The conversion below was inspired by:
  //   http://www.unicode.org/Public/PROGRAMS/CVTUTF/ConvertUTF.c
  // First, we determine how many bytes the result will require.
  typedef uint8_t UTF8;

  unsigned short bytesToWrite = 0;
  if (UcnVal < (UTF32)0x80)
    bytesToWrite = 1;
  else if (UcnVal < (UTF32)0x800)
    bytesToWrite = 2;
  else if (UcnVal < (UTF32)0x10000)
    bytesToWrite = 3;
  else
    bytesToWrite = 4;

  const unsigned byteMask = 0xBF;
  const unsigned byteMark = 0x80;

  // Once the bits are split out into bytes of UTF8, this is a mask OR-ed
  // into the first byte, depending on how many bytes follow.
  static const UTF8 firstByteMark[5] = {
    0x00, 0x00, 0xC0, 0xE0, 0xF0
  };
  // Finally, we write the bytes into ResultBuf.
  ResultBuf += bytesToWrite;
  switch (bytesToWrite) { // note: everything falls through.
    case 4: *--ResultBuf = (UTF8)((UcnVal | byteMark) & byteMask); UcnVal >>= 6;
    case 3: *--ResultBuf = (UTF8)((UcnVal | byteMark) & byteMask); UcnVal >>= 6;
    case 2: *--ResultBuf = (UTF8)((UcnVal | byteMark) & byteMask); UcnVal >>= 6;
    case 1: *--ResultBuf = (UTF8) (UcnVal | firstByteMark[bytesToWrite]);
  }
  // Update the buffer.
  ResultBuf += bytesToWrite;
}


///       integer-constant: [C99 6.4.4.1]
///         decimal-constant integer-suffix
///         octal-constant integer-suffix
///         hexadecimal-constant integer-suffix
///       decimal-constant:
///         nonzero-digit
///         decimal-constant digit
///       octal-constant:
///         0
///         octal-constant octal-digit
///       hexadecimal-constant:
///         hexadecimal-prefix hexadecimal-digit
///         hexadecimal-constant hexadecimal-digit
///       hexadecimal-prefix: one of
///         0x 0X
///       integer-suffix:
///         unsigned-suffix [long-suffix]
///         unsigned-suffix [long-long-suffix]
///         long-suffix [unsigned-suffix]
///         long-long-suffix [unsigned-sufix]
///       nonzero-digit:
///         1 2 3 4 5 6 7 8 9
///       octal-digit:
///         0 1 2 3 4 5 6 7
///       hexadecimal-digit:
///         0 1 2 3 4 5 6 7 8 9
///         a b c d e f
///         A B C D E F
///       unsigned-suffix: one of
///         u U
///       long-suffix: one of
///         l L
///       long-long-suffix: one of
///         ll LL
///
///       floating-constant: [C99 6.4.4.2]
///         TODO: add rules...
///
NumericLiteralParser::
NumericLiteralParser(const char *begin, const char *end,
                     SourceLocation TokLoc, Preprocessor &pp)
  : PP(pp), ThisTokBegin(begin), ThisTokEnd(end) {

  // This routine assumes that the range begin/end matches the regex for integer
  // and FP constants (specifically, the 'pp-number' regex), and assumes that
  // the byte at "*end" is both valid and not part of the regex.  Because of
  // this, it doesn't have to check for 'overscan' in various places.
  assert(!isalnum(*end) && *end != '.' && *end != '_' &&
         "Lexer didn't maximally munch?");

  s = DigitsBegin = begin;
  saw_exponent = false;
  saw_period = false;
  isLong = false;
  isUnsigned = false;
  isLongLong = false;
  isFloat = false;
  isImaginary = false;
  isMicrosoftInteger = false;
  hadError = false;

  if (*s == '0') { // parse radix
    ParseNumberStartingWithZero(TokLoc);
    if (hadError)
      return;
  } else { // the first digit is non-zero
    radix = 10;
    s = SkipDigits(s);
    if (s == ThisTokEnd) {
      // Done.
    } else if (isxdigit(*s) && !(*s == 'e' || *s == 'E')) {
      PP.Diag(PP.AdvanceToTokenCharacter(TokLoc, s-begin),
              diag::err_invalid_decimal_digit) << std::string(s, s+1);
      hadError = true;
      return;
    } else if (*s == '.') {
      s++;
      saw_period = true;
      s = SkipDigits(s);
    }
    if ((*s == 'e' || *s == 'E')) { // exponent
      const char *Exponent = s;
      s++;
      saw_exponent = true;
      if (*s == '+' || *s == '-')  s++; // sign
      const char *first_non_digit = SkipDigits(s);
      if (first_non_digit != s) {
        s = first_non_digit;
      } else {
        PP.Diag(PP.AdvanceToTokenCharacter(TokLoc, Exponent-begin),
                diag::err_exponent_has_no_digits);
        hadError = true;
        return;
      }
    }
  }

  SuffixBegin = s;

  // Parse the suffix.  At this point we can classify whether we have an FP or
  // integer constant.
  bool isFPConstant = isFloatingLiteral();

  // Loop over all of the characters of the suffix.  If we see something bad,
  // we break out of the loop.
  for (; s != ThisTokEnd; ++s) {
    switch (*s) {
    case 'f':      // FP Suffix for "float"
    case 'F':
      if (!isFPConstant) break;  // Error for integer constant.
      if (isFloat || isLong) break; // FF, LF invalid.
      isFloat = true;
      continue;  // Success.
    case 'u':
    case 'U':
      if (isFPConstant) break;  // Error for floating constant.
      if (isUnsigned) break;    // Cannot be repeated.
      isUnsigned = true;
      continue;  // Success.
    case 'l':
    case 'L':
      if (isLong || isLongLong) break;  // Cannot be repeated.
      if (isFloat) break;               // LF invalid.

      // Check for long long.  The L's need to be adjacent and the same case.
      if (s+1 != ThisTokEnd && s[1] == s[0]) {
        if (isFPConstant) break;        // long long invalid for floats.
        isLongLong = true;
        ++s;  // Eat both of them.
      } else {
        isLong = true;
      }
      continue;  // Success.
    case 'i':
      if (PP.getLangOptions().Microsoft) {
        if (isFPConstant || isUnsigned || isLong || isLongLong) break;

        // Allow i8, i16, i32, i64, and i128.
        if (s + 1 != ThisTokEnd) {
          switch (s[1]) {
            case '8':
              s += 2; // i8 suffix
              isMicrosoftInteger = true;
              break;
            case '1':
              if (s + 2 == ThisTokEnd) break;
              if (s[2] == '6') s += 3; // i16 suffix
              else if (s[2] == '2') {
                if (s + 3 == ThisTokEnd) break;
                if (s[3] == '8') s += 4; // i128 suffix
              }
              isMicrosoftInteger = true;
              break;
            case '3':
              if (s + 2 == ThisTokEnd) break;
              if (s[2] == '2') s += 3; // i32 suffix
              isMicrosoftInteger = true;
              break;
            case '6':
              if (s + 2 == ThisTokEnd) break;
              if (s[2] == '4') s += 3; // i64 suffix
              isMicrosoftInteger = true;
              break;
            default:
              break;
          }
          break;
        }
      }
      // fall through.
    case 'I':
    case 'j':
    case 'J':
      if (isImaginary) break;   // Cannot be repeated.
      PP.Diag(PP.AdvanceToTokenCharacter(TokLoc, s-begin),
              diag::ext_imaginary_constant);
      isImaginary = true;
      continue;  // Success.
    }
    // If we reached here, there was an error.
    break;
  }

  // Report an error if there are any.
  if (s != ThisTokEnd) {
    PP.Diag(PP.AdvanceToTokenCharacter(TokLoc, s-begin),
            isFPConstant ? diag::err_invalid_suffix_float_constant :
                           diag::err_invalid_suffix_integer_constant)
      << std::string(SuffixBegin, ThisTokEnd);
    hadError = true;
    return;
  }
}

/// ParseNumberStartingWithZero - This method is called when the first character
/// of the number is found to be a zero.  This means it is either an octal
/// number (like '04') or a hex number ('0x123a') a binary number ('0b1010') or
/// a floating point number (01239.123e4).  Eat the prefix, determining the
/// radix etc.
void NumericLiteralParser::ParseNumberStartingWithZero(SourceLocation TokLoc) {
  assert(s[0] == '0' && "Invalid method call");
  s++;

  // Handle a hex number like 0x1234.
  if ((*s == 'x' || *s == 'X') && (isxdigit(s[1]) || s[1] == '.')) {
    s++;
    radix = 16;
    DigitsBegin = s;
    s = SkipHexDigits(s);
    if (s == ThisTokEnd) {
      // Done.
    } else if (*s == '.') {
      s++;
      saw_period = true;
      s = SkipHexDigits(s);
    }
    // A binary exponent can appear with or with a '.'. If dotted, the
    // binary exponent is required.
    if (*s == 'p' || *s == 'P') {
      const char *Exponent = s;
      s++;
      saw_exponent = true;
      if (*s == '+' || *s == '-')  s++; // sign
      const char *first_non_digit = SkipDigits(s);
      if (first_non_digit == s) {
        PP.Diag(PP.AdvanceToTokenCharacter(TokLoc, Exponent-ThisTokBegin),
                diag::err_exponent_has_no_digits);
        hadError = true;
        return;
      }
      s = first_non_digit;

      if (!PP.getLangOptions().HexFloats)
        PP.Diag(TokLoc, diag::ext_hexconstant_invalid);
    } else if (saw_period) {
      PP.Diag(PP.AdvanceToTokenCharacter(TokLoc, s-ThisTokBegin),
              diag::err_hexconstant_requires_exponent);
      hadError = true;
    }
    return;
  }

  // Handle simple binary numbers 0b01010
  if (*s == 'b' || *s == 'B') {
    // 0b101010 is a GCC extension.
    PP.Diag(TokLoc, diag::ext_binary_literal);
    ++s;
    radix = 2;
    DigitsBegin = s;
    s = SkipBinaryDigits(s);
    if (s == ThisTokEnd) {
      // Done.
    } else if (isxdigit(*s)) {
      PP.Diag(PP.AdvanceToTokenCharacter(TokLoc, s-ThisTokBegin),
              diag::err_invalid_binary_digit) << std::string(s, s+1);
      hadError = true;
    }
    // Other suffixes will be diagnosed by the caller.
    return;
  }

  // For now, the radix is set to 8. If we discover that we have a
  // floating point constant, the radix will change to 10. Octal floating
  // point constants are not permitted (only decimal and hexadecimal).
  radix = 8;
  DigitsBegin = s;
  s = SkipOctalDigits(s);
  if (s == ThisTokEnd)
    return; // Done, simple octal number like 01234

  // If we have some other non-octal digit that *is* a decimal digit, see if
  // this is part of a floating point number like 094.123 or 09e1.
  if (isdigit(*s)) {
    const char *EndDecimal = SkipDigits(s);
    if (EndDecimal[0] == '.' || EndDecimal[0] == 'e' || EndDecimal[0] == 'E') {
      s = EndDecimal;
      radix = 10;
    }
  }

  // If we have a hex digit other than 'e' (which denotes a FP exponent) then
  // the code is using an incorrect base.
  if (isxdigit(*s) && *s != 'e' && *s != 'E') {
    PP.Diag(PP.AdvanceToTokenCharacter(TokLoc, s-ThisTokBegin),
            diag::err_invalid_octal_digit) << std::string(s, s+1);
    hadError = true;
    return;
  }

  if (*s == '.') {
    s++;
    radix = 10;
    saw_period = true;
    s = SkipDigits(s); // Skip suffix.
  }
  if (*s == 'e' || *s == 'E') { // exponent
    const char *Exponent = s;
    s++;
    radix = 10;
    saw_exponent = true;
    if (*s == '+' || *s == '-')  s++; // sign
    const char *first_non_digit = SkipDigits(s);
    if (first_non_digit != s) {
      s = first_non_digit;
    } else {
      PP.Diag(PP.AdvanceToTokenCharacter(TokLoc, Exponent-ThisTokBegin),
              diag::err_exponent_has_no_digits);
      hadError = true;
      return;
    }
  }
}


/// GetIntegerValue - Convert this numeric literal value to an APInt that
/// matches Val's input width.  If there is an overflow, set Val to the low bits
/// of the result and return true.  Otherwise, return false.
bool NumericLiteralParser::GetIntegerValue(llvm::APInt &Val) {
  // Fast path: Compute a conservative bound on the maximum number of
  // bits per digit in this radix. If we can't possibly overflow a
  // uint64 based on that bound then do the simple conversion to
  // integer. This avoids the expensive overflow checking below, and
  // handles the common cases that matter (small decimal integers and
  // hex/octal values which don't overflow).
  unsigned MaxBitsPerDigit = 1;
  while ((1U << MaxBitsPerDigit) < radix)
    MaxBitsPerDigit += 1;
  if ((SuffixBegin - DigitsBegin) * MaxBitsPerDigit <= 64) {
    uint64_t N = 0;
    for (s = DigitsBegin; s != SuffixBegin; ++s)
      N = N*radix + HexDigitValue(*s);

    // This will truncate the value to Val's input width. Simply check
    // for overflow by comparing.
    Val = N;
    return Val.getZExtValue() != N;
  }

  Val = 0;
  s = DigitsBegin;

  llvm::APInt RadixVal(Val.getBitWidth(), radix);
  llvm::APInt CharVal(Val.getBitWidth(), 0);
  llvm::APInt OldVal = Val;

  bool OverflowOccurred = false;
  while (s < SuffixBegin) {
    unsigned C = HexDigitValue(*s++);

    // If this letter is out of bound for this radix, reject it.
    assert(C < radix && "NumericLiteralParser ctor should have rejected this");

    CharVal = C;

    // Add the digit to the value in the appropriate radix.  If adding in digits
    // made the value smaller, then this overflowed.
    OldVal = Val;

    // Multiply by radix, did overflow occur on the multiply?
    Val *= RadixVal;
    OverflowOccurred |= Val.udiv(RadixVal) != OldVal;

    // Add value, did overflow occur on the value?
    //   (a + b) ult b  <=> overflow
    Val += CharVal;
    OverflowOccurred |= Val.ult(CharVal);
  }
  return OverflowOccurred;
}

llvm::APFloat NumericLiteralParser::
GetFloatValue(const llvm::fltSemantics &Format, bool* isExact) {
  using llvm::APFloat;
  using llvm::StringRef;

  unsigned n = std::min(SuffixBegin - ThisTokBegin, ThisTokEnd - ThisTokBegin);

  APFloat V (Format, APFloat::fcZero, false);
  APFloat::opStatus status;

  status = V.convertFromString(StringRef(ThisTokBegin, n),
                               APFloat::rmNearestTiesToEven);

  if (isExact)
    *isExact = status == APFloat::opOK;

  return V;
}


CharLiteralParser::CharLiteralParser(const char *begin, const char *end,
                                     SourceLocation Loc, Preprocessor &PP) {
  // At this point we know that the character matches the regex "L?'.*'".
  HadError = false;

  // Determine if this is a wide character.
  IsWide = begin[0] == 'L';
  if (IsWide) ++begin;

  // Skip over the entry quote.
  assert(begin[0] == '\'' && "Invalid token lexed");
  ++begin;

  // FIXME: The "Value" is an uint64_t so we can handle char literals of
  // upto 64-bits.
  // FIXME: This extensively assumes that 'char' is 8-bits.
  assert(PP.getTargetInfo().getCharWidth() == 8 &&
         "Assumes char is 8 bits");
  assert(PP.getTargetInfo().getIntWidth() <= 64 &&
         (PP.getTargetInfo().getIntWidth() & 7) == 0 &&
         "Assumes sizeof(int) on target is <= 64 and a multiple of char");
  assert(PP.getTargetInfo().getWCharWidth() <= 64 &&
         "Assumes sizeof(wchar) on target is <= 64");

  // This is what we will use for overflow detection
  llvm::APInt LitVal(PP.getTargetInfo().getIntWidth(), 0);

  unsigned NumCharsSoFar = 0;
  while (begin[0] != '\'') {
    uint64_t ResultChar;
    if (begin[0] != '\\')     // If this is a normal character, consume it.
      ResultChar = *begin++;
    else                      // Otherwise, this is an escape character.
      ResultChar = ProcessCharEscape(begin, end, HadError, Loc, IsWide, PP);

    // If this is a multi-character constant (e.g. 'abc'), handle it.  These are
    // implementation defined (C99 6.4.4.4p10).
    if (NumCharsSoFar) {
      if (IsWide) {
        // Emulate GCC's (unintentional?) behavior: L'ab' -> L'b'.
        LitVal = 0;
      } else {
        // Narrow character literals act as though their value is concatenated
        // in this implementation, but warn on overflow.
        if (LitVal.countLeadingZeros() < 8)
          PP.Diag(Loc, diag::warn_char_constant_too_large);
        LitVal <<= 8;
      }
    }

    LitVal = LitVal + ResultChar;
    ++NumCharsSoFar;
  }

  // If this is the second character being processed, do special handling.
  if (NumCharsSoFar > 1) {
    // Warn about discarding the top bits for multi-char wide-character
    // constants (L'abcd').
    if (IsWide)
      PP.Diag(Loc, diag::warn_extraneous_wide_char_constant);
    else if (NumCharsSoFar != 4)
      PP.Diag(Loc, diag::ext_multichar_character_literal);
    else
      PP.Diag(Loc, diag::ext_four_char_character_literal);
    IsMultiChar = true;
  } else
    IsMultiChar = false;

  // Transfer the value from APInt to uint64_t
  Value = LitVal.getZExtValue();

  // If this is a single narrow character, sign extend it (e.g. '\xFF' is "-1")
  // if 'char' is signed for this target (C99 6.4.4.4p10).  Note that multiple
  // character constants are not sign extended in the this implementation:
  // '\xFF\xFF' = 65536 and '\x0\xFF' = 255, which matches GCC.
  if (!IsWide && NumCharsSoFar == 1 && (Value & 128) &&
      PP.getLangOptions().CharIsSigned)
    Value = (signed char)Value;
}


///       string-literal: [C99 6.4.5]
///          " [s-char-sequence] "
///         L" [s-char-sequence] "
///       s-char-sequence:
///         s-char
///         s-char-sequence s-char
///       s-char:
///         any source character except the double quote ",
///           backslash \, or newline character
///         escape-character
///         universal-character-name
///       escape-character: [C99 6.4.4.4]
///         \ escape-code
///         universal-character-name
///       escape-code:
///         character-escape-code
///         octal-escape-code
///         hex-escape-code
///       character-escape-code: one of
///         n t b r f v a
///         \ ' " ?
///       octal-escape-code:
///         octal-digit
///         octal-digit octal-digit
///         octal-digit octal-digit octal-digit
///       hex-escape-code:
///         x hex-digit
///         hex-escape-code hex-digit
///       universal-character-name:
///         \u hex-quad
///         \U hex-quad hex-quad
///       hex-quad:
///         hex-digit hex-digit hex-digit hex-digit
///
StringLiteralParser::
StringLiteralParser(const Token *StringToks, unsigned NumStringToks,
                    Preprocessor &pp) : PP(pp) {
  // Scan all of the string portions, remember the max individual token length,
  // computing a bound on the concatenated string length, and see whether any
  // piece is a wide-string.  If any of the string portions is a wide-string
  // literal, the result is a wide-string literal [C99 6.4.5p4].
  MaxTokenLength = StringToks[0].getLength();
  SizeBound = StringToks[0].getLength()-2;  // -2 for "".
  AnyWide = StringToks[0].is(tok::wide_string_literal);

  hadError = false;

  // Implement Translation Phase #6: concatenation of string literals
  /// (C99 5.1.1.2p1).  The common case is only one string fragment.
  for (unsigned i = 1; i != NumStringToks; ++i) {
    // The string could be shorter than this if it needs cleaning, but this is a
    // reasonable bound, which is all we need.
    SizeBound += StringToks[i].getLength()-2;  // -2 for "".

    // Remember maximum string piece length.
    if (StringToks[i].getLength() > MaxTokenLength)
      MaxTokenLength = StringToks[i].getLength();

    // Remember if we see any wide strings.
    AnyWide |= StringToks[i].is(tok::wide_string_literal);
  }

  // Include space for the null terminator.
  ++SizeBound;

  // TODO: K&R warning: "traditional C rejects string constant concatenation"

  // Get the width in bytes of wchar_t.  If no wchar_t strings are used, do not
  // query the target.  As such, wchar_tByteWidth is only valid if AnyWide=true.
  wchar_tByteWidth = ~0U;
  if (AnyWide) {
    wchar_tByteWidth = PP.getTargetInfo().getWCharWidth();
    assert((wchar_tByteWidth & 7) == 0 && "Assumes wchar_t is byte multiple!");
    wchar_tByteWidth /= 8;
  }

  // The output buffer size needs to be large enough to hold wide characters.
  // This is a worst-case assumption which basically corresponds to L"" "long".
  if (AnyWide)
    SizeBound *= wchar_tByteWidth;

  // Size the temporary buffer to hold the result string data.
  ResultBuf.resize(SizeBound);

  // Likewise, but for each string piece.
  llvm::SmallString<512> TokenBuf;
  TokenBuf.resize(MaxTokenLength);

  // Loop over all the strings, getting their spelling, and expanding them to
  // wide strings as appropriate.
  ResultPtr = &ResultBuf[0];   // Next byte to fill in.

  Pascal = false;

  for (unsigned i = 0, e = NumStringToks; i != e; ++i) {
    const char *ThisTokBuf = &TokenBuf[0];
    // Get the spelling of the token, which eliminates trigraphs, etc.  We know
    // that ThisTokBuf points to a buffer that is big enough for the whole token
    // and 'spelled' tokens can only shrink.
    unsigned ThisTokLen = PP.getSpelling(StringToks[i], ThisTokBuf);
    const char *ThisTokEnd = ThisTokBuf+ThisTokLen-1;  // Skip end quote.

    // TODO: Input character set mapping support.

    // Skip L marker for wide strings.
    bool ThisIsWide = false;
    if (ThisTokBuf[0] == 'L') {
      ++ThisTokBuf;
      ThisIsWide = true;
    }

    assert(ThisTokBuf[0] == '"' && "Expected quote, lexer broken?");
    ++ThisTokBuf;

    // Check if this is a pascal string
    if (pp.getLangOptions().PascalStrings && ThisTokBuf + 1 != ThisTokEnd &&
        ThisTokBuf[0] == '\\' && ThisTokBuf[1] == 'p') {

      // If the \p sequence is found in the first token, we have a pascal string
      // Otherwise, if we already have a pascal string, ignore the first \p
      if (i == 0) {
        ++ThisTokBuf;
        Pascal = true;
      } else if (Pascal)
        ThisTokBuf += 2;
    }

    while (ThisTokBuf != ThisTokEnd) {
      // Is this a span of non-escape characters?
      if (ThisTokBuf[0] != '\\') {
        const char *InStart = ThisTokBuf;
        do {
          ++ThisTokBuf;
        } while (ThisTokBuf != ThisTokEnd && ThisTokBuf[0] != '\\');

        // Copy the character span over.
        unsigned Len = ThisTokBuf-InStart;
        if (!AnyWide) {
          memcpy(ResultPtr, InStart, Len);
          ResultPtr += Len;
        } else {
          // Note: our internal rep of wide char tokens is always little-endian.
          for (; Len; --Len, ++InStart) {
            *ResultPtr++ = InStart[0];
            // Add zeros at the end.
            for (unsigned i = 1, e = wchar_tByteWidth; i != e; ++i)
              *ResultPtr++ = 0;
          }
        }
        continue;
      }
      // Is this a Universal Character Name escape?
      if (ThisTokBuf[1] == 'u' || ThisTokBuf[1] == 'U') {
        ProcessUCNEscape(ThisTokBuf, ThisTokEnd, ResultPtr,
                         hadError, StringToks[i].getLocation(), ThisIsWide, PP);
        continue;
      }
      // Otherwise, this is a non-UCN escape character.  Process it.
      unsigned ResultChar = ProcessCharEscape(ThisTokBuf, ThisTokEnd, hadError,
                                              StringToks[i].getLocation(),
                                              ThisIsWide, PP);

      // Note: our internal rep of wide char tokens is always little-endian.
      *ResultPtr++ = ResultChar & 0xFF;

      if (AnyWide) {
        for (unsigned i = 1, e = wchar_tByteWidth; i != e; ++i)
          *ResultPtr++ = ResultChar >> i*8;
      }
    }
  }

  if (Pascal) {
    ResultBuf[0] = ResultPtr-&ResultBuf[0]-1;

    // Verify that pascal strings aren't too large.
    if (GetStringLength() > 256) {
      PP.Diag(StringToks[0].getLocation(), diag::err_pascal_string_too_long)
        << SourceRange(StringToks[0].getLocation(),
                       StringToks[NumStringToks-1].getLocation());
      hadError = 1;
      return;
    }
  }
}


/// getOffsetOfStringByte - This function returns the offset of the
/// specified byte of the string data represented by Token.  This handles
/// advancing over escape sequences in the string.
unsigned StringLiteralParser::getOffsetOfStringByte(const Token &Tok,
                                                    unsigned ByteNo,
                                                    Preprocessor &PP) {
  // Get the spelling of the token.
  llvm::SmallString<16> SpellingBuffer;
  SpellingBuffer.resize(Tok.getLength());

  const char *SpellingPtr = &SpellingBuffer[0];
  unsigned TokLen = PP.getSpelling(Tok, SpellingPtr);

  assert(SpellingPtr[0] != 'L' && "Doesn't handle wide strings yet");


  const char *SpellingStart = SpellingPtr;
  const char *SpellingEnd = SpellingPtr+TokLen;

  // Skip over the leading quote.
  assert(SpellingPtr[0] == '"' && "Should be a string literal!");
  ++SpellingPtr;

  // Skip over bytes until we find the offset we're looking for.
  while (ByteNo) {
    assert(SpellingPtr < SpellingEnd && "Didn't find byte offset!");

    // Step over non-escapes simply.
    if (*SpellingPtr != '\\') {
      ++SpellingPtr;
      --ByteNo;
      continue;
    }

    // Otherwise, this is an escape character.  Advance over it.
    bool HadError = false;
    ProcessCharEscape(SpellingPtr, SpellingEnd, HadError,
                      Tok.getLocation(), false, PP);
    assert(!HadError && "This method isn't valid on erroneous strings");
    --ByteNo;
  }

  return SpellingPtr-SpellingStart;
}

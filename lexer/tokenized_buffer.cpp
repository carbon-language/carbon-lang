// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "lexer/tokenized_buffer.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <iterator>
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

struct TrailingComment : SimpleDiagnostic<TrailingComment> {
  static constexpr llvm::StringLiteral ShortName = "syntax-comments";
  static constexpr llvm::StringLiteral Message =
      "Trailing comments are not permitted.";
};

struct NoWhitespaceAfterCommentIntroducer
    : SimpleDiagnostic<NoWhitespaceAfterCommentIntroducer> {
  static constexpr llvm::StringLiteral ShortName = "syntax-comments";
  static constexpr llvm::StringLiteral Message =
      "Whitespace is required after '//'.";
};

struct UnmatchedClosing : SimpleDiagnostic<UnmatchedClosing> {
  static constexpr llvm::StringLiteral ShortName = "syntax-balanced-delimiters";
  static constexpr llvm::StringLiteral Message =
      "Closing symbol without a corresponding opening symbol.";
};

struct MismatchedClosing : SimpleDiagnostic<MismatchedClosing> {
  static constexpr llvm::StringLiteral ShortName = "syntax-balanced-delimiters";
  static constexpr llvm::StringLiteral Message =
      "Closing symbol does not match most recent opening symbol.";
};

struct EmptyDigitSequence : SimpleDiagnostic<EmptyDigitSequence> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";
  static constexpr llvm::StringLiteral Message =
      "Empty digit sequence in numeric literal.";
};

struct InvalidDigit {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";

  struct Substitutions {
    char digit;
    int radix;
  };
  static auto Format(const Substitutions& subst) -> std::string {
    return llvm::formatv("Invalid digit '{0}' in {1} numeric literal.",
                         subst.digit,
                         (subst.radix == 2
                              ? "binary"
                              : subst.radix == 16 ? "hexadecimal" : "decimal"))
        .str();
  }
};

struct InvalidDigitSeparator : SimpleDiagnostic<InvalidDigitSeparator> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";
  static constexpr llvm::StringLiteral Message =
      "Misplaced digit separator in numeric literal.";
};

struct IrregularDigitSeparators {
  static constexpr llvm::StringLiteral ShortName =
      "syntax-irregular-digit-separators";

  struct Substitutions {
    int radix;
  };
  static auto Format(const Substitutions& subst) -> std::string {
    assert((subst.radix == 10 || subst.radix == 16) && "unexpected radix");
    return llvm::formatv(
               "Digit separators in {0} number should appear every {1} "
               "characters from the right.",
               (subst.radix == 10 ? "decimal" : "hexadecimal"),
               (subst.radix == 10 ? "3" : "4"))
        .str();
  }
};

struct UnknownBaseSpecifier : SimpleDiagnostic<UnknownBaseSpecifier> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";
  static constexpr llvm::StringLiteral Message =
      "Unknown base specifier in numeric literal.";
};

struct BinaryRealLiteral : SimpleDiagnostic<BinaryRealLiteral> {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";
  static constexpr llvm::StringLiteral Message =
      "Binary real number literals are not supported.";
};

struct WrongRealLiteralExponent {
  static constexpr llvm::StringLiteral ShortName = "syntax-invalid-number";

  struct Substitutions {
    char expected;
  };
  static auto Format(const Substitutions& subst) -> std::string {
    return llvm::formatv("Expected '{0}' to introduce exponent.",
                         subst.expected)
        .str();
  }
};

struct UnrecognizedCharacters : SimpleDiagnostic<UnrecognizedCharacters> {
  static constexpr llvm::StringLiteral ShortName =
      "syntax-unrecognized-characters";
  static constexpr llvm::StringLiteral Message =
      "Encountered unrecognized characters while parsing.";
};

// TODO(zygoloid): Update this to match whatever we decide qualifies as
// acceptable whitespace.
static bool isSpace(char c) { return c == ' ' || c == '\n' || c == '\t'; }

static bool isLower(char c) { return 'a' <= c && c <= 'z'; }

namespace {
struct NumericLiteral {
  llvm::StringRef text;

  // The offset of the '.'. Set to text.size() if none is present.
  int radix_point;

  // The offset of the alphabetical character introducing the exponent. In a
  // valid literal, this will be an 'e' or a 'p', and may be followed by a '+'
  // or a '-', but for error recovery, this may simply be the last lowercase
  // letter in the invalid token. Always greater than or equal to radix_point.
  // Set to text.size() if none is present.
  int exponent;
};
}  // namespace

static auto TakeLeadingNumericLiteral(llvm::StringRef source_text)
    -> NumericLiteral {
  NumericLiteral result;

  if (source_text.empty() || !llvm::isDigit(source_text.front()))
    return result;

  bool seen_plus_minus = false;
  bool seen_radix_point = false;
  bool seen_potential_exponent = false;

  // Greedily consume all following characters that might be part of a numeric
  // literal. This allows us to produce better diagnostics on invalid literals.
  //
  // TODO(zygoloid): Update lexical rules to specify that a numeric literal
  // cannot be immediately followed by an alphanumeric character.
  int i = 1, n = source_text.size();
  for (; i != n; ++i) {
    char c = source_text[i];
    if (llvm::isAlnum(c) || c == '_') {
      if (isLower(c) && seen_radix_point && !seen_plus_minus) {
        result.exponent = i;
        seen_potential_exponent = true;
      }
      continue;
    }

    // Exactly one `.` can be part of the literal, but only if it's followed by
    // an alphanumeric character.
    if (c == '.' && i + 1 != n && llvm::isAlnum(source_text[i + 1]) &&
        !seen_radix_point) {
      result.radix_point = i;
      seen_radix_point = true;
      continue;
    }

    // A `+` or `-` continues the literal only if it's preceded by a lowercase
    // letter (which will be 'e' or 'p' or part of an invalid literal) and
    // followed by an alphanumeric character. This '+' or '-' cannot be an
    // operator because a literal cannot end in a lowercase letter.
    if ((c == '+' || c == '-') && seen_potential_exponent &&
        result.exponent == i - 1 && i + 1 != n &&
        llvm::isAlnum(source_text[i + 1])) {
      // This is not possible because we don't update result.exponent after we
      // see a '+' or '-'.
      assert(!seen_plus_minus && "should only consume one + or -");
      seen_plus_minus = true;
      continue;
    }

    break;
  }

  result.text = source_text.substr(0, i);
  if (!seen_radix_point)
    result.radix_point = i;
  if (!seen_potential_exponent)
    result.exponent = i;

  return result;
}

namespace {
// Parser for numeric literal tokens.
//
// Responsible for checking that a numeric literal is valid and meaningful and
// either diagnosing or extracting its meaning.
class NumericLiteralParser {
 public:
  NumericLiteralParser(DiagnosticEmitter& emitter, NumericLiteral literal)
      : emitter(emitter), literal(literal) {
    int_part = literal.text.substr(0, literal.radix_point);
    if (int_part.consume_front("0x")) {
      radix = 16;
    } else if (int_part.consume_front("0b")) {
      radix = 2;
    }

    fract_part = literal.text.substr(
        literal.radix_point + 1, literal.exponent - literal.radix_point - 1);

    exponent_part = literal.text.substr(literal.exponent + 1);
    if (!exponent_part.consume_front("+")) {
      exponent_is_negative = exponent_part.consume_front("-");
    }
  }

  auto IsInteger() -> bool {
    return literal.radix_point == static_cast<int>(literal.text.size());
  }

  enum CheckResult {
    // The token is valid.
    Valid,
    // The token is invalid, but we've diagnosed and recovered from the error.
    RecoverableError,
    // The token is invalid, and we've diagnosed, but we can't assign meaning
    // to it.
    UnrecoverableError,
  };

  // Check that the numeric literal token is syntactically valid and
  // meaningful, and diagnose if not.
  auto Check() -> CheckResult {
    if (!CheckLeadingZero() || !CheckIntPart() || !CheckFractionalPart() ||
        !CheckExponentPart())
      return UnrecoverableError;
    return recovered_from_error ? RecoverableError : Valid;
  }

  auto GetMantissa() -> llvm::APInt {
    const char* end = IsInteger() ? int_part.end() : fract_part.end();
    llvm::StringRef digits(int_part.begin(), end - int_part.begin());
    return ParseInteger(digits, radix, mantissa_needs_cleaning);
  }

  auto GetExponent() -> llvm::APInt {
    // Compute the effective exponent from the specified exponent, if any,
    // and the position of the radix point.
    llvm::APInt exponent(64, 0);
    if (!exponent_part.empty()) {
      exponent = ParseInteger(exponent_part, 10, exponent_needs_cleaning);

      // The exponent is a signed integer, and the number we just parsed is
      // non-negative, so ensure we have a wide enough representation to
      // include a sign bit. Also make sure the exponent isn't too narrow so
      // the calculation below can't lose information through overflow.
      if (exponent.isSignBitSet() || exponent.getBitWidth() < 64) {
        exponent = exponent.zext(std::max(64u, exponent.getBitWidth() + 1));
      }
      if (exponent_is_negative) {
        exponent.negate();
      }
    }

    // Each character after the decimal point reduces the effective exponent.
    int excess_exponent = fract_part.size();
    if (radix == 16) {
      excess_exponent *= 4;
    }
    exponent -= excess_exponent;
    if (exponent_is_negative && !exponent.isNegative()) {
      // We overflowed. Note that we can only overflow by a little, and only
      // from negative to positive, because exponent is at least 64 bits wide
      // and excess_exponent is bounded above by four times the size of the
      // input buffer, which we assume fits into 32 bits.
      exponent = exponent.zext(exponent.getBitWidth() + 1);
      exponent.setSignBit();
    }
    return exponent;
  }

 private:
  struct CheckDigitSequenceResult {
    bool ok;
    bool has_digit_separators = false;
  };

  // Check that a digit sequence is valid: that it contains one or more digits,
  // contains only digits in the specified base, and that any digit separators
  // are present and correctly positioned.
  auto CheckDigitSequence(llvm::StringRef text, int radix,
                          bool allow_digit_separators = true)
      -> CheckDigitSequenceResult {
    assert((radix == 2 || radix == 10 || radix == 16) && "unknown radix");

    std::bitset<256> valid_digits;
    if (radix == 2) {
      for (char c : "01")
        valid_digits[static_cast<unsigned char>(c)] = true;
    } else if (radix == 10) {
      for (char c : "0123456789")
        valid_digits[static_cast<unsigned char>(c)] = true;
    } else {
      for (char c : "0123456789ABCDEF")
        valid_digits[static_cast<unsigned char>(c)] = true;
    }

    int num_digit_separators = 0;

    for (int i = 0, n = text.size(); i != n; ++i) {
      char c = text[i];
      if (valid_digits[static_cast<unsigned char>(c)]) {
        continue;
      }

      if (c == '_') {
        // A digit separator cannot appear at the start of a digit sequence,
        // next to another digit separator, or at the end.
        if (!allow_digit_separators || i == 0 || text[i - 1] == '_' ||
            i + 1 == n) {
          emitter.EmitError<InvalidDigitSeparator>();
          recovered_from_error = true;
        }
        ++num_digit_separators;
        continue;
      }

      emitter.EmitError<InvalidDigit>({.digit = c, .radix = radix});
      return {.ok = false};
    }

    if (num_digit_separators == static_cast<int>(text.size())) {
      emitter.EmitError<EmptyDigitSequence>();
      return {.ok = false};
    }

    // Check that digit separators occur in exactly the expected positions.
    if (num_digit_separators && radix != 2)
      CheckDigitSeparatorPlacement(text, radix, num_digit_separators);

    return {.ok = true, .has_digit_separators = (num_digit_separators != 0)};
  }

  // Given a number with digit separators, check that the digit separators are
  // correctly positioned.
  auto CheckDigitSeparatorPlacement(llvm::StringRef text, int radix,
                                    int num_digit_separators) -> void {
    assert((radix == 10 || radix == 16) &&
           "unexpected radix for digit separator checks");
    assert(std::count(text.begin(), text.end(), '_') == num_digit_separators &&
           "given wrong number of digit separators");

    auto diagnose_irregular_digit_separators = [&] {
      emitter.EmitError<IrregularDigitSeparators>({.radix = radix});
      recovered_from_error = true;
    };

    // For decimal and hexadecimal digit sequences, digit separators must form
    // groups of 3 or 4 digits (4 or 5 characters), respectively.
    int stride = (radix == 10 ? 4 : 5);
    int remaining_digit_separators = num_digit_separators;
    for (auto pos = text.end(); pos - text.begin() >= stride; /*in loop*/) {
      pos -= stride;
      if (*pos != '_')
        return diagnose_irregular_digit_separators();

      --remaining_digit_separators;
    }

    // Check there weren't any other digit separators.
    if (remaining_digit_separators)
      diagnose_irregular_digit_separators();
  };

  // Check that we don't have a '0' prefix on a non-zero decimal integer.
  auto CheckLeadingZero() -> bool {
    if (radix == 10 && int_part.startswith("0") && int_part != "0") {
      emitter.EmitError<UnknownBaseSpecifier>();
      return false;
    }
    return true;
  }

  // Check the integer part (before the '.', if any) is valid.
  auto CheckIntPart() -> bool {
    auto int_result = CheckDigitSequence(int_part, radix);
    mantissa_needs_cleaning |= int_result.has_digit_separators;
    return int_result.ok;
  }

  // Check the fractional part (after the '.' and before the exponent, if any)
  // is valid.
  auto CheckFractionalPart() -> bool {
    if (IsInteger()) {
      return true;
    }

    if (radix == 2) {
      emitter.EmitError<BinaryRealLiteral>();
      recovered_from_error = true;
      // Carry on and parse the binary real literal anyway.
    }

    // We need to remove a '.' from the mantissa.
    mantissa_needs_cleaning = true;

    return CheckDigitSequence(fract_part, radix,
                              /*allow_digit_separators=*/false)
        .ok;
  }

  // Check the exponent part (if any) is valid.
  auto CheckExponentPart() -> bool {
    if (literal.exponent == static_cast<int>(literal.text.size())) {
      return true;
    }

    char expected_exponent_kind = (radix == 10 ? 'e' : 'p');
    if (literal.text[literal.exponent] != expected_exponent_kind) {
      emitter.EmitError<WrongRealLiteralExponent>(
          {.expected = expected_exponent_kind});
      return false;
    }

    auto exponent_result = CheckDigitSequence(exponent_part, 10);
    exponent_needs_cleaning = exponent_result.has_digit_separators;
    return exponent_result.ok;
  }

  // Parse a string that is known to be a valid base-radix integer into an
  // APInt.  If needs_cleaning is true, the string may additionally contain '_'
  // and '.' characters that should be ignored.
  //
  // Ignoring '.' is used when parsing a real literal. For example, when
  // parsing 123.456e7, we want to decompose it into an integer mantissa
  // (123456) and an exponent (7 - 3 = 2), and this routine is given the
  // "123.456" to parse as the mantissa.
  static auto ParseInteger(llvm::StringRef digits, int radix,
                           bool needs_cleaning) -> llvm::APInt {
    llvm::SmallString<32> cleaned;
    if (needs_cleaning) {
      cleaned.reserve(digits.size());
      std::remove_copy_if(digits.begin(), digits.end(),
                          std::back_inserter(cleaned),
                          [](char c) { return c == '_' || c == '.'; });
      digits = cleaned;
    }

    llvm::APInt value;
    if (digits.getAsInteger(radix, value)) {
      llvm_unreachable("should never fail");
    }
    return value;
  }

 private:
  DiagnosticEmitter& emitter;
  NumericLiteral literal;

  // The radix of the literal: 2, 10, or 16, for a prefix of '0b', no prefix,
  // or '0x', respectively.
  int radix = 10;

  // The various components of a numeric literal:
  //
  //     [radix] int_part [. fract_part [[ep] [+-] exponent_part]]
  llvm::StringRef int_part;
  llvm::StringRef fract_part;
  llvm::StringRef exponent_part;

  // Do we need to remove any special characters (digit separator or radix
  // point) before interpreting the mantissa or exponent as an integer?
  bool mantissa_needs_cleaning = false;
  bool exponent_needs_cleaning = false;

  // True if we found a `-` before `exponent_part`.
  bool exponent_is_negative = false;

  // True if we produced an error but recovered.
  bool recovered_from_error = false;
};
}  // namespace

// Implementation of the lexer logic itself.
//
// The design is that lexing can loop over the source buffer, consuming it into
// tokens by calling into this API. This class handles the state and breaks down
// the different lexing steps that may be used. It directly updates the provided
// tokenized buffer with the lexed tokens.
class TokenizedBuffer::Lexer {
  TokenizedBuffer& buffer;
  DiagnosticEmitter& emitter;

  Line current_line;
  LineInfo* current_line_info;

  int current_column = 0;
  bool set_indent = false;

  llvm::SmallVector<Token, 8> open_groups;

 public:
  Lexer(TokenizedBuffer& buffer, DiagnosticEmitter& emitter)
      : buffer(buffer),
        emitter(emitter),
        current_line(buffer.AddLine({0, 0, 0})),
        current_line_info(&buffer.GetLineInfo(current_line)) {}

  // Symbolic result of a lexing action. This indicates whether we successfully
  // lexed a token, or whether other lexing actions should be attempted.
  //
  // While it wraps a simple boolean state, its API both helps make the failures
  // more self documenting, and by consuming the actual token constructively
  // when one is produced, it helps ensure the correct result is returned.
  class LexResult {
    bool formed_token;
    explicit LexResult(bool formed_token) : formed_token(formed_token) {}

   public:
    // Consumes (and discard) a valid token to construct a result
    // indicating a token has been produced.
    LexResult(Token) : LexResult(true) {}

    // Returns a result indicating no token was produced.
    static LexResult NoMatch() { return LexResult(false); }

    // Tests whether a token was produced by the lexing routine, and
    // the lexer can continue forming tokens.
    explicit operator bool() const { return formed_token; }
  };

  auto SkipWhitespace(llvm::StringRef& source_text) -> bool {
    while (!source_text.empty()) {
      // We only support line-oriented commenting and lex comments as-if they
      // were whitespace.
      if (source_text.startswith("//")) {
        // Any comment must be the only non-whitespace on the line.
        if (set_indent) {
          emitter.EmitError<TrailingComment>();
          buffer.has_errors = true;
        }
        // The introducer '//' must be followed by whitespace or EOF.
        if (source_text.size() > 2 && !isSpace(source_text[2])) {
          emitter.EmitError<NoWhitespaceAfterCommentIntroducer>();
          buffer.has_errors = true;
        }
        while (!source_text.empty() && source_text.front() != '\n') {
          ++current_column;
          source_text = source_text.drop_front();
        }
        if (source_text.empty()) {
          break;
        }
      }

      switch (source_text.front()) {
        default:
          // If we find a non-whitespace character without exhausting the
          // buffer, return true to continue lexing.
          assert(!isSpace(source_text.front()));
          return true;

        case '\n':
          // New lines are special in order to track line structure.
          current_line_info->length = current_column;
          // If this is the last character in the source, directly return here
          // to avoid creating an empty line.
          source_text = source_text.drop_front();
          if (source_text.empty()) {
            return false;
          }

          // Otherwise, add a line and set up to continue lexing.
          current_line = buffer.AddLine(
              {current_line_info->start + current_column + 1, 0, 0});
          current_line_info = &buffer.GetLineInfo(current_line);
          current_column = 0;
          set_indent = false;
          continue;

        case ' ':
        case '\t':
          // Skip other forms of whitespace while tracking column.
          // FIXME: This obviously needs looooots more work to handle unicode
          // whitespace as well as special handling to allow better tokenization
          // of operators. This is just a stub to check that our column
          // management works.
          ++current_column;
          source_text = source_text.drop_front();
          continue;
      }
    }

    assert(source_text.empty() && "Cannot reach here w/o finishing the text!");
    // Update the line length as this is also the end of a line.
    current_line_info->length = current_column;
    return false;
  }

  auto LexNumericLiteral(llvm::StringRef& source_text) -> LexResult {
    NumericLiteral literal = TakeLeadingNumericLiteral(source_text);
    if (literal.text.empty()) {
      return LexResult::NoMatch();
    }

    int int_column = current_column;
    current_column += literal.text.size();
    source_text = source_text.drop_front(literal.text.size());

    if (!set_indent) {
      current_line_info->indent = int_column;
      set_indent = true;
    }

    NumericLiteralParser literal_parser(emitter, literal);

    switch (literal_parser.Check()) {
      case NumericLiteralParser::UnrecoverableError: {
        auto token = buffer.AddToken({
            .kind = TokenKind::Error(),
            .token_line = current_line,
            .column = int_column,
            .error_length = static_cast<int32_t>(literal.text.size()),
        });
        buffer.has_errors = true;
        return token;
      }

      case NumericLiteralParser::RecoverableError:
        buffer.has_errors = true;
        break;

      case NumericLiteralParser::Valid:
        break;
    }

    if (literal_parser.IsInteger()) {
      auto token = buffer.AddToken({.kind = TokenKind::IntegerLiteral(),
                                    .token_line = current_line,
                                    .column = int_column});
      buffer.GetTokenInfo(token).literal_index =
          buffer.literal_int_storage.size();
      buffer.literal_int_storage.push_back(literal_parser.GetMantissa());
      return token;
    } else {
      auto token = buffer.AddToken({.kind = TokenKind::RealLiteral(),
                                    .token_line = current_line,
                                    .column = int_column});
      buffer.GetTokenInfo(token).literal_index =
          buffer.literal_int_storage.size();
      buffer.literal_int_storage.push_back(literal_parser.GetMantissa());
      buffer.literal_int_storage.push_back(literal_parser.GetExponent());
      return token;
    }
  }

  auto LexSymbolToken(llvm::StringRef& source_text) -> LexResult {
    TokenKind kind = llvm::StringSwitch<TokenKind>(source_text)
#define CARBON_SYMBOL_TOKEN(Name, Spelling) \
  .StartsWith(Spelling, TokenKind::Name())
#include "lexer/token_registry.def"
                         .Default(TokenKind::Error());
    if (kind == TokenKind::Error()) {
      return LexResult::NoMatch();
    }

    if (!set_indent) {
      current_line_info->indent = current_column;
      set_indent = true;
    }

    CloseInvalidOpenGroups(kind);

    Token token = buffer.AddToken(
        {.kind = kind, .token_line = current_line, .column = current_column});
    current_column += kind.GetFixedSpelling().size();
    source_text = source_text.drop_front(kind.GetFixedSpelling().size());

    // Opening symbols just need to be pushed onto our queue of opening groups.
    if (kind.IsOpeningSymbol()) {
      open_groups.push_back(token);
      return token;
    }

    // Only closing symbols need further special handling.
    if (!kind.IsClosingSymbol()) {
      return token;
    }

    TokenInfo& closing_token_info = buffer.GetTokenInfo(token);

    // Check that there is a matching opening symbol before we consume this as
    // a closing symbol.
    if (open_groups.empty()) {
      closing_token_info.kind = TokenKind::Error();
      closing_token_info.error_length = kind.GetFixedSpelling().size();
      buffer.has_errors = true;

      emitter.EmitError<UnmatchedClosing>();
      // Note that this still returns true as we do consume a symbol.
      return token;
    }

    // Finally can handle a normal closing symbol.
    Token opening_token = open_groups.pop_back_val();
    TokenInfo& opening_token_info = buffer.GetTokenInfo(opening_token);
    opening_token_info.closing_token = token;
    closing_token_info.opening_token = opening_token;
    return token;
  }

  // Closes all open groups that cannot remain open across the symbol `K`.
  // Users may pass `Error` to close all open groups.
  auto CloseInvalidOpenGroups(TokenKind kind) -> void {
    if (!kind.IsClosingSymbol() && kind != TokenKind::Error()) {
      return;
    }

    while (!open_groups.empty()) {
      Token opening_token = open_groups.back();
      TokenKind opening_kind = buffer.GetTokenInfo(opening_token).kind;
      if (kind == opening_kind.GetClosingSymbol()) {
        return;
      }

      open_groups.pop_back();
      buffer.has_errors = true;
      emitter.EmitError<MismatchedClosing>();

      // TODO: do a smarter backwards scan for where to put the closing
      // token.
      Token closing_token =
          buffer.AddToken({.kind = opening_kind.GetClosingSymbol(),
                           .is_recovery = true,
                           .token_line = current_line,
                           .column = current_column});
      TokenInfo& opening_token_info = buffer.GetTokenInfo(opening_token);
      TokenInfo& closing_token_info = buffer.GetTokenInfo(closing_token);
      opening_token_info.closing_token = closing_token;
      closing_token_info.opening_token = opening_token;
    }
  }

  auto GetOrCreateIdentifier(llvm::StringRef text) -> Identifier {
    auto insert_result = buffer.identifier_map.insert(
        {text, Identifier(buffer.identifier_infos.size())});
    if (insert_result.second) {
      buffer.identifier_infos.push_back({text});
    }
    return insert_result.first->second;
  }

  auto LexKeywordOrIdentifier(llvm::StringRef& source_text) -> LexResult {
    if (!llvm::isAlpha(source_text.front()) && source_text.front() != '_') {
      return LexResult::NoMatch();
    }

    if (!set_indent) {
      current_line_info->indent = current_column;
      set_indent = true;
    }

    // Take the valid characters off the front of the source buffer.
    llvm::StringRef identifier_text = source_text.take_while(
        [](char c) { return llvm::isAlnum(c) || c == '_'; });
    assert(!identifier_text.empty() && "Must have at least one character!");
    int identifier_column = current_column;
    current_column += identifier_text.size();
    source_text = source_text.drop_front(identifier_text.size());

    // Check if the text matches a keyword token, and if so use that.
    TokenKind kind = llvm::StringSwitch<TokenKind>(identifier_text)
#define CARBON_KEYWORD_TOKEN(Name, Spelling) .Case(Spelling, TokenKind::Name())
#include "lexer/token_registry.def"
                         .Default(TokenKind::Error());
    if (kind != TokenKind::Error()) {
      return buffer.AddToken({.kind = kind,
                              .token_line = current_line,
                              .column = identifier_column});
    }

    // Otherwise we have a generic identifier.
    return buffer.AddToken({.kind = TokenKind::Identifier(),
                            .token_line = current_line,
                            .column = identifier_column,
                            .id = GetOrCreateIdentifier(identifier_text)});
  }

  auto LexError(llvm::StringRef& source_text) -> LexResult {
    llvm::StringRef error_text = source_text.take_while([](char c) {
      if (llvm::isAlnum(c)) {
        return false;
      }
      switch (c) {
        case '_':
        case '\t':
        case '\n':
          return false;
      }
      return llvm::StringSwitch<bool>(llvm::StringRef(&c, 1))
#define CARBON_SYMBOL_TOKEN(Name, Spelling) .StartsWith(Spelling, false)
#include "lexer/token_registry.def"
          .Default(true);
    });
    if (error_text.empty()) {
      // TODO: Reimplement this to use the lexer properly. In the meantime,
      // guarantee that we eat at least one byte.
      error_text = source_text.take_front(1);
    }

    // Longer errors get to be two tokens.
    error_text = error_text.substr(0, std::numeric_limits<int32_t>::max());
    auto token = buffer.AddToken(
        {.kind = TokenKind::Error(),
         .token_line = current_line,
         .column = current_column,
         .error_length = static_cast<int32_t>(error_text.size())});
    // TODO: #19 - Need to convert to the diagnostics library.
    llvm::errs() << "ERROR: Line " << buffer.GetLineNumber(token) << ", Column "
                 << buffer.GetColumnNumber(token)
                 << ": Unrecognized characters!\n";

    current_column += error_text.size();
    source_text = source_text.drop_front(error_text.size());
    buffer.has_errors = true;
    return token;
  }
};

auto TokenizedBuffer::Lex(SourceBuffer& source, DiagnosticEmitter& emitter)
    -> TokenizedBuffer {
  TokenizedBuffer buffer(source);
  Lexer lexer(buffer, emitter);

  llvm::StringRef source_text = source.Text();
  while (lexer.SkipWhitespace(source_text)) {
    // Each time we find non-whitespace characters, try each kind of token we
    // support lexing, from simplest to most complex.
    Lexer::LexResult result = lexer.LexSymbolToken(source_text);
    if (!result) {
      result = lexer.LexKeywordOrIdentifier(source_text);
    }
    if (!result) {
      result = lexer.LexNumericLiteral(source_text);
    }
    if (!result) {
      result = lexer.LexError(source_text);
    }
    assert(result && "No token was lexed.");
  }

  lexer.CloseInvalidOpenGroups(TokenKind::Error());
  return buffer;
}

auto TokenizedBuffer::GetKind(Token token) const -> TokenKind {
  return GetTokenInfo(token).kind;
}

auto TokenizedBuffer::GetLine(Token token) const -> Line {
  return GetTokenInfo(token).token_line;
}

auto TokenizedBuffer::GetLineNumber(Token token) const -> int {
  return GetLineNumber(GetLine(token));
}

auto TokenizedBuffer::GetColumnNumber(Token token) const -> int {
  return GetTokenInfo(token).column + 1;
}

auto TokenizedBuffer::GetTokenText(Token token) const -> llvm::StringRef {
  auto& token_info = GetTokenInfo(token);
  llvm::StringRef fixed_spelling = token_info.kind.GetFixedSpelling();
  if (!fixed_spelling.empty()) {
    return fixed_spelling;
  }

  if (token_info.kind == TokenKind::Error()) {
    auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    return source->Text().substr(token_start, token_info.error_length);
  }

  // Refer back to the source text to preserve oddities like radix or digit
  // separators the author included.
  if (token_info.kind == TokenKind::IntegerLiteral() ||
      token_info.kind == TokenKind::RealLiteral()) {
    auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    return TakeLeadingNumericLiteral(source->Text().substr(token_start)).text;
  }

  assert(token_info.kind == TokenKind::Identifier() &&
         "Only identifiers have stored text!");
  return GetIdentifierText(token_info.id);
}

auto TokenizedBuffer::GetIdentifier(Token token) const -> Identifier {
  auto& token_info = GetTokenInfo(token);
  assert(token_info.kind == TokenKind::Identifier() &&
         "The token must be an identifier!");
  return token_info.id;
}

auto TokenizedBuffer::GetIntegerLiteral(Token token) const
    -> const llvm::APInt& {
  auto& token_info = GetTokenInfo(token);
  assert(token_info.kind == TokenKind::IntegerLiteral() &&
         "The token must be an integer literal!");
  return literal_int_storage[token_info.literal_index];
}

auto TokenizedBuffer::GetRealLiteral(Token token) const -> RealLiteralValue {
  auto& token_info = GetTokenInfo(token);
  assert(token_info.kind == TokenKind::RealLiteral() &&
         "The token must be a real literal!");

  // Note that every real literal is at least three characters long, so we can
  // safely look at the second character to determine whether we have a decimal
  // or hexadecimal literal.
  auto& line_info = GetLineInfo(token_info.token_line);
  int64_t token_start = line_info.start + token_info.column;
  char second_char = source->Text()[token_start + 1];
  bool is_decimal = second_char != 'x' && second_char != 'b';

  return RealLiteralValue(this, token_info.literal_index, is_decimal);
}

auto TokenizedBuffer::GetMatchedClosingToken(Token opening_token) const
    -> Token {
  auto& opening_token_info = GetTokenInfo(opening_token);
  assert(opening_token_info.kind.IsOpeningSymbol() &&
         "The token must be an opening group symbol!");
  return opening_token_info.closing_token;
}

auto TokenizedBuffer::GetMatchedOpeningToken(Token closing_token) const
    -> Token {
  auto& closing_token_info = GetTokenInfo(closing_token);
  assert(closing_token_info.kind.IsClosingSymbol() &&
         "The token must be an closing group symbol!");
  return closing_token_info.opening_token;
}

auto TokenizedBuffer::IsRecoveryToken(Token token) const -> bool {
  return GetTokenInfo(token).is_recovery;
}

auto TokenizedBuffer::GetLineNumber(Line line) const -> int {
  return line.index + 1;
}

auto TokenizedBuffer::GetIndentColumnNumber(Line line) const -> int {
  return GetLineInfo(line).indent + 1;
}

auto TokenizedBuffer::GetIdentifierText(Identifier identifier) const
    -> llvm::StringRef {
  return identifier_infos[identifier.index].text;
}

auto TokenizedBuffer::PrintWidths::Widen(const PrintWidths& widths) -> void {
  index = std::max(widths.index, index);
  kind = std::max(widths.kind, kind);
  column = std::max(widths.column, column);
  line = std::max(widths.line, line);
  indent = std::max(widths.indent, indent);
}

// Compute the printed width of a number. When numbers are printed in decimal,
// the number of digits needed is is one more than the log-base-10 of the value.
// We handle a value of `zero` explicitly.
//
// This routine requires its argument to be *non-negative*.
static auto ComputeDecimalPrintedWidth(int number) -> int {
  assert(number >= 0 && "Negative numbers are not supported.");
  if (number == 0) {
    return 1;
  }

  return static_cast<int>(std::log10(number)) + 1;
}

auto TokenizedBuffer::GetTokenPrintWidths(Token token) const -> PrintWidths {
  PrintWidths widths = {};
  widths.index = ComputeDecimalPrintedWidth(token_infos.size());
  widths.kind = GetKind(token).Name().size();
  widths.line = ComputeDecimalPrintedWidth(GetLineNumber(token));
  widths.column = ComputeDecimalPrintedWidth(GetColumnNumber(token));
  widths.indent =
      ComputeDecimalPrintedWidth(GetIndentColumnNumber(GetLine(token)));
  return widths;
}

auto TokenizedBuffer::Print(llvm::raw_ostream& output_stream) const -> void {
  if (Tokens().begin() == Tokens().end()) {
    return;
  }

  PrintWidths widths = {};
  widths.index = ComputeDecimalPrintedWidth((token_infos.size()));
  for (Token token : Tokens()) {
    widths.Widen(GetTokenPrintWidths(token));
  }

  for (Token token : Tokens()) {
    PrintToken(output_stream, token, widths);
    output_stream << "\n";
  }
}

auto TokenizedBuffer::PrintToken(llvm::raw_ostream& output_stream,
                                 Token token) const -> void {
  PrintToken(output_stream, token, {});
}

auto TokenizedBuffer::PrintToken(llvm::raw_ostream& output_stream, Token token,
                                 PrintWidths widths) const -> void {
  widths.Widen(GetTokenPrintWidths(token));
  int token_index = token.index;
  auto& token_info = GetTokenInfo(token);
  llvm::StringRef token_text = GetTokenText(token);

  // Output the main chunk using one format string. We have to do the
  // justification manually in order to use the dynamically computed widths
  // and get the quotes included.
  output_stream << llvm::formatv(
      "token: { index: {0}, kind: {1}, line: {2}, column: {3}, indent: {4}, "
      "spelling: '{5}'",
      llvm::format_decimal(token_index, widths.index),
      llvm::right_justify(
          (llvm::Twine("'") + token_info.kind.Name() + "'").str(),
          widths.kind + 2),
      llvm::format_decimal(GetLineNumber(token_info.token_line), widths.line),
      llvm::format_decimal(GetColumnNumber(token), widths.column),
      llvm::format_decimal(GetIndentColumnNumber(token_info.token_line),
                           widths.indent),
      token_text);

  if (token_info.kind == TokenKind::Identifier()) {
    output_stream << ", identifier: " << GetIdentifier(token).index;
  } else if (token_info.kind.IsOpeningSymbol()) {
    output_stream << ", closing_token: " << GetMatchedClosingToken(token).index;
  } else if (token_info.kind.IsClosingSymbol()) {
    output_stream << ", opening_token: " << GetMatchedOpeningToken(token).index;
  }

  if (token_info.is_recovery) {
    output_stream << ", recovery: true";
  }

  output_stream << " }";
}

auto TokenizedBuffer::GetLineInfo(Line line) -> LineInfo& {
  return line_infos[line.index];
}

auto TokenizedBuffer::GetLineInfo(Line line) const -> const LineInfo& {
  return line_infos[line.index];
}

auto TokenizedBuffer::AddLine(LineInfo info) -> Line {
  line_infos.push_back(info);
  return Line(static_cast<int>(line_infos.size()) - 1);
}

auto TokenizedBuffer::GetTokenInfo(Token token) -> TokenInfo& {
  return token_infos[token.index];
}

auto TokenizedBuffer::GetTokenInfo(Token token) const -> const TokenInfo& {
  return token_infos[token.index];
}

auto TokenizedBuffer::AddToken(TokenInfo info) -> Token {
  token_infos.push_back(info);
  return Token(static_cast<int>(token_infos.size()) - 1);
}

}  // namespace Carbon

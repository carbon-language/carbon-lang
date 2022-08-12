// Copyright 2003-2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: Package name conflicts with member class RE2!
package RE2 api;

// C++ interface to the re2 regular-expression library.
// RE2 supports Perl-style regular expressions (with extensions like
// \d, \w, \s, ...).
//
// -----------------------------------------------------------------------
// REGEXP SYNTAX:
//
// This module uses the re2 library and hence supports
// its syntax for regular expressions, which is similar to Perl's with
// some of the more complicated things thrown away.  In particular,
// backreferences and generalized assertions are not available, nor is \Z.
//
// See https://github.com/google/re2/wiki/Syntax for the syntax
// supported by RE2, and a comparison with PCRE and PERL regexps.
//
// For those not familiar with Perl's regular expressions,
// here are some examples of the most commonly used extensions:
//
//   "hello (\\w+) world"  -- \w matches a "word" character
//   "version (\\d+)"      -- \d matches a digit
//   "hello\\s+world"      -- \s matches any whitespace character
//   "\\b(\\w+)\\b"        -- \b matches non-empty string at word boundary
//   "(?i)hello"           -- (?i) turns on case-insensitive matching
//   "/\\*(.*?)\\*/"       -- .*? matches . minimum no. of times possible
//
// The double backslashes are needed when writing C++ string literals.
// However, they should NOT be used when writing C++11 raw string literals:
//
//   R"(hello (\w+) world)"  -- \w matches a "word" character
//   R"(version (\d+))"      -- \d matches a digit
//   R"(hello\s+world)"      -- \s matches any whitespace character
//   R"(\b(\w+)\b)"          -- \b matches non-empty string at word boundary
//   R"((?i)hello)"          -- (?i) turns on case-insensitive matching
//   R"(/\*(.*?)\*/)"        -- .*? matches . minimum no. of times possible
//
// When using UTF-8 encoding, case-insensitive matching will perform
// simple case folding, not full case folding.
//
// -----------------------------------------------------------------------
// MATCHING INTERFACE:
//
// The "FullMatch" operation checks that supplied text matches a
// supplied pattern exactly.
//
// Example: successful match
//    CHECK(RE2::FullMatch("hello", "h.*o"));
//
// Example: unsuccessful match (requires full match):
//    CHECK(!RE2::FullMatch("hello", "e"));
//
// -----------------------------------------------------------------------
// UTF-8 AND THE MATCHING INTERFACE:
//
// By default, the pattern and input text are interpreted as UTF-8.
// The RE2::Latin1 option causes them to be interpreted as Latin-1.
//
// Example:
//    CHECK(RE2::FullMatch(utf8_string, RE2(utf8_pattern)));
//    CHECK(RE2::FullMatch(latin1_string, RE2(latin1_pattern, RE2::Latin1)));
//
// -----------------------------------------------------------------------
// MATCHING WITH SUBSTRING EXTRACTION:
//
// You can supply extra pointer arguments to extract matched substrings.
// On match failure, none of the pointees will have been modified.
// On match success, the substrings will be converted (as necessary) and
// their values will be assigned to their pointees until all conversions
// have succeeded or one conversion has failed.
// On conversion failure, the pointees will be in an indeterminate state
// because the caller has no way of knowing which conversion failed.
// However, conversion cannot fail for types like string and StringPiece
// that do not inspect the substring contents. Hence, in the common case
// where all of the pointees are of such types, failure is always due to
// match failure and thus none of the pointees will have been modified.
//
// Example: extracts "ruby" into "s" and 1234 into "i"
//    int i;
//    std::string s;
//    CHECK(RE2::FullMatch("ruby:1234", "(\\w+):(\\d+)", &s, &i));
//
// Example: fails because string cannot be stored in integer
//    CHECK(!RE2::FullMatch("ruby", "(.*)", &i));
//
// Example: fails because there aren't enough sub-patterns
//    CHECK(!RE2::FullMatch("ruby:1234", "\\w+:\\d+", &s));
//
// Example: does not try to extract any extra sub-patterns
//    CHECK(RE2::FullMatch("ruby:1234", "(\\w+):(\\d+)", &s));
//
// Example: does not try to extract into NULL
//    CHECK(RE2::FullMatch("ruby:1234", "(\\w+):(\\d+)", NULL, &i));
//
// Example: integer overflow causes failure
//    CHECK(!RE2::FullMatch("ruby:1234567891234", "\\w+:(\\d+)", &i));
//
// NOTE(rsc): Asking for substrings slows successful matches quite a bit.
// This may get a little faster in the future, but right now is slower
// than PCRE.  On the other hand, failed matches run *very* fast (faster
// than PCRE), as do matches without substring extraction.
//
// -----------------------------------------------------------------------
// PARTIAL MATCHES
//
// You can use the "PartialMatch" operation when you want the pattern
// to match any substring of the text.
//
// Example: simple search for a string:
//      CHECK(RE2::PartialMatch("hello", "ell"));
//
// Example: find first number in a string
//      int number;
//      CHECK(RE2::PartialMatch("x*100 + 20", "(\\d+)", &number));
//      CHECK_EQ(number, 100);
//
// -----------------------------------------------------------------------
// PRE-COMPILED REGULAR EXPRESSIONS
//
// RE2 makes it easy to use any string as a regular expression, without
// requiring a separate compilation step.
//
// If speed is of the essence, you can create a pre-compiled "RE2"
// object from the pattern and use it multiple times.  If you do so,
// you can typically parse text faster than with sscanf.
//
// Example: precompile pattern for faster matching:
//    RE2 pattern("h.*o");
//    while (ReadLine(&str)) {
//      if (RE2::FullMatch(str, pattern)) ...;
//    }
//
// -----------------------------------------------------------------------
// SCANNING TEXT INCREMENTALLY
//
// The "Consume" operation may be useful if you want to repeatedly
// match regular expressions at the front of a string and skip over
// them as they match.  This requires use of the "StringPiece" type,
// which represents a sub-range of a real string.
//
// Example: read lines of the form "var = value" from a string.
//      std::string contents = ...;     // Fill string somehow
//      StringPiece input(contents);    // Wrap a StringPiece around it
//
//      std::string var;
//      int value;
//      while (RE2::Consume(&input, "(\\w+) = (\\d+)\n", &var, &value)) {
//        ...;
//      }
//
// Each successful call to "Consume" will set "var/value", and also
// advance "input" so it points past the matched text.  Note that if the
// regular expression matches an empty string, input will advance
// by 0 bytes.  If the regular expression being used might match
// an empty string, the loop body must check for this case and either
// advance the string or break out of the loop.
//
// The "FindAndConsume" operation is similar to "Consume" but does not
// anchor your match at the beginning of the string.  For example, you
// could extract all words from a string by repeatedly calling
//     RE2::FindAndConsume(&input, "(\\w+)", &word)
//
// -----------------------------------------------------------------------
// USING VARIABLE NUMBER OF ARGUMENTS
//
// The above operations require you to know the number of arguments
// when you write the code.  This is not always possible or easy (for
// example, the regular expression may be calculated at run time).
// You can use the "N" version of the operations when the number of
// match arguments are determined at run time.
//
// Example:
//   const RE2::Arg* args[10];
//   int n;
//   // ... populate args with pointers to RE2::Arg values ...
//   // ... set n to the number of RE2::Arg objects ...
//   bool match = RE2::FullMatchN(input, pattern, args, n);
//
// The last statement is equivalent to
//
//   bool match = RE2::FullMatch(input, pattern,
//                               *args[0], *args[1], ..., *args[n - 1]);
//
// -----------------------------------------------------------------------
// PARSING HEX/OCTAL/C-RADIX NUMBERS
//
// By default, if you pass a pointer to a numeric value, the
// corresponding text is interpreted as a base-10 number.  You can
// instead wrap the pointer with a call to one of the operators Hex(),
// Octal(), or CRadix() to interpret the text in another base.  The
// CRadix operator interprets C-style "0" (base-8) and "0x" (base-16)
// prefixes, but defaults to base-10.
//
// Example:
//   int a, b, c, d;
//   CHECK(RE2::FullMatch("100 40 0100 0x40", "(.*) (.*) (.*) (.*)",
//         RE2::Octal(&a), RE2::Hex(&b), RE2::CRadix(&c), RE2::CRadix(&d));
// will leave 64 in a, b, c, and d.

import Cpp library "<algorithm>";
import Cpp library "<map>";
import Cpp library "<mutex>";
import Cpp library "<vector>";

// TODO: How to express target-specific conditional compilation?
// TODO: #if defined(__APPLE__)
// TODO: #include <TargetConditionals.h>
// TODO: #endif

// TODO: How to forward declare classes from another library?
// Is a physical dependency on the library required?
// TODO: namespace re2 {
// TODO: class Prog;
// TODO: class Regexp;
// TODO: }  // namespace re2

private interface Parse4ary;

// Interface for regular expression matching.  Also corresponds to a
// pre-compiled regular expression.  An "RE2" object is safe for
// concurrent use by multiple threads.
class RE2 {
  // We convert user-passed pointers into special Arg objects
  class Arg;
  class Options;

  // Defined in set.h.
  class Set;

  // TODO: Assuming a C++-like enum syntax for now.
  enum ErrorCode {
    NoError = 0,

    // Unexpected error
    ErrorInternal,

    // Parse errors
    // bad escape sequence
    ErrorBadEscape,
    // bad character class
    ErrorBadCharClass,
    // bad character class range
    ErrorBadCharRange,
    // missing closing ]
    ErrorMissingBracket,
    // missing closing )
    ErrorMissingParen,
    // unexpected closing )
    ErrorUnexpectedParen,
    // trailing \ at end of regexp
    ErrorTrailingBackslash,
    // repeat argument missing, e.g. "*"
    ErrorRepeatArgument,
    // bad repetition argument
    ErrorRepeatSize,
    // bad repetition operator
    ErrorRepeatOp,
    // bad perl operator
    ErrorBadPerlOp,
    // invalid UTF-8 in regexp
    ErrorBadUTF8,
    // bad named capture group
    ErrorBadNamedCapture,
    // pattern too large (compile failed)
    ErrorPatternTooLarge
  }

  // Predefined common options.
  // If you need more complicated things, instantiate
  // an Option class, possibly passing one of these to
  // the Option constructor, change the settings, and pass that
  // Option class to the RE2 constructor.
  enum CannedOptions {
    DefaultOptions = 0,
    // treat input as Latin-1 (default UTF-8)
    Latin1,
    // POSIX syntax, leftmost-longest match
    POSIX,
    // do not log about regexp parse errors
    Quiet
  }

  fn Make(pattern: StringPiece) -> RE2;
  fn Make(pattern: StringPiece, options: Options) -> RE2;

  // TODO: Should a Carbonic RE2 support these?
  impl StringView as ImplicitAs(RE2) {
    fn Convert[me: Self]() -> RE2 { return Make(me); }
  }
  impl String as ImplicitAs(RE2) {
    fn Convert[me: Self]() -> RE2 { return Make(me); }
  }
  impl StringPiece as ImplicitAs(RE2) {
    fn Convert[me: Self]() -> RE2 { return Make(me); }
  }

  impl as Destroyable;

  // Returns whether RE2 was created properly.
  fn ok[me: Self]() -> bool { return me.error_code() == ErrorCode.NoError; }

  // The string specification for this RE2.  E.g.
  //   RE2 re("ab*c?d+");
  //   re.pattern();    // "ab*c?d+"
  fn pattern[me: Self]() -> String { return me.pattern_; }

  // If RE2 could not be created properly, returns an error string.
  // Else returns the empty string.
  fn error[me: Self]() -> String { return *me.error_; }

  // If RE2 could not be created properly, returns an error code.
  // Else returns RE2::NoError (== 0).
  fn error_code[me: Self]() -> ErrorCode { return me.error_code_; }

  // If RE2 could not be created properly, returns the offending
  // portion of the regexp.
  fn error_arg[me: Self]() -> String { return me.error_arg_; }

  // Returns the program size, a very approximate measure of a regexp's "cost".
  // Larger numbers are more expensive than smaller numbers.
  fn ProgramSize[me: Self]() -> i32;
  fn ReverseProgramSize[me: Self]() -> i32;

  // If histogram is not null, outputs the program fanout
  // as a histogram bucketed by powers of 2.
  // Returns the number of the largest non-empty bucket.
  fn ProgramFanout[me: Self](histogram: Cpp.std.vector(i32)*) -> i32;
  fn ReverseProgramFanout[me: Self](histogram: Cpp.std.vector(i32)*) -> i32;

  // Returns the underlying Regexp; not for general use.
  // Returns entire_regexp_ so that callers don't need
  // to know about prefix_ and prefix_foldcase_.
  fn Regexp[me: Self]() -> package.Regexp* { return me.entire_regexp_; }

  /***** The array-based matching interface ******/

  // The functions here have names ending in 'N' and are used to implement
  // the functions whose names are the prefix before the 'N'. It is sometimes
  // useful to invoke them directly, but the syntax is awkward, so the 'N'-less
  // versions should be preferred.
  // TODO: pointer with const pointee
  fn FullMatchN(text: StringPiece, re: Self,
                args: Array(const Arg*), n: i32) -> bool;
  fn PartialMatchN(text: StringPiece, re: Self,
                   args: Array(const Arg*), n: i32) -> bool;
  fn ConsumeN(input: StringPiece*, re: Self,
              args: Array(const Arg*), n: i32) -> bool;
  fn FindAndConsumeN(input: StringPiece*, re: RE2,
                     args: Array(const Arg*), n: i32) -> bool;

  private fn Apply[template F:! Type, SP:! Type](f: F, sp: SP, re: Self) {
    return f(sp, re, nullptr, 0);
  }

  // TODO: (variadics)
  // TODO: template <typename F, typename SP, typename... A>
  // TODO: static inline bool Apply(F f, SP sp, const RE2& re, const A&... a) {
  // TODO:   const Arg* const args[] = {&a...};
  // TODO:   const int n = sizeof...(a);
  // TODO:   return f(sp, re, args, n);
  // TODO: }

  // In order to allow FullMatch() et al. to be called with a varying number
  // of arguments of varying types, we use two layers of variadic templates.
  // The first layer constructs the temporary Arg objects. The second layer
  // (above) constructs the array of pointers to the temporary Arg objects.

  /***** The useful part: the matching interface *****/

  // Matches "text" against "re".  If pointer arguments are
  // supplied, copies matched sub-patterns into them.
  //
  // You can pass in a "const char*" or a "std::string" for "text".
  // You can pass in a "const char*" or a "std::string" or a "RE2" for "re".
  //
  // The provided pointer arguments can be pointers to any scalar numeric
  // type, or one of:
  //    std::string     (matched piece is copied to string)
  //    StringPiece     (StringPiece is mutated to point to matched piece)
  //    T               (where "bool T::ParseFrom(const char*, size_t)" exists)
  //    (void*)NULL     (the corresponding matched sub-pattern is not copied)
  //
  // Returns true iff all of the following conditions are satisfied:
  //   a. "text" matches "re" fully - from the beginning to the end of "text".
  //   b. The number of matched sub-patterns is >= number of supplied pointers.
  //   c. The "i"th argument has a suitable type for holding the
  //      string captured as the "i"th sub-pattern.  If you pass in
  //      NULL for the "i"th argument, or pass fewer arguments than
  //      number of sub-patterns, the "i"th captured sub-pattern is
  //      ignored.
  //
  // CAVEAT: An optional sub-pattern that does not exist in the
  // matched string is assigned the empty string.  Therefore, the
  // following will return false (because the empty string is not a
  // valid number):
  //    int number;
  //    RE2::FullMatch("abc", "[a-z]+(\\d+)?", &number);
  fn FullMatch(text: StringPiece, re: Self) -> bool {
    return Apply(FullMatchN, text, re);
  }
  // TODO: template <typename... A>
  // TODO: static bool FullMatch(const StringPiece& text, const RE2& re, A&&... a) {
  // TODO:   return Apply(FullMatchN, text, re, Arg(std::forward<A>(a))...);
  // TODO: }

  // Like FullMatch(), except that "re" is allowed to match a substring
  // of "text".
  //
  // Returns true iff all of the following conditions are satisfied:
  //   a. "text" matches "re" partially - for some substring of "text".
  //   b. The number of matched sub-patterns is >= number of supplied pointers.
  //   c. The "i"th argument has a suitable type for holding the
  //      string captured as the "i"th sub-pattern.  If you pass in
  //      NULL for the "i"th argument, or pass fewer arguments than
  //      number of sub-patterns, the "i"th captured sub-pattern is
  //      ignored.
  fn PartialMatch(text: StringPiece, re: Self) -> bool {
    return Apply(PartialMatchN, text, re);
  }
  // TODO: template <typename... A>
  // TODO: static bool PartialMatch(const StringPiece& text, const RE2& re, A&&... a) {
  // TODO:   return Apply(PartialMatchN, text, re, Arg(std::forward<A>(a))...);
  // TODO: }

  // Like FullMatch() and PartialMatch(), except that "re" has to match
  // a prefix of the text, and "input" is advanced past the matched
  // text.  Note: "input" is modified iff this routine returns true
  // and "re" matched a non-empty substring of "input".
  //
  // Returns true iff all of the following conditions are satisfied:
  //   a. "input" matches "re" partially - for some prefix of "input".
  //   b. The number of matched sub-patterns is >= number of supplied pointers.
  //   c. The "i"th argument has a suitable type for holding the
  //      string captured as the "i"th sub-pattern.  If you pass in
  //      NULL for the "i"th argument, or pass fewer arguments than
  //      number of sub-patterns, the "i"th captured sub-pattern is
  //      ignored.
  fn Consume(input: StringPiece*, re: Self) {
    return Apply(ConsumeN, input, re);
  }
  // TODO: template <typename... A>
  // TODO: static bool Consume(StringPiece* input, const RE2& re, A&&... a) {
  // TODO:   return Apply(ConsumeN, input, re, Arg(std::forward<A>(a))...);
  // TODO: }

  // Like Consume(), but does not anchor the match at the beginning of
  // the text.  That is, "re" need not start its match at the beginning
  // of "input".  For example, "FindAndConsume(s, "(\\w+)", &word)" finds
  // the next word in "s" and stores it in "word".
  //
  // Returns true iff all of the following conditions are satisfied:
  //   a. "input" matches "re" partially - for some substring of "input".
  //   b. The number of matched sub-patterns is >= number of supplied pointers.
  //   c. The "i"th argument has a suitable type for holding the
  //      string captured as the "i"th sub-pattern.  If you pass in
  //      NULL for the "i"th argument, or pass fewer arguments than
  //      number of sub-patterns, the "i"th captured sub-pattern is
  //      ignored.
  fn FindAndConsume(input: StringPiece*, re: Self) {
    return Apply(FindAndConsumeN, input, re);
  }
  // TODO: template <typename... A>
  // TODO: static bool FindAndConsume(StringPiece* input, const RE2& re, A&&... a) {
  // TODO:   return Apply(FindAndConsumeN, input, re, Arg(std::forward<A>(a))...);
  // TODO: }

  // Replace the first match of "re" in "str" with "rewrite".
  // Within "rewrite", backslash-escaped digits (\1 to \9) can be
  // used to insert text matching corresponding parenthesized group
  // from the pattern.  \0 in "rewrite" refers to the entire matching
  // text.  E.g.,
  //
  //   std::string s = "yabba dabba doo";
  //   CHECK(RE2::Replace(&s, "b+", "d"));
  //
  // will leave "s" containing "yada dabba doo"
  //
  // Returns true if the pattern matches and a replacement occurs,
  // false otherwise.
  fn Replace(str: String*, re: Self, rewrite: StringPiece) -> bool;

  // Like Replace(), except replaces successive non-overlapping occurrences
  // of the pattern in the string with the rewrite. E.g.
  //
  //   std::string s = "yabba dabba doo";
  //   CHECK(RE2::GlobalReplace(&s, "b+", "d"));
  //
  // will leave "s" containing "yada dada doo"
  // Replacements are not subject to re-matching.
  //
  // Because GlobalReplace only replaces non-overlapping matches,
  // replacing "ana" within "banana" makes only one replacement, not two.
  //
  // Returns the number of replacements made.
  fn GlobalReplace(str: String*, re: Self, rewrite: StringPiece) -> i32;

  // Like Replace, except that if the pattern matches, "rewrite"
  // is copied into "out" with substitutions.  The non-matching
  // portions of "text" are ignored.
  //
  // Returns true iff a match occurred and the extraction happened
  // successfully;  if no match occurs, the string is left unaffected.
  //
  // REQUIRES: "text" must not alias any part of "*out".
  fn Extract(text: StringPiece,
             re: Self,
             rewrite: StringPiece,
             out: String*)
    -> bool;

  // Escapes all potentially meaningful regexp characters in
  // 'unquoted'.  The returned string, used as a regular expression,
  // will match exactly the original string.  For example,
  //           1.5-2.0?
  // may become:
  //           1\.5\-2\.0\?
  fn QuoteMeta(unquoted: StringPiece) -> String;

  // Computes range for any strings matching regexp. The min and max can in
  // some cases be arbitrarily precise, so the caller gets to specify the
  // maximum desired length of string returned.
  //
  // Assuming PossibleMatchRange(&min, &max, N) returns successfully, any
  // string s that is an anchored match for this regexp satisfies
  //   min <= s && s <= max.
  //
  // Note that PossibleMatchRange() will only consider the first copy of an
  // infinitely repeated element (i.e., any regexp element followed by a '*' or
  // '+' operator). Regexps with "{N}" constructions are not affected, as those
  // do not compile down to infinite repetitions.
  //
  // Returns true on success, false on error.
  fn PossibleMatchRange[me: Self](min: String*, max: String*, maxlen: i32);

  // Generic matching interface

  // Type of match.
  enum Anchor {
    // No anchoring
    UNANCHORED,
    // Anchor at start only
    ANCHOR_START,
    // Anchor at start and end
    ANCHOR_BOTH
  }

  // Return the number of capturing subpatterns, or -1 if the
  // regexp wasn't valid on construction.  The overall match ($0)
  // does not count: if the regexp is "(a)(b)", returns 2.
  fn NumberOfCapturingGroups[me: Self]() -> i32 { return me.num_captures_; }

  // Return a map from names to capturing indices.
  // The map records the index of the leftmost group
  // with the given name.
  // NOTE: Originally returned by reference with comment "valid until re is deleted".
  fn NamedCapturingGroups[me: Self]() -> Map(String, i32);

  // Return a map from capturing indices to names.
  // The map has no entries for unnamed groups.
  // NOTE: Originally returned by reference with comment "valid until re is deleted".
  fn CapturingGroupNames[me: Self]() -> Map(i32, String);

  // General matching routine.
  // Match against text starting at offset startpos
  // and stopping the search at offset endpos.
  // Returns true if match found, false if not.
  // On a successful match, fills in submatch[] (up to nsubmatch entries)
  // with information about submatches.
  // I.e. matching RE2("(foo)|(bar)baz") on "barbazbla" will return true, with
  // submatch[0] = "barbaz", submatch[1].data() = NULL, submatch[2] = "bar",
  // submatch[3].data() = NULL, ..., up to submatch[nsubmatch-1].data() = NULL.
  // Caveat: submatch[] may be clobbered even on match failure.
  //
  // Don't ask for more match information than you will use:
  // runs much faster with nsubmatch == 1 than nsubmatch > 1, and
  // runs even faster if nsubmatch == 0.
  // Doesn't make sense to use nsubmatch > 1 + NumberOfCapturingGroups(),
  // but will be handled correctly.
  //
  // Passing text == StringPiece(NULL, 0) will be handled like any other
  // empty string, but note that on return, it will not be possible to tell
  // whether submatch i matched the empty string or did not match:
  // either way, submatch[i].data() == NULL.
  fn Match[me: Self](text: StringPiece,
                     startpos: i64,
                     endpos: i64,
                     re_anchor: Anchor,
                     submatch: ArrayIterator(StringPiece),
                     nsubmatch: i32)
    -> bool;

  // Check that the given rewrite string is suitable for use with this
  // regular expression.  It checks that:
  //   * The regular expression has enough parenthesized subexpressions
  //     to satisfy all of the \N tokens in rewrite
  //   * The rewrite string doesn't have any syntax errors.  E.g.,
  //     '\' followed by anything other than a digit or '\'.
  // A true return value guarantees that Replace() and Extract() won't
  // fail because of a bad rewrite string.
  fn CheckRewriteString[me: Self](rewrite: StringPiece, error: String*) -> bool;

  // Returns the maximum submatch needed for the rewrite to be done by
  // Replace(). E.g. if rewrite == "foo \\2,\\1", returns 2.
  fn MaxSubmatch(rewrite: StringPiece) -> i32;

  // Append the "rewrite" string, with backslash substitutions from "vec",
  // to string "out".
  // Returns true on success.  This method can fail because of a malformed
  // rewrite string.  CheckRewriteString guarantees that the rewrite will
  // be successful.
  fn Rewrite[me: Self](out: String*, rewrite: StringPiece,
                       vec: ArrayIterator(StringPiece), veclen: i32)
    -> bool;

  // Constructor options
  class Options {
    // The options are (defaults in parentheses):
    //
    //   utf8             (true)  text and pattern are UTF-8; otherwise Latin-1
    //   posix_syntax     (false) restrict regexps to POSIX egrep syntax
    //   longest_match    (false) search for longest match, not first match
    //   log_errors       (true)  log syntax and execution errors to ERROR
    //   max_mem          (see below)  approx. max memory footprint of RE2
    //   literal          (false) interpret string as literal, not regexp
    //   never_nl         (false) never match \n, even if it is in regexp
    //   dot_nl           (false) dot matches everything including new line
    //   never_capture    (false) parse all parens as non-capturing
    //   case_sensitive   (true)  match is case-sensitive (regexp can override
    //                              with (?i) unless in posix_syntax mode)
    //
    // The following options are only consulted when posix_syntax == true.
    // When posix_syntax == false, these features are always enabled and
    // cannot be turned off; to perform multi-line matching in that case,
    // begin the regexp with (?m).
    //   perl_classes     (false) allow Perl's \d \s \w \D \S \W
    //   word_boundary    (false) allow Perl's \b \B (word boundary and not)
    //   one_line         (false) ^ and $ only match beginning and end of text
    //
    // The max_mem option controls how much memory can be used
    // to hold the compiled form of the regexp (the Prog) and
    // its cached DFA graphs.  Code Search placed limits on the number
    // of Prog instructions and DFA states: 10,000 for both.
    // In RE2, those limits would translate to about 240 KB per Prog
    // and perhaps 2.5 MB per DFA (DFA state sizes vary by regexp; RE2 does a
    // better job of keeping them small than Code Search did).
    // Each RE2 has two Progs (one forward, one reverse), and each Prog
    // can have two DFAs (one first match, one longest match).
    // That makes 4 DFAs:
    //
    //   forward, first-match    - used for UNANCHORED or ANCHOR_START searches
    //                               if opt.longest_match() == false
    //   forward, longest-match  - used for all ANCHOR_BOTH searches,
    //                               and the other two kinds if
    //                               opt.longest_match() == true
    //   reverse, first-match    - never used
    //   reverse, longest-match  - used as second phase for unanchored searches
    //
    // The RE2 memory budget is statically divided between the two
    // Progs and then the DFAs: two thirds to the forward Prog
    // and one third to the reverse Prog.  The forward Prog gives half
    // of what it has left over to each of its DFAs.  The reverse Prog
    // gives it all to its longest-match DFA.
    //
    // Once a DFA fills its budget, it flushes its cache and starts over.
    // If this happens too often, RE2 falls back on the NFA implementation.

    // For now, make the default budget something close to Code Search.
    // TODO: How to define a class-scope constant?
    let kDefaultMaxMem:! i32 = 8 << 20;

    enum Encoding {
      EncodingUTF8 = 1,
      EncodingLatin1
    }

    // TODO: A `;` after this would be nicer than a `{}`.
    impl as DefaultValue where .Value = {
      .encoding_ = EncodingUTF8,
      .posix_syntax_ = false,
      .longest_match_ = false,
      .log_errors_ = true,
      .max_mem_ = kDefaultMaxMem,
      .literal_ = false,
      .never_nl_ = false,
      .dot_nl_ = false,
      .never_capture_ = false,
      .case_sensitive_ = true,
      .perl_classes_ = false,
      .word_boundary_ = false,
      .one_line_ = false} {}

    impl CannedOptions as ImplicitAs(Self);

    fn encoding[me: Self]() -> Encoding { return me.encoding_; }
    fn set_encoding[addr me: Self*](encoding: Encoding) { me->encoding_ = encoding; }

    fn posix_syntax[me: Self]() -> bool { return me.posix_syntax_; }
    fn set_posix_syntax[addr me: Self*](b: bool) { me->posix_syntax_ = b; }

    fn longest_match[me: Self]() -> bool { return me.longest_match_; }
    fn set_longest_match[addr me: Self*](b: bool) { me->longest_match_ = b; }

    fn log_errors[me: Self]() -> bool { return me.log_errors_; }
    fn set_log_errors[addr me: Self*](b: bool) { me->log_errors_ = b; }

    fn max_mem[me: Self]() -> i64 { return me.max_mem_; }
    fn set_max_mem[addr me: Self*](m: i64) { me->max_mem_ = m; }

    fn literal[me: Self]() -> bool { return me.literal_; }
    fn set_literal[addr me: Self*](b: bool) { me->literal_ = b; }

    fn never_nl[me: Self]() -> bool { return me.never_nl_; }
    fn set_never_nl[addr me: Self*](b: bool) { me->never_nl_ = b; }

    fn dot_nl[me: Self]() -> bool { return me.dot_nl_; }
    fn set_dot_nl[addr me: Self*](b: bool) { me->dot_nl_ = b; }

    fn never_capture[me: Self]() -> bool { return me.never_capture_; }
    fn set_never_capture[addr me: Self*](b: bool) { me->never_capture_ = b; }

    fn case_sensitive[me: Self]() -> bool { return me.case_sensitive_; }
    fn set_case_sensitive[addr me: Self*](b: bool) { me->case_sensitive_ = b; }

    fn perl_classes[me: Self]() -> bool { return me.perl_classes_; }
    fn set_perl_classes[addr me: Self*](b: bool) { me->perl_classes_ = b; }

    fn word_boundary[me: Self]() -> bool { return me.word_boundary_; }
    fn set_word_boundary[addr me: Self*](b: bool) { me->word_boundary_ = b; }

    fn one_line[me: Self]() -> bool { return me.one_line_; }
    fn set_one_line[addr me: Self*](b: bool) { me->one_line_ = b; }

    fn Copy[addr me: Self*](src: Options) {
      *me = src;
    }

    fn ParseFlags[me: Self]() -> i32;

    private var encoding_: Encoding;
    private var posix_syntax_: bool;
    private var longest_match_: bool;
    private var log_errors_: bool;
    private var max_mem_: i64;
    private var literal_: bool;
    private var never_nl_: bool;
    private var dot_nl_: bool;
    private var never_capture_: bool;
    private var case_sensitive_: bool;
    private var perl_classes_: bool;
    private var word_boundary_: bool;
    private var one_line_: bool;
  };

  // Returns the options set in the constructor.
  fn options[me: Self]() -> Options { return me.options_; }

  // Argument converters; see below.
  // TODO: Should these be package members not class members in Carbon
  // so you use `RE2.Hex` not `RE2.RE2.Hex`?
  fn CRadix[T:! Parse4ary](ptr: T*) -> Self.Arg;
  fn Hex[T:! Parse4ary](ptr: T*) -> Self.Arg;
  fn Octal[T:! Parse4ary](ptr: T*) -> Self.Arg;

  private fn Init[addr me: Self](pattern: StringPiece, options: Options);

  private fn DoMatch[me: Self](text: StringPiece,
                               re_anchor: Anchor,
                               consumed: i64*,
                               // TODO: Pointer to `const Arg`.
                               args: Array(Arg*),
                               n: i32)
    -> bool;

  fn ReverseProg[me: Self]() -> package.Prog*;

  // string regular expression
  private var pattern_: String;
  // option flags
  private var options_: Options;
  // parsed regular expression
  private var entire_regexp_: package.Regexp*;
  // error indicator (or points to empty string)
  // TODO: pointer to `const String`
  private var error_: String*;
  // error code
  private var error_code_: ErrorCode;
  // fragment of regexp showing error
  private var error_arg_: String;
  // required prefix (before suffix_regexp_)
  private var prefix_: String;
  // prefix_ is ASCII case-insensitive
  private var prefix_foldcase_: bool;
  // parsed regular expression, prefix_ removed
  private var suffix_regexp_: package.Regexp*;
  // compiled program for regexp
  private var prog_: package.Prog*;
  // number of capturing groups
  private var num_captures_: i32;
  // can use prog_->SearchOnePass?
  private var is_one_pass_: bool;

  // TODO: Rest of the member variables are mutable.

  // Reverse Prog for DFA execution only
  private var rprog_: package.Prog*;
  // Map from capture names to indices
  // TODO: pointer to const map
  private var named_groups_: Map(String, i32)*;
  // Map from capture indices to names
  // TODO: pointer to const map
  private var group_names_: Map(i32, String)*;

  private var rprog_once_: Cpp.std.once_flag;
  private var named_groups_once_: Cpp.std.once_flag;
  private var group_names_once_: Cpp.std.once_flag;
};

/***** Implementation details *****/

private interface Parse3ary {
  fn Parse(str: StringView, n: i64, dest: Self*) -> bool;
}
impl void as Parse3ary;
impl String as Parse3ary;
impl StringPiece as Parse3ary;
impl Char as Parse3ary;
impl f32 as Parse3ary;
impl f64 as Parse3ary;

private interface Parse4ary {
  fn Parse(str: StringView, n: i64, dest: Self*, radix: i32) -> bool;
}
impl i16 as Parse4ary;
impl u16 as Parse4ary;
impl i32 as Parse4ary;
impl u32 as Parse4ary;
impl i64 as Parse4ary;
impl u64 as Parse4ary;

interface ParseFrom {
  fn Parse(str: StringView, n: i64) -> bool;
}

class RE2.Arg {
  fn Make() -> Self { return Make(nullptr); }
  // TODO: Can we put an irrefutable pattern here?
  // TODO: Is 'nullptr' an irrefutable pattern of type nullptr_t (whatever we call that)?
  fn Make(nullptr) -> Self { return Make(nullptr as NullArg*); }

  interface Parseable {
    fn Parse[addr me: Self*](str: StringView, n: i64) -> bool;
  }
  match_first {
    impl [T:! Parse3ary] T as Parseable {
      fn Parse[addr me: Self*](str: StringView, n: i64) -> bool {
        return T.Parse(str, n, me);
      }
    }
    impl [T:! Parse4ary] T as Parseable {
      fn Parse[addr me: Self*](str: StringView, n: i64) -> bool {
        return T.Parse(str, n, me, 10);
      }
    }
    impl [T:! ParseFrom] T as Parseable {
      fn Parse[addr me: Self*](str: StringView, n: i64) -> bool {
        if (me == nullptr) { return true; }
        return T.Parse(str, n, me);
      }
    }
  }

  private class NullArg {}
  impl NullArg as Parseable {
    fn Parse[addr me: Self*](str: StringView, n: i64) -> bool {
      return true;
    }
  }

  fn Make[T:! Parseable](ptr: T*) {
    return {.type_ = T, .arg_ = ptr};
  }

  fn Parse[me: Self](str: StringView, n: i64) -> bool {
    return me.arg_->Parse(str, n);
  }

  // TODO: Existential types or `DynPtr(Parseable)`.
  private let type_: Parseable;
  private var arg_: Nullable(type_*);
}

private adapter ParseAsBase(T:! Parse4ary, base: i32) for T {
  impl as Self.Arg.Parseable {
    fn Parse[addr me: Self*](str: StringView, n: i64) -> bool {
      return T.Parse(str, n, me, base);
    }
  }
}

fn RE2.CRadix[T:! Parse4ary](ptr: T*) -> Self.Arg {
  return Self.Arg.Make(ptr as ParseAsBase(T, 0)*);
}

fn RE2.Hex[T:! Parse4ary](ptr: T*) -> Self.Arg {
  return Self.Arg.Make(ptr as ParseAsBase(T, 16)*);
}

fn RE2.Octal[T:! Parse4ary](ptr: T*) -> Self.Arg {
  return Self.Arg.Make(ptr as ParseAsBase(T, 8)*);
}

// Helper for writing global or static RE2s safely.
// Write
//     static LazyRE2 re = {".*"};
// and then use *re instead of writing
//     static RE2 re(".*");
// The former is more careful about multithreaded
// situations than the latter.
//
// N.B. This class never deletes the RE2 object that
// it constructs: that's a feature, so that it can be used
// for global and function static variables.
class LazyRE2 {
  class NoArg {}

  alias element_type = RE2;  // support std::pointer_traits

  // Permit implicit conversion from a struct.
  // TODO: Think about how this interacts with the access check for the `As`
  // and `ImplicitAs` conversions from structs to classes.
  impl {.pattern_: StringPiece} as ImplicitAs(Self) {}
  impl {.pattern_: StringPiece, .options_: RE2.CannedOptions} as ImplicitAs(Self) {}

  // Pretend to be a pointer to Type (never NULL due to on-demand creation):
  impl as Pointer where .Pointee = RE2 {
    fn Resolve[me: Self]() -> Pointee* { return me.get(); }
  }

  // Named accessor/initializer:
  fn get[addr me: Self*]() -> RE* {
    Cpp.std.call_once(once_, Self.Init, me);
    return ptr_;
  }

  var pattern_: StringPiece;
  var options_: RE2.CannedOptions;

  // TODO: mutable?
  private var ptr_: RE2*;
  private var once_: Cpp.std.once_flag;

  private fn Init(lazy_re2: LazyRE2*) {
    lazy_re2->ptr_ = heap.New!(RE2.Make(lazy_re2->pattern_, lazy_re2->options_));
  }
}

// TODO: namespace hooks {
// TODO:
// TODO: // Most platforms support thread_local. Older versions of iOS don't support
// TODO: // thread_local, but for the sake of brevity, we lump together all versions
// TODO: // of Apple platforms that aren't macOS. If an iOS application really needs
// TODO: // the context pointee someday, we can get more specific then...
// TODO: //
// TODO: // As per https://github.com/google/re2/issues/325, thread_local support in
// TODO: // MinGW seems to be buggy. (FWIW, Abseil folks also avoid it.)
// TODO: #define RE2_HAVE_THREAD_LOCAL
// TODO: #if (defined(__APPLE__) && !(defined(TARGET_OS_OSX) && TARGET_OS_OSX)) || defined(__MINGW32__)
// TODO: #undef RE2_HAVE_THREAD_LOCAL
// TODO: #endif
// TODO:
// TODO: // A hook must not make any assumptions regarding the lifetime of the context
// TODO: // pointee beyond the current invocation of the hook. Pointers and references
// TODO: // obtained via the context pointee should be considered invalidated when the
// TODO: // hook returns. Hence, any data about the context pointee (e.g. its pattern)
// TODO: // would have to be copied in order for it to be kept for an indefinite time.
// TODO: //
// TODO: // A hook must not use RE2 for matching. Control flow reentering RE2::Match()
// TODO: // could result in infinite mutual recursion. To discourage that possibility,
// TODO: // RE2 will not maintain the context pointer correctly when used in that way.
// TODO: #ifdef RE2_HAVE_THREAD_LOCAL
// TODO: extern thread_local const RE2* context;
// TODO: #endif
// TODO:
// TODO: struct DFAStateCacheReset {
// TODO:   int64_t state_budget;
// TODO:   size_t state_cache_size;
// TODO: };
// TODO:
// TODO: struct DFASearchFailure {
// TODO:   // Nothing yet...
// TODO: };
// TODO:
// TODO: #define DECLARE_HOOK(type)                  \
// TODO:   using type##Callback = void(const type&); \
// TODO:   void Set##type##Hook(type##Callback* cb); \
// TODO:   type##Callback* Get##type##Hook();
// TODO:
// TODO: DECLARE_HOOK(DFAStateCacheReset)
// TODO: DECLARE_HOOK(DFASearchFailure)
// TODO:
// TODO: #undef DECLARE_HOOK
// TODO:
// TODO: }  // namespace hooks

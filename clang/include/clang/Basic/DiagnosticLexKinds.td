//==--- DiagnosticLexKinds.td - liblex diagnostics ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Lexer Diagnostics
//===----------------------------------------------------------------------===//

let Component = "Lex", CategoryName = "Lexical or Preprocessor Issue" in {

def null_in_string : Warning<"null character(s) preserved in string literal">;
def null_in_char : Warning<"null character(s) preserved in character literal">;
def null_in_file : Warning<"null character ignored">;
def warn_nested_block_comment : Warning<"'/*' within block comment">,
  InGroup<Comment>;
def escaped_newline_block_comment_end : Warning<
  "escaped newline between */ characters at block comment end">,
  InGroup<Comment>;
def backslash_newline_space : Warning<
  "backslash and newline separated by space">;

// Trigraphs.
def trigraph_ignored : Warning<"trigraph ignored">, InGroup<Trigraphs>;
def trigraph_ignored_block_comment : Warning<
  "ignored trigraph would end block comment">, InGroup<Trigraphs>;
def trigraph_ends_block_comment : Warning<"trigraph ends block comment">,
    InGroup<Trigraphs>;
def trigraph_converted : Warning<"trigraph converted to '%0' character">,
    InGroup<Trigraphs>;

def ext_multi_line_bcpl_comment : Extension<"multi-line // comment">,
    InGroup<Comment>;
def ext_bcpl_comment : Extension<
  "// comments are not allowed in this language">,
  InGroup<Comment>;
def ext_no_newline_eof : Extension<"no newline at end of file">;
def ext_backslash_newline_eof : Extension<"backslash-newline at end of file">;
def ext_dollar_in_identifier : Extension<"'$' in identifier">;
def charize_microsoft_ext : Extension<"@# is a microsoft extension">;

def ext_token_used : Extension<"extension used">;

def err_unterminated_string : Error<"missing terminating '\"' character">;
def err_unterminated_char : Error<"missing terminating ' character">;
def err_empty_character : Error<"empty character constant">;
def err_unterminated_block_comment : Error<"unterminated /* comment">;
def err_invalid_character_to_charify : Error<
  "invalid argument to convert to character">;
def err_unterminated___pragma : Error<"missing terminating ')' character">;

def err_conflict_marker : Error<"version control conflict marker in file">;

def ext_multichar_character_literal : ExtWarn<
  "multi-character character constant">, InGroup<MultiChar>;
def ext_four_char_character_literal : Extension<
  "multi-character character constant">, InGroup<FourByteMultiChar>;
  

// Literal
def ext_nonstandard_escape : Extension<
  "use of non-standard escape character '\\%0'">;
def ext_unknown_escape : ExtWarn<"unknown escape sequence '\\%0'">;
def err_hex_escape_no_digits : Error<"\\x used with no following hex digits">;
def err_ucn_escape_no_digits : Error<"\\u used with no following hex digits">;
def err_ucn_escape_invalid : Error<"invalid universal character">;
def err_ucn_escape_incomplete : Error<"incomplete universal character name">;
def err_ucn_escape_too_big : Error<"universal character name is too long">;
def err_invalid_decimal_digit : Error<"invalid digit '%0' in decimal constant">;
def err_invalid_binary_digit : Error<"invalid digit '%0' in binary constant">;
def err_invalid_octal_digit : Error<"invalid digit '%0' in octal constant">;
def err_invalid_suffix_integer_constant : Error<
  "invalid suffix '%0' on integer constant">;
def err_invalid_suffix_float_constant : Error<
  "invalid suffix '%0' on floating constant">;
def warn_extraneous_wide_char_constant : Warning<
  "extraneous characters in wide character constant ignored">;
def warn_char_constant_too_large : Warning<
  "character constant too long for its type">;
def err_exponent_has_no_digits : Error<"exponent has no digits">;
def ext_imaginary_constant : Extension<"imaginary constants are an extension">;
def err_hexconstant_requires_exponent : Error<
  "hexadecimal floating constants require an exponent">;
def ext_hexconstant_cplusplus : Extension<
  "hexadecimal floating constants are a C99 feature that is incompatible with "
  "C++0x">, InGroup<CXXHexFloats>;
def ext_hexconstant_invalid : Extension<
  "hexadecimal floating constants are a C99 feature">;
def ext_binary_literal : Extension<
  "binary integer literals are an extension">;
def err_pascal_string_too_long : Error<"Pascal string is too long">;
def warn_octal_escape_too_large : ExtWarn<"octal escape sequence out of range">;
def warn_hex_escape_too_large : ExtWarn<"hex escape sequence out of range">;
def ext_string_too_long : Extension<"string literal of length %0 exceeds "
  "maximum length %1 that %select{C90|ISO C99|C++}2 compilers are required to "
  "support">, InGroup<OverlengthStrings>;
  
//===----------------------------------------------------------------------===//
// PTH Diagnostics
//===----------------------------------------------------------------------===//
def err_pth_cannot_read : Error<
    "PTH file '%0' could not be read">;
def err_invalid_pth_file : Error<
    "invalid or corrupt PTH file '%0'">;

//===----------------------------------------------------------------------===//
// Preprocessor Diagnostics
//===----------------------------------------------------------------------===//
def pp_hash_warning : Warning<"#warning%0">, InGroup<PoundWarning>;
def pp_include_next_in_primary : Warning<
  "#include_next in primary source file">;
def pp_include_macros_out_of_predefines : Error<
  "the #__include_macros directive is only for internal use by -imacros">;
def pp_include_next_absolute_path : Warning<"#include_next with absolute path">;
def ext_c99_whitespace_required_after_macro_name : ExtWarn<
  "ISO C99 requires whitespace after the macro name">;
def ext_missing_whitespace_after_macro_name : ExtWarn<
  "whitespace required after macro name">;
def warn_missing_whitespace_after_macro_name : Warning<
  "whitespace recommended after macro name">;
  
def pp_pragma_once_in_main_file : Warning<"#pragma once in main file">;
def pp_pragma_sysheader_in_main_file : Warning<
  "#pragma system_header ignored in main file">;
def pp_poisoning_existing_macro : Warning<"poisoning existing macro">;
def pp_out_of_date_dependency : Warning<
  "current file is older than dependency %0">;
def pp_undef_builtin_macro : Warning<"undefining builtin macro">;
def pp_redef_builtin_macro : Warning<"redefining builtin macro">,
  InGroup<DiagGroup<"builtin-macro-redefined">>;
def pp_macro_not_used : Warning<"macro is not used">, DefaultIgnore,
  InGroup<DiagGroup<"unused-macros">>;
def warn_pp_undef_identifier : Warning<
  "%0 is not defined, evaluates to 0">,
  InGroup<DiagGroup<"undef">>, DefaultIgnore;

def pp_invalid_string_literal : Warning<
  "invalid string literal, ignoring final '\\'">;
def warn_pp_expr_overflow : Warning<
  "integer overflow in preprocessor expression">;
def warn_pp_convert_lhs_to_positive : Warning<
  "left side of operator converted from negative value to unsigned: %0">;
def warn_pp_convert_rhs_to_positive : Warning<
  "right side of operator converted from negative value to unsigned: %0">;

def ext_pp_import_directive : Extension<"#import is a language extension">;
def ext_pp_ident_directive : Extension<"#ident is a language extension">;
def ext_pp_include_next_directive : Extension<
  "#include_next is a language extension">;
def ext_pp_warning_directive : Extension<"#warning is a language extension">;

def ext_pp_extra_tokens_at_eol : ExtWarn<
  "extra tokens at end of #%0 directive">, InGroup<ExtraTokens>;
  
def ext_pp_comma_expr : Extension<"comma operator in operand of #if">;
def ext_pp_bad_vaargs_use : Extension<
  "__VA_ARGS__ can only appear in the expansion of a C99 variadic macro">;
def ext_pp_macro_redef : ExtWarn<"%0 macro redefined">;
def ext_variadic_macro : Extension<"variadic macros were introduced in C99">,
  InGroup<VariadicMacros>;
def ext_named_variadic_macro : Extension<
  "named variadic macros are a GNU extension">, InGroup<VariadicMacros>;
def ext_embedded_directive : Extension<
  "embedding a directive within macro arguments is not portable">;
def ext_missing_varargs_arg : Extension<
  "varargs argument missing, but tolerated as an extension">;
def ext_empty_fnmacro_arg : Extension<
  "empty macro arguments were standardized in C99">;

def err_pp_invalid_directive : Error<"invalid preprocessing directive">;
def err_pp_hash_error : Error<"#error%0">;
def err_pp_file_not_found : Error<"'%0' file not found">, DefaultFatal;
def err_pp_error_opening_file : Error<
  "error opening file '%0': %1">, DefaultFatal;
def err_pp_empty_filename : Error<"empty filename">;
def err_pp_include_too_deep : Error<"#include nested too deeply">;
def err_pp_expects_filename : Error<"expected \"FILENAME\" or <FILENAME>">;
def err_pp_macro_not_identifier : Error<"macro names must be identifiers">;
def err_pp_missing_macro_name : Error<"macro name missing">;
def err_pp_missing_rparen_in_macro_def : Error<
  "missing ')' in macro parameter list">;
def err_pp_invalid_tok_in_arg_list : Error<
  "invalid token in macro parameter list">;
def err_pp_expected_ident_in_arg_list : Error<
  "expected identifier in macro parameter list">;
def err_pp_expected_comma_in_arg_list : Error<
  "expected comma in macro parameter list">;
def err_pp_duplicate_name_in_arg_list : Error<
  "duplicate macro parameter name %0">;
def err_pp_stringize_not_parameter : Error<
  "'#' is not followed by a macro parameter">;
def err_pp_malformed_ident : Error<"invalid #ident directive">;
def err_pp_unterminated_conditional : Error<
  "unterminated conditional directive">;
def pp_err_else_after_else : Error<"#else after #else">;
def pp_err_elif_after_else : Error<"#elif after #else">;
def pp_err_else_without_if : Error<"#else without #if">;
def pp_err_elif_without_if : Error<"#elif without #if">;
def err_pp_endif_without_if : Error<"#endif without #if">;
def err_pp_expected_value_in_expr : Error<"expected value in expression">;
def err_pp_missing_val_before_operator : Error<"missing value before operator">;
def err_pp_expected_rparen : Error<"expected ')' in preprocessor expression">;
def err_pp_expected_eol : Error<
  "expected end of line in preprocessor expression">;
def err_pp_defined_requires_identifier : Error<
  "operator 'defined' requires an identifier">;
def err_pp_missing_lparen : Error<"missing '(' after '%0'">;
def err_pp_missing_rparen : Error<"missing ')' after '%0'">;
def err_pp_colon_without_question : Error<"':' without preceding '?'">;
def err_pp_division_by_zero : Error<
  "division by zero in preprocessor expression">;
def err_pp_remainder_by_zero : Error<
  "remainder by zero in preprocessor expression">;
def err_pp_expr_bad_token_binop : Error<
  "token is not a valid binary operator in a preprocessor subexpression">;
def err_pp_expr_bad_token_start_expr : Error<
  "invalid token at start of a preprocessor expression">;
def err_pp_invalid_poison : Error<"can only poison identifier tokens">;
def err_pp_used_poisoned_id : Error<"attempt to use a poisoned identifier">;

def err_feature_check_malformed : Error<
  "builtin feature check macro requires a parenthesized identifier">;

def err__Pragma_malformed : Error<
  "_Pragma takes a parenthesized string literal">;
def err_pragma_comment_malformed : Error<
  "pragma comment requires parenthesized identifier and optional string">;
def err_pragma_message_malformed : Error<
  "pragma message requires parenthesized string">;
def err_pragma_push_pop_macro_malformed : Error<
   "pragma %0 requires a parenthesized string">;
def warn_pragma_pop_macro_no_push : Warning<
   "pragma pop_macro could not pop '%0', no matching push_macro">;
def warn_pragma_message : Warning<"%0">;
def warn_pragma_ignored : Warning<"unknown pragma ignored">,
   InGroup<UnknownPragmas>, DefaultIgnore;
def ext_stdc_pragma_ignored : ExtWarn<"unknown pragma in STDC namespace">,
   InGroup<UnknownPragmas>;
def ext_stdc_pragma_syntax :
   ExtWarn<"expected 'ON' or 'OFF' or 'DEFAULT' in pragma">,
   InGroup<UnknownPragmas>;
def ext_stdc_pragma_syntax_eom :
   ExtWarn<"expected end of macro in STDC pragma">,
   InGroup<UnknownPragmas>;
def warn_stdc_fenv_access_not_supported :
   Warning<"pragma STDC FENV_ACCESS ON is not supported, ignoring pragma">,
   InGroup<UnknownPragmas>;
def warn_pragma_diagnostic_gcc_invalid :
   ExtWarn<"pragma diagnostic expected 'error', 'warning', 'ignored', or"
            " 'fatal'">,
   InGroup<UnknownPragmas>;
def warn_pragma_diagnostic_clang_invalid :
   ExtWarn<"pragma diagnostic expected 'error', 'warning', 'ignored', 'fatal'"
            " 'push', or 'pop'">,
   InGroup<UnknownPragmas>;
def warn_pragma_diagnostic_clang_cannot_ppp :
   ExtWarn<"pragma diagnostic pop could not pop, no matching push">,
   InGroup<UnknownPragmas>;
def warn_pragma_diagnostic_invalid_option :
   ExtWarn<"pragma diagnostic expected option name (e.g. \"-Wundef\")">,
   InGroup<UnknownPragmas>;
def warn_pragma_diagnostic_invalid_token :
   ExtWarn<"unexpected token in pragma diagnostic">,
   InGroup<UnknownPragmas>;
def warn_pragma_diagnostic_unknown_warning :
   ExtWarn<"unknown warning group '%0', ignored">,
   InGroup<UnknownPragmas>;
// - #pragma __debug
def warn_pragma_debug_unexpected_command : Warning<
  "unexpected debug command '%0'">;

def err_pragma_comment_unknown_kind : Error<"unknown kind of pragma comment">;
def err_defined_macro_name : Error<"'defined' cannot be used as a macro name">;
def err_paste_at_start : Error<
  "'##' cannot appear at start of macro expansion">;
def err_paste_at_end : Error<"'##' cannot appear at end of macro expansion">;
def ext_paste_comma : Extension<
  "Use of comma pasting extension is non-portable">;
def err_unterm_macro_invoc : Error<
  "unterminated function-like macro invocation">;
def err_too_many_args_in_macro_invoc : Error<
  "too many arguments provided to function-like macro invocation">;
def err_too_few_args_in_macro_invoc : Error<
  "too few arguments provided to function-like macro invocation">;
def err_pp_bad_paste : Error<
  "pasting formed '%0', an invalid preprocessing token">;
def err_pp_bad_paste_ms : Warning<
  "pasting formed '%0', an invalid preprocessing token">, DefaultError,
  InGroup<DiagGroup<"invalid-token-paste">>;
def err_pp_operator_used_as_macro_name : Error<
  "C++ operator '%0' cannot be used as a macro name">;
def err_pp_illegal_floating_literal : Error<
  "floating point literal in preprocessor expression">;
def err_pp_line_requires_integer : Error<
  "#line directive requires a positive integer argument">;
def err_pp_line_invalid_filename : Error<
  "invalid filename for #line directive">;
def warn_pp_line_decimal : Warning<
  "#line directive interprets number as decimal, not octal">;
def err_pp_line_digit_sequence : Error<
  "#line directive requires a simple digit sequence">;
def err_pp_linemarker_requires_integer : Error<
  "line marker directive requires a positive integer argument">;
def err_pp_linemarker_invalid_filename : Error<
  "invalid filename for line marker directive">;
def err_pp_linemarker_invalid_flag : Error<
  "invalid flag line marker directive">;
def err_pp_linemarker_invalid_pop : Error<
  "invalid line marker flag '2': cannot pop empty include stack">;
def ext_pp_line_too_big : Extension<
  "C requires #line number to be less than %0, allowed as extension">;

}

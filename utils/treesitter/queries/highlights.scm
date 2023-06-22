; Part of the Carbon Language project, under the Apache License v2.0 with LLVM
; Exceptions. See /LICENSE for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

(comment) @comment
(builtin_type) @type.builtin
(sized_type_literal) @type.builtin
(bool_literal) @constant.builtin
(escape_sequence) @constant.character.escape
(string_literal) @string
(integer_literal) @constant.builtin
(float_literal) @constant.builtin

; function declaration or call expression => function
(function_declaration (declared_name (ident) @function))
(call_expression (ident) @function)

; upper case => type
((ident) @type
  (#match? @type "^[A-Z]"))

; lower case => variable
((ident) @variable
  (#match? @variable "^[a-z_]"))

[
  "("
  ")"
  "{"
  "}"
  "["
  "]"
] @punctuation.bracket

[
  "."
  ";"
  ","
  ":"
  ":!"
  "=>"
] @punctuation.delimiter

"->" @punctuation

[
  "+"
  "-"
  (binary_star)
  "/"
  "%"
  "=="
  "!="
  "<"
  "<="
  ">"
  ">="
  "not"
  "and"
  "or"
  "|"
  "&"
  "^"
  ">>"
  "<<"
  (prefix_star)
  (postfix_star)
  "++"
  "--"
] @operator

[
  "abstract"
  "addr"
  "alias"
  "api"
  "auto"
  "base"
  "break"
  "case"
  "class"
  "constraint"
  "continue"
  "default"
  "else"
  "extends"
  "fn"
  "for"
  "if"
  "impl"
  "impls"
  "import"
  "in"
  "interface"
  "let"
  "library"
  "match"
  "namespace"
  "package"
  "return"
  "returned"
  "template"
  "then"
  "var"
  "var"
  "virtual"
  "where"
  "while"
] @keyword

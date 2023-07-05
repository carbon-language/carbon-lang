; Part of the Carbon Language project, under the Apache License v2.0 with LLVM
; Exceptions. See /LICENSE for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; This maps syntax node patterns to highlighting scopes.
; The scopes are used themes and editors to select style for that node.

(comment) @comment
(builtin_type) @type.builtin
(bool_literal) @constant.builtin
(escape_sequence) @constant.character.escape
(string_literal) @string
(numeric_literal) @constant.builtin
(numeric_type_literal) @type.builtin

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
  "*" ; prefix star
  (postfix_star)
  "++"
  "--"
] @operator

; keywords not used in grammar.js are commented out
[
  "abstract"
  ; "adapt"
  "addr"
  "alias"
  "and"
  "api"
  "as"
  "auto"
  "base"
  "break"
  "case"
  "class"
  "constraint"
  "continue"
  "default"
  "else"
  "extend"
  ; "final"
  "fn"
  "for"
  "forall"
  ; "friend"
  "if"
  "impl"
  "impls"
  "import"
  "in"
  "interface"
  "let"
  "library"
  ; "like"
  "match"
  "namespace"
  "not"
  ; "observe"
  "or"
  ; "override"
  "package"
  ; "partial"
  ; "private"
  ; "protected"
  "require"
  "return"
  "returned"
  "Self"
  "then"
  "type"
  "var"
  "virtual"
  "where"
  "while"
] @keyword

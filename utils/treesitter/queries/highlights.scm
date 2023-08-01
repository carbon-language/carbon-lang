; Part of the Carbon Language project, under the Apache License v2.0 with LLVM
; Exceptions. See /LICENSE for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; This maps syntax node patterns to highlighting scopes.
; The scopes are used themes and editors to select style for that node.

(comment) @comment
(builtin_type) @type.builtin
(bool_literal) @constant.builtin
(string) @string
(numeric_literal) @constant.builtin
(numeric_type_literal) @type.builtin

; function declaration or call expression => function
(function_declaration (declared_name (ident) @function))
(call_expression (ident) @function)

(namespace_declaration (declared_name) @namespace)
(interface_declaration (declared_name) @type)
(constraint_declaration (declared_name) @type)
(class_declaration (declared_name) @type)
(choice_declaration (declared_name) @type)
(binding_lhs) @variable

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
  "choice"
  "class"
  "constraint"
  "continue"
  "default"
  "destructor"
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
  "template"
  "then"
  "type"
  "var"
  "virtual"
  "where"
  "while"
] @keyword

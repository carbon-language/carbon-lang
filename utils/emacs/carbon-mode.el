;;; carbon-mode.el --- major mode for editing Carbon source code  -*- lexical-binding: t; -*-

;; Version: 0.0.0
;; Author: Lesley Lai
;; Url: https://github.com/rust-lang/rust-mode
;; Keywords: languages
;; Package-Requires: ((emacs "28.0"))

;; Part of the Carbon Language project, under the Apache License v2.0 with LLVM
;; Exceptions. See /LICENSE for license information.
;; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

;;; Commentary:

;; This package implements a major-mode for editing Carbon source code.

;; TODOs
;; Highlighting for variable definition

;;; Code:

(eval-when-compile
  (require 'rx))

;;; Customization

(defgroup carbon-mode nil
  "Support for Carbon code."
  :link '(url-link "https://github.com/carbon-language/carbon-lang")
  :group 'languages)

(defface carbon-function-face '((t :inherit font-lock-function-name-face))
  "Face for the function definitions."
  :group 'carbon-mode)

(defface carbon-class-face '((t :inherit font-lock-type-face))
  "Face for the class/interface/constraint definitions."
  :group 'carbon-mode)

;;; Utils

;; Create a capture group for inner
(defun carbon-re-grab (inner)
  (concat "\\(" inner "\\)"))

;; Create a non-capturing group for inner
(defun carbon-re-shy (inner)
  (concat "\\(?:" inner "\\)"))

;;; Syntax

(defconst carbon-keywords
  '(      "abstract" "addr" "alias" "and" "api" "as" "auto" "base" "break" "case" "class"
          "constraint" "continue" "default" "else" "extends" "external" "final" "fn" "for" "forall"
          "friend" "if" "impl" "import" "in" "interface" "is" "let" "library" "like" "match" "me"
          "namespace" "not" "observe" "or" "override" "package" "partial" "private" "protected"
          "return" "returned" "then" "_" "var" "virtual" "where" "while"))

(defconst carbon-literals '("false" "true"))

(defconst carbon-re-identifier "[_A-Za-z][_A-Za-z0-9]*")

(defconst carbon-re-number-literal
  (rx (or (regex "[1-9][_0-9]*\\(\\.[_0-9]+\\(e[-+]?[1-9][0-9]*\\)\\)?")
          (regex "0x[_0-9A-F]+\\(\\.[_0-9A-F]+\\(p[-+]?[1-9][0-9]*\\)?\\)?")
          (regex "0b[_01]+"))))

(defconst carbon-re-class-def
  (concat (carbon-re-shy (rx (seq (or "class"
                                      "interface"
                                      "constraint")
                                  (+ space))))
          (carbon-re-grab carbon-re-identifier)))

(defconst carbon-re-fn-def
  (concat (carbon-re-shy "fn[[:space:]]+")
          (carbon-re-grab carbon-re-identifier)))

(defconst carbon-re-builtin-types (rx (seq (or "i"
                                               "u"
                                               "f")
                                           (or "8"
                                               "16"
                                               "32"
                                               "64"
                                               "128"))))

(defvar carbon-font-lock-keywords
  `((,carbon-re-builtin-types . 'font-lock-type-face)
    (,carbon-re-number-literal . 'font-lock-constant-face)
    (,(regexp-opt carbon-literals 'words) . 'font-lock-constant-face)
    (,(regexp-opt carbon-keywords 'words) . 'font-lock-keyword-face)
    (,carbon-re-fn-def 1 'carbon-function-face)
    (,carbon-re-class-def 1 'carbon-class-face)))

;; Syntax table
(defvar carbon-mode-syntax-table
  (let ((table (make-syntax-table)))
    ;; Strings
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?\\ "\\" table)

    ;; Comments
    (modify-syntax-entry ?/  ". 12b" table)
    (modify-syntax-entry ?\n "> b"    table)
    (modify-syntax-entry ?\^m "> b"   table) table)
  "Syntax definitions and helpers.")

;; Put everything together
(define-derived-mode carbon-mode prog-mode
  "Carbon"
  "Major mode for editing code in the Carbon programming language."
  :syntax-table carbon-mode-syntax-table

  ;; Highlighting
  (setq-local font-lock-defaults '((carbon-font-lock-keywords)))

  ;; Comment insertion
  (setq-local comment-start "// ")
  (setq-local comment-end   ""))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.carbon\\'" . carbon-mode))

(provide 'carbon-mode)

;; carbon-mode.el ends here

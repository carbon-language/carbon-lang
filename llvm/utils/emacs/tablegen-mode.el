;; Maintainer:  The LLVM team, http://llvm.cs.uiuc.edu/
;; Description: Major mode for TableGen description files (part of LLVM project)
;; Updated:     2003-08-11

;; Create mode-specific tables.
(defvar tablegen-mode-syntax-table nil
  "Syntax table used while in TableGen mode.")

(defvar tablegen-font-lock-keywords
  (list
   ;; Comments
   '("\/\/.*" . font-lock-comment-face)
   ;; Strings
   '("\"[^\"]+\"" . font-lock-string-face)
   ;; Hex constants
   '("0x[0-9A-Fa-f]+" . font-lock-preprocessor-face)
   ;; Binary constants
   '("0b[01]+" . font-lock-preprocessor-face)
   ;; Integer literals
   '("[-]?[0-9]+" . font-lock-preprocessor-face)
   ;; Floating point constants
   '("[-+]?[0-9]+\.[0-9]*\([eE][-+]?[0-9]+\)?" . font-lock-preprocessor-face)
   ;; Keywords
   '("include\\|def\\|let\\|in\\|code\\|dag\\|field" . font-lock-keyword-face)
   ;; Types
   '("class\\|int\\|string\\|list\\|bits?" . font-lock-type-face)
   )
  "Syntax highlighting for TableGen"
  )

;; ---------------------- Syntax table ---------------------------
;; Shamelessly ripped from jasmin.el
;; URL: http://www.neilvandyke.org/jasmin-emacs/jasmin.el.html

(if (not tablegen-mode-syntax-table)
    (progn
      (setq tablegen-mode-syntax-table (make-syntax-table))
      (mapcar (function (lambda (n)
                          (modify-syntax-entry (aref n 0)
                                               (aref n 1)
                                               tablegen-mode-syntax-table)))
              '(
                ;; whitespace (` ')
                [?\^m " "]
                [?\f  " "]
                [?\n  " "]
                [?\t  " "]
                [?\   " "]
                ;; word constituents (`w')
                ;;[?<  "w"]
                ;;[?>  "w"]
                [?\%  "w"]
                ;;[?_  "w  "]
                ;; comments
                [?\;  "< "]
                [?\n  "> "]
                ;;[?\r  "> "]
                ;;[?\^m "> "]
                ;; symbol constituents (`_')
                ;; punctuation (`.')
                ;; open paren (`(')
                [?\( "("]
                [?\[ "("]
                [?\{ "("]
                ;; close paren (`)')
                [?\) ")"]
                [?\] ")"]
                [?\} ")"]
                ;; string quote ('"')
                [?\" "\""]
                ))))

;; --------------------- Abbrev table -----------------------------

(defvar tablegen-mode-abbrev-table nil
  "Abbrev table used while in TableGen mode.")
(define-abbrev-table 'tablegen-mode-abbrev-table ())

(defvar tablegen-mode-hook nil)
(defvar tablegen-mode-map nil)   ; Create a mode-specific keymap.

(if (not tablegen-mode-map)
    ()  ; Do not change the keymap if it is already set up.
  (setq tablegen-mode-map (make-sparse-keymap))
  (define-key tablegen-mode-map "\t" 'tab-to-tab-stop)
  (define-key tablegen-mode-map "\es" 'center-line)
  (define-key tablegen-mode-map "\eS" 'center-paragraph))


(defun tablegen-mode ()
  "Major mode for editing TableGen description files.
  \\{tablegen-mode-map}
  Runs tablegen-mode-hook on startup."
  (interactive)
  (kill-all-local-variables)
  (use-local-map tablegen-mode-map)         ; Provides the local keymap.
  (setq major-mode 'tablegen-mode)          

  (make-local-variable	'font-lock-defaults)
  (setq major-mode 'tablegen-mode           ; This is how describe-mode
                                            ;   finds the doc string to print.
	mode-name "TableGen"                      ; This name goes into the modeline.
	font-lock-defaults `(tablegen-font-lock-keywords))

  (setq local-abbrev-table tablegen-mode-abbrev-table)
  (set-syntax-table tablegen-mode-syntax-table)
  (run-hooks 'tablegen-mode-hook))          ; Finally, this permits the user to
                                            ;   customize the mode with a hook.

;; Associate .td files with tablegen-mode
(setq auto-mode-alist (append '(("\\.td$" . tablegen-mode)) auto-mode-alist))

(provide 'tablegen-mode)
;; end of tablegen-mode.el

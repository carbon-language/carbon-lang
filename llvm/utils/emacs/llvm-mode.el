;; Maintainer:  The LLVM team, http://llvm.cs.uiuc.edu/
;; Description: Major mode for the LLVM assembler language.
;; Updated:     2003-06-02

;; Create mode-specific tables.
(defvar llvm-mode-syntax-table nil
  "Syntax table used while in LLVM mode.")

(defvar llvm-font-lock-keywords
  (list
   ;; Comments
   '(";.*" . font-lock-comment-face)
   ;; Variables
   '("%[-a-zA-Z$\._][-a-zA-Z$\._0-9]*" . font-lock-variable-name-face)
   ;; Labels
   '("[-a-zA-Z$\._0-9]+:" . font-lock-variable-name-face)
   ;; Strings
   '("\"[^\"]+\"" . font-lock-string-face)
   ;; Unnamed variable slots
   '("%[-]?[0-9]+" . font-lock-variable-name-face)
   ;; Integer literals
   '("[-]?[0-9]+" . font-lock-preprocessor-face)
   ;; Floating point constants
   '("[-+]?[0-9]+\.[0-9]*\([eE][-+]?[0-9]+\)?" . font-lock-preprocessor-face)
   ;; Hex constants
   '("0x[0-9A-Fa-f]+" . font-lock-preprocessor-face)
   ;; Keywords
   '("begin\\|end\\|true\\|false\\|declare\\|global\\|constant\\|const\\|internal\\|linkonce\\|weak\\|appending\\|uninitialized\\|implementation\\|\\.\\.\\.\\|null\\|to\\|except\\|not\\|target\\|endian\\|little\\|big\\|pointersize\\|volatile" . font-lock-keyword-face)
   ;; Types
   '("void\\|bool\\|sbyte\\|ubyte\\|u?short\\|u?int\\|u?long\\|float\\|double\\|type\\|label\\|opaque" . font-lock-type-face)
   ;; Arithmetic and Logical Operators
   '("add\\|sub\\|mul\\|div\\|rem\\|and\\|or\\|xor\\|set\\(ne\\|eq\\|lt\\|gt\\|le\\|ge\\)" . font-lock-keyword-face)
   ;; Special instructions
   '("phi\\|call\\|cast\\|to\\|shl\\|shr\\|vaarg\\|vanext" . font-lock-keyword-face)
   ;; Control instructions
   '("ret\\|br\\|switch\\|invoke\\|unwind" . font-lock-keyword-face)
   ;; Memory operators
   '("malloc\\|alloca\\|free\\|load\\|store\\|getelementptr" . font-lock-keyword-face)
   )
  "Syntax highlighting for LLVM"
  )

;; ---------------------- Syntax table ---------------------------
;; Shamelessly ripped from jasmin.el
;; URL: http://www.neilvandyke.org/jasmin-emacs/jasmin.el.html

(if (not llvm-mode-syntax-table)
    (progn
      (setq llvm-mode-syntax-table (make-syntax-table))
      (mapcar (function (lambda (n)
                          (modify-syntax-entry (aref n 0)
                                               (aref n 1)
                                               llvm-mode-syntax-table)))
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

(defvar llvm-mode-abbrev-table nil
  "Abbrev table used while in LLVM mode.")
(define-abbrev-table 'llvm-mode-abbrev-table ())

(defvar llvm-mode-hook nil)
(defvar llvm-mode-map nil)   ; Create a mode-specific keymap.

(if (not llvm-mode-map)
    ()  ; Do not change the keymap if it is already set up.
  (setq llvm-mode-map (make-sparse-keymap))
  (define-key llvm-mode-map "\t" 'tab-to-tab-stop)
  (define-key llvm-mode-map "\es" 'center-line)
  (define-key llvm-mode-map "\eS" 'center-paragraph))


(defun llvm-mode ()
  "Major mode for editing LLVM source files.
  \\{llvm-mode-map}
  Runs llvm-mode-hook on startup."
  (interactive)
  (kill-all-local-variables)
  (use-local-map llvm-mode-map)         ; Provides the local keymap.
  (setq major-mode 'llvm-mode)          

  (make-local-variable	'font-lock-defaults)
  (setq major-mode 'llvm-mode           ; This is how describe-mode
                                        ;   finds the doc string to print.
	mode-name "LLVM"                ; This name goes into the modeline.
	font-lock-defaults `(llvm-font-lock-keywords))

  (setq local-abbrev-table llvm-mode-abbrev-table)
  (set-syntax-table llvm-mode-syntax-table)
  (run-hooks 'llvm-mode-hook))          ; Finally, this permits the user to
                                        ;   customize the mode with a hook.

;; Associate .ll files with llvm-mode
(setq auto-mode-alist
   (append '(("\\.ll$" . llvm-mode) ("\\.llx$" . llvm-mode)) auto-mode-alist))

(provide 'llvm-mode)
;; end of llvm-mode.el

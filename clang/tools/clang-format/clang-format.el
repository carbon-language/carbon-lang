;;; clang-format.el --- Format code using clang-format

;; Keywords: tools, c
;; Package-Requires: ((json "1.3"))

;;; Commentary:

;; This package allows to filter code through clang-format to fix its formatting.
;; clang-format is a tool that formats C/C++/Obj-C code according to a set of
;; style options, see <http://clang.llvm.org/docs/ClangFormatStyleOptions.html>.
;; Note that clang-format 3.4 or newer is required.

;; clang-format.el is available via MELPA and can be installed via
;;
;;   M-x package-install clang-format
;;
;; when ("melpa" . "http://melpa.org/packages/") is included in
;; `package-archives'. Alternatively, ensure the directory of this
;; file is in your `load-path' and add
;;
;;   (require 'clang-format)
;;
;; to your .emacs configuration.

;; You may also want to bind `clang-format-region' to a key:
;;
;;   (global-set-key [C-M-tab] 'clang-format-region)

;;; Code:

(require 'json)

(defgroup clang-format nil
  "Format code using clang-format."
  :group 'tools)

(defcustom clang-format-executable
  (or (executable-find "clang-format")
      "clang-format")
  "Location of the clang-format executable.

A string containing the name or the full path of the executable."
  :group 'clang-format
  :type 'string
  :risky t)

(defcustom clang-format-style "file"
  "Style argument to pass to clang-format.

By default clang-format will load the style configuration from
a file named .clang-format located in one of the parent directories
of the buffer."
  :group 'clang-format
  :type 'string
  :safe #'stringp)
(make-variable-buffer-local 'clang-format-style)

;;;###autoload
(defun clang-format-region (start end &optional style)
  "Use clang-format to format the code between START and END according to STYLE.
If called interactively uses the region or the current statement if there
is no active region.  If no style is given uses `clang-format-style'."
  (interactive
   (if (use-region-p)
       (list (region-beginning) (region-end))
     (list (point) (point))))

  (unless style
    (setq style clang-format-style))

  (let* ((temp-file (make-temp-file "clang-format"))
         (keep-stderr (list t temp-file))
         (window-starts
          (mapcar (lambda (w) (list w (window-start w)))
                  (get-buffer-window-list)))
         (status)
         (stderr)
         (json))

    (unwind-protect
        (setq status
              (call-process-region
               (point-min) (point-max) clang-format-executable
               'delete keep-stderr nil

               "-assume-filename" (or (buffer-file-name) "")
               "-style" style
               "-offset" (number-to-string (1- start))
               "-length" (number-to-string (- end start))
               "-cursor" (number-to-string (1- (point))))
              stderr
              (with-temp-buffer
                (insert-file-contents temp-file)
                (when (> (point-max) (point-min))
                  (insert ": "))
                (buffer-substring-no-properties
                 (point-min) (line-end-position))))
      (delete-file temp-file))

    (cond
     ((stringp status)
      (error "(clang-format killed by signal %s%s)" status stderr))
     ((not (equal 0 status))
      (error "(clang-format failed with code %d%s)" status stderr))
     (t (message "(clang-format succeeded%s)" stderr)))

    (goto-char (point-min))
    (setq json (json-read-from-string
                (buffer-substring-no-properties
                 (point-min) (line-end-position))))

    (delete-region (point-min) (line-beginning-position 2))
    (mapc (lambda (w) (apply #'set-window-start w))
          window-starts)
    (goto-char (1+ (cdr (assoc 'Cursor json))))))

;;;###autoload
(defun clang-format-buffer (&optional style)
  "Use clang-format to format the current buffer according to STYLE."
  (interactive)
  (clang-format-region (point-min) (point-max) style))

;;;###autoload
(defalias 'clang-format 'clang-format-region)

(provide 'clang-format)
;;; clang-format.el ends here

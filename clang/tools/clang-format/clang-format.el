;;; Clang-format emacs integration for use with C/Objective-C/C++.

;; This defines a function clang-format-region that you can bind to a key.
;; A minimal .emacs would contain:
;;
;;   (load "<path-to-clang>/tools/clang-format/clang-format.el")
;;   (global-set-key [C-M-tab] 'clang-format-region)
;;
;; Depending on your configuration and coding style, you might need to modify
;; 'style' in clang-format, below.

(require 'json)

;; *Location of the clang-format binary. If it is on your PATH, a full path name
;; need not be specified.
(defvar clang-format-binary "clang-format")

(defun clang-format-region ()
  "Use clang-format to format the currently active region."
  (interactive)
  (let ((beg (if mark-active
                 (region-beginning)
               (min (line-beginning-position) (1- (point-max)))))
        (end (if mark-active
                 (region-end)
               (line-end-position))))
    (clang-format beg end)))

(defun clang-format-buffer ()
  "Use clang-format to format the current buffer."
  (clang-format (point-min) (point-max)))

(defun clang-format (begin end)
  "Use clang-format to format the code between BEGIN and END."
  (let* ((orig-windows (get-buffer-window-list (current-buffer)))
         (orig-window-starts (mapcar #'window-start orig-windows))
         (orig-point (point))
         (style "LLVM"))
    (unwind-protect
        (call-process-region (point-min) (point-max) clang-format-binary t t nil
                             "-offset" (number-to-string (1- begin))
                             "-length" (number-to-string (- end begin))
                             "-cursor" (number-to-string (point))
                             "-style" style)
      (goto-char (point-min))
      (let ((json-output (json-read-from-string
                           (buffer-substring-no-properties
                             (point-min) (line-beginning-position 2)))))
        (delete-region (point-min) (line-beginning-position 2))
        (goto-char (cdr (assoc 'Cursor json-output)))
        (dotimes (index (length orig-windows))
          (set-window-start (nth index orig-windows)
                            (nth index orig-window-starts)))))))

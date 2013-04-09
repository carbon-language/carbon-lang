;;; Clang-format emacs integration for use with C/Objective-C/C++.

;; This defines a function clang-format-region that you can bind to a key.
;; A minimal .emacs would contain:
;;
;;   (load "<path-to-clang>/tools/clang/clang-format/clang-format.el")
;;   (global-set-key [C-M-tab] 'clang-format-region)
;;
;; Depending on your configuration and coding style, you might need to modify
;; 'style' and 'binary' below.
(defun clang-format-region ()
  (interactive)
  (let ((orig-file buffer-file-name)
        (orig-point (point))
        (orig-mark (mark t))
        (orig-mark-active mark-active)
        (binary "clang-format")
        (style "LLVM")
        replacement-text replaced beg end)
    (basic-save-buffer)
    (save-restriction
      (widen)
      (if mark-active
          (setq beg (1- (region-beginning))
                end (1- (region-end)))
        (setq beg (1- (line-beginning-position))
              end (1- (line-end-position))))
      (with-temp-buffer
        (call-process
         binary orig-file '(t nil) t
         "-offset" (number-to-string beg)
         "-length" (number-to-string (- end beg))
         "-style" style)
        (setq replacement-text
              (buffer-substring-no-properties (point-min) (point-max))))
      (unless (string= replacement-text
                       (buffer-substring-no-properties (point-min) (point-max)))
        (delete-region (point-min) (point-max))
        (insert replacement-text)
        (setq replaced t)))
    (ignore-errors
      (when orig-mark
        (push-mark orig-mark)
        (when orig-mark-active
          (activate-mark)
          (setq deactivate-mark nil)))
      (goto-char orig-point))))

;;; Clang-format emacs integration for use with C/Objective-C/C++.

;; This defines a function clang-format-region that you can bind to a key.
;; A minimal .emacs would contain:
;;
;;   (load "<path-to-clang>/tools/clang-format/clang-format.el")
;;   (global-set-key [C-M-tab] 'clang-format-region)
;;
;; Depending on your configuration and coding style, you might need to modify
;; 'style' and 'binary' below.
(defun clang-format-region ()
  (interactive)
  (let ((orig-window-start (window-start))
        (orig-point (point))
        (binary "clang-format")
        (style "LLVM"))
    (if mark-active
        (setq beg (region-beginning)
              end (region-end))
      (setq beg (line-beginning-position)
            end (line-end-position)))
    (call-process-region (point-min) (point-max) binary t t nil
                         "-offset" (number-to-string (1- beg))
                         "-length" (number-to-string (- end beg))
                         "-style" style)
    (goto-char orig-point)
    (set-window-start (selected-window) orig-window-start)))

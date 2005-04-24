" LLVM coding guidelines conformance for VIM
" Maintainer: LLVM Team, http://llvm.cs.uiuc.edu
" Updated:    2005-04-24
" WARNING:    Read before you source in all these commands and macros!  Some
"             of them may change VIM behavior that you depend on and the
"             settings here may depend on other settings that you may have.

" Wrap text at 80 cols
set textwidth=80

" A tab produces a 2-space indentation
set tabstop=2
set shiftwidth=2
set expandtab

" Enable filetype detection
filetype on

" LLVM Makefiles can have names such as Makefile.rules or TEST.nightly.Makefile,
" so it's important to categorize them as such.
augroup filetype
  au! BufRead,BufNewFile *Makefile*     set filetype=make
augroup END

" In Makefiles, don't expand tabs to spaces, since we need the actual tabs
autocmd FileType make set noexpandtab

" Useful macros for cleaning up code to conform to LLVM coding guidelines

" Delete trailing whitespace and tabs at the end of each line
map :dtws :%s/[\ \t]\+$//

" Convert all tab characters to two spaces
map :untab :%s/\t/  /g

" LLVM coding guidelines conformance for VIM
"
" Maintainer: The LLVM Team, http://llvm.org
" WARNING:    Read before you source in all these commands and macros!  Some
"             of them may change VIM behavior that you depend on.
"
" You can run VIM with these settings without changing your current setup with:
" $ vim -u /path/to/llvm/utils/vim/vimrc

" It's VIM, not VI
set nocompatible

" Wrap text at 80 cols
set textwidth=80

" A tab produces a 2-space indentation
set tabstop=2
set shiftwidth=2
set expandtab

" Highlight trailing whitespace
highlight WhitespaceEOL ctermbg=DarkYellow guibg=DarkYellow
match WhitespaceEOL /\s\+$/

" Optional
" C/C++ programming helpers
set cindent
" Don't indent switch case labels beyond the switch.
set cinoptions=:0
" Add and delete spaces in increments of `shiftwidth' for tabs
set smarttab

" Highlight syntax in programming languages
syntax on

" Enable filetype detection
filetype on

" LLVM Makefiles can have names such as Makefile.rules or TEST.nightly.Makefile,
" so it's important to categorize them as such.
augroup filetype
  au! BufRead,BufNewFile *Makefile* set filetype=make
augroup END

" In Makefiles, don't expand tabs to spaces, since we need the actual tabs
autocmd FileType make set noexpandtab

" Useful macros for cleaning up code to conform to LLVM coding guidelines

" Delete trailing whitespace and tabs at the end of each line
command! DeleteTrailingWs :%s/[\ \t]\+$//

" Convert all tab characters to two spaces
command! Untab :%s/\t/  /g

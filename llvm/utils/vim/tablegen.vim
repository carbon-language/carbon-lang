" Vim syntax file
" Language:   TableGen
" Maintainer: The LLVM team, http://llvm.cs.uiuc.edu/
" Updated:    2003-08-11

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn case match

syn keyword tgKeyword   def let in code dag field
syn keyword tgType      class int string list bit bits 
syn match   tgNumber    /\<\d\+\>/
syn match   tgNumber    /\<\d\+\.\d*\>/
syn match   tgComment   /\/\/.*$/
" FIXME: this does not capture multi-line C-style comments
syn match   tgComment   /\/\*.*\*\//
syn region  tgString    start=/"/ skip=/\\"/ end=/"/

if version >= 508 || !exists("did_c_syn_inits")
  if version < 508
    let did_c_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink tgKeyword Statement
  HiLink tgType Type
  HiLink tgNumber Number
  HiLink tgComment Comment
  HiLink tgString String

  delcommand HiLink
endif
 
let b:current_syntax = "tablegen"

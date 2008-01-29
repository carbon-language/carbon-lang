" Vim syntax file
" Language:   llvm
" Maintainer: The LLVM team, http://llvm.org/
" Updated:    2003-06-02

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn case match

syn keyword llvmType void float double
syn keyword llvmType x86_fp80 fp128 ppc_fp128
syn keyword llvmType type label opaque
syn match   llvmType /\<i\d\+\>/

syn keyword llvmStatement add sub mul sdiv udiv fdiv srem urem frem
syn keyword llvmStatement and or xor
syn keyword llvmStatement icmp fcmp
syn keyword llvmStatement eq ne ugt uge ult ule sgt sge slt sle
syn keyword llvmStatement false oeq ogt oge olt ole one ord ueq ugt uge
syn keyword llvmStatement ult ule une uno true

syn keyword llvmStatement phi tail call to select shl lshr ashr va_arg
syn keyword llvmStatement trunc zext sext
syn keyword llvmStatement fptrunc fpext fptoui fptosi uitofp sitofp
syn keyword llvmStatement ptrtoint inttoptr bitcast
syn keyword llvmStatement ret br switch invoke unwind unreachable
syn keyword llvmStatement malloc alloca free load store getelementptr

syn keyword llvmStatement true false zeroinitializer
syn keyword llvmStatement define declare global constant const
syn keyword llvmStatement internal uninitialized external implementation
syn keyword llvmStatement linkonce weak appending
syn keyword llvmStatement undef null to except target datalayout triple deplibs
syn keyword llvmStatement big little volatile fastcc coldcc cc
syn keyword llvmStatement extractelement insertelement shufflevector
syn keyword llvmStatement module asm align

"syn match llvmFunction /%[a-zA-Z\$._\-][a-zA-Z\$._\-0-9]*/
syn match  llvmNumber /\<\d\+\>/
syn match  llvmNumber /\<\d\+\.\d*\>/

syn match  llvmComment /;.*$/
syn region llvmString start=/"/ skip=/\\"/ end=/"/
syn match  llvmLabel /[\-a-zA-Z\$._0-9]*:/


if version >= 508 || !exists("did_c_syn_inits")
  if version < 508
    let did_c_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink llvmType Type
  HiLink llvmStatement Statement
  HiLink llvmNumber Number
  HiLink llvmComment Comment
  HiLink llvmString String
  HiLink llvmLabel Label

  delcommand HiLink
endif
 
let b:current_syntax = "llvm"

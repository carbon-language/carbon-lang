" Vim syntax file
" Language:   llvm
" Maintainer: The LLVM team, http://llvm.org/
" Version:      $Revision$

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn case match

" Types.
" Types also include struct, array, vector, etc. but these don't
" benefit as much from having dedicated highlighting rules.
syn keyword llvmType void half float double x86_fp80 fp128 ppc_fp128
syn keyword llvmType label metadata x86_mmx
syn keyword llvmType type label opaque
syn match   llvmType /\<i\d\+\>/

" Instructions.
" The true and false tokens can be used for comparison opcodes, but it's
" much more common for these tokens to be used for boolean constants.
syn keyword llvmStatement add alloca and arcp ashr atomicrmw bitcast br call
syn keyword llvmStatement cmpxchg eq exact extractelement extractvalue fadd fast
syn keyword llvmStatement fcmp fdiv fence fmul fpext fptosi fptoui fptrunc free
syn keyword llvmStatement frem fsub getelementptr icmp inbounds indirectbr
syn keyword llvmStatement insertelement insertvalue inttoptr invoke landingpad
syn keyword llvmStatement load lshr malloc max min mul nand ne ninf nnan nsw nsz
syn keyword llvmStatement nuw oeq oge ogt ole olt one or ord phi ptrtoint resume
syn keyword llvmStatement ret sdiv select sext sge sgt shl shufflevector sitofp
syn keyword llvmStatement sle slt srem store sub switch trunc udiv ueq uge ugt
syn keyword llvmStatement uitofp ule ult umax umin une uno unreachable unwind
syn keyword llvmStatement urem va_arg xchg xor zext

" Keywords.
syn keyword llvmKeyword acq_rel acquire address_safety addrspace alias align
syn keyword llvmKeyword alignstack alwaysinline appending arm_aapcs_vfpcc
syn keyword llvmKeyword arm_aapcscc arm_apcscc asm atomic available_externally
syn keyword llvmKeyword blockaddress byval c catch cc ccc cleanup coldcc common
syn keyword llvmKeyword constant datalayout declare default define deplibs
syn keyword llvmKeyword dllexport dllimport except extern_weak external fastcc
syn keyword llvmKeyword filter gc global hidden initialexec inlinehint inreg
syn keyword llvmKeyword intel_ocl_bicc inteldialect internal linker_private
syn keyword llvmKeyword linker_private_weak linker_private_weak_def_auto
syn keyword llvmKeyword linkonce linkonce_odr linkonce_odr_auto_hide
syn keyword llvmKeyword localdynamic localexec minsize module monotonic
syn keyword llvmKeyword msp430_intrcc naked nest noalias nocapture
syn keyword llvmKeyword noimplicitfloat noinline nonlazybind noredzone noreturn
syn keyword llvmKeyword nounwind optsize personality private protected
syn keyword llvmKeyword ptx_device ptx_kernel readnone readonly release
syn keyword llvmKeyword returns_twice section seq_cst sideeffect signext
syn keyword llvmKeyword singlethread spir_func spir_kernel sret ssp sspreq tail
syn keyword llvmKeyword target thread_local to triple unnamed_addr unordered
syn keyword llvmKeyword uwtable volatile weak weak_odr x86_fastcallcc
syn keyword llvmKeyword x86_stdcallcc x86_thiscallcc zeroext

" Obsolete keywords.
syn keyword llvmError  getresult begin end

" Misc syntax.
syn match   llvmNoName /[%@]\d\+\>/
syn match   llvmNumber /-\?\<\d\+\>/
syn match   llvmFloat  /-\?\<\d\+\.\d*\(e[+-]\d\+\)\?\>/
syn match   llvmFloat  /\<0x\x\+\>/
syn keyword llvmBoolean true false
syn keyword llvmConstant zeroinitializer undef null
syn match   llvmComment /;.*$/
syn region  llvmString start=/"/ skip=/\\"/ end=/"/
syn match   llvmLabel /[-a-zA-Z$._][-a-zA-Z$._0-9]*:/
syn match   llvmIdentifier /[%@][-a-zA-Z$._][-a-zA-Z$._0-9]*/

" Syntax-highlight dejagnu test commands.
syn match  llvmSpecialComment /;\s*RUN:.*$/
syn match  llvmSpecialComment /;\s*PR\d*\s*$/
syn match  llvmSpecialComment /;\s*END\.\s*$/
syn match  llvmSpecialComment /;\s*XFAIL:.*$/

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
  HiLink llvmKeyword Keyword
  HiLink llvmBoolean Boolean
  HiLink llvmFloat Float
  HiLink llvmNoName Identifier
  HiLink llvmConstant Constant
  HiLink llvmSpecialComment SpecialComment
  HiLink llvmError Error
  HiLink llvmIdentifier Identifier

  delcommand HiLink
endif

let b:current_syntax = "llvm"

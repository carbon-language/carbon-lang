; Bitcode compatibility test for llvm 3.6.2
;
; N.b: This is 3.6.2-compatible IR. The CHECK lines occasionally differ from
;      the IR used to generate the bitcode, and may need to be updated. These
;      locations are tagged with an 'XXX'.

; RUN: llvm-dis < %s.bc | FileCheck %s

target datalayout = "E"
; CHECK: target datalayout = "E"

target triple = "x86_64-apple-macosx10.10.0"
; CHECK: target triple = "x86_64-apple-macosx10.10.0"

;; Module-level assembly
module asm "beep boop"
; CHECK: module asm "beep boop"

;; Comdats
$comdat.any = comdat any
; CHECK: $comdat.any = comdat any
$comdat.exactmatch = comdat exactmatch
; CHECK: $comdat.exactmatch = comdat exactmatch
$comdat.largest = comdat largest
; CHECK: $comdat.largest = comdat largest
$comdat.noduplicates = comdat nodeduplicate
; CHECK: $comdat.noduplicates = comdat nodeduplicate
$comdat.samesize = comdat samesize
; CHECK: $comdat.samesize = comdat samesize

; Global Variables -- comdat
@comdat.any = global i32 0, comdat
; CHECK: @comdat.any = global i32 0, comdat
@comdat.exactmatch = global i32 0, comdat
; CHECK: @comdat.exactmatch = global i32 0, comdat
@comdat.largest = global i32 0, comdat
; CHECK: @comdat.largest = global i32 0, comdat
@comdat.noduplicates = global i32 0, comdat
; CHECK: @comdat.noduplicates = global i32 0, comdat
@comdat.samesize = global i32 0, comdat
; CHECK: @comdat.samesize = global i32 0, comdat

; Force two globals from different comdats into sections with the same name.
$comdat1 = comdat any
$comdat2 = comdat any
@g.comdat1 = global i32 0, section "SharedSection", comdat($comdat1)
; CHECK: @g.comdat1 = global i32 0, section "SharedSection", comdat($comdat1)
@g.comdat2 = global i32 0, section "SharedSection", comdat($comdat2)
; CHECK: @g.comdat2 = global i32 0, section "SharedSection", comdat($comdat2)

;; Constants
@const.true = constant i1 true
; CHECK: @const.true = constant i1 true
@const.false = constant i1 false
; CHECK: @const.false = constant i1 false
@const.int = constant i32 zeroinitializer
; CHECK: @const.int = constant i32 0
@const.float = constant double 0.0
; CHECK: @const.float = constant double 0.0
@const.null = constant i8* null
; CHECK: @const.null = constant i8* null
%const.struct.type = type { i32, i8 }
%const.struct.type.packed = type <{ i32, i8 }>
@const.struct = constant %const.struct.type { i32 -1, i8 undef }
; CHECK: @const.struct = constant %const.struct.type { i32 -1, i8 undef }
@const.struct.packed = constant %const.struct.type.packed <{ i32 -1, i8 1 }>
; CHECK: @const.struct.packed = constant %const.struct.type.packed <{ i32 -1, i8 1 }>
@const.array = constant [2 x i32] [i32 -3, i32 -4]
; CHECK: @const.array = constant [2 x i32] [i32 -3, i32 -4]
@const.vector = constant <2 x i32> <i32 -5, i32 -6>
; CHECK: @const.vector = constant <2 x i32> <i32 -5, i32 -6>

;; Global Variables
; Format: [@<GlobalVarName> =] [Linkage] [Visibility] [DLLStorageClass]
;         [ThreadLocal] [unnamed_addr] [AddrSpace] [ExternallyInitialized]
;         <global | constant> <Type> [<InitializerConstant>]
;         [, section "name"] [, align <Alignment>]

; Global Variables -- Simple
@g1 = global i32 0
; CHECK: @g1 = global i32 0
@g2 = constant i32 0
; CHECK: @g2 = constant i32 0

; Global Variables -- Linkage
@g.private = private global i32 0
; CHECK: @g.private = private global i32 0
@g.internal = internal global i32 0
; CHECK: @g.internal = internal global i32 0
@g.available_externally = available_externally global i32 0
; CHECK: @g.available_externally = available_externally global i32 0
@g.linkonce = linkonce global i32 0
; CHECK: @g.linkonce = linkonce global i32 0
@g.weak = weak global i32 0
; CHECK: @g.weak = weak global i32 0
@g.common = common global i32 0
; CHECK: @g.common = common global i32 0
@g.appending = appending global [4 x i8] c"test"
; CHECK: @g.appending = appending global [4 x i8] c"test"
@g.extern_weak = extern_weak global i32
; CHECK: @g.extern_weak = extern_weak global i32
@g.linkonce_odr = linkonce_odr global i32 0
; CHECK: @g.linkonce_odr = linkonce_odr global i32 0
@g.weak_odr = weak_odr global i32 0
; CHECK: @g.weak_odr = weak_odr global i32 0
@g.external = external global i32
; CHECK: @g.external = external global i32

; Global Variables -- Visibility
@g.default = default global i32 0
; CHECK: @g.default = global i32 0
@g.hidden = hidden global i32 0
; CHECK: @g.hidden = hidden global i32 0
@g.protected = protected global i32 0
; CHECK: @g.protected = protected global i32 0

; Global Variables -- DLLStorageClass
@g.dlldefault = default global i32 0
; CHECK: @g.dlldefault = global i32 0
@g.dllimport = external dllimport global i32
; CHECK: @g.dllimport = external dllimport global i32
@g.dllexport = dllexport global i32 0
; CHECK: @g.dllexport = dllexport global i32 0

; Global Variables -- ThreadLocal
@g.notthreadlocal = global i32 0
; CHECK: @g.notthreadlocal = global i32 0
@g.generaldynamic = thread_local global i32 0
; CHECK: @g.generaldynamic = thread_local global i32 0
@g.localdynamic = thread_local(localdynamic) global i32 0
; CHECK: @g.localdynamic = thread_local(localdynamic) global i32 0
@g.initialexec = thread_local(initialexec) global i32 0
; CHECK: @g.initialexec = thread_local(initialexec) global i32 0
@g.localexec = thread_local(localexec) global i32 0
; CHECK: @g.localexec = thread_local(localexec) global i32 0

; Global Variables -- unnamed_addr
@g.unnamed_addr = unnamed_addr global i32 0
; CHECK: @g.unnamed_addr = unnamed_addr global i32 0

; Global Variables -- AddrSpace
@g.addrspace = addrspace(1) global i32 0
; CHECK: @g.addrspace = addrspace(1) global i32 0

; Global Variables -- ExternallyInitialized
@g.externally_initialized = external externally_initialized global i32
; CHECK: @g.externally_initialized = external externally_initialized global i32

; Global Variables -- section
@g.section = global i32 0, section "_DATA"
; CHECK: @g.section = global i32 0, section "_DATA"

; Global Variables -- align
@g.align = global i32 0, align 4
; CHECK: @g.align = global i32 0, align 4

; Global Variables -- Intrinsics
%pri.func.data = type { i32, void ()*, i8* }
@g.used1 = global i32 0
@g.used2 = global i32 0
@g.used3 = global i8 0
declare void @g.f1()
@llvm.used = appending global [1 x i32*] [i32* @g.used1], section "llvm.metadata"
; CHECK: @llvm.used = appending global [1 x i32*] [i32* @g.used1], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x i32*] [i32* @g.used2], section "llvm.metadata"
; CHECK: @llvm.compiler.used = appending global [1 x i32*] [i32* @g.used2], section "llvm.metadata"
@llvm.global_ctors = appending global [1 x %pri.func.data] [%pri.func.data { i32 0, void ()* @g.f1, i8* @g.used3 }], section "llvm.metadata"
; CHECK: @llvm.global_ctors = appending global [1 x %pri.func.data] [%pri.func.data { i32 0, void ()* @g.f1, i8* @g.used3 }], section "llvm.metadata"
@llvm.global_dtors = appending global [1 x %pri.func.data] [%pri.func.data { i32 0, void ()* @g.f1, i8* @g.used3 }], section "llvm.metadata"
; CHECK: @llvm.global_dtors = appending global [1 x %pri.func.data] [%pri.func.data { i32 0, void ()* @g.f1, i8* @g.used3 }], section "llvm.metadata"

;; Aliases
; Format: @<Name> = [Linkage] [Visibility] [DLLStorageClass] [ThreadLocal]
;                   [unnamed_addr] alias <AliaseeTy> @<Aliasee>

; Aliases -- Linkage
@a.private = private alias i32* @g.private
; CHECK: @a.private = private alias i32, i32* @g.private
@a.internal = internal alias i32* @g.internal
; CHECK: @a.internal = internal alias i32, i32* @g.internal
@a.linkonce = linkonce alias i32* @g.linkonce
; CHECK: @a.linkonce = linkonce alias i32, i32* @g.linkonce
@a.weak = weak alias i32* @g.weak
; CHECK: @a.weak = weak alias i32, i32* @g.weak
@a.linkonce_odr = linkonce_odr alias i32* @g.linkonce_odr
; CHECK: @a.linkonce_odr = linkonce_odr alias i32, i32* @g.linkonce_odr
@a.weak_odr = weak_odr alias i32* @g.weak_odr
; CHECK: @a.weak_odr = weak_odr alias i32, i32* @g.weak_odr
@a.external = external alias i32* @g1
; CHECK: @a.external = alias i32, i32* @g1

; Aliases -- Visibility
@a.default = default alias i32* @g.default
; CHECK: @a.default = alias i32, i32* @g.default
@a.hidden = hidden alias i32* @g.hidden
; CHECK: @a.hidden = hidden alias i32, i32* @g.hidden
@a.protected = protected alias i32* @g.protected
; CHECK: @a.protected = protected alias i32, i32* @g.protected

; Aliases -- DLLStorageClass
@a.dlldefault = default alias i32* @g.dlldefault
; CHECK: @a.dlldefault = alias i32, i32* @g.dlldefault
@a.dllimport = dllimport alias i32* @g1
; CHECK: @a.dllimport = dllimport alias i32, i32* @g1
@a.dllexport = dllexport alias i32* @g.dllexport
; CHECK: @a.dllexport = dllexport alias i32, i32* @g.dllexport

; Aliases -- ThreadLocal
@a.notthreadlocal = alias i32* @g.notthreadlocal
; CHECK: @a.notthreadlocal = alias i32, i32* @g.notthreadlocal
@a.generaldynamic = thread_local alias i32* @g.generaldynamic
; CHECK: @a.generaldynamic = thread_local alias i32, i32* @g.generaldynamic
@a.localdynamic = thread_local(localdynamic) alias i32* @g.localdynamic
; CHECK: @a.localdynamic = thread_local(localdynamic) alias i32, i32* @g.localdynamic
@a.initialexec = thread_local(initialexec) alias i32* @g.initialexec
; CHECK: @a.initialexec = thread_local(initialexec) alias i32, i32* @g.initialexec
@a.localexec = thread_local(localexec) alias i32* @g.localexec
; CHECK: @a.localexec = thread_local(localexec) alias i32, i32* @g.localexec

; Aliases -- unnamed_addr
@a.unnamed_addr = unnamed_addr alias i32* @g.unnamed_addr
; CHECK: @a.unnamed_addr = unnamed_addr alias i32, i32* @g.unnamed_addr

;; Functions
; Format: define [linkage] [visibility] [DLLStorageClass]
;         [cconv] [ret attrs]
;         <ResultType> @<FunctionName> ([argument list])
;         [unnamed_addr] [fn Attrs] [section "name"] [comdat [($name)]]
;         [align N] [gc] [prefix Constant]
;         { ... }

; Functions -- Simple
declare void @f1 ()
; CHECK: declare void @f1()

define void @f2 () {
; CHECK: define void @f2()
entry:
  ret void
}

; Functions -- linkage
define private void @f.private() {
; CHECK: define private void @f.private()
entry:
  ret void
}
define internal void @f.internal() {
; CHECK: define internal void @f.internal()
entry:
  ret void
}
define available_externally void @f.available_externally() {
; CHECK: define available_externally void @f.available_externally()
entry:
  ret void
}
define linkonce void @f.linkonce() {
; CHECK: define linkonce void @f.linkonce()
entry:
  ret void
}
define weak void @f.weak() {
; CHECK: define weak void @f.weak()
entry:
  ret void
}
define linkonce_odr void @f.linkonce_odr() {
; CHECK: define linkonce_odr void @f.linkonce_odr()
entry:
  ret void
}
define weak_odr void @f.weak_odr() {
; CHECK: define weak_odr void @f.weak_odr()
entry:
  ret void
}
declare external void @f.external()
; CHECK: declare void @f.external()
declare extern_weak void @f.extern_weak()
; CHECK: declare extern_weak void @f.extern_weak()

; Functions -- visibility
declare default void @f.default()
; CHECK: declare void @f.default()
declare hidden void @f.hidden()
; CHECK: declare hidden void @f.hidden()
declare protected void @f.protected()
; CHECK: declare protected void @f.protected()

; Functions -- DLLStorageClass
declare dllimport void @f.dllimport()
; CHECK: declare dllimport void @f.dllimport()
declare dllexport void @f.dllexport()
; CHECK: declare dllexport void @f.dllexport()

; Functions -- cconv (Calling conventions)
declare ccc void @f.ccc()
; CHECK: declare void @f.ccc()
declare fastcc void @f.fastcc()
; CHECK: declare fastcc void @f.fastcc()
declare coldcc void @f.coldcc()
; CHECK: declare coldcc void @f.coldcc()
declare cc10 void @f.cc10()
; CHECK: declare ghccc void @f.cc10()
declare ghccc void @f.ghccc()
; CHECK: declare ghccc void @f.ghccc()
declare cc11 void @f.cc11()
; CHECK: declare cc11 void @f.cc11()
declare webkit_jscc void @f.webkit_jscc()
; CHECK: declare webkit_jscc void @f.webkit_jscc()
declare anyregcc void @f.anyregcc()
; CHECK: declare anyregcc void @f.anyregcc()
declare preserve_mostcc void @f.preserve_mostcc()
; CHECK: declare preserve_mostcc void @f.preserve_mostcc()
declare preserve_allcc void @f.preserve_allcc()
; CHECK: declare preserve_allcc void @f.preserve_allcc()
declare cc64 void @f.cc64()
; CHECK: declare x86_stdcallcc void @f.cc64()
declare x86_stdcallcc void @f.x86_stdcallcc()
; CHECK: declare x86_stdcallcc void @f.x86_stdcallcc()
declare cc65 void @f.cc65()
; CHECK: declare x86_fastcallcc void @f.cc65()
declare x86_fastcallcc void @f.x86_fastcallcc()
; CHECK: declare x86_fastcallcc void @f.x86_fastcallcc()
declare cc66 void @f.cc66()
; CHECK: declare arm_apcscc void @f.cc66()
declare arm_apcscc void @f.arm_apcscc()
; CHECK: declare arm_apcscc void @f.arm_apcscc()
declare cc67 void @f.cc67()
; CHECK: declare arm_aapcscc void @f.cc67()
declare arm_aapcscc void @f.arm_aapcscc()
; CHECK: declare arm_aapcscc void @f.arm_aapcscc()
declare cc68 void @f.cc68()
; CHECK: declare arm_aapcs_vfpcc void @f.cc68()
declare arm_aapcs_vfpcc void @f.arm_aapcs_vfpcc()
; CHECK: declare arm_aapcs_vfpcc void @f.arm_aapcs_vfpcc()
declare cc69 void @f.cc69()
; CHECK: declare msp430_intrcc void @f.cc69()
declare msp430_intrcc void @f.msp430_intrcc()
; CHECK: declare msp430_intrcc void @f.msp430_intrcc()
declare cc70 void @f.cc70()
; CHECK: declare x86_thiscallcc void @f.cc70()
declare x86_thiscallcc void @f.x86_thiscallcc()
; CHECK: declare x86_thiscallcc void @f.x86_thiscallcc()
declare cc71 void @f.cc71()
; CHECK: declare ptx_kernel void @f.cc71()
declare ptx_kernel void @f.ptx_kernel()
; CHECK: declare ptx_kernel void @f.ptx_kernel()
declare cc72 void @f.cc72()
; CHECK: declare ptx_device void @f.cc72()
declare ptx_device void @f.ptx_device()
; CHECK: declare ptx_device void @f.ptx_device()
declare cc75 void @f.cc75()
; CHECK: declare spir_func void @f.cc75()
declare spir_func void @f.spir_func()
; CHECK: declare spir_func void @f.spir_func()
declare cc76 void @f.cc76()
; CHECK: declare spir_kernel void @f.cc76()
declare spir_kernel void @f.spir_kernel()
; CHECK: declare spir_kernel void @f.spir_kernel()
declare cc77 void @f.cc77()
; CHECK: declare intel_ocl_bicc void @f.cc77()
declare intel_ocl_bicc void @f.intel_ocl_bicc()
; CHECK: declare intel_ocl_bicc void @f.intel_ocl_bicc()
declare cc78 void @f.cc78()
; CHECK: declare x86_64_sysvcc void @f.cc78()
declare x86_64_sysvcc void @f.x86_64_sysvcc()
; CHECK: declare x86_64_sysvcc void @f.x86_64_sysvcc()
declare cc79 void @f.cc79()
; CHECK: declare win64cc void @f.cc79()
declare win64cc void @f.x86_64_win64cc()
; CHECK: declare win64cc void @f.x86_64_win64cc()
declare cc80 void @f.cc80()
; CHECK: declare x86_vectorcallcc void @f.cc80()
declare x86_vectorcallcc void @f.x86_vectorcallcc()
; CHECK: declare x86_vectorcallcc void @f.x86_vectorcallcc()
declare cc1023 void @f.cc1023()
; CHECK: declare cc1023 void @f.cc1023()

; Functions -- ret attrs (Return attributes)
declare zeroext i64 @f.zeroext()
; CHECK: declare zeroext i64 @f.zeroext()
declare signext i64 @f.signext()
; CHECK: declare signext i64 @f.signext()
declare inreg i32* @f.inreg()
; CHECK: declare inreg i32* @f.inreg()
declare noalias i32* @f.noalias()
; CHECK: declare noalias i32* @f.noalias()
declare nonnull i32* @f.nonnull()
; CHECK: declare nonnull i32* @f.nonnull()
declare dereferenceable(4) i32* @f.dereferenceable4()
; CHECK: declare dereferenceable(4) i32* @f.dereferenceable4()
declare dereferenceable(8) i32* @f.dereferenceable8()
; CHECK: declare dereferenceable(8) i32* @f.dereferenceable8()
declare dereferenceable(16) i32* @f.dereferenceable16()
; CHECK: declare dereferenceable(16) i32* @f.dereferenceable16()

; Functions -- Parameter attributes
declare void @f.param.zeroext(i8 zeroext)
; CHECK: declare void @f.param.zeroext(i8 zeroext)
declare void @f.param.signext(i8 signext)
; CHECK: declare void @f.param.signext(i8 signext)
declare void @f.param.inreg(i8 inreg)
; CHECK: declare void @f.param.inreg(i8 inreg)
declare void @f.param.byval({ i8, i8 }* byval({ i8, i8 }))
; CHECK: declare void @f.param.byval({ i8, i8 }* byval({ i8, i8 }))
declare void @f.param.inalloca(i8* inalloca)
; CHECK: declare void @f.param.inalloca(i8* inalloca(i8))
declare void @f.param.sret(i8* sret(i8))
; CHECK: declare void @f.param.sret(i8* sret(i8))
declare void @f.param.noalias(i8* noalias)
; CHECK: declare void @f.param.noalias(i8* noalias)
declare void @f.param.nocapture(i8* nocapture)
; CHECK: declare void @f.param.nocapture(i8* nocapture)
declare void @f.param.nest(i8* nest)
; CHECK: declare void @f.param.nest(i8* nest)
declare i8* @f.param.returned(i8* returned)
; CHECK: declare i8* @f.param.returned(i8* returned)
declare void @f.param.nonnull(i8* nonnull)
; CHECK: declare void @f.param.nonnull(i8* nonnull)
declare void @f.param.dereferenceable(i8* dereferenceable(4))
; CHECK: declare void @f.param.dereferenceable(i8* dereferenceable(4))

; Functions -- unnamed_addr
declare void @f.unnamed_addr() unnamed_addr
; CHECK: declare void @f.unnamed_addr() unnamed_addr

; Functions -- fn Attrs (Function attributes)
declare void @f.alignstack4() alignstack(4)
; CHECK: declare void @f.alignstack4() #0
declare void @f.alignstack8() alignstack(8)
; CHECK: declare void @f.alignstack8() #1
declare void @f.alwaysinline() alwaysinline
; CHECK: declare void @f.alwaysinline() #2
declare void @f.cold() cold
; CHECK: declare void @f.cold() #3
declare void @f.inlinehint() inlinehint
; CHECK: declare void @f.inlinehint() #4
declare void @f.jumptable() unnamed_addr jumptable
; CHECK: declare void @f.jumptable() unnamed_addr #5
declare void @f.minsize() minsize
; CHECK: declare void @f.minsize() #6
declare void @f.naked() naked
; CHECK: declare void @f.naked() #7
declare void @f.nobuiltin() nobuiltin
; CHECK: declare void @f.nobuiltin() #8
declare void @f.noduplicate() noduplicate
; CHECK: declare void @f.noduplicate() #9
declare void @f.noimplicitfloat() noimplicitfloat
; CHECK: declare void @f.noimplicitfloat() #10
declare void @f.noinline() noinline
; CHECK: declare void @f.noinline() #11
declare void @f.nonlazybind() nonlazybind
; CHECK: declare void @f.nonlazybind() #12
declare void @f.noredzone() noredzone
; CHECK: declare void @f.noredzone() #13
declare void @f.noreturn() noreturn
; CHECK: declare void @f.noreturn() #14
declare void @f.nounwind() nounwind
; CHECK: declare void @f.nounwind() #15
declare void @f.optnone() noinline optnone
; CHECK: declare void @f.optnone() #16
declare void @f.optsize() optsize
; CHECK: declare void @f.optsize() #17
declare void @f.readnone() readnone
; CHECK: declare void @f.readnone() #18
declare void @f.readonly() readonly
; CHECK: declare void @f.readonly() #19
declare void @f.returns_twice() returns_twice
; CHECK: declare void @f.returns_twice() #20
declare void @f.sanitize_address() sanitize_address
; CHECK: declare void @f.sanitize_address() #21
declare void @f.sanitize_memory() sanitize_memory
; CHECK: declare void @f.sanitize_memory() #22
declare void @f.sanitize_thread() sanitize_thread
; CHECK: declare void @f.sanitize_thread() #23
declare void @f.ssp() ssp
; CHECK: declare void @f.ssp() #24
declare void @f.sspreq() sspreq
; CHECK: declare void @f.sspreq() #25
declare void @f.sspstrong() sspstrong
; CHECK: declare void @f.sspstrong() #26
declare void @f.uwtable() uwtable
; CHECK: declare void @f.uwtable() #27
declare void @f.kvpair() "cpu"="cortex-a8"
; CHECK:declare void @f.kvpair() #28

; Functions -- section
declare void @f.section() section "80"
; CHECK: declare void @f.section() section "80"

; Functions -- comdat
define void @f.comdat_any() comdat($comdat.any) {
; CHECK: define void @f.comdat_any() comdat($comdat.any)
entry:
  ret void
}
define void @f.comdat_exactmatch() comdat($comdat.exactmatch) {
; CHECK: define void @f.comdat_exactmatch() comdat($comdat.exactmatch)
entry:
  ret void
}
define void @f.comdat_largest() comdat($comdat.largest) {
; CHECK: define void @f.comdat_largest() comdat($comdat.largest)
entry:
  ret void
}
define void @f.comdat_noduplicates() comdat($comdat.noduplicates) {
; CHECK: define void @f.comdat_noduplicates() comdat($comdat.noduplicates)
entry:
  ret void
}
define void @f.comdat_samesize() comdat($comdat.samesize) {
; CHECK: define void @f.comdat_samesize() comdat($comdat.samesize)
entry:
  ret void
}

; Functions -- align
declare void @f.align2() align 2
; CHECK: declare void @f.align2() align 2
declare void @f.align4() align 4
; CHECK: declare void @f.align4() align 4
declare void @f.align8() align 8
; CHECK: declare void @f.align8() align 8

; Functions -- GC
declare void @f.gcshadow() gc "shadow-stack"
; CHECK: declare void @f.gcshadow() gc "shadow-stack"

; Functions -- Prefix data
declare void @f.prefixi32() prefix i32 1684365668
; CHECK: declare void @f.prefixi32() prefix i32 1684365668
declare void @f.prefixarray() prefix [4 x i32] [i32 0, i32 1, i32 2, i32 3]
; CHECK: declare void @f.prefixarray() prefix [4 x i32] [i32 0, i32 1, i32 2, i32 3]

;; Atomic Memory Ordering Constraints
define void @atomics(i32* %word) {
  %cmpxchg.0 = cmpxchg i32* %word, i32 0, i32 4 monotonic monotonic
  ; CHECK: %cmpxchg.0 = cmpxchg i32* %word, i32 0, i32 4 monotonic monotonic
  %cmpxchg.1 = cmpxchg i32* %word, i32 0, i32 5 acq_rel monotonic
  ; CHECK: %cmpxchg.1 = cmpxchg i32* %word, i32 0, i32 5 acq_rel monotonic
  %cmpxchg.2 = cmpxchg i32* %word, i32 0, i32 6 acquire monotonic
  ; CHECK: %cmpxchg.2 = cmpxchg i32* %word, i32 0, i32 6 acquire monotonic
  %cmpxchg.3 = cmpxchg i32* %word, i32 0, i32 7 release monotonic
  ; CHECK: %cmpxchg.3 = cmpxchg i32* %word, i32 0, i32 7 release monotonic
  %cmpxchg.4 = cmpxchg i32* %word, i32 0, i32 8 seq_cst monotonic
  ; CHECK: %cmpxchg.4 = cmpxchg i32* %word, i32 0, i32 8 seq_cst monotonic
  %cmpxchg.5 = cmpxchg weak i32* %word, i32 0, i32 9 seq_cst monotonic
  ; CHECK: %cmpxchg.5 = cmpxchg weak i32* %word, i32 0, i32 9 seq_cst monotonic
  %cmpxchg.6 = cmpxchg volatile i32* %word, i32 0, i32 10 seq_cst monotonic
  ; CHECK: %cmpxchg.6 = cmpxchg volatile i32* %word, i32 0, i32 10 seq_cst monotonic
  %cmpxchg.7 = cmpxchg weak volatile i32* %word, i32 0, i32 11 syncscope("singlethread") seq_cst monotonic
  ; CHECK: %cmpxchg.7 = cmpxchg weak volatile i32* %word, i32 0, i32 11 syncscope("singlethread") seq_cst monotonic
  %atomicrmw.xchg = atomicrmw xchg i32* %word, i32 12 monotonic
  ; CHECK: %atomicrmw.xchg = atomicrmw xchg i32* %word, i32 12 monotonic
  %atomicrmw.add = atomicrmw add i32* %word, i32 13 monotonic
  ; CHECK: %atomicrmw.add = atomicrmw add i32* %word, i32 13 monotonic
  %atomicrmw.sub = atomicrmw sub i32* %word, i32 14 monotonic
  ; CHECK: %atomicrmw.sub = atomicrmw sub i32* %word, i32 14 monotonic
  %atomicrmw.and = atomicrmw and i32* %word, i32 15 monotonic
  ; CHECK: %atomicrmw.and = atomicrmw and i32* %word, i32 15 monotonic
  %atomicrmw.nand = atomicrmw nand i32* %word, i32 16 monotonic
  ; CHECK: %atomicrmw.nand = atomicrmw nand i32* %word, i32 16 monotonic
  %atomicrmw.or = atomicrmw or i32* %word, i32 17 monotonic
  ; CHECK: %atomicrmw.or = atomicrmw or i32* %word, i32 17 monotonic
  %atomicrmw.xor = atomicrmw xor i32* %word, i32 18 monotonic
  ; CHECK: %atomicrmw.xor = atomicrmw xor i32* %word, i32 18 monotonic
  %atomicrmw.max = atomicrmw max i32* %word, i32 19 monotonic
  ; CHECK: %atomicrmw.max = atomicrmw max i32* %word, i32 19 monotonic
  %atomicrmw.min = atomicrmw volatile min i32* %word, i32 20 monotonic
  ; CHECK: %atomicrmw.min = atomicrmw volatile min i32* %word, i32 20 monotonic
  %atomicrmw.umax = atomicrmw umax i32* %word, i32 21 syncscope("singlethread") monotonic
  ; CHECK: %atomicrmw.umax = atomicrmw umax i32* %word, i32 21 syncscope("singlethread") monotonic
  %atomicrmw.umin = atomicrmw volatile umin i32* %word, i32 22 syncscope("singlethread") monotonic
  ; CHECK: %atomicrmw.umin = atomicrmw volatile umin i32* %word, i32 22 syncscope("singlethread") monotonic
  fence acquire
  ; CHECK: fence acquire
  fence release
  ; CHECK: fence release
  fence acq_rel
  ; CHECK: fence acq_rel
  fence syncscope("singlethread") seq_cst
  ; CHECK: fence syncscope("singlethread") seq_cst

  ; XXX: The parser spits out the load type here.
  %ld.1 = load atomic i32* %word monotonic, align 4
  ; CHECK: %ld.1 = load atomic i32, i32* %word monotonic, align 4
  %ld.2 = load atomic volatile i32* %word acquire, align 8
  ; CHECK: %ld.2 = load atomic volatile i32, i32* %word acquire, align 8
  %ld.3 = load atomic volatile i32* %word syncscope("singlethread") seq_cst, align 16
  ; CHECK: %ld.3 = load atomic volatile i32, i32* %word syncscope("singlethread") seq_cst, align 16

  store atomic i32 23, i32* %word monotonic, align 4
  ; CHECK: store atomic i32 23, i32* %word monotonic, align 4
  store atomic volatile i32 24, i32* %word monotonic, align 4
  ; CHECK: store atomic volatile i32 24, i32* %word monotonic, align 4
  store atomic volatile i32 25, i32* %word syncscope("singlethread") monotonic, align 4
  ; CHECK: store atomic volatile i32 25, i32* %word syncscope("singlethread") monotonic, align 4
  ret void
}

;; Fast Math Flags
define void @fastmathflags(float %op1, float %op2) {
  %f.nnan = fadd nnan float %op1, %op2
  ; CHECK: %f.nnan = fadd nnan float %op1, %op2
  %f.ninf = fadd ninf float %op1, %op2
  ; CHECK: %f.ninf = fadd ninf float %op1, %op2
  %f.nsz = fadd nsz float %op1, %op2
  ; CHECK: %f.nsz = fadd nsz float %op1, %op2
  %f.arcp = fadd arcp float %op1, %op2
  ; CHECK: %f.arcp = fadd arcp float %op1, %op2
  %f.fast = fadd fast float %op1, %op2
  ; CHECK: %f.fast = fadd fast float %op1, %op2
  ret void
}

;; Type System
%opaquety = type opaque
define void @typesystem() {
  %p0 = bitcast i8* null to i32 (i32)*
  ; CHECK: %p0 = bitcast i8* null to i32 (i32)*
  %p1 = bitcast i8* null to void (i8*)*
  ; CHECK: %p1 = bitcast i8* null to void (i8*)*
  %p2 = bitcast i8* null to i32 (i8*, ...)*
  ; CHECK: %p2 = bitcast i8* null to i32 (i8*, ...)*
  %p3 = bitcast i8* null to { i32, i8 } (i8*, ...)*
  ; CHECK: %p3 = bitcast i8* null to { i32, i8 } (i8*, ...)*
  %p4 = bitcast i8* null to <{ i32, i8 }> (i8*, ...)*
  ; CHECK: %p4 = bitcast i8* null to <{ i32, i8 }> (i8*, ...)*
  %p5 = bitcast i8* null to <{ i32, i8 }> (<{ i8*, i64 }>*, ...)*
  ; CHECK: %p5 = bitcast i8* null to <{ i32, i8 }> (<{ i8*, i64 }>*, ...)*

  %t0 = alloca i1942652
  ; CHECK: %t0 = alloca i1942652
  %t1 = alloca half
  ; CHECK: %t1 = alloca half
  %t2 = alloca float
  ; CHECK: %t2 = alloca float
  %t3 = alloca double
  ; CHECK: %t3 = alloca double
  %t4 = alloca fp128
  ; CHECK: %t4 = alloca fp128
  %t5 = alloca x86_fp80
  ; CHECK: %t5 = alloca x86_fp80
  %t6 = alloca ppc_fp128
  ; CHECK: %t6 = alloca ppc_fp128
  %t7 = alloca x86_mmx
  ; CHECK: %t7 = alloca x86_mmx
  %t8 = alloca %opaquety*
  ; CHECK: %t8 = alloca %opaquety*

  ret void
}

;; Inline Assembler Expressions
define void @inlineasm(i32 %arg) {
  call i32 asm "bswap $0", "=r,r"(i32 %arg)
  ; CHECK: call i32 asm "bswap $0", "=r,r"(i32 %arg)
  call i32 asm sideeffect "blt $1, $2, $3", "=r,r,rm"(i32 %arg, i32 %arg)
  ; CHECK: call i32 asm sideeffect "blt $1, $2, $3", "=r,r,rm"(i32 %arg, i32 %arg)
  ret void
}

;; Instructions

; Instructions -- Terminators
define void @instructions.terminators(i8 %val) { ; XXX: landingpad changed.
; CHECK: define void @instructions.terminators(i8 %val) personality i32 ()* @personality_handler

  br i1 false, label %iftrue, label %iffalse
  ; CHECK: br i1 false, label %iftrue, label %iffalse
  br label %iftrue
  ; CHECK: br label %iftrue
iftrue:
  ret void
  ; CHECK: ret void
iffalse:

  switch i8 %val, label %defaultdest [
  ; CHECK: switch i8 %val, label %defaultdest [
         i8 0, label %defaultdest.0
         ; CHECK: i8 0, label %defaultdest.0
         i8 1, label %defaultdest.1
         ; CHECK: i8 1, label %defaultdest.1
         i8 2, label %defaultdest.2
         ; CHECK: i8 2, label %defaultdest.2
  ]
  ; CHECK: ]
defaultdest:
  ret void
defaultdest.0:
  ret void
defaultdest.1:
  ret void
defaultdest.2:

  indirectbr i8* blockaddress(@instructions.terminators, %defaultdest.2), [label %defaultdest.2]
  ; CHECK: indirectbr i8* blockaddress(@instructions.terminators, %defaultdest.2), [label %defaultdest.2]
  indirectbr i8* blockaddress(@instructions.terminators, %defaultdest.2), [label %defaultdest.2, label %defaultdest.2]
  ; CHECK: indirectbr i8* blockaddress(@instructions.terminators, %defaultdest.2), [label %defaultdest.2, label %defaultdest.2]

  invoke fastcc void @f.fastcc()
  ; CHECK: invoke fastcc void @f.fastcc()
         to label %defaultdest unwind label %exc
         ; CHECK: to label %defaultdest unwind label %exc
exc:
  %cleanup = landingpad i32 personality i32()* @personality_handler cleanup

  resume i32 undef
  ; CHECK: resume i32 undef
  unreachable
  ; CHECK: unreachable

  ret void
}

; Instructions -- Binary Operations
define void @instructions.binops(i8 %op1, i8 %op2) {
  ; nuw x nsw
  add i8 %op1, %op2
  ; CHECK: add i8 %op1, %op2
  add nuw i8 %op1, %op2
  ; CHECK: add nuw i8 %op1, %op2
  add nsw i8 %op1, %op2
  ; CHECK: add nsw i8 %op1, %op2
  add nuw nsw i8 %op1, %op2
  ; CHECK: add nuw nsw i8 %op1, %op2
  sub i8 %op1, %op2
  ; CHECK: sub i8 %op1, %op2
  sub nuw i8 %op1, %op2
  ; CHECK: sub nuw i8 %op1, %op2
  sub nsw i8 %op1, %op2
  ; CHECK: sub nsw i8 %op1, %op2
  sub nuw nsw i8 %op1, %op2
  ; CHECK: sub nuw nsw i8 %op1, %op2
  mul i8 %op1, %op2
  ; CHECK: mul i8 %op1, %op2
  mul nuw i8 %op1, %op2
  ; CHECK: mul nuw i8 %op1, %op2
  mul nsw i8 %op1, %op2
  ; CHECK: mul nsw i8 %op1, %op2
  mul nuw nsw i8 %op1, %op2
  ; CHECK: mul nuw nsw i8 %op1, %op2

  ; exact
  udiv i8 %op1, %op2
  ; CHECK: udiv i8 %op1, %op2
  udiv exact i8 %op1, %op2
  ; CHECK: udiv exact i8 %op1, %op2
  sdiv i8 %op1, %op2
  ; CHECK: sdiv i8 %op1, %op2
  sdiv exact i8 %op1, %op2
  ; CHECK: sdiv exact i8 %op1, %op2

  ; none
  urem i8 %op1, %op2
  ; CHECK: urem i8 %op1, %op2
  srem i8 %op1, %op2
  ; CHECK: srem i8 %op1, %op2

  ret void
}

; Instructions -- Bitwise Binary Operations
define void @instructions.bitwise_binops(i8 %op1, i8 %op2) {
  ; nuw x nsw
  shl i8 %op1, %op2
  ; CHECK: shl i8 %op1, %op2
  shl nuw i8 %op1, %op2
  ; CHECK: shl nuw i8 %op1, %op2
  shl nsw i8 %op1, %op2
  ; CHECK: shl nsw i8 %op1, %op2
  shl nuw nsw i8 %op1, %op2
  ; CHECK: shl nuw nsw i8 %op1, %op2

  ; exact
  lshr i8 %op1, %op2
  ; CHECK: lshr i8 %op1, %op2
  lshr exact i8 %op1, %op2
  ; CHECK: lshr exact i8 %op1, %op2
  ashr i8 %op1, %op2
  ; CHECK: ashr i8 %op1, %op2
  ashr exact i8 %op1, %op2
  ; CHECK: ashr exact i8 %op1, %op2

  ; none
  and i8 %op1, %op2
  ; CHECK: and i8 %op1, %op2
  or i8 %op1, %op2
  ; CHECK: or i8 %op1, %op2
  xor i8 %op1, %op2
  ; CHECK: xor i8 %op1, %op2

  ret void
}

; Instructions -- Vector Operations
define void @instructions.vectorops(<4 x float> %vec, <4 x float> %vec2) {
  extractelement <4 x float> %vec, i8 0
  ; CHECK: extractelement <4 x float> %vec, i8 0
  insertelement <4 x float> %vec, float 3.500000e+00, i8 0
  ; CHECK: insertelement <4 x float> %vec, float 3.500000e+00, i8 0
  shufflevector <4 x float> %vec, <4 x float> %vec2, <2 x i32> zeroinitializer
  ; CHECK: shufflevector <4 x float> %vec, <4 x float> %vec2, <2 x i32> zeroinitializer

  ret void
}

; Instructions -- Aggregate Operations
define void @instructions.aggregateops({ i8, i32 } %up, <{ i8, i32 }> %p,
                                       [3 x i8] %arr, { i8, { i32 }} %n,
                                       <2 x i8*> %pvec, <2 x i64> %offsets) {
  extractvalue { i8, i32 } %up, 0
  ; CHECK: extractvalue { i8, i32 } %up, 0
  extractvalue <{ i8, i32 }> %p, 1
  ; CHECK: extractvalue <{ i8, i32 }> %p, 1
  extractvalue [3 x i8] %arr, 2
  ; CHECK: extractvalue [3 x i8] %arr, 2
  extractvalue { i8, { i32 } } %n, 1, 0
  ; CHECK: extractvalue { i8, { i32 } } %n, 1, 0

  insertvalue { i8, i32 } %up, i8 1, 0
  ; CHECK: insertvalue { i8, i32 } %up, i8 1, 0
  insertvalue <{ i8, i32 }> %p, i32 2, 1
  ; CHECK: insertvalue <{ i8, i32 }> %p, i32 2, 1
  insertvalue [3 x i8] %arr, i8 0, 0
  ; CHECK: insertvalue [3 x i8] %arr, i8 0, 0
  insertvalue { i8, { i32 } } %n, i32 0, 1, 0
  ; CHECK: insertvalue { i8, { i32 } } %n, i32 0, 1, 0

  %up.ptr = alloca { i8, i32 }
  %p.ptr = alloca <{ i8, i32 }>
  %arr.ptr = alloca [3 x i8]
  %n.ptr = alloca { i8, { i32 } }

  ; XXX: The parser spits out the load type here.
  getelementptr { i8, i32 }* %up.ptr, i8 0
  ; CHECK: getelementptr { i8, i32 }, { i8, i32 }* %up.ptr, i8 0
  getelementptr <{ i8, i32 }>* %p.ptr, i8 1
  ; CHECK: getelementptr <{ i8, i32 }>, <{ i8, i32 }>* %p.ptr, i8 1
  getelementptr [3 x i8]* %arr.ptr, i8 2
  ; CHECK: getelementptr [3 x i8], [3 x i8]* %arr.ptr, i8 2
  getelementptr { i8, { i32 } }* %n.ptr, i32 0, i32 1
  ; CHECK: getelementptr { i8, { i32 } }, { i8, { i32 } }* %n.ptr, i32 0, i32 1
  getelementptr inbounds { i8, { i32 } }* %n.ptr, i32 1, i32 0
  ; CHECK: getelementptr inbounds { i8, { i32 } }, { i8, { i32 } }* %n.ptr, i32 1, i32 0
  getelementptr <2 x i8*> %pvec, <2 x i64> %offsets
  ; CHECK: getelementptr i8, <2 x i8*> %pvec, <2 x i64> %offsets

  ret void
}

; Instructions -- Memory Access and Addressing Operations
!7 = !{i32 1}
!8 = !{}
!9 = !{i64 4}
define void @instructions.memops(i32** %base) {
  alloca i32, i8 4, align 4
  ; CHECK: alloca i32, i8 4, align 4
  alloca inalloca i32, i8 4, align 4
  ; CHECK: alloca inalloca i32, i8 4, align 4

  load i32** %base, align 8, !invariant.load !7, !nontemporal !8, !nonnull !7, !dereferenceable !9
  ; CHECK: load i32*, i32** %base, align 8, !invariant.load !7, !nontemporal !8, !nonnull !7, !dereferenceable !9
  load volatile i32** %base, align 8, !invariant.load !7, !nontemporal !8, !nonnull !7, !dereferenceable !9
  ; CHECK: load volatile i32*, i32** %base, align 8, !invariant.load !7, !nontemporal !8, !nonnull !7, !dereferenceable !9

  store i32* null, i32** %base, align 4, !nontemporal !8
  ; CHECK: store i32* null, i32** %base, align 4, !nontemporal !8
  store volatile i32* null, i32** %base, align 4, !nontemporal !8
  ; CHECK: store volatile i32* null, i32** %base, align 4, !nontemporal !8

  ret void
}

; Instructions -- Conversion Operations
define void @instructions.conversions() {
  trunc i32 -1 to i1
  ; CHECK: trunc i32 -1 to i1
  zext i32 -1 to i64
  ; CHECK: zext i32 -1 to i64
  sext i32 -1 to i64
  ; CHECK: sext i32 -1 to i64
  fptrunc float undef to half
  ; CHECK: fptrunc float undef to half
  fpext half undef to float
  ; CHECK: fpext half undef to float
  fptoui float undef to i32
  ; CHECK: fptoui float undef to i32
  fptosi float undef to i32
  ; CHECK: fptosi float undef to i32
  uitofp i32 1 to float
  ; CHECK: uitofp i32 1 to float
  sitofp i32 -1 to float
  ; CHECK: sitofp i32 -1 to float
  ptrtoint i8* null to i64
  ; CHECK: ptrtoint i8* null to i64
  inttoptr i64 0 to i8*
  ; CHECK: inttoptr i64 0 to i8*
  bitcast i32 0 to i32
  ; CHECK: bitcast i32 0 to i32
  addrspacecast i32* null to i32 addrspace(1)*
  ; CHECK: addrspacecast i32* null to i32 addrspace(1)*

  ret void
}

; Instructions -- Other Operations
define void @instructions.other(i32 %op1, i32 %op2, half %fop1, half %fop2) {
entry:
  icmp eq  i32 %op1, %op2
  ; CHECK: icmp eq  i32 %op1, %op2
  icmp ne  i32 %op1, %op2
  ; CHECK: icmp ne  i32 %op1, %op2
  icmp ugt i32 %op1, %op2
  ; CHECK: icmp ugt i32 %op1, %op2
  icmp uge i32 %op1, %op2
  ; CHECK: icmp uge i32 %op1, %op2
  icmp ult i32 %op1, %op2
  ; CHECK: icmp ult i32 %op1, %op2
  icmp ule i32 %op1, %op2
  ; CHECK: icmp ule i32 %op1, %op2
  icmp sgt i32 %op1, %op2
  ; CHECK: icmp sgt i32 %op1, %op2
  icmp sge i32 %op1, %op2
  ; CHECK: icmp sge i32 %op1, %op2
  icmp slt i32 %op1, %op2
  ; CHECK: icmp slt i32 %op1, %op2
  icmp sle i32 %op1, %op2
  ; CHECK: icmp sle i32 %op1, %op2

  fcmp false half %fop1, %fop2
  ; CHECK: fcmp false half %fop1, %fop2
  fcmp oeq   half %fop1, %fop2
  ; CHECK: fcmp oeq   half %fop1, %fop2
  fcmp ogt   half %fop1, %fop2
  ; CHECK: fcmp ogt   half %fop1, %fop2
  fcmp oge   half %fop1, %fop2
  ; CHECK: fcmp oge   half %fop1, %fop2
  fcmp olt   half %fop1, %fop2
  ; CHECK: fcmp olt   half %fop1, %fop2
  fcmp ole   half %fop1, %fop2
  ; CHECK: fcmp ole   half %fop1, %fop2
  fcmp one   half %fop1, %fop2
  ; CHECK: fcmp one   half %fop1, %fop2
  fcmp ord   half %fop1, %fop2
  ; CHECK: fcmp ord   half %fop1, %fop2
  fcmp ueq   half %fop1, %fop2
  ; CHECK: fcmp ueq   half %fop1, %fop2
  fcmp ugt   half %fop1, %fop2
  ; CHECK: fcmp ugt   half %fop1, %fop2
  fcmp uge   half %fop1, %fop2
  ; CHECK: fcmp uge   half %fop1, %fop2
  fcmp ult   half %fop1, %fop2
  ; CHECK: fcmp ult   half %fop1, %fop2
  fcmp ule   half %fop1, %fop2
  ; CHECK: fcmp ule   half %fop1, %fop2
  fcmp une   half %fop1, %fop2
  ; CHECK: fcmp une   half %fop1, %fop2
  fcmp uno   half %fop1, %fop2
  ; CHECK: fcmp uno   half %fop1, %fop2
  fcmp true  half %fop1, %fop2
  ; CHECK: fcmp true  half %fop1, %fop2

  br label %exit
L1:
  %v1 = add i32 %op1, %op2
  br label %exit
L2:
  %v2 = add i32 %op1, %op2
  br label %exit
exit:
  phi i32 [ %v1, %L1 ], [ %v2, %L2 ], [ %op1, %entry ]
  ; CHECK: phi i32 [ %v1, %L1 ], [ %v2, %L2 ], [ %op1, %entry ]

  select i1 true, i32 0, i32 1
  ; CHECK: select i1 true, i32 0, i32 1
  select <2 x i1> <i1 true, i1 false>, <2 x i8> <i8 2, i8 3>, <2 x i8> <i8 3, i8 2>
  ; CHECK: select <2 x i1> <i1 true, i1 false>, <2 x i8> <i8 2, i8 3>, <2 x i8> <i8 3, i8 2>

  call void @f.nobuiltin() builtin
  ; CHECK: call void @f.nobuiltin() #35

  call fastcc noalias i32* @f.noalias() noinline
  ; CHECK: call fastcc noalias i32* @f.noalias() #11
  tail call ghccc nonnull i32* @f.nonnull() minsize
  ; CHECK: tail call ghccc nonnull i32* @f.nonnull() #6

  ret void
}

define void @instructions.call_musttail(i8* inalloca %val) {
  musttail call void @f.param.inalloca(i8* inalloca %val)
  ; CHECK: musttail call void @f.param.inalloca(i8* inalloca(i8) %val)

  ret void
}

declare void @llvm.donothing() nounwind readnone

declare i32 @personality_handler()

define void @instructions.landingpad() {
; CHECK: define void @instructions.landingpad() personality i32 ()* @personality_handler

  invoke void @llvm.donothing() to label %proceed unwind label %catch1
  invoke void @llvm.donothing() to label %proceed unwind label %catch2
  invoke void @llvm.donothing() to label %proceed unwind label %catch3
  invoke void @llvm.donothing() to label %proceed unwind label %catch4

catch1:
  landingpad i32
  ; CHECK: landingpad i32
             personality i32()* @personality_handler
             cleanup
             ; CHECK: cleanup
  br label %proceed

catch2:
  landingpad i32
  ; CHECK: landingpad i32
             personality i32()* @personality_handler
             cleanup
             ; CHECK: cleanup
             catch i32* null
             ; CHECK: catch i32* null
  br label %proceed

catch3:
  landingpad i32
  ; CHECK: landingpad i32
             personality i32()* @personality_handler
             cleanup
             ; CHECK: cleanup
             catch i32* null
             ; CHECK: catch i32* null
             catch i32* null
             ; CHECK: catch i32* null
  br label %proceed

catch4:
  landingpad i32
  ; CHECK: landingpad i32
             personality i32()* @personality_handler
             filter [2 x i32] zeroinitializer
             ; CHECK: filter [2 x i32] zeroinitializer
  br label %proceed

proceed:
  ret void
}

;; Intrinsic Functions

; Intrinsic Functions -- Variable Argument Handling
declare void @llvm.va_start(i8*)
declare void @llvm.va_copy(i8*, i8*)
declare void @llvm.va_end(i8*)
define void @instructions.va_arg(i8* %v, ...) {
  %ap = alloca i8*
  %ap2 = bitcast i8** %ap to i8*

  call void @llvm.va_start(i8* %ap2)
  ; CHECK: call void @llvm.va_start(i8* %ap2)

  va_arg i8* %ap2, i32
  ; CHECK: va_arg i8* %ap2, i32

  call void @llvm.va_copy(i8* %v, i8* %ap2)
  ; CHECK: call void @llvm.va_copy(i8* %v, i8* %ap2)

  call void @llvm.va_end(i8* %ap2)
  ; CHECK: call void @llvm.va_end(i8* %ap2)

  ret void
}

; Intrinsic Functions -- Accurate Garbage Collection
declare void @llvm.gcroot(i8**, i8*)
declare i8* @llvm.gcread(i8*, i8**)
declare void @llvm.gcwrite(i8*, i8*, i8**)
define void @intrinsics.gc() gc "shadow-stack" {
  %ptrloc = alloca i8*
  call void @llvm.gcroot(i8** %ptrloc, i8* null)
  ; CHECK: call void @llvm.gcroot(i8** %ptrloc, i8* null)

  call i8* @llvm.gcread(i8* null, i8** %ptrloc)
  ; CHECK: call i8* @llvm.gcread(i8* null, i8** %ptrloc)

  %ref = alloca i8
  call void @llvm.gcwrite(i8* %ref, i8* null, i8** %ptrloc)
  ; CHECK: call void @llvm.gcwrite(i8* %ref, i8* null, i8** %ptrloc)

  ret void
}

; Intrinsic Functions -- Code Generation
declare i8* @llvm.returnaddress(i32)
declare i8* @llvm.frameaddress(i32)
declare i32 @llvm.read_register.i32(metadata)
declare i64 @llvm.read_register.i64(metadata)
declare void @llvm.write_register.i32(metadata, i32)
declare void @llvm.write_register.i64(metadata, i64)
declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)
declare void @llvm.prefetch(i8*, i32, i32, i32)
declare void @llvm.pcmarker(i32)
declare i64 @llvm.readcyclecounter()
declare void @llvm.clear_cache(i8*, i8*)
declare void @llvm.instrprof_increment(i8*, i64, i32, i32)

!10 = !{!"rax"}
define void @intrinsics.codegen() {
  call i8* @llvm.returnaddress(i32 1)
  ; CHECK: call i8* @llvm.returnaddress(i32 1)
  call i8* @llvm.frameaddress(i32 1)
  ; CHECK: call i8* @llvm.frameaddress.p0i8(i32 1)

  call i32 @llvm.read_register.i32(metadata !10)
  ; CHECK: call i32 @llvm.read_register.i32(metadata !10)
  call i64 @llvm.read_register.i64(metadata !10)
  ; CHECK: call i64 @llvm.read_register.i64(metadata !10)
  call void @llvm.write_register.i32(metadata !10, i32 0)
  ; CHECK: call void @llvm.write_register.i32(metadata !10, i32 0)
  call void @llvm.write_register.i64(metadata !10, i64 0)
  ; CHECK: call void @llvm.write_register.i64(metadata !10, i64 0)

  %stack = call i8* @llvm.stacksave()
  ; CHECK: %stack = call i8* @llvm.stacksave()
  call void @llvm.stackrestore(i8* %stack)
  ; CHECK: call void @llvm.stackrestore(i8* %stack)

  call void @llvm.prefetch(i8* %stack, i32 0, i32 3, i32 0)
  ; CHECK: call void @llvm.prefetch.p0i8(i8* %stack, i32 0, i32 3, i32 0)

  call void @llvm.pcmarker(i32 1)
  ; CHECK: call void @llvm.pcmarker(i32 1)

  call i64 @llvm.readcyclecounter()
  ; CHECK: call i64 @llvm.readcyclecounter()

  call void @llvm.clear_cache(i8* null, i8* null)
  ; CHECK: call void @llvm.clear_cache(i8* null, i8* null)

  call void @llvm.instrprof_increment(i8* null, i64 0, i32 0, i32 0)
  ; CHECK: call void @llvm.instrprof_increment(i8* null, i64 0, i32 0, i32 0)

  ret void
}

; CHECK: attributes #0 = { alignstack=4 }
; CHECK: attributes #1 = { alignstack=8 }
; CHECK: attributes #2 = { alwaysinline }
; CHECK: attributes #3 = { cold }
; CHECK: attributes #4 = { inlinehint }
; CHECK: attributes #5 = { jumptable }
; CHECK: attributes #6 = { minsize }
; CHECK: attributes #7 = { naked }
; CHECK: attributes #8 = { nobuiltin }
; CHECK: attributes #9 = { noduplicate }
; CHECK: attributes #10 = { noimplicitfloat }
; CHECK: attributes #11 = { noinline }
; CHECK: attributes #12 = { nonlazybind }
; CHECK: attributes #13 = { noredzone }
; CHECK: attributes #14 = { noreturn }
; CHECK: attributes #15 = { nounwind }
; CHECK: attributes #16 = { noinline optnone }
; CHECK: attributes #17 = { optsize }
; CHECK: attributes #18 = { readnone }
; CHECK: attributes #19 = { readonly }
; CHECK: attributes #20 = { returns_twice }
; CHECK: attributes #21 = { sanitize_address }
; CHECK: attributes #22 = { sanitize_memory }
; CHECK: attributes #23 = { sanitize_thread }
; CHECK: attributes #24 = { ssp }
; CHECK: attributes #25 = { sspreq }
; CHECK: attributes #26 = { sspstrong }
; CHECK: attributes #27 = { uwtable }
; CHECK: attributes #28 = { "cpu"="cortex-a8" }
; CHECK: attributes #29 = { nofree nosync nounwind readnone willreturn }
; CHECK: attributes #30 = { nofree nosync nounwind willreturn }
; CHECK: attributes #31 = { argmemonly nounwind readonly }
; CHECK: attributes #32 = { argmemonly nounwind }
; CHECK: attributes #33 = { nounwind readonly }
; CHECK: attributes #34 = { inaccessiblemem_or_argmemonly nofree nosync nounwind willreturn }
; CHECK: attributes #35 = { builtin }

;; Metadata

; Metadata -- Module flags
!llvm.module.flags = !{!0, !1, !2, !4, !5, !6}
; CHECK: !llvm.module.flags = !{!0, !1, !2, !4, !5, !6}

!0 = !{i32 1, !"mod1", i32 0}
; CHECK: !0 = !{i32 1, !"mod1", i32 0}
!1 = !{i32 2, !"mod2", i32 0}
; CHECK: !1 = !{i32 2, !"mod2", i32 0}
!2 = !{i32 3, !"mod3", !3}
; CHECK: !2 = !{i32 3, !"mod3", !3}
!3 = !{!"mod6", !0}
; CHECK: !3 = !{!"mod6", !0}
!4 = !{i32 4, !"mod4", i32 0}
; CHECK: !4 = !{i32 4, !"mod4", i32 0}
!5 = !{i32 5, !"mod5", !0}
; CHECK: !5 = !{i32 5, !"mod5", !0}
!6 = !{i32 6, !"mod6", !0}
; CHECK: !6 = !{i32 6, !"mod6", !0}

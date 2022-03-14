; RUN: llc -mtriple x86_64-unknown-unknown -exception-model sjlj -verify-machineinstrs=0 -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s --check-prefix=NUM
; RUN: llc -mtriple x86_64-unknown-unknown -exception-model sjlj -verify-machineinstrs=0 -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s --check-prefix=SJLJ

; NUM-COUNT-3: endbr64

;SJLJ:       main:                                  # @main
;SJLJ-NEXT: .Lfunc_begin0:
;SJLJ-NEXT: # %bb.0:                                # %entry
;SJLJ-NEXT:         endbr64
;SJLJ-NEXT:         pushq   %rbp
;SJLJ:               callq   _Unwind_SjLj_Register
;SJLJ-NEXT: .Ltmp0:
;SJLJ-NEXT:         callq   _Z3foov
;SJLJ-NEXT: .Ltmp1:
;SJLJ-NEXT: # %bb.1:                                # %invoke.cont
;SJLJ-NEXT:         movl
;SJLJ-NEXT: .LBB0_7:                                # %return
;SJLJ:               callq   _Unwind_SjLj_Unregister
;SJLJ:               retq
;SJLJ-NEXT: .LBB0_9:
;SJLJ-NEXT:         endbr64
;SJLJ-NEXT:         movl
;SJLJ-NEXT:         cmpl
;SJLJ-NEXT:         jb      .LBB0_10
;SJLJ-NEXT: # %bb.11:
;SJLJ-NEXT:         ud2
;SJLJ-NEXT: .LBB0_10:
;SJLJ-NEXT:         leaq    .LJTI0_0(%rip), %rcx
;SJLJ-NEXT:         jmpq    *(%rcx,%rax,8)
;SJLJ-NEXT: .LBB0_2:                                # %lpad
;SJLJ-NEXT: .Ltmp2:
;SJLJ-NEXT:         endbr64
;SJLJ:               jne     .LBB0_4
;SJLJ-NEXT: # %bb.3:                                # %catch3
;SJLJ:               callq   __cxa_begin_catch
;SJLJ:               jmp     .LBB0_6
;SJLJ-NEXT: .LBB0_4:                                # %catch.fallthrough
;SJLJ-NEXT:         cmpl
;SJLJ-NEXT:         jne     .LBB0_8
;SJLJ-NEXT: # %bb.5:                                # %catch
;SJLJ:               callq   __cxa_begin_catch
;SJLJ:               cmpb
;SJLJ-NEXT: .LBB0_6:                                # %return
;SJLJ:               callq   __cxa_end_catch
;SJLJ-NEXT:         jmp     .LBB0_7
;SJLJ-NEXT: .LBB0_8:                                # %eh.resume
;SJLJ-NEXT:         movl
;SJLJ-NEXT: .Lfunc_end0:
;SJLJ:      .LJTI0_0:
;SJLJ-NEXT:         .quad   .LBB0_2

@_ZTIi = external dso_local constant i8*
@_ZTIc = external dso_local constant i8*

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @main() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  %retval = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %x = alloca i8, align 1
  %x4 = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  invoke void @_Z3foov()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* bitcast (i8** @_ZTIc to i8*)
  %1 = extractvalue { i8*, i32 } %0, 0
  store i8* %1, i8** %exn.slot, align 8
  %2 = extractvalue { i8*, i32 } %0, 1
  store i32 %2, i32* %ehselector.slot, align 4
  br label %catch.dispatch

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot, align 4
  %3 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #3
  %matches = icmp eq i32 %sel, %3
  br i1 %matches, label %catch3, label %catch.fallthrough

catch3:                                           ; preds = %catch.dispatch
  %exn5 = load i8*, i8** %exn.slot, align 8
  %4 = call i8* @__cxa_begin_catch(i8* %exn5) #3
  %5 = bitcast i8* %4 to i32*
  %6 = load i32, i32* %5, align 4
  store i32 %6, i32* %x4, align 4
  %7 = load i32, i32* %x4, align 4
  %cmp6 = icmp ne i32 %7, 5
  %conv7 = zext i1 %cmp6 to i32
  store i32 %conv7, i32* %retval, align 4
  call void @__cxa_end_catch() #3
  br label %return

catch.fallthrough:                                ; preds = %catch.dispatch
  %8 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIc to i8*)) #3
  %matches1 = icmp eq i32 %sel, %8
  br i1 %matches1, label %catch, label %eh.resume

catch:                                            ; preds = %catch.fallthrough
  %exn = load i8*, i8** %exn.slot, align 8
  %9 = call i8* @__cxa_begin_catch(i8* %exn) #3
  %10 = load i8, i8* %9, align 1
  store i8 %10, i8* %x, align 1
  %11 = load i8, i8* %x, align 1
  %conv = sext i8 %11 to i32
  %cmp = icmp ne i32 %conv, 3
  %conv2 = zext i1 %cmp to i32
  store i32 %conv2, i32* %retval, align 4
  call void @__cxa_end_catch() #3
  br label %return

try.cont:                                         ; preds = %invoke.cont
  store i32 1, i32* %retval, align 4
  br label %return

return:                                           ; preds = %try.cont, %catch3, %catch
  %12 = load i32, i32* %retval, align 4
  ret i32 %12

eh.resume:                                        ; preds = %catch.fallthrough
  %exn8 = load i8*, i8** %exn.slot, align 8
  %sel9 = load i32, i32* %ehselector.slot, align 4
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn8, 0
  %lpad.val10 = insertvalue { i8*, i32 } %lpad.val, i32 %sel9, 1
  resume { i8*, i32 } %lpad.val10
}

declare dso_local void @_Z3foov() #1

declare dso_local i32 @__gxx_personality_sj0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

declare dso_local i8* @__cxa_begin_catch(i8*)

declare dso_local void @__cxa_end_catch()

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"cf-protection-return", i32 1}
!2 = !{i32 4, !"cf-protection-branch", i32 1}

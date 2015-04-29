; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S < %s | FileCheck %s

; Notionally based on this C++ source:
; int liveout_catch(int p) {
;   int val = p + 1;
;   try {
;     might_throw();
;   } catch (int) {
;     val++;
;   }
;   return val;
; }

declare void @llvm.eh.begincatch(i8*, i8*)
declare void @llvm.eh.endcatch()
declare void @might_throw()
declare i32 @__CxxFrameHandler3(...)
declare i32 @llvm.eh.typeid.for(i8*)

@typeinfo.int = external global i32

define i32 @liveout_catch(i32 %p) {
entry:
  %val.entry = add i32 %p, 1
  invoke void @might_throw()
      to label %ret unwind label %lpad

lpad:
  %ehvals = landingpad { i8*, i32 } personality i32 (...)* @__CxxFrameHandler3
      cleanup
      catch i32* @typeinfo.int
  %ehptr = extractvalue { i8*, i32 } %ehvals, 0
  %sel = extractvalue { i8*, i32 } %ehvals, 1
  %int_sel = call i32 @llvm.eh.typeid.for(i8* bitcast (i32* @typeinfo.int to i8*))
  %match = icmp eq i32 %sel, %int_sel
  br i1 %match, label %catchit, label %resume

catchit:
  call void @llvm.eh.begincatch(i8* %ehptr, i8* null)
  %val.lpad = add i32 %val.entry, 1
  call void @llvm.eh.endcatch()
  br label %ret

ret:
  %rv = phi i32 [%val.entry, %entry], [%val.lpad, %catchit]
  ret i32 %rv

resume:
  resume {i8*, i32} %ehvals
}

; CHECK-LABEL: define i32 @liveout_catch(i32 %p)
; CHECK: %val.entry = add i32 %p, 1
; CHECK-NEXT: store i32 %val.entry, i32* %val.entry.reg2mem
; CHECK: invoke void @might_throw()
;
; CHECK: landingpad
; CHECK: indirectbr i8* {{.*}}, [label %catchit.split]
;
; CHECK: catchit.split:
; CHECK: load i32, i32* %val.lpad.reg2mem
; CHECK: br label %ret
;
; CHECK: ret:
; CHECK: %rv = phi i32 [ {{.*}}, %entry ], [ {{.*}}, %catchit.split ]
; CHECK: ret i32

; CHECK-LABEL: define internal i8* @liveout_catch.catch(i8*, i8*)
; CHECK: %[[val:[^ ]*]] = load i32, i32*
; CHECK-NEXT: %[[val_lpad:[^ ]*]] = add i32 %[[val]], 1
; CHECK-NEXT: store i32 %[[val_lpad]], i32*
; CHECK: ret i8* blockaddress(@liveout_catch, %catchit.split)

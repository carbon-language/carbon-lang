; RUN: llc < %s -mtriple=powerpc-apple-darwin10 -relocation-model=pic -disable-fp-elim | FileCheck %s
; <rdar://problem/7604010>

%struct.NSString = type opaque
%struct._DTOpaqueAssertStruct = type { }
%struct.objc_selector = type opaque

@_cmd = constant %struct._DTOpaqueAssertStruct zeroinitializer ; <%struct._DTOpaqueAssertStruct*> [#uses=1]
@OBJC_IMAGE_INFO = private constant [2 x i32] zeroinitializer, section "__OBJC, __image_info,regular" ; <[2 x i32]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast ([2 x i32]* @OBJC_IMAGE_INFO to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define void @_DTAssertionFailureHandler(i8* %objp, i8* %selp, i32 %lineNumber, %struct.NSString* %msgFormat, ...) nounwind ssp {
entry:
  %objp_addr = alloca i8*                         ; <i8**> [#uses=1]
  %selp_addr = alloca i8*                         ; <i8**> [#uses=3]
  %lineNumber_addr = alloca i32                   ; <i32*> [#uses=1]
  %msgFormat_addr = alloca %struct.NSString*      ; <%struct.NSString**> [#uses=1]
  %iftmp.0 = alloca %struct.objc_selector*        ; <%struct.objc_selector**> [#uses=3]
  %sel = alloca %struct.objc_selector*            ; <%struct.objc_selector**> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i8* %objp, i8** %objp_addr
  store i8* %selp, i8** %selp_addr
  store i32 %lineNumber, i32* %lineNumber_addr
  store %struct.NSString* %msgFormat, %struct.NSString** %msgFormat_addr
  %0 = load i8** %selp_addr, align 4              ; <i8*> [#uses=1]
  %1 = icmp ne i8* %0, bitcast (%struct._DTOpaqueAssertStruct* @_cmd to i8*) ; <i1> [#uses=1]
  br i1 %1, label %bb, label %bb1

bb:                                               ; preds = %entry
  %2 = load i8** %selp_addr, align 4              ; <i8*> [#uses=1]
  %3 = bitcast i8* %2 to %struct.objc_selector**  ; <%struct.objc_selector**> [#uses=1]
  %4 = load %struct.objc_selector** %3, align 4   ; <%struct.objc_selector*> [#uses=1]
  store %struct.objc_selector* %4, %struct.objc_selector** %iftmp.0, align 4
  br label %bb2

bb1:                                              ; preds = %entry
  store %struct.objc_selector* null, %struct.objc_selector** %iftmp.0, align 4
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %5 = load %struct.objc_selector** %iftmp.0, align 4 ; <%struct.objc_selector*> [#uses=1]
  store %struct.objc_selector* %5, %struct.objc_selector** %sel, align 4
  br label %return

return:                                           ; preds = %bb2
  ret void

; CHECK:      __cmd:
; CHECK-NEXT: .space 1
}

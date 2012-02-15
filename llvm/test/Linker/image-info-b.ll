; RUN: true
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%0 = type opaque
%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._objc_cache*, i8* (i8*, i8*)**, %struct._class_ro_t* }
%struct._objc_cache = type opaque
%struct._class_ro_t = type { i32, i32, i32, i8*, i8*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._ivar_list_t*, i8*, %struct._prop_list_t* }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_protocol_list = type { i64, [0 x %struct._protocol_t*] }
%struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8** }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { i64*, i8*, i8*, i32, i32 }
%struct._message_ref_t = type { i8*, i8* }

@"OBJC_CLASS_$_NSAutoreleasePool" = external global %struct._class_t
@"\01L_OBJC_CLASSLIST_REFERENCES_$_" = internal global %struct._class_t* @"OBJC_CLASS_$_NSAutoreleasePool", section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_METH_VAR_NAME_" = internal global [6 x i8] c"alloc\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01l_objc_msgSend_fixup_alloc" = weak hidden global { i8* (i8*, %struct._message_ref_t*, ...)*, i8* } { i8* (i8*, %struct._message_ref_t*, ...)* @objc_msgSend_fixup, i8* getelementptr inbounds ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0) }, section "__DATA, __objc_msgrefs, coalesced", align 16
@"\01L_OBJC_METH_VAR_NAME_1" = internal global [5 x i8] c"init\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global i8* getelementptr inbounds ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_1", i32 0, i32 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@llvm.used = appending global [4 x i8*] [i8* bitcast (%struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_" to i8*), i8* getelementptr inbounds ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_1", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_" to i8*)], section "llvm.metadata"

define i32 @main() uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %pool = alloca %0*, align 8
  %pond = alloca %0*, align 8
  store i32 0, i32* %retval
  %0 = load %struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_"
  %1 = bitcast %struct._class_t* %0 to i8*
  %msgSend_fn = load i8** getelementptr inbounds (%struct._message_ref_t* bitcast ({ i8* (i8*, %struct._message_ref_t*, ...)*, i8* }* @"\01l_objc_msgSend_fixup_alloc" to %struct._message_ref_t*), i32 0, i32 0)
  %2 = bitcast i8* %msgSend_fn to i8* (i8*, %struct._message_ref_t*)*
  %call = call i8* %2(i8* %1, %struct._message_ref_t* bitcast ({ i8* (i8*, %struct._message_ref_t*, ...)*, i8* }* @"\01l_objc_msgSend_fixup_alloc" to %struct._message_ref_t*))
  %3 = bitcast i8* %call to %0*
  %4 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", !invariant.load !4
  %5 = bitcast %0* %3 to i8*
  %call1 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %5, i8* %4)
  %6 = bitcast i8* %call1 to %0*
  store %0* %6, %0** %pool, align 8
  %call2 = call %0* @_Z3foov()
  store %0* %call2, %0** %pond, align 8
  ret i32 0
}

declare i8* @objc_msgSend_fixup(i8*, %struct._message_ref_t*, ...)

declare i8* @objc_msgSend(i8*, i8*, ...)

declare %0* @_Z3foov()

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
!1 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
!2 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!3 = metadata !{i32 1, metadata !"Objective-C Garbage Collection", i32 2}
!4 = metadata !{}

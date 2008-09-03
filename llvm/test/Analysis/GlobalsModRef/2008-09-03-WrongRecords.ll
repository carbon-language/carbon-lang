; RUN: llvm-as < %s | opt -globalsmodref-aa -markmodref | llvm-dis | not grep {define.*read}

	%struct.FRAME.ce3806g = type { %struct.string___XUB, %struct.string___XUB, %struct.string___XUB, %struct.string___XUB }
	%struct.FRAME.ce3806g__fxio__put__4 = type { i32, i32, i32, %struct.system__file_control_block__pstring*, i32, i32, i8 }
	%struct.ada__streams__root_stream_type = type { %struct.ada__tags__dispatch_table* }
	%struct.ada__tags__dispatch_table = type { [1 x i32] }
	%struct.ada__text_io__text_afcb = type { %struct.system__file_control_block__afcb, i32, i32, i32, i32, i32, %struct.ada__text_io__text_afcb*, i8, i8 }
	%struct.exception = type { i8, i8, i32, i8*, i8*, i32, i8* }
	%struct.string___XUB = type { i32, i32 }
	%struct.system__file_control_block__afcb = type { %struct.ada__streams__root_stream_type, i32, %struct.system__file_control_block__pstring, %struct.system__file_control_block__pstring, i8, i8, i8, i8, i8, i8, i8, %struct.system__file_control_block__afcb*, %struct.system__file_control_block__afcb* }
	%struct.system__file_control_block__pstring = type { i8*, %struct.string___XUB* }
	%struct.system__standard_library__exception_data = type { i8, i8, i32, i32, %struct.system__standard_library__exception_data*, i32, void ()* }
@.str = internal constant [12 x i8] c"system.ads\00\00"		; <[12 x i8]*> [#uses=1]
@.str1 = internal constant [14 x i8] c"a-tifiio.adb\00\00"		; <[14 x i8]*> [#uses=1]
@system__soft_links__abort_undefer = external global void ()*		; <void ()**> [#uses=6]
@.str2 = internal constant [47 x i8] c"a-tifiio.adb:327 instantiated at ce3806g.adb:52"		; <[47 x i8]*> [#uses=1]
@C.354.2200 = internal constant %struct.string___XUB { i32 1, i32 47 }		; <%struct.string___XUB*> [#uses=2]
@ada__io_exceptions__data_error = external global %struct.exception		; <%struct.exception*> [#uses=1]
@constraint_error = external global %struct.exception		; <%struct.exception*> [#uses=2]
@__gnat_all_others_value = external constant i32		; <i32*> [#uses=21]
@.str3 = internal constant [10 x i8] c"0123456789"		; <[10 x i8]*> [#uses=2]
@ada__text_io__current_out = external global %struct.ada__text_io__text_afcb*		; <%struct.ada__text_io__text_afcb**> [#uses=1]
@.str4 = internal constant [126 x i8] c"CHECK THAT FIXED_IO PUT OPERATES ON FILES OF MODE OUT_FILE AND IF NO FILE IS SPECIFIED THE CURRENT DEFAULT OUTPUT FILE IS USED"		; <[126 x i8]*> [#uses=1]
@C.131.1559 = internal constant %struct.string___XUB { i32 1, i32 126 }		; <%struct.string___XUB*> [#uses=1]
@.str5 = internal constant [7 x i8] c"CE3806G"		; <[7 x i8]*> [#uses=1]
@C.132.1562 = internal constant %struct.string___XUB { i32 1, i32 7 }		; <%struct.string___XUB*> [#uses=1]
@incompleteF.1176.b = internal global i1 false		; <i1*> [#uses=2]
@incomplete.1177 = internal global %struct.exception { i8 0, i8 65, i32 23, i8* getelementptr ([23 x i8]* @incompleteE.1174, i32 0, i32 0), i8* null, i32 0, i8* null }		; <%struct.exception*> [#uses=15]
@incompleteE.1174 = internal global [23 x i8] c"CE3806G.B_1.INCOMPLETE\00"		; <[23 x i8]*> [#uses=1]
@.str6 = internal constant [0 x i8] zeroinitializer		; <[0 x i8]*> [#uses=1]
@C.136.1568 = internal constant %struct.string___XUB { i32 1, i32 0 }		; <%struct.string___XUB*> [#uses=8]
@.str7 = internal constant [50 x i8] c"USE_ERROR RAISED ON TEXT CREATE WITH OUT_FILE MODE"		; <[50 x i8]*> [#uses=1]
@C.139.1577 = internal constant %struct.string___XUB { i32 1, i32 50 }		; <%struct.string___XUB*> [#uses=1]
@.str8 = internal constant [14 x i8] c"ce3806g.adb:65"		; <[14 x i8]*> [#uses=1]
@C.140.1580 = internal constant %struct.string___XUB { i32 1, i32 14 }		; <%struct.string___XUB*> [#uses=3]
@.str9 = internal constant [51 x i8] c"NAME_ERROR RAISED ON TEXT CREATE WITH OUT_FILE MODE"		; <[51 x i8]*> [#uses=1]
@C.143.1585 = internal constant %struct.string___XUB { i32 1, i32 51 }		; <%struct.string___XUB*> [#uses=1]
@.str10 = internal constant [14 x i8] c"ce3806g.adb:69"		; <[14 x i8]*> [#uses=1]
@.str12 = internal constant [47 x i8] c"USE_ERROR RAISED ON TEXT OPEN WITH IN_FILE MODE"		; <[47 x i8]*> [#uses=1]
@.str13 = internal constant [14 x i8] c"ce3806g.adb:88"		; <[14 x i8]*> [#uses=1]
@.str14 = internal constant [33 x i8] c"VALUE INCORRECT - FIXED FROM FILE"		; <[33 x i8]*> [#uses=1]
@C.162.1637 = internal constant %struct.string___XUB { i32 1, i32 33 }		; <%struct.string___XUB*> [#uses=1]
@.str15 = internal constant [36 x i8] c"VALUE INCORRECT - FIXED FROM DEFAULT"		; <[36 x i8]*> [#uses=1]
@C.164.1642 = internal constant %struct.string___XUB { i32 1, i32 36 }		; <%struct.string___XUB*> [#uses=1]
@ada__io_exceptions__use_error = external global %struct.exception		; <%struct.exception*> [#uses=4]
@ada__io_exceptions__name_error = external global %struct.exception		; <%struct.exception*> [#uses=2]

define void @_ada_ce3806g() {
entry:
	%A.266.i = alloca %struct.string___XUB		; <%struct.string___XUB*> [#uses=3]
	%A.257.i = alloca %struct.string___XUB		; <%struct.string___XUB*> [#uses=3]
	%0 = alloca [12 x i8]		; <[12 x i8]*> [#uses=1]
	%A.301.i = alloca %struct.string___XUB		; <%struct.string___XUB*> [#uses=3]
	%A.292.i = alloca %struct.string___XUB		; <%struct.string___XUB*> [#uses=3]
	%1 = alloca [12 x i8]		; <[12 x i8]*> [#uses=1]
	%2 = alloca %struct.system__file_control_block__pstring, align 8		; <%struct.system__file_control_block__pstring*> [#uses=3]
	%3 = alloca %struct.system__file_control_block__pstring, align 8		; <%struct.system__file_control_block__pstring*> [#uses=3]
	%4 = alloca %struct.system__file_control_block__pstring, align 8		; <%struct.system__file_control_block__pstring*> [#uses=3]
	%5 = alloca %struct.system__file_control_block__pstring, align 8		; <%struct.system__file_control_block__pstring*> [#uses=3]
	%FRAME.356 = alloca %struct.FRAME.ce3806g		; <%struct.FRAME.ce3806g*> [#uses=20]
	call void @report__test( i8* getelementptr ([7 x i8]* @.str5, i32 0, i32 0), %struct.string___XUB* @C.132.1562, i8* getelementptr ([126 x i8]* @.str4, i32 0, i32 0), %struct.string___XUB* @C.131.1559 )
	%6 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3		; <%struct.string___XUB*> [#uses=1]
	call void @system__secondary_stack__ss_mark( %struct.string___XUB* noalias sret %6 )
	%.b = load i1* @incompleteF.1176.b		; <i1> [#uses=1]
	br i1 %.b, label %bb11, label %bb

bb:		; preds = %entry
	invoke void @system__exception_table__register_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @incomplete.1177 to %struct.system__standard_library__exception_data*) )
			to label %invcont unwind label %lpad

invcont:		; preds = %bb
	store i1 true, i1* @incompleteF.1176.b
	br label %bb11

bb11:		; preds = %entry, %invcont
	%7 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2		; <%struct.string___XUB*> [#uses=1]
	invoke void @system__secondary_stack__ss_mark( %struct.string___XUB* noalias sret %7 )
			to label %invcont12 unwind label %lpad228

invcont12:		; preds = %bb11
	invoke void @report__legal_file_name( %struct.system__file_control_block__pstring* noalias sret %5, i32 1, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.136.1568 )
			to label %invcont17 unwind label %lpad232

invcont17:		; preds = %invcont12
	%elt18 = getelementptr %struct.system__file_control_block__pstring* %5, i32 0, i32 0		; <i8**> [#uses=1]
	%val19 = load i8** %elt18, align 8		; <i8*> [#uses=1]
	%elt20 = getelementptr %struct.system__file_control_block__pstring* %5, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	%val21 = load %struct.string___XUB** %elt20		; <%struct.string___XUB*> [#uses=1]
	%8 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__create( %struct.ada__text_io__text_afcb* null, i8 2, i8* %val19, %struct.string___XUB* %val21, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.136.1568 )
			to label %invcont26 unwind label %lpad232		; <%struct.ada__text_io__text_afcb*> [#uses=2]

invcont26:		; preds = %invcont17
	%9 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2, i32 0		; <i32*> [#uses=1]
	%10 = load i32* %9, align 8		; <i32> [#uses=1]
	%11 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2, i32 1		; <i32*> [#uses=1]
	%12 = load i32* %11, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %10, i32 %12 )
			to label %bb73 unwind label %lpad228

bb32:		; preds = %lpad232
	call void @__gnat_begin_handler( i8* %eh_ptr233 ) nounwind
	%13 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %13( )
			to label %invcont33 unwind label %lpad240

invcont33:		; preds = %bb32
	invoke void @report__not_applicable( i8* getelementptr ([50 x i8]* @.str7, i32 0, i32 0), %struct.string___XUB* @C.139.1577 )
			to label %invcont38 unwind label %lpad240

invcont38:		; preds = %invcont33
	invoke void @__gnat_raise_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @incomplete.1177 to %struct.system__standard_library__exception_data*), i8* getelementptr ([14 x i8]* @.str8, i32 0, i32 0), %struct.string___XUB* @C.140.1580 ) noreturn
			to label %invcont43 unwind label %lpad240

invcont43:		; preds = %invcont38
	unreachable

bb47:		; preds = %ppad291
	call void @__gnat_begin_handler( i8* %eh_ptr233 ) nounwind
	%14 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %14( )
			to label %invcont49 unwind label %lpad248

invcont49:		; preds = %bb47
	invoke void @report__not_applicable( i8* getelementptr ([51 x i8]* @.str9, i32 0, i32 0), %struct.string___XUB* @C.143.1585 )
			to label %invcont54 unwind label %lpad248

invcont54:		; preds = %invcont49
	invoke void @__gnat_raise_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @incomplete.1177 to %struct.system__standard_library__exception_data*), i8* getelementptr ([14 x i8]* @.str10, i32 0, i32 0), %struct.string___XUB* @C.140.1580 ) noreturn
			to label %invcont59 unwind label %lpad248

invcont59:		; preds = %invcont54
	unreachable

bb73:		; preds = %invcont26
	invoke void @report__legal_file_name( %struct.system__file_control_block__pstring* noalias sret %4, i32 2, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.136.1568 )
			to label %invcont78 unwind label %lpad228

invcont78:		; preds = %bb73
	%elt79 = getelementptr %struct.system__file_control_block__pstring* %4, i32 0, i32 0		; <i8**> [#uses=1]
	%val80 = load i8** %elt79, align 8		; <i8*> [#uses=1]
	%elt81 = getelementptr %struct.system__file_control_block__pstring* %4, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	%val82 = load %struct.string___XUB** %elt81		; <%struct.string___XUB*> [#uses=1]
	%15 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__create( %struct.ada__text_io__text_afcb* null, i8 2, i8* %val80, %struct.string___XUB* %val82, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.136.1568 )
			to label %invcont87 unwind label %lpad228		; <%struct.ada__text_io__text_afcb*> [#uses=2]

invcont87:		; preds = %invcont78
	invoke void @ada__text_io__set_output( %struct.ada__text_io__text_afcb* %15 )
			to label %invcont88 unwind label %lpad228

invcont88:		; preds = %invcont87
	%16 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1		; <%struct.string___XUB*> [#uses=1]
	invoke void @system__secondary_stack__ss_mark( %struct.string___XUB* noalias sret %16 )
			to label %invcont89 unwind label %lpad228

invcont89:		; preds = %invcont88
	%17 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%.sub.i = getelementptr [12 x i8]* %1, i32 0, i32 0		; <i8*> [#uses=2]
	%18 = getelementptr %struct.string___XUB* %A.292.i, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %18, align 8
	%19 = getelementptr %struct.string___XUB* %A.292.i, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 12, i32* %19, align 4
	%20 = invoke fastcc i32 @ce3806g__fxio__put__4.1215( i8* %.sub.i, %struct.string___XUB* %A.292.i, i8 signext -3 )
			to label %invcont.i unwind label %lpad.i		; <i32> [#uses=1]

invcont.i:		; preds = %invcont89
	%21 = getelementptr %struct.string___XUB* %A.301.i, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %21, align 8
	%22 = getelementptr %struct.string___XUB* %A.301.i, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %20, i32* %22, align 4
	invoke void @ada__text_io__generic_aux__put_item( %struct.ada__text_io__text_afcb* %8, i8* %.sub.i, %struct.string___XUB* %A.301.i )
			to label %bb94 unwind label %lpad.i

lpad.i:		; preds = %invcont.i, %invcont89
	%eh_ptr.i = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select62.i = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr.i, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	call void @llvm.stackrestore( i8* %17 )
	%23 = invoke i32 (...)* @_Unwind_Resume( i8* %eh_ptr.i )
			to label %.noexc unwind label %lpad252		; <i32> [#uses=0]

.noexc:		; preds = %lpad.i
	unreachable

bb94:		; preds = %invcont.i
	%24 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%.sub.i360 = getelementptr [12 x i8]* %0, i32 0, i32 0		; <i8*> [#uses=2]
	%25 = getelementptr %struct.string___XUB* %A.257.i, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %25, align 8
	%26 = getelementptr %struct.string___XUB* %A.257.i, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 12, i32* %26, align 4
	%27 = invoke fastcc i32 @ce3806g__fxio__put__4.1215( i8* %.sub.i360, %struct.string___XUB* %A.257.i, i8 signext -1 )
			to label %invcont.i361 unwind label %lpad.i364		; <i32> [#uses=1]

invcont.i361:		; preds = %bb94
	%28 = getelementptr %struct.string___XUB* %A.266.i, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %28, align 8
	%29 = getelementptr %struct.string___XUB* %A.266.i, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %27, i32* %29, align 4
	%30 = load %struct.ada__text_io__text_afcb** @ada__text_io__current_out, align 4		; <%struct.ada__text_io__text_afcb*> [#uses=1]
	invoke void @ada__text_io__generic_aux__put_item( %struct.ada__text_io__text_afcb* %30, i8* %.sub.i360, %struct.string___XUB* %A.266.i )
			to label %invcont95 unwind label %lpad.i364

lpad.i364:		; preds = %invcont.i361, %bb94
	%eh_ptr.i362 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select62.i363 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr.i362, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	call void @llvm.stackrestore( i8* %24 )
	%31 = invoke i32 (...)* @_Unwind_Resume( i8* %eh_ptr.i362 )
			to label %.noexc365 unwind label %lpad252		; <i32> [#uses=0]

.noexc365:		; preds = %lpad.i364
	unreachable

invcont95:		; preds = %invcont.i361
	%32 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__close( %struct.ada__text_io__text_afcb* %8 )
			to label %invcont96 unwind label %lpad252		; <%struct.ada__text_io__text_afcb*> [#uses=1]

invcont96:		; preds = %invcont95
	%33 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0		; <%struct.string___XUB*> [#uses=1]
	invoke void @system__secondary_stack__ss_mark( %struct.string___XUB* noalias sret %33 )
			to label %invcont97 unwind label %lpad252

invcont97:		; preds = %invcont96
	invoke void @report__legal_file_name( %struct.system__file_control_block__pstring* noalias sret %3, i32 1, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.136.1568 )
			to label %invcont102 unwind label %lpad256

invcont102:		; preds = %invcont97
	%elt103 = getelementptr %struct.system__file_control_block__pstring* %3, i32 0, i32 0		; <i8**> [#uses=1]
	%val104 = load i8** %elt103, align 8		; <i8*> [#uses=1]
	%elt105 = getelementptr %struct.system__file_control_block__pstring* %3, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	%val106 = load %struct.string___XUB** %elt105		; <%struct.string___XUB*> [#uses=1]
	%34 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__open( %struct.ada__text_io__text_afcb* %32, i8 0, i8* %val104, %struct.string___XUB* %val106, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.136.1568 )
			to label %invcont111 unwind label %lpad256		; <%struct.ada__text_io__text_afcb*> [#uses=2]

invcont111:		; preds = %invcont102
	%35 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%36 = load i32* %35, align 8		; <i32> [#uses=1]
	%37 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%38 = load i32* %37, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %36, i32 %38 )
			to label %bb143 unwind label %lpad252

bb117:		; preds = %lpad256
	call void @__gnat_begin_handler( i8* %eh_ptr257 ) nounwind
	%39 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %39( )
			to label %invcont119 unwind label %lpad264

invcont119:		; preds = %bb117
	invoke void @report__not_applicable( i8* getelementptr ([47 x i8]* @.str12, i32 0, i32 0), %struct.string___XUB* @C.354.2200 )
			to label %invcont124 unwind label %lpad264

invcont124:		; preds = %invcont119
	invoke void @__gnat_raise_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @incomplete.1177 to %struct.system__standard_library__exception_data*), i8* getelementptr ([14 x i8]* @.str13, i32 0, i32 0), %struct.string___XUB* @C.140.1580 ) noreturn
			to label %invcont129 unwind label %lpad264

invcont129:		; preds = %invcont124
	unreachable

bb143:		; preds = %invcont111
	%40 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__standard_output( )
			to label %invcont144 unwind label %lpad252		; <%struct.ada__text_io__text_afcb*> [#uses=1]

invcont144:		; preds = %bb143
	invoke void @ada__text_io__set_output( %struct.ada__text_io__text_afcb* %40 )
			to label %invcont145 unwind label %lpad252

invcont145:		; preds = %invcont144
	%41 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__close( %struct.ada__text_io__text_afcb* %15 )
			to label %invcont146 unwind label %lpad252		; <%struct.ada__text_io__text_afcb*> [#uses=1]

invcont146:		; preds = %invcont145
	invoke void @report__legal_file_name( %struct.system__file_control_block__pstring* noalias sret %2, i32 2, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.136.1568 )
			to label %invcont151 unwind label %lpad252

invcont151:		; preds = %invcont146
	%elt152 = getelementptr %struct.system__file_control_block__pstring* %2, i32 0, i32 0		; <i8**> [#uses=1]
	%val153 = load i8** %elt152, align 8		; <i8*> [#uses=1]
	%elt154 = getelementptr %struct.system__file_control_block__pstring* %2, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	%val155 = load %struct.string___XUB** %elt154		; <%struct.string___XUB*> [#uses=1]
	%42 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__open( %struct.ada__text_io__text_afcb* %41, i8 0, i8* %val153, %struct.string___XUB* %val155, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.136.1568 )
			to label %invcont160 unwind label %lpad252		; <%struct.ada__text_io__text_afcb*> [#uses=2]

invcont160:		; preds = %invcont151
	%43 = invoke fastcc i8 @ce3806g__fxio__get.1137( %struct.ada__text_io__text_afcb* %34 ) signext
			to label %invcont161 unwind label %lpad252		; <i8> [#uses=1]

invcont161:		; preds = %invcont160
	%44 = icmp eq i8 %43, -3		; <i1> [#uses=1]
	br i1 %44, label %bb169, label %bb163

bb163:		; preds = %invcont161
	invoke void @report__failed( i8* getelementptr ([33 x i8]* @.str14, i32 0, i32 0), %struct.string___XUB* @C.162.1637 )
			to label %bb169 unwind label %lpad252

bb169:		; preds = %bb163, %invcont161
	%45 = invoke fastcc i8 @ce3806g__fxio__get.1137( %struct.ada__text_io__text_afcb* %42 ) signext
			to label %invcont170 unwind label %lpad252		; <i8> [#uses=1]

invcont170:		; preds = %bb169
	%46 = icmp eq i8 %45, -1		; <i1> [#uses=1]
	br i1 %46, label %bb187, label %bb172

bb172:		; preds = %invcont170
	invoke void @report__failed( i8* getelementptr ([36 x i8]* @.str15, i32 0, i32 0), %struct.string___XUB* @C.164.1642 )
			to label %bb187 unwind label %lpad252

bb187:		; preds = %bb172, %invcont170
	%47 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1, i32 0		; <i32*> [#uses=1]
	%48 = load i32* %47, align 8		; <i32> [#uses=1]
	%49 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1, i32 1		; <i32*> [#uses=1]
	%50 = load i32* %49, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %48, i32 %50 )
			to label %bb193 unwind label %lpad228

bb193:		; preds = %bb187
	%51 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__delete( %struct.ada__text_io__text_afcb* %34 )
			to label %invcont194 unwind label %lpad268		; <%struct.ada__text_io__text_afcb*> [#uses=0]

invcont194:		; preds = %bb193
	%52 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__delete( %struct.ada__text_io__text_afcb* %42 )
			to label %bb221 unwind label %lpad268		; <%struct.ada__text_io__text_afcb*> [#uses=0]

bb196:		; preds = %lpad268
	call void @__gnat_begin_handler( i8* %eh_ptr269 ) nounwind
	%53 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %53( )
			to label %bb203 unwind label %lpad276

bb203:		; preds = %bb196
	invoke void @__gnat_end_handler( i8* %eh_ptr269 )
			to label %bb221 unwind label %lpad272

bb205:		; preds = %ppad304
	call void @__gnat_begin_handler( i8* %eh_exception.1 ) nounwind
	%54 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %54( )
			to label %bb212 unwind label %lpad284

bb212:		; preds = %bb205
	invoke void @__gnat_end_handler( i8* %eh_exception.1 )
			to label %bb221 unwind label %lpad280

bb221:		; preds = %bb212, %bb203, %invcont194
	%55 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3, i32 0		; <i32*> [#uses=1]
	%56 = load i32* %55, align 8		; <i32> [#uses=1]
	%57 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3, i32 1		; <i32*> [#uses=1]
	%58 = load i32* %57, align 4		; <i32> [#uses=1]
	call void @system__secondary_stack__ss_release( i32 %56, i32 %58 )
	call void @report__result( )
	ret void

lpad:		; preds = %bb
	%eh_ptr = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select227 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	br label %ppad

lpad228:		; preds = %ppad294, %ppad288, %bb187, %invcont88, %invcont87, %invcont78, %bb73, %invcont26, %bb11
	%eh_ptr229 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select231 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr229, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad304

lpad232:		; preds = %invcont17, %invcont12
	%eh_ptr233 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=6]
	%eh_select235 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr233, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @ada__io_exceptions__use_error, %struct.exception* @ada__io_exceptions__name_error, %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=3]
	%eh_typeid = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @ada__io_exceptions__use_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%59 = icmp eq i32 %eh_select235, %eh_typeid		; <i1> [#uses=1]
	br i1 %59, label %bb32, label %ppad291

lpad236:		; preds = %lpad240
	%eh_ptr237 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select239 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr237, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad288

lpad240:		; preds = %invcont38, %invcont33, %bb32
	%eh_ptr241 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select243 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr241, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	invoke void @__gnat_end_handler( i8* %eh_ptr233 )
			to label %ppad288 unwind label %lpad236

lpad244:		; preds = %lpad248
	%eh_ptr245 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select247 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr245, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad288

lpad248:		; preds = %invcont54, %invcont49, %bb47
	%eh_ptr249 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select251 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr249, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	invoke void @__gnat_end_handler( i8* %eh_ptr233 )
			to label %ppad288 unwind label %lpad244

lpad252:		; preds = %ppad295, %bb172, %bb169, %bb163, %invcont160, %invcont151, %invcont146, %invcont145, %invcont144, %bb143, %invcont111, %invcont96, %invcont95, %lpad.i364, %lpad.i
	%eh_ptr253 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select255 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr253, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad294

lpad256:		; preds = %invcont102, %invcont97
	%eh_ptr257 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=4]
	%eh_select259 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr257, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @ada__io_exceptions__use_error, %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=2]
	%eh_typeid297 = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @ada__io_exceptions__use_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%60 = icmp eq i32 %eh_select259, %eh_typeid297		; <i1> [#uses=1]
	br i1 %60, label %bb117, label %ppad295

lpad260:		; preds = %lpad264
	%eh_ptr261 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select263 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr261, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad295

lpad264:		; preds = %invcont124, %invcont119, %bb117
	%eh_ptr265 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select267 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr265, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	invoke void @__gnat_end_handler( i8* %eh_ptr257 )
			to label %ppad295 unwind label %lpad260

lpad268:		; preds = %invcont194, %bb193
	%eh_ptr269 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=5]
	%eh_select271 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr269, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @ada__io_exceptions__use_error, %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=2]
	%eh_typeid301 = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @ada__io_exceptions__use_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%61 = icmp eq i32 %eh_select271, %eh_typeid301		; <i1> [#uses=1]
	br i1 %61, label %bb196, label %ppad304

lpad272:		; preds = %lpad276, %bb203
	%eh_ptr273 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select275 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr273, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad304

lpad276:		; preds = %bb196
	%eh_ptr277 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select279 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr277, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	invoke void @__gnat_end_handler( i8* %eh_ptr269 )
			to label %ppad304 unwind label %lpad272

lpad280:		; preds = %lpad284, %bb212
	%eh_ptr281 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select283 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr281, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	br label %ppad

lpad284:		; preds = %bb205
	%eh_ptr285 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select287 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr285, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	invoke void @__gnat_end_handler( i8* %eh_exception.1 )
			to label %ppad unwind label %lpad280

ppad:		; preds = %ppad304, %lpad284, %lpad280, %lpad
	%eh_exception.2 = phi i8* [ %eh_ptr281, %lpad280 ], [ %eh_ptr, %lpad ], [ %eh_ptr285, %lpad284 ], [ %eh_exception.1, %ppad304 ]		; <i8*> [#uses=1]
	%62 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3, i32 0		; <i32*> [#uses=1]
	%63 = load i32* %62, align 8		; <i32> [#uses=1]
	%64 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3, i32 1		; <i32*> [#uses=1]
	%65 = load i32* %64, align 4		; <i32> [#uses=1]
	call void @system__secondary_stack__ss_release( i32 %63, i32 %65 )
	%66 = call i32 (...)* @_Unwind_Resume( i8* %eh_exception.2 )		; <i32> [#uses=0]
	unreachable

ppad288:		; preds = %ppad291, %lpad248, %lpad240, %lpad244, %lpad236
	%eh_exception.0 = phi i8* [ %eh_ptr245, %lpad244 ], [ %eh_ptr237, %lpad236 ], [ %eh_ptr241, %lpad240 ], [ %eh_ptr249, %lpad248 ], [ %eh_ptr233, %ppad291 ]		; <i8*> [#uses=1]
	%eh_selector.0 = phi i32 [ %eh_select247, %lpad244 ], [ %eh_select239, %lpad236 ], [ %eh_select243, %lpad240 ], [ %eh_select251, %lpad248 ], [ %eh_select235, %ppad291 ]		; <i32> [#uses=1]
	%67 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2, i32 0		; <i32*> [#uses=1]
	%68 = load i32* %67, align 8		; <i32> [#uses=1]
	%69 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2, i32 1		; <i32*> [#uses=1]
	%70 = load i32* %69, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %68, i32 %70 )
			to label %ppad304 unwind label %lpad228

ppad291:		; preds = %lpad232
	%eh_typeid292 = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @ada__io_exceptions__name_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%71 = icmp eq i32 %eh_select235, %eh_typeid292		; <i1> [#uses=1]
	br i1 %71, label %bb47, label %ppad288

ppad294:		; preds = %ppad295, %lpad252
	%eh_exception.4 = phi i8* [ %eh_ptr253, %lpad252 ], [ %eh_exception.3, %ppad295 ]		; <i8*> [#uses=1]
	%eh_selector.4 = phi i32 [ %eh_select255, %lpad252 ], [ %eh_selector.3, %ppad295 ]		; <i32> [#uses=1]
	%72 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1, i32 0		; <i32*> [#uses=1]
	%73 = load i32* %72, align 8		; <i32> [#uses=1]
	%74 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1, i32 1		; <i32*> [#uses=1]
	%75 = load i32* %74, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %73, i32 %75 )
			to label %ppad304 unwind label %lpad228

ppad295:		; preds = %lpad264, %lpad256, %lpad260
	%eh_exception.3 = phi i8* [ %eh_ptr261, %lpad260 ], [ %eh_ptr257, %lpad256 ], [ %eh_ptr265, %lpad264 ]		; <i8*> [#uses=1]
	%eh_selector.3 = phi i32 [ %eh_select263, %lpad260 ], [ %eh_select259, %lpad256 ], [ %eh_select267, %lpad264 ]		; <i32> [#uses=1]
	%76 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%77 = load i32* %76, align 8		; <i32> [#uses=1]
	%78 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%79 = load i32* %78, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %77, i32 %79 )
			to label %ppad294 unwind label %lpad252

ppad304:		; preds = %ppad294, %ppad288, %lpad276, %lpad268, %lpad272, %lpad228
	%eh_exception.1 = phi i8* [ %eh_ptr229, %lpad228 ], [ %eh_ptr273, %lpad272 ], [ %eh_ptr269, %lpad268 ], [ %eh_ptr277, %lpad276 ], [ %eh_exception.0, %ppad288 ], [ %eh_exception.4, %ppad294 ]		; <i8*> [#uses=4]
	%eh_selector.1 = phi i32 [ %eh_select231, %lpad228 ], [ %eh_select275, %lpad272 ], [ %eh_select271, %lpad268 ], [ %eh_select279, %lpad276 ], [ %eh_selector.0, %ppad288 ], [ %eh_selector.4, %ppad294 ]		; <i32> [#uses=1]
	%eh_typeid305 = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @incomplete.1177, i32 0, i32 0) )		; <i32> [#uses=1]
	%80 = icmp eq i32 %eh_selector.1, %eh_typeid305		; <i1> [#uses=1]
	br i1 %80, label %bb205, label %ppad
}

define internal fastcc i8 @ce3806g__fxio__get.1137(%struct.ada__text_io__text_afcb* %file) signext {
entry:
	%0 = invoke x86_fp80 @ada__text_io__float_aux__get( %struct.ada__text_io__text_afcb* %file, i32 0 )
			to label %invcont unwind label %lpad		; <x86_fp80> [#uses=3]

invcont:		; preds = %entry
	%1 = fcmp ult x86_fp80 %0, 0xKFFFEFFFFFFFFFFFFFFFF		; <i1> [#uses=1]
	%2 = fcmp ugt x86_fp80 %0, 0xK7FFEFFFFFFFFFFFFFFFF		; <i1> [#uses=1]
	%or.cond = or i1 %1, %2		; <i1> [#uses=1]
	br i1 %or.cond, label %bb2, label %bb4

bb2:		; preds = %invcont
	invoke void @__gnat_rcheck_12( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 1 ) noreturn
			to label %invcont3 unwind label %lpad

invcont3:		; preds = %bb2
	unreachable

bb4:		; preds = %invcont
	%3 = mul x86_fp80 %0, 0xK40008000000000000000		; <x86_fp80> [#uses=5]
	%4 = fcmp ult x86_fp80 %3, 0xKC0068000000000000000		; <i1> [#uses=1]
	%5 = fcmp ugt x86_fp80 %3, 0xK4005FE00000000000000		; <i1> [#uses=1]
	%or.cond1 = or i1 %4, %5		; <i1> [#uses=1]
	br i1 %or.cond1, label %bb8, label %bb10

bb8:		; preds = %bb4
	invoke void @__gnat_rcheck_10( i8* getelementptr ([14 x i8]* @.str1, i32 0, i32 0), i32 324 ) noreturn
			to label %invcont9 unwind label %lpad

invcont9:		; preds = %bb8
	unreachable

bb10:		; preds = %bb4
	%6 = fcmp ult x86_fp80 %3, 0xK00000000000000000000		; <i1> [#uses=1]
	br i1 %6, label %bb13, label %bb12

bb12:		; preds = %bb10
	%7 = add x86_fp80 %3, 0xK3FFDFFFFFFFFFFFFFFFF		; <x86_fp80> [#uses=1]
	br label %bb14

bb13:		; preds = %bb10
	%8 = sub x86_fp80 %3, 0xK3FFDFFFFFFFFFFFFFFFF		; <x86_fp80> [#uses=1]
	br label %bb14

bb14:		; preds = %bb13, %bb12
	%iftmp.339.0.in = phi x86_fp80 [ %8, %bb13 ], [ %7, %bb12 ]		; <x86_fp80> [#uses=1]
	%iftmp.339.0 = fptosi x86_fp80 %iftmp.339.0.in to i8		; <i8> [#uses=2]
	%9 = add i8 %iftmp.339.0, 20		; <i8> [#uses=1]
	%10 = icmp ugt i8 %9, 40		; <i1> [#uses=1]
	br i1 %10, label %bb16, label %bb22

bb16:		; preds = %bb14
	invoke void @__gnat_rcheck_12( i8* getelementptr ([14 x i8]* @.str1, i32 0, i32 0), i32 324 ) noreturn
			to label %invcont17 unwind label %lpad

invcont17:		; preds = %bb16
	unreachable

bb22:		; preds = %bb14
	ret i8 %iftmp.339.0

bb23:		; preds = %lpad
	tail call void @__gnat_begin_handler( i8* %eh_ptr ) nounwind
	%11 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %11( )
			to label %invcont24 unwind label %lpad33

invcont24:		; preds = %bb23
	invoke void @__gnat_raise_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @ada__io_exceptions__data_error to %struct.system__standard_library__exception_data*), i8* getelementptr ([47 x i8]* @.str2, i32 0, i32 0), %struct.string___XUB* @C.354.2200 ) noreturn
			to label %invcont27 unwind label %lpad33

invcont27:		; preds = %invcont24
	unreachable

lpad:		; preds = %bb16, %bb8, %bb2, %entry
	%eh_ptr = tail call i8* @llvm.eh.exception( )		; <i8*> [#uses=4]
	%eh_select32 = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @constraint_error, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	%eh_typeid = tail call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @constraint_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%12 = icmp eq i32 %eh_select32, %eh_typeid		; <i1> [#uses=1]
	br i1 %12, label %bb23, label %Unwind

lpad33:		; preds = %invcont24, %bb23
	%eh_ptr34 = tail call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select36 = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr34, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	tail call void @__gnat_end_handler( i8* %eh_ptr )
	br label %Unwind

Unwind:		; preds = %lpad, %lpad33
	%eh_exception.0 = phi i8* [ %eh_ptr34, %lpad33 ], [ %eh_ptr, %lpad ]		; <i8*> [#uses=1]
	%13 = tail call i32 (...)* @_Unwind_Resume( i8* %eh_exception.0 )		; <i32> [#uses=0]
	unreachable
}

define internal fastcc i32 @ce3806g__fxio__put__4.1215(i8* %to.0, %struct.string___XUB* %to.1, i8 signext %item) {
entry:
	%0 = alloca { i64, i64 }		; <{ i64, i64 }*> [#uses=3]
	%1 = alloca i64		; <i64*> [#uses=3]
	%to_addr = alloca %struct.system__file_control_block__pstring		; <%struct.system__file_control_block__pstring*> [#uses=3]
	%FRAME.358 = alloca %struct.FRAME.ce3806g__fxio__put__4		; <%struct.FRAME.ce3806g__fxio__put__4*> [#uses=9]
	%2 = getelementptr %struct.system__file_control_block__pstring* %to_addr, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* %to.0, i8** %2, align 8
	%3 = getelementptr %struct.system__file_control_block__pstring* %to_addr, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	store %struct.string___XUB* %to.1, %struct.string___XUB** %3
	%4 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=9]
	store %struct.system__file_control_block__pstring* %to_addr, %struct.system__file_control_block__pstring** %4, align 4
	%5 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 0		; <i32*> [#uses=3]
	store i32 3, i32* %5, align 8
	%6 = getelementptr %struct.string___XUB* %to.1, i32 0, i32 0		; <i32*> [#uses=1]
	%7 = load i32* %6, align 4		; <i32> [#uses=3]
	%8 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 2		; <i32*> [#uses=3]
	store i32 %7, i32* %8, align 8
	%9 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=9]
	store i32 %7, i32* %9, align 8
	%item.lobit = lshr i8 %item, 7		; <i8> [#uses=1]
	%10 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 6		; <i8*> [#uses=3]
	store i8 %item.lobit, i8* %10, align 8
	%11 = add i32 %7, -1		; <i32> [#uses=1]
	%12 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=18]
	store i32 %11, i32* %12, align 4
	%13 = sext i8 %item to i64		; <i64> [#uses=2]
	%14 = call i64 @system__exn_lli__exn_long_long_integer( i64 10, i32 1 ) readnone		; <i64> [#uses=1]
	%15 = call i64 @system__exn_lli__exn_long_long_integer( i64 10, i32 0 ) readnone		; <i64> [#uses=1]
	%16 = mul i64 %15, -2		; <i64> [#uses=1]
	%savedstack = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%17 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	store i64 0, i64* %1, align 8
	%18 = getelementptr { i64, i64 }* %0, i32 0, i32 0		; <i64*> [#uses=1]
	%19 = getelementptr { i64, i64 }* %0, i32 0, i32 1		; <i64*> [#uses=1]
	%20 = icmp eq i64 %13, 0		; <i1> [#uses=1]
	br i1 %20, label %ce3806g__fxio__put__put_scaled__4.1346.exit, label %bb63.i

bb63.i:		; preds = %entry
	%yy.0.i = sub i64 0, %14		; <i64> [#uses=1]
	invoke void @system__arith_64__scaled_divide( { i64, i64 }* noalias sret %0, i64 %13, i64 %yy.0.i, i64 %16, i8 1 )
			to label %invcont.i unwind label %lpad.i

invcont.i:		; preds = %bb63.i
	%21 = load i64* %18, align 8		; <i64> [#uses=1]
	store i64 %21, i64* %1, align 8
	%22 = load i64* %19, align 8		; <i64> [#uses=0]
	%23 = invoke i64 @system__exn_lli__exn_long_long_integer( i64 10, i32 0 ) readnone
			to label %ce3806g__fxio__put__put_scaled__4.1346.exit unwind label %lpad.i		; <i64> [#uses=0]

lpad.i:		; preds = %invcont.i, %bb63.i
	%eh_ptr.i = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select103.i = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr.i, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	call void @llvm.stackrestore( i8* %17 )
	%24 = call i32 (...)* @_Unwind_Resume( i8* %eh_ptr.i )		; <i32> [#uses=0]
	unreachable

ce3806g__fxio__put__put_scaled__4.1346.exit:		; preds = %invcont.i, %entry
	%25 = load i64* %1, align 8		; <i64> [#uses=1]
	call fastcc void @ce3806g__fxio__put__put_int64__4.1339( %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i64 %25, i32 -1 )
	call void @llvm.stackrestore( i8* %savedstack )
	%26 = load i32* %12, align 4		; <i32> [#uses=4]
	%27 = load i32* %8, align 8		; <i32> [#uses=1]
	%28 = icmp slt i32 %26, %27		; <i1> [#uses=1]
	%.pre = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=1]
	br i1 %28, label %bb71, label %bb72.preheader

bb71:		; preds = %ce3806g__fxio__put__put_scaled__4.1346.exit
	%29 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=2]
	store i32 0, i32* %29, align 4
	br label %bb72.preheader

bb72.preheader:		; preds = %ce3806g__fxio__put__put_scaled__4.1346.exit, %bb71
	%.pre-phi = phi i32* [ %29, %bb71 ], [ %.pre, %ce3806g__fxio__put__put_scaled__4.1346.exit ]		; <i32*> [#uses=5]
	%30 = load i32* %.pre-phi, align 4		; <i32> [#uses=2]
	%31 = icmp slt i32 %30, -1		; <i1> [#uses=1]
	br i1 %31, label %bb103, label %bb74

bb74:		; preds = %bb102, %bb72.preheader
	%.rle10 = phi i32 [ %26, %bb72.preheader ], [ %119, %bb102 ]		; <i32> [#uses=1]
	%.rle = phi i32 [ %30, %bb72.preheader ], [ %117, %bb102 ]		; <i32> [#uses=4]
	%32 = phi i32 [ %26, %bb72.preheader ], [ %119, %bb102 ]		; <i32> [#uses=1]
	%33 = load i32* %8, align 8		; <i32> [#uses=1]
	%34 = add i32 %33, -1		; <i32> [#uses=1]
	%35 = icmp eq i32 %32, %34		; <i1> [#uses=1]
	br i1 %35, label %bb76, label %bb98

bb76:		; preds = %bb74
	%36 = icmp slt i32 %.rle, 1		; <i1> [#uses=1]
	br i1 %36, label %bb80, label %bb102

bb80:		; preds = %bb76
	%37 = icmp sgt i32 %.rle, -1		; <i1> [#uses=1]
	%.op = add i32 %.rle, 2		; <i32> [#uses=1]
	%38 = select i1 %37, i32 %.op, i32 2		; <i32> [#uses=1]
	%39 = load i8* %10, align 8		; <i8> [#uses=1]
	%40 = zext i8 %39 to i32		; <i32> [#uses=1]
	%41 = add i32 %38, %40		; <i32> [#uses=2]
	%42 = load i32* %5, align 8		; <i32> [#uses=1]
	%43 = icmp sgt i32 %41, %42		; <i1> [#uses=1]
	br i1 %43, label %bb88, label %bb85

bb85:		; preds = %bb80, %bb87
	%indvar4 = phi i32 [ %indvar.next5, %bb87 ], [ 0, %bb80 ]		; <i32> [#uses=2]
	%j.0 = add i32 %indvar4, %41		; <i32> [#uses=1]
	%44 = load i32* %12, align 4		; <i32> [#uses=1]
	%45 = add i32 %44, 1		; <i32> [#uses=2]
	store i32 %45, i32* %12, align 4
	%46 = load i32* %9, align 8		; <i32> [#uses=1]
	%47 = load %struct.system__file_control_block__pstring** %4, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%48 = getelementptr %struct.system__file_control_block__pstring* %47, i32 0, i32 0		; <i8**> [#uses=1]
	%49 = load i8** %48, align 4		; <i8*> [#uses=1]
	%50 = sub i32 %45, %46		; <i32> [#uses=1]
	%51 = getelementptr i8* %49, i32 %50		; <i8*> [#uses=1]
	store i8 32, i8* %51, align 1
	%52 = load i32* %5, align 8		; <i32> [#uses=1]
	%53 = icmp eq i32 %52, %j.0		; <i1> [#uses=1]
	br i1 %53, label %bb88, label %bb87

bb87:		; preds = %bb85
	%indvar.next5 = add i32 %indvar4, 1		; <i32> [#uses=1]
	br label %bb85

bb88:		; preds = %bb85, %bb80
	%54 = load i8* %10, align 8		; <i8> [#uses=1]
	%toBool89 = icmp eq i8 %54, 0		; <i1> [#uses=1]
	br i1 %toBool89, label %bb91, label %bb90

bb90:		; preds = %bb88
	%55 = load i32* %12, align 4		; <i32> [#uses=1]
	%56 = add i32 %55, 1		; <i32> [#uses=2]
	store i32 %56, i32* %12, align 4
	%57 = load i32* %9, align 8		; <i32> [#uses=1]
	%58 = load %struct.system__file_control_block__pstring** %4, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%59 = getelementptr %struct.system__file_control_block__pstring* %58, i32 0, i32 0		; <i8**> [#uses=1]
	%60 = load i8** %59, align 4		; <i8*> [#uses=1]
	%61 = sub i32 %56, %57		; <i32> [#uses=1]
	%62 = getelementptr i8* %60, i32 %61		; <i8*> [#uses=1]
	store i8 45, i8* %62, align 1
	br label %bb91

bb91:		; preds = %bb88, %bb90
	%63 = load i32* %.pre-phi, align 4		; <i32> [#uses=1]
	%64 = icmp slt i32 %63, 0		; <i1> [#uses=1]
	br i1 %64, label %bb93, label %bb97

bb93:		; preds = %bb91
	%65 = load i32* %12, align 4		; <i32> [#uses=1]
	%66 = add i32 %65, 1		; <i32> [#uses=2]
	store i32 %66, i32* %12, align 4
	%67 = load i32* %9, align 8		; <i32> [#uses=1]
	%68 = load %struct.system__file_control_block__pstring** %4, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%69 = getelementptr %struct.system__file_control_block__pstring* %68, i32 0, i32 0		; <i8**> [#uses=1]
	%70 = load i8** %69, align 4		; <i8*> [#uses=1]
	%71 = sub i32 %66, %67		; <i32> [#uses=1]
	%72 = getelementptr i8* %70, i32 %71		; <i8*> [#uses=1]
	store i8 48, i8* %72, align 1
	%73 = load i32* %12, align 4		; <i32> [#uses=1]
	%74 = add i32 %73, 1		; <i32> [#uses=2]
	store i32 %74, i32* %12, align 4
	%75 = load i32* %9, align 8		; <i32> [#uses=1]
	%76 = load %struct.system__file_control_block__pstring** %4, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%77 = getelementptr %struct.system__file_control_block__pstring* %76, i32 0, i32 0		; <i8**> [#uses=1]
	%78 = load i8** %77, align 4		; <i8*> [#uses=1]
	%79 = sub i32 %74, %75		; <i32> [#uses=1]
	%80 = getelementptr i8* %78, i32 %79		; <i8*> [#uses=1]
	store i8 46, i8* %80, align 1
	%81 = load i32* %.pre-phi, align 4		; <i32> [#uses=2]
	%82 = icmp sgt i32 %81, -2		; <i1> [#uses=1]
	br i1 %82, label %bb97, label %bb96

bb96:		; preds = %bb96, %bb93
	%indvar = phi i32 [ 0, %bb93 ], [ %indvar.next, %bb96 ]		; <i32> [#uses=2]
	%83 = load i32* %12, align 4		; <i32> [#uses=1]
	%84 = add i32 %83, 1		; <i32> [#uses=2]
	store i32 %84, i32* %12, align 4
	%85 = load i32* %9, align 8		; <i32> [#uses=1]
	%86 = load %struct.system__file_control_block__pstring** %4, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%87 = getelementptr %struct.system__file_control_block__pstring* %86, i32 0, i32 0		; <i8**> [#uses=1]
	%88 = load i8** %87, align 4		; <i8*> [#uses=1]
	%89 = sub i32 %84, %85		; <i32> [#uses=1]
	%90 = getelementptr i8* %88, i32 %89		; <i8*> [#uses=1]
	store i8 48, i8* %90, align 1
	%j8.01 = add i32 %indvar, %81		; <i32> [#uses=1]
	%91 = add i32 %j8.01, 1		; <i32> [#uses=1]
	%phitmp = icmp sgt i32 %91, -2		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %phitmp, label %bb97, label %bb96

bb97:		; preds = %bb93, %bb96, %bb91
	%92 = load i32* %12, align 4		; <i32> [#uses=1]
	%93 = add i32 %92, 1		; <i32> [#uses=2]
	store i32 %93, i32* %12, align 4
	%94 = load i32* %9, align 8		; <i32> [#uses=1]
	%95 = load %struct.system__file_control_block__pstring** %4, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%96 = getelementptr %struct.system__file_control_block__pstring* %95, i32 0, i32 0		; <i8**> [#uses=1]
	%97 = load i8** %96, align 4		; <i8*> [#uses=1]
	%98 = sub i32 %93, %94		; <i32> [#uses=1]
	%99 = getelementptr i8* %97, i32 %98		; <i8*> [#uses=1]
	store i8 48, i8* %99, align 1
	br label %bb102

bb98:		; preds = %bb74
	%100 = icmp eq i32 %.rle, -1		; <i1> [#uses=1]
	br i1 %100, label %bb100, label %bb101

bb100:		; preds = %bb98
	%101 = add i32 %.rle10, 1		; <i32> [#uses=2]
	store i32 %101, i32* %12, align 4
	%102 = load i32* %9, align 8		; <i32> [#uses=1]
	%103 = load %struct.system__file_control_block__pstring** %4, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%104 = getelementptr %struct.system__file_control_block__pstring* %103, i32 0, i32 0		; <i8**> [#uses=1]
	%105 = load i8** %104, align 4		; <i8*> [#uses=1]
	%106 = sub i32 %101, %102		; <i32> [#uses=1]
	%107 = getelementptr i8* %105, i32 %106		; <i8*> [#uses=1]
	store i8 46, i8* %107, align 1
	br label %bb101

bb101:		; preds = %bb98, %bb100
	%108 = load i32* %12, align 4		; <i32> [#uses=1]
	%109 = add i32 %108, 1		; <i32> [#uses=2]
	store i32 %109, i32* %12, align 4
	%110 = load i32* %9, align 8		; <i32> [#uses=1]
	%111 = load %struct.system__file_control_block__pstring** %4, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%112 = getelementptr %struct.system__file_control_block__pstring* %111, i32 0, i32 0		; <i8**> [#uses=1]
	%113 = load i8** %112, align 4		; <i8*> [#uses=1]
	%114 = sub i32 %109, %110		; <i32> [#uses=1]
	%115 = getelementptr i8* %113, i32 %114		; <i8*> [#uses=1]
	store i8 48, i8* %115, align 1
	br label %bb102

bb102:		; preds = %bb76, %bb101, %bb97
	%116 = load i32* %.pre-phi, align 4		; <i32> [#uses=1]
	%117 = add i32 %116, -1		; <i32> [#uses=3]
	store i32 %117, i32* %.pre-phi, align 4
	%118 = icmp slt i32 %117, -1		; <i1> [#uses=1]
	%119 = load i32* %12, align 4		; <i32> [#uses=3]
	br i1 %118, label %bb103, label %bb74

bb103:		; preds = %bb102, %bb72.preheader
	%.lcssa = phi i32 [ %26, %bb72.preheader ], [ %119, %bb102 ]		; <i32> [#uses=1]
	ret i32 %.lcssa
}

declare x86_fp80 @ada__text_io__float_aux__get(%struct.ada__text_io__text_afcb*, i32)

declare void @__gnat_rcheck_12(i8*, i32) noreturn

declare void @__gnat_rcheck_10(i8*, i32) noreturn

declare i8* @llvm.eh.exception() nounwind

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...) nounwind

declare i32 @llvm.eh.typeid.for.i32(i8*) nounwind

declare void @__gnat_begin_handler(i8*) nounwind

declare void @__gnat_raise_exception(%struct.system__standard_library__exception_data*, i8*, %struct.string___XUB*) noreturn

declare void @__gnat_end_handler(i8*)

declare i32 @__gnat_eh_personality(...)

declare i32 @_Unwind_Resume(...)

define internal fastcc void @ce3806g__fxio__put__put_int64__4.1339(%struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i64 %x, i32 %scale) nounwind {
entry:
	%0 = icmp eq i64 %x, 0		; <i1> [#uses=1]
	br i1 %0, label %return, label %bb

bb:		; preds = %entry
	%1 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 1		; <i32*> [#uses=7]
	store i32 %scale, i32* %1, align 4
	%2 = add i64 %x, 9		; <i64> [#uses=1]
	%3 = icmp ugt i64 %2, 18		; <i1> [#uses=1]
	br i1 %3, label %bb18, label %bb19

bb18:		; preds = %bb
	%4 = add i32 %scale, 1		; <i32> [#uses=1]
	%5 = sdiv i64 %x, 10		; <i64> [#uses=1]
	tail call fastcc void @ce3806g__fxio__put__put_int64__4.1339( %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i64 %5, i32 %4 )
	br label %bb19

bb19:		; preds = %bb, %bb18
	%6 = srem i64 %x, 10		; <i64> [#uses=3]
	%neg = sub i64 0, %6		; <i64> [#uses=1]
	%abscond = icmp sgt i64 %6, -1		; <i1> [#uses=1]
	%abs = select i1 %abscond, i64 %6, i64 %neg		; <i64> [#uses=3]
	%7 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=16]
	%8 = load i32* %7, align 4		; <i32> [#uses=2]
	%9 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 2		; <i32*> [#uses=1]
	%10 = load i32* %9, align 4		; <i32> [#uses=1]
	%11 = add i32 %10, -1		; <i32> [#uses=1]
	%12 = icmp eq i32 %8, %11		; <i1> [#uses=1]
	br i1 %12, label %bb23, label %bb44

bb23:		; preds = %bb19
	%13 = icmp ne i64 %abs, 0		; <i1> [#uses=1]
	%14 = load i32* %1, align 4		; <i32> [#uses=3]
	%15 = icmp slt i32 %14, 1		; <i1> [#uses=1]
	%16 = or i1 %15, %13		; <i1> [#uses=1]
	br i1 %16, label %bb27, label %bb48

bb27:		; preds = %bb23
	%17 = icmp sgt i32 %14, -1		; <i1> [#uses=1]
	%.op = add i32 %14, 2		; <i32> [#uses=1]
	%18 = select i1 %17, i32 %.op, i32 2		; <i32> [#uses=1]
	%19 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 6		; <i8*> [#uses=2]
	%20 = load i8* %19, align 1		; <i8> [#uses=1]
	%21 = zext i8 %20 to i32		; <i32> [#uses=1]
	%22 = add i32 %18, %21		; <i32> [#uses=2]
	%23 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 0		; <i32*> [#uses=2]
	%24 = load i32* %23, align 4		; <i32> [#uses=1]
	%25 = icmp sgt i32 %22, %24		; <i1> [#uses=1]
	br i1 %25, label %bb34, label %bb31.preheader

bb31.preheader:		; preds = %bb27
	%26 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%27 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	br label %bb31

bb31:		; preds = %bb31.preheader, %bb33
	%indvar = phi i32 [ 0, %bb31.preheader ], [ %indvar.next, %bb33 ]		; <i32> [#uses=2]
	%j.0 = add i32 %indvar, %22		; <i32> [#uses=1]
	%28 = load i32* %7, align 4		; <i32> [#uses=1]
	%29 = add i32 %28, 1		; <i32> [#uses=2]
	store i32 %29, i32* %7, align 4
	%30 = load i32* %26, align 4		; <i32> [#uses=1]
	%31 = load %struct.system__file_control_block__pstring** %27, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%32 = getelementptr %struct.system__file_control_block__pstring* %31, i32 0, i32 0		; <i8**> [#uses=1]
	%33 = load i8** %32, align 4		; <i8*> [#uses=1]
	%34 = sub i32 %29, %30		; <i32> [#uses=1]
	%35 = getelementptr i8* %33, i32 %34		; <i8*> [#uses=1]
	store i8 32, i8* %35, align 1
	%36 = load i32* %23, align 4		; <i32> [#uses=1]
	%37 = icmp eq i32 %36, %j.0		; <i1> [#uses=1]
	br i1 %37, label %bb34, label %bb33

bb33:		; preds = %bb31
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb31

bb34:		; preds = %bb31, %bb27
	%38 = load i8* %19, align 1		; <i8> [#uses=1]
	%toBool35 = icmp eq i8 %38, 0		; <i1> [#uses=1]
	br i1 %toBool35, label %bb37, label %bb36

bb36:		; preds = %bb34
	%39 = load i32* %7, align 4		; <i32> [#uses=1]
	%40 = add i32 %39, 1		; <i32> [#uses=2]
	store i32 %40, i32* %7, align 4
	%41 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%42 = load i32* %41, align 4		; <i32> [#uses=1]
	%43 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%44 = load %struct.system__file_control_block__pstring** %43, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%45 = getelementptr %struct.system__file_control_block__pstring* %44, i32 0, i32 0		; <i8**> [#uses=1]
	%46 = load i8** %45, align 4		; <i8*> [#uses=1]
	%47 = sub i32 %40, %42		; <i32> [#uses=1]
	%48 = getelementptr i8* %46, i32 %47		; <i8*> [#uses=1]
	store i8 45, i8* %48, align 1
	br label %bb37

bb37:		; preds = %bb34, %bb36
	%49 = load i32* %1, align 4		; <i32> [#uses=1]
	%50 = icmp slt i32 %49, 0		; <i1> [#uses=1]
	br i1 %50, label %bb39, label %bb43

bb39:		; preds = %bb37
	%51 = load i32* %7, align 4		; <i32> [#uses=1]
	%52 = add i32 %51, 1		; <i32> [#uses=2]
	store i32 %52, i32* %7, align 4
	%53 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=3]
	%54 = load i32* %53, align 4		; <i32> [#uses=1]
	%55 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=3]
	%56 = load %struct.system__file_control_block__pstring** %55, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%57 = getelementptr %struct.system__file_control_block__pstring* %56, i32 0, i32 0		; <i8**> [#uses=1]
	%58 = load i8** %57, align 4		; <i8*> [#uses=1]
	%59 = sub i32 %52, %54		; <i32> [#uses=1]
	%60 = getelementptr i8* %58, i32 %59		; <i8*> [#uses=1]
	store i8 48, i8* %60, align 1
	%61 = load i32* %7, align 4		; <i32> [#uses=1]
	%62 = add i32 %61, 1		; <i32> [#uses=2]
	store i32 %62, i32* %7, align 4
	%63 = load i32* %53, align 4		; <i32> [#uses=1]
	%64 = load %struct.system__file_control_block__pstring** %55, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%65 = getelementptr %struct.system__file_control_block__pstring* %64, i32 0, i32 0		; <i8**> [#uses=1]
	%66 = load i8** %65, align 4		; <i8*> [#uses=1]
	%67 = sub i32 %62, %63		; <i32> [#uses=1]
	%68 = getelementptr i8* %66, i32 %67		; <i8*> [#uses=1]
	store i8 46, i8* %68, align 1
	%69 = load i32* %1, align 4		; <i32> [#uses=2]
	%70 = icmp sgt i32 %69, -2		; <i1> [#uses=1]
	br i1 %70, label %bb43, label %bb42

bb42:		; preds = %bb42, %bb39
	%indvar52 = phi i32 [ 0, %bb39 ], [ %indvar.next53, %bb42 ]		; <i32> [#uses=2]
	%71 = load i32* %7, align 4		; <i32> [#uses=1]
	%72 = add i32 %71, 1		; <i32> [#uses=2]
	store i32 %72, i32* %7, align 4
	%73 = load i32* %53, align 4		; <i32> [#uses=1]
	%74 = load %struct.system__file_control_block__pstring** %55, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%75 = getelementptr %struct.system__file_control_block__pstring* %74, i32 0, i32 0		; <i8**> [#uses=1]
	%76 = load i8** %75, align 4		; <i8*> [#uses=1]
	%77 = sub i32 %72, %73		; <i32> [#uses=1]
	%78 = getelementptr i8* %76, i32 %77		; <i8*> [#uses=1]
	store i8 48, i8* %78, align 1
	%j15.050 = add i32 %indvar52, %69		; <i32> [#uses=1]
	%79 = add i32 %j15.050, 1		; <i32> [#uses=1]
	%phitmp = icmp sgt i32 %79, -2		; <i1> [#uses=1]
	%indvar.next53 = add i32 %indvar52, 1		; <i32> [#uses=1]
	br i1 %phitmp, label %bb43, label %bb42

bb43:		; preds = %bb39, %bb42, %bb37
	%80 = trunc i64 %abs to i32		; <i32> [#uses=1]
	%81 = getelementptr [10 x i8]* @.str3, i32 0, i32 %80		; <i8*> [#uses=1]
	%82 = load i8* %81, align 1		; <i8> [#uses=1]
	%83 = load i32* %7, align 4		; <i32> [#uses=1]
	%84 = add i32 %83, 1		; <i32> [#uses=2]
	store i32 %84, i32* %7, align 4
	%85 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%86 = load i32* %85, align 4		; <i32> [#uses=1]
	%87 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%88 = load %struct.system__file_control_block__pstring** %87, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%89 = getelementptr %struct.system__file_control_block__pstring* %88, i32 0, i32 0		; <i8**> [#uses=1]
	%90 = load i8** %89, align 4		; <i8*> [#uses=1]
	%91 = sub i32 %84, %86		; <i32> [#uses=1]
	%92 = getelementptr i8* %90, i32 %91		; <i8*> [#uses=1]
	store i8 %82, i8* %92, align 1
	br label %bb48

bb44:		; preds = %bb19
	%93 = load i32* %1, align 4		; <i32> [#uses=1]
	%94 = icmp eq i32 %93, -1		; <i1> [#uses=1]
	%.pre = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%.pre55 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	br i1 %94, label %bb46, label %bb47

bb46:		; preds = %bb44
	%95 = add i32 %8, 1		; <i32> [#uses=2]
	store i32 %95, i32* %7, align 4
	%96 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=2]
	%97 = load i32* %96, align 4		; <i32> [#uses=1]
	%98 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=2]
	%99 = load %struct.system__file_control_block__pstring** %98, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%100 = getelementptr %struct.system__file_control_block__pstring* %99, i32 0, i32 0		; <i8**> [#uses=1]
	%101 = load i8** %100, align 4		; <i8*> [#uses=1]
	%102 = sub i32 %95, %97		; <i32> [#uses=1]
	%103 = getelementptr i8* %101, i32 %102		; <i8*> [#uses=1]
	store i8 46, i8* %103, align 1
	br label %bb47

bb47:		; preds = %bb44, %bb46
	%.pre-phi56 = phi %struct.system__file_control_block__pstring** [ %98, %bb46 ], [ %.pre55, %bb44 ]		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%.pre-phi = phi i32* [ %96, %bb46 ], [ %.pre, %bb44 ]		; <i32*> [#uses=1]
	%104 = trunc i64 %abs to i32		; <i32> [#uses=1]
	%105 = getelementptr [10 x i8]* @.str3, i32 0, i32 %104		; <i8*> [#uses=1]
	%106 = load i8* %105, align 1		; <i8> [#uses=1]
	%107 = load i32* %7, align 4		; <i32> [#uses=1]
	%108 = add i32 %107, 1		; <i32> [#uses=2]
	store i32 %108, i32* %7, align 4
	%109 = load i32* %.pre-phi, align 4		; <i32> [#uses=1]
	%110 = load %struct.system__file_control_block__pstring** %.pre-phi56, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%111 = getelementptr %struct.system__file_control_block__pstring* %110, i32 0, i32 0		; <i8**> [#uses=1]
	%112 = load i8** %111, align 4		; <i8*> [#uses=1]
	%113 = sub i32 %108, %109		; <i32> [#uses=1]
	%114 = getelementptr i8* %112, i32 %113		; <i8*> [#uses=1]
	store i8 %106, i8* %114, align 1
	br label %bb48

bb48:		; preds = %bb23, %bb47, %bb43
	%115 = load i32* %1, align 4		; <i32> [#uses=1]
	%116 = add i32 %115, -1		; <i32> [#uses=1]
	store i32 %116, i32* %1, align 4
	ret void

return:		; preds = %entry
	ret void
}

declare i8* @llvm.stacksave() nounwind

declare void @system__arith_64__scaled_divide({ i64, i64 }* noalias sret, i64, i64, i64, i8)

declare i64 @system__exn_lli__exn_long_long_integer(i64, i32) readnone

declare void @llvm.stackrestore(i8*) nounwind

declare void @ada__text_io__generic_aux__put_item(%struct.ada__text_io__text_afcb*, i8*, %struct.string___XUB*)

declare void @report__test(i8*, %struct.string___XUB*, i8*, %struct.string___XUB*)

declare void @system__secondary_stack__ss_mark(%struct.string___XUB* noalias sret)

declare void @system__exception_table__register_exception(%struct.system__standard_library__exception_data*)

declare void @report__legal_file_name(%struct.system__file_control_block__pstring* noalias sret, i32, i8*, %struct.string___XUB*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__create(%struct.ada__text_io__text_afcb*, i8, i8*, %struct.string___XUB*, i8*, %struct.string___XUB*)

declare void @system__secondary_stack__ss_release(i32, i32)

declare void @report__not_applicable(i8*, %struct.string___XUB*)

declare void @ada__text_io__set_output(%struct.ada__text_io__text_afcb*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__close(%struct.ada__text_io__text_afcb*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__open(%struct.ada__text_io__text_afcb*, i8, i8*, %struct.string___XUB*, i8*, %struct.string___XUB*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__standard_output()

declare void @report__failed(i8*, %struct.string___XUB*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__delete(%struct.ada__text_io__text_afcb*)

declare void @report__result()

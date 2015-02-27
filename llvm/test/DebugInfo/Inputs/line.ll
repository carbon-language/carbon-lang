; From source:
; int f(int a, int b) {
;   return a   //
;          &&  //
;          b;
; }

; Check that the comparison of 'a' is attributed to line 2, not 3.

; CHECK: .loc{{ +}}1{{ +}}2
; CHECK-NOT: .loc{{ }}
; CHECK: cmp

; Function Attrs: nounwind uwtable
define i32 @_Z1fii(i32 %a, i32 %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32, i32* %a.addr, align 4, !dbg !10
  %tobool = icmp ne i32 %0, 0, !dbg !10
  br i1 %tobool, label %land.rhs, label %land.end, !dbg !11

land.rhs:                                         ; preds = %entry
  %1 = load i32, i32* %b.addr, align 4, !dbg !12
  %tobool1 = icmp ne i32 %1, 0, !dbg !12
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %2 = phi i1 [ false, %entry ], [ %tobool1, %land.rhs ]
  %conv = zext i1 %2 to i32, !dbg !10
  ret i32 %conv, !dbg !13
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{!"0x11\004\00clang version 3.7.0 (trunk 227472) (llvm/trunk 227476)\000\00\000\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/line.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"line.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00f\00f\00\001\000\001\000\000\00256\000\001", !1, !5, !6, null, i32 (i32, i32)* @_Z1fii, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = !{!"0x29", !1}                               ; [ DW_TAG_file_type ] [/tmp/dbginfo/line.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.7.0 (trunk 227472) (llvm/trunk 227476)"}
!10 = !MDLocation(line: 2, scope: !4)
!11 = !MDLocation(line: 3, scope: !4)
!12 = !MDLocation(line: 4, scope: !4)
!13 = !MDLocation(line: 2, scope: !4)

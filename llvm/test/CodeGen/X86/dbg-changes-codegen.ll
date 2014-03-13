; RUN: llc < %s -march=x86-64 | FileCheck %s

; The Peephole optimizer should fold the load into the cmp even with debug info.
; CHECK-LABEL: _ZN3Foo3batEv
; CHECK-NOT: movq pfoo
; CHECK: cmpq {{%[a-z]+}}, pfoo(%rip)
;
; CHECK-LABEL: _Z3bazv
; CHECK-NOT: movq wibble2
; CHECK: cmpq {{%[a-z]+}}, wibble2(%rip)

; Regenerate test with this command: 
;   clang -emit-llvm -S -O2 -g
; from this source:
;   struct Foo {
;     bool bat();
;     bool operator==(Foo &arg) { return (this == &arg); }
;   };
;   Foo *pfoo;
;   bool Foo::bat() { return (*this == *pfoo); }
;
;   struct Wibble {
;     int x;
;   } *wibble1, *wibble2;
;   struct Flibble {
;     void bar(Wibble *c) {
;       if (c < wibble2)
;         wibble2 = 0;
;       c->x = 0;
;     }
;   } flibble;
;   void baz() { flibble.bar(wibble1); }

%struct.Foo = type { i8 }
%struct.Wibble = type { i32 }
%struct.Flibble = type { i8 }

@pfoo = global %struct.Foo* null, align 8
@wibble1 = global %struct.Wibble* null, align 8
@wibble2 = global %struct.Wibble* null, align 8
@flibble = global %struct.Flibble zeroinitializer, align 1

; Function Attrs: nounwind readonly uwtable
define zeroext i1 @_ZN3Foo3batEv(%struct.Foo* %this) #0 align 2 {
entry:
  %0 = load %struct.Foo** @pfoo, align 8
  tail call void @llvm.dbg.value(metadata !{%struct.Foo* %0}, i64 0, metadata !62)
  %cmp.i = icmp eq %struct.Foo* %0, %this
  ret i1 %cmp.i
}

; Function Attrs: nounwind uwtable
define void @_Z3bazv() #1 {
entry:
  %0 = load %struct.Wibble** @wibble1, align 8
  tail call void @llvm.dbg.value(metadata !64, i64 0, metadata !65)
  %1 = load %struct.Wibble** @wibble2, align 8
  %cmp.i = icmp ugt %struct.Wibble* %1, %0
  br i1 %cmp.i, label %if.then.i, label %_ZN7Flibble3barEP6Wibble.exit

if.then.i:                                        ; preds = %entry
  store %struct.Wibble* null, %struct.Wibble** @wibble2, align 8
  br label %_ZN7Flibble3barEP6Wibble.exit

_ZN7Flibble3barEP6Wibble.exit:                    ; preds = %entry, %if.then.i
  %x.i = getelementptr inbounds %struct.Wibble* %0, i64 0, i32 0
  store i32 0, i32* %x.i, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #2

attributes #0 = { nounwind readonly uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }


!17 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, null} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from Foo]
!45 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from Flibble]
!62 = metadata !{i32 786689, null, metadata !"arg", null, i32 33554436, metadata !17, i32 0, null} ; [ DW_TAG_arg_variable ] [arg] [line 4]
!64 = metadata !{%struct.Flibble* undef}
!65 = metadata !{i32 786689, null, metadata !"this", null, i32 16777229, metadata !45, i32 1088, null} ; [ DW_TAG_arg_variable ] [this] [line 13]

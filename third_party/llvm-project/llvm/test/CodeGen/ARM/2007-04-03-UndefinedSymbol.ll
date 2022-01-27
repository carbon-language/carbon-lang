; RUN: llc -mtriple arm-apple-darwin -relocation-model pic -filetype asm -o - %s | FileCheck %s

%struct.B = type { i32 }
%struct.anon = type { void (%struct.B*)*, i32 }
@str = internal constant [7 x i8] c"i, %d\0A\00"
@str1 = internal constant [7 x i8] c"j, %d\0A\00"

define internal void @_ZN1B1iEv(%struct.B* %this) {
entry:
  %tmp1 = getelementptr %struct.B, %struct.B* %this, i32 0, i32 0
  %tmp2 = load i32, i32* %tmp1
  %tmp4 = tail call i32 (i8*, ...) @printf(i8* getelementptr ([7 x i8], [7 x i8]* @str, i32 0, i32 0), i32 %tmp2)
  ret void
}

declare i32 @printf(i8*, ...)

define internal void @_ZN1B1jEv(%struct.B* %this) {
entry:
  %tmp1 = getelementptr %struct.B, %struct.B* %this, i32 0, i32 0
  %tmp2 = load i32, i32* %tmp1
  %tmp4 = tail call i32 (i8*, ...) @printf(i8* getelementptr ([7 x i8], [7 x i8]* @str1, i32 0, i32 0), i32 %tmp2)
  ret void
}

define i32 @main() {
entry:
  %b.i29 = alloca %struct.B, align 4
  %b.i1 = alloca %struct.B, align 4
  %b.i = alloca %struct.B, align 4
  %tmp2.i = getelementptr %struct.B, %struct.B* %b.i, i32 0, i32 0
  store i32 4, i32* %tmp2.i
  br i1 icmp eq (i64 and (i64 zext (i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32) to i64), i64 4294967296), i64 0), label %_Z3fooiM1BFvvE.exit, label %cond_true.i

cond_true.i:
  %b2.i = bitcast %struct.B* %b.i to i8*
  %ctg23.i = getelementptr i8, i8* %b2.i, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32) to i64), i64 32) to i32), i32 1)
  %tmp121314.i = bitcast i8* %ctg23.i to i32 (...)***
  %tmp15.i = load i32 (...)**, i32 (...)*** %tmp121314.i
  %tmp151.i = bitcast i32 (...)** %tmp15.i to i8*
  %ctg2.i = getelementptr i8, i8* %tmp151.i, i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32)
  %tmp2021.i = bitcast i8* %ctg2.i to i32 (...)**
  %tmp22.i = load i32 (...)*, i32 (...)** %tmp2021.i
  %tmp2223.i = bitcast i32 (...)* %tmp22.i to void (%struct.B*)*
  br label %_Z3fooiM1BFvvE.exit

_Z3fooiM1BFvvE.exit:
  %iftmp.2.0.i = phi void (%struct.B*)* [ %tmp2223.i, %cond_true.i ], [ inttoptr (i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32) to void (%struct.B*)*), %entry ]
  %b4.i = bitcast %struct.B* %b.i to i8*
  %ctg25.i = getelementptr i8, i8* %b4.i, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32) to i64), i64 32) to i32), i32 1)
  %tmp3031.i = bitcast i8* %ctg25.i to %struct.B*
  call void %iftmp.2.0.i(%struct.B* %tmp3031.i)
  %tmp2.i30 = getelementptr %struct.B, %struct.B* %b.i29, i32 0, i32 0
  store i32 6, i32* %tmp2.i30
  br i1 icmp eq (i64 and (i64 zext (i32 ptrtoint (void (%struct.B*)* @_ZN1B1jEv to i32) to i64), i64 4294967296), i64 0), label %_Z3fooiM1BFvvE.exit56, label %cond_true.i46

cond_true.i46:
  %b2.i35 = bitcast %struct.B* %b.i29 to i8*
  %ctg23.i36 = getelementptr i8, i8* %b2.i35, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (void (%struct.B*)* @_ZN1B1jEv to i32) to i64), i64 32) to i32), i32 1)
  %tmp121314.i37 = bitcast i8* %ctg23.i36 to i32 (...)***
  %tmp15.i38 = load i32 (...)**, i32 (...)*** %tmp121314.i37
  %tmp151.i41 = bitcast i32 (...)** %tmp15.i38 to i8*
  %ctg2.i42 = getelementptr i8, i8* %tmp151.i41, i32 ptrtoint (void (%struct.B*)* @_ZN1B1jEv to i32)
  %tmp2021.i43 = bitcast i8* %ctg2.i42 to i32 (...)**
  %tmp22.i44 = load i32 (...)*, i32 (...)** %tmp2021.i43
  %tmp2223.i45 = bitcast i32 (...)* %tmp22.i44 to void (%struct.B*)*
  br label %_Z3fooiM1BFvvE.exit56

_Z3fooiM1BFvvE.exit56:
  %iftmp.2.0.i49 = phi void (%struct.B*)* [ %tmp2223.i45, %cond_true.i46 ], [ inttoptr (i32 ptrtoint (void (%struct.B*)* @_ZN1B1jEv to i32) to void (%struct.B*)*), %_Z3fooiM1BFvvE.exit ]
  %b4.i53 = bitcast %struct.B* %b.i29 to i8*
  %ctg25.i54 = getelementptr i8, i8* %b4.i53, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (void (%struct.B*)* @_ZN1B1jEv to i32) to i64), i64 32) to i32), i32 1)
  %tmp3031.i55 = bitcast i8* %ctg25.i54 to %struct.B*
  call void %iftmp.2.0.i49(%struct.B* %tmp3031.i55)
  %tmp2.i2 = getelementptr %struct.B, %struct.B* %b.i1, i32 0, i32 0
  store i32 -1, i32* %tmp2.i2
  br i1 icmp eq (i64 and (i64 zext (i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32) to i64), i64 4294967296), i64 0), label %_Z3fooiM1BFvvE.exit28, label %cond_true.i18

cond_true.i18:
  %b2.i7 = bitcast %struct.B* %b.i1 to i8*
  %ctg23.i8 = getelementptr i8, i8* %b2.i7, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32) to i64), i64 32) to i32), i32 1)
  %tmp121314.i9 = bitcast i8* %ctg23.i8 to i32 (...)***
  %tmp15.i10 = load i32 (...)**, i32 (...)*** %tmp121314.i9
  %tmp151.i13 = bitcast i32 (...)** %tmp15.i10 to i8*
  %ctg2.i14 = getelementptr i8, i8* %tmp151.i13, i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32)
  %tmp2021.i15 = bitcast i8* %ctg2.i14 to i32 (...)**
  %tmp22.i16 = load i32 (...)*, i32 (...)** %tmp2021.i15
  %tmp2223.i17 = bitcast i32 (...)* %tmp22.i16 to void (%struct.B*)*
  br label %_Z3fooiM1BFvvE.exit28

_Z3fooiM1BFvvE.exit28:
  %iftmp.2.0.i21 = phi void (%struct.B*)* [ %tmp2223.i17, %cond_true.i18 ], [ inttoptr (i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32) to void (%struct.B*)*), %_Z3fooiM1BFvvE.exit56 ]
  %b4.i25 = bitcast %struct.B* %b.i1 to i8*
  %ctg25.i26 = getelementptr i8, i8* %b4.i25, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (void (%struct.B*)* @_ZN1B1iEv to i32) to i64), i64 32) to i32), i32 1)
  %tmp3031.i27 = bitcast i8* %ctg25.i26 to %struct.B*
  call void %iftmp.2.0.i21(%struct.B* %tmp3031.i27)
  ret i32 0
}

; CHECK-NOT: LPC9


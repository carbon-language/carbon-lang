; RUN: sed -e "s/ORDER/unordered/" %s | llc -march=hexagon -disable-hexagon-amodeopt=0 | FileCheck %s
; RUN: sed -e "s/ORDER/monotonic/" %s | llc -march=hexagon -disable-hexagon-amodeopt=0 | FileCheck %s
; RUN: sed -e "s/ORDER/acquire/" %s | llc -march=hexagon -disable-hexagon-amodeopt=0 | FileCheck %s
; RUN: sed -e "s/ORDER/seq_cst/" %s | llc -march=hexagon -disable-hexagon-amodeopt=0 | FileCheck %s

%struct.Obj = type { [100 x i32] }

@i8Src   = global i8 0,  align 1
@i8Dest  = global i8 0,  align 1
@i16Src  = global i16 0, align 2
@i16Dest = global i16 0, align 2
@i32Src  = global i32 0, align 4
@i32Dest = global i32 0, align 4
@i64Src  = global i64 0, align 8
@i64Dest = global i64 0, align 8
@ptrSrc  = global %struct.Obj* null, align 4
@ptrDest = global %struct.Obj* null, align 4

define void @load_i8() #0 {
entry:
  %i8Tmp = load atomic i8, i8* @i8Src ORDER, align 1
  store i8 %i8Tmp, i8* @i8Dest, align 1
  ret void
}
; CHECK-LABEL: load_i8:
; CHECK: [[TMP_REG:r[0-9]+]] = memub(gp+#i8Src)
; CHECK: memb(gp+#i8Dest) = [[TMP_REG]]

define void @load_i16() #0 {
entry:
  %i16Tmp = load atomic i16, i16* @i16Src ORDER, align 2
  store i16 %i16Tmp, i16* @i16Dest, align 2
  ret void
}
; CHECK-LABEL: load_i16:
; CHECK: [[TMP_REG:r[0-9]+]] = memuh(gp+#i16Src)
; CHECK: memh(gp+#i16Dest) = [[TMP_REG]]

define void @load_i32() #0 {
entry:
  %i32Tmp = load atomic i32, i32* @i32Src ORDER, align 4
  store i32 %i32Tmp, i32* @i32Dest, align 4
  ret void
}
; CHECK-LABEL: load_i32:
; CHECK: [[TMP_REG:r[0-9]+]] = memw(gp+#i32Src)
; CHECK: memw(gp+#i32Dest) = [[TMP_REG]]

define void @load_i64() #0 {
entry:
  %i64Tmp = load atomic i64, i64* @i64Src ORDER, align 8
  store i64 %i64Tmp, i64* @i64Dest, align 8
  ret void
}
; CHECK-LABEL: load_i64:
; CHECK: [[TMP_REG:r[0-9]+:[0-9]+]] = memd(gp+#i64Src)
; CHECK: memd(gp+#i64Dest) = [[TMP_REG]]

define void @load_ptr() #0 {
entry:
  %ptrTmp = load atomic i32, i32* bitcast (%struct.Obj** @ptrSrc to i32*) ORDER, align 4
  store i32 %ptrTmp, i32* bitcast (%struct.Obj** @ptrDest to i32*), align 4
  ret void
}
; CHECK-LABEL: load_ptr:
; CHECK: [[TMP_REG:r[0-9]+]] = memw(gp+#ptrSrc)
; CHECK: memw(gp+#ptrDest) = [[TMP_REG]]


; RUN: llc < %s -march=mipsel -mcpu=mips32 -O0 -relocation-model=pic \
; RUN:      -fast-isel-abort=3 | FileCheck %s
; RUN: llc < %s -march=mipsel -mcpu=mips32r2 -O0 -relocation-model=pic \
; RUN:      -fast-isel-abort=3 | FileCheck %s

@sj = global i32 200, align 4
@sk = global i32 -47, align 4
@uj = global i32 200, align 4
@uk = global i32 43, align 4
@si = common global i32 0, align 4
@ui = common global i32 0, align 4

define void @rems() {
  ; CHECK-LABEL:  rems:

  ; CHECK:            lui     $[[GOT1:[0-9]+]], %hi(_gp_disp)
  ; CHECK:            addiu   $[[GOT2:[0-9]+]], $[[GOT1]], %lo(_gp_disp)
  ; CHECK:            addu    $[[GOT:[0-9]+]], $[[GOT2:[0-9]+]], $25
  ; CHECK-DAG:        lw      $[[I_ADDR:[0-9]+]], %got(si)($[[GOT]])
  ; CHECK-DAG:        lw      $[[K_ADDR:[0-9]+]], %got(sk)($[[GOT]])
  ; CHECK-DAG:        lw      $[[J_ADDR:[0-9]+]], %got(sj)($[[GOT]])
  ; CHECK-DAG:        lw      $[[J:[0-9]+]], 0($[[J_ADDR]])
  ; CHECK-DAG:        lw      $[[K:[0-9]+]], 0($[[K_ADDR]])
  ; CHECK-DAG:        div     $zero, $[[J]], $[[K]]
  ; CHECK-DAG:        teq     $[[K]], $zero, 7
  ; CHECK-DAG:        mfhi    $[[RESULT:[0-9]+]]
  ; CHECK:            sw      $[[RESULT]], 0($[[I_ADDR]])
  %1 = load i32, i32* @sj, align 4
  %2 = load i32, i32* @sk, align 4
  %rem = srem i32 %1, %2
  store i32 %rem, i32* @si, align 4
  ret void
}

; Function Attrs: noinline nounwind
define void @remu() {
  ; CHECK-LABEL:  remu:

  ; CHECK:            lui     $[[GOT1:[0-9]+]], %hi(_gp_disp)
  ; CHECK:            addiu   $[[GOT2:[0-9]+]], $[[GOT1]], %lo(_gp_disp)
  ; CHECK:            addu    $[[GOT:[0-9]+]], $[[GOT2:[0-9]+]], $25
  ; CHECK-DAG:        lw      $[[I_ADDR:[0-9]+]], %got(ui)($[[GOT]])
  ; CHECK-DAG:        lw      $[[K_ADDR:[0-9]+]], %got(uk)($[[GOT]])
  ; CHECK-DAG:        lw      $[[J_ADDR:[0-9]+]], %got(uj)($[[GOT]])
  ; CHECK-DAG:        lw      $[[J:[0-9]+]], 0($[[J_ADDR]])
  ; CHECK-DAG:        lw      $[[K:[0-9]+]], 0($[[K_ADDR]])
  ; CHECK-DAG:        divu    $zero, $[[J]], $[[K]]
  ; CHECK-DAG:        teq     $[[K]], $zero, 7
  ; CHECK-DAG:        mfhi    $[[RESULT:[0-9]+]]
  ; CHECK:            sw      $[[RESULT]], 0($[[I_ADDR]])
  %1 = load i32, i32* @uj, align 4
  %2 = load i32, i32* @uk, align 4
  %rem = urem i32 %1, %2
  store i32 %rem, i32* @ui, align 4
  ret void
}

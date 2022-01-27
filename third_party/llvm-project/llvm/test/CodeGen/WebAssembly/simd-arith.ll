; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128,SIMD128-SLOW

; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 -fast-isel | FileCheck %s --check-prefixes CHECK,SIMD128,SIMD128-FAST

; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -fast-isel | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; check that a non-test run (including explicit locals pass) at least finishes
; RUN: llc < %s -O0 -mattr=+simd128
; RUN: llc < %s -O2 -mattr=+simd128

; Test that basic SIMD128 arithmetic operations assemble as expected.

target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
; CHECK-LABEL: add_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype add_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.add $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @add_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = add <16 x i8> %x, %y
  ret <16 x i8> %a
}

; CHECK-LABEL: sub_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype sub_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.sub $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @sub_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = sub <16 x i8> %x, %y
  ret <16 x i8> %a
}

; i8x16.mul is not in spec
; CHECK-LABEL: mul_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NOT: i8x16.mul
; SIMD128: i8x16.extract_lane_u
; SIMD128: i32.mul
define <16 x i8> @mul_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = mul <16 x i8> %x, %y
  ret <16 x i8> %a
}

; CHECK-LABEL: min_s_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype min_s_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.min_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @min_s_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %c = icmp slt <16 x i8> %x, %y
  %a = select <16 x i1> %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %a
}

; CHECK-LABEL: min_u_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype min_u_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.min_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @min_u_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %c = icmp ult <16 x i8> %x, %y
  %a = select <16 x i1> %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %a
}

; CHECK-LABEL: max_s_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype max_s_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.max_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @max_s_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %c = icmp sgt <16 x i8> %x, %y
  %a = select <16 x i1> %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %a
}

; CHECK-LABEL: max_u_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype max_u_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.max_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @max_u_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %c = icmp ugt <16 x i8> %x, %y
  %a = select <16 x i1> %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %a
}

; CHECK-LABEL: avgr_u_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype avgr_u_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.avgr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @avgr_u_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = add nuw <16 x i8> %x, %y
  %b = add nuw <16 x i8> %a, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                              i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %c = udiv <16 x i8> %b, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2,
                           i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  ret <16 x i8> %c
}

; CHECK-LABEL: avgr_u_v16i8_wrap:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype avgr_u_v16i8_wrap (v128, v128) -> (v128){{$}}
; SIMD128-NOT: i8x16.avgr_u
define <16 x i8> @avgr_u_v16i8_wrap(<16 x i8> %x, <16 x i8> %y) {
  %a = add <16 x i8> %x, %y
  %b = add <16 x i8> %a, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                          i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %c = udiv <16 x i8> %b, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2,
                           i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
  ret <16 x i8> %c
}

; CHECK-LABEL: abs_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype abs_v16i8 (v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.abs $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @abs_v16i8(<16 x i8> %x) {
  %a = sub <16 x i8> zeroinitializer, %x
  %b = icmp slt <16 x i8> %x, zeroinitializer
  %c = select <16 x i1> %b, <16 x i8> %a, <16 x i8> %x
  ret <16 x i8> %c
}

; CHECK-LABEL: neg_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype neg_v16i8 (v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.neg $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @neg_v16i8(<16 x i8> %x) {
  %a = sub <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                      i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>,
                     %x
  ret <16 x i8> %a
}

; CHECK-LABEL: shl_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shl_v16i8 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shl $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shl_v16i8(<16 x i8> %v, i8 %x) {
  %t = insertelement <16 x i8> undef, i8 %x, i32 0
  %s = shufflevector <16 x i8> %t, <16 x i8> undef,
    <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0,
                i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %a = shl <16 x i8> %v, %s
  ret <16 x i8> %a
}

; CHECK-LABEL: shl_const_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shl_const_v16i8 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 5
; SIMD128-NEXT: i8x16.shl $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shl_const_v16i8(<16 x i8> %v) {
  %a = shl <16 x i8> %v,
    <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5,
     i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  ret <16 x i8> %a
}

; CHECK-LABEL: shl_vec_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shl_vec_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i32.const $push[[M0:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[M1:[0-9]+]]=, $pop[[L1]], $pop[[M0]]{{$}}
; SIMD128-NEXT: i32.shl $push[[M2:[0-9]+]]=, $pop[[L0]], $pop[[M1]]
; SIMD128-NEXT: i8x16.splat $push[[M3:[0-9]+]]=, $pop[[M2]]
; Skip 14 lanes
; SIMD128:      i8x16.extract_lane_u $push[[L4:[0-9]+]]=, $0, 15{{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[L5:[0-9]+]]=, $1, 15{{$}}
; SIMD128-NEXT: i32.const $push[[M4:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[M5:[0-9]+]]=, $pop[[L5]], $pop[[M4]]{{$}}
; SIMD128-NEXT: i32.shl $push[[M6:[0-9]+]]=, $pop[[L4]], $pop[[M5]]{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[R:[0-9]+]]=, $pop[[M7:[0-9]+]], 15, $pop[[M6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shl_vec_v16i8(<16 x i8> %v, <16 x i8> %x) {
  %a = shl <16 x i8> %v, %x
  ret <16 x i8> %a
}

; CHECK-LABEL: shr_s_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shr_s_v16i8 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shr_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shr_s_v16i8(<16 x i8> %v, i8 %x) {
  %t = insertelement <16 x i8> undef, i8 %x, i32 0
  %s = shufflevector <16 x i8> %t, <16 x i8> undef,
    <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0,
                i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %a = ashr <16 x i8> %v, %s
  ret <16 x i8> %a
}

; CHECK-LABEL: shr_s_vec_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shr_s_vec_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.extract_lane_s $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i32.const $push[[M0:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[M1:[0-9]+]]=, $pop[[L1]], $pop[[M0]]{{$}}
; SIMD128-NEXT: i32.shr_s $push[[M2:[0-9]+]]=, $pop[[L0]], $pop[[M1]]
; SIMD128-NEXT: i8x16.splat $push[[M3:[0-9]+]]=, $pop[[M2]]
; Skip 14 lanes
; SIMD128:      i8x16.extract_lane_s $push[[L4:[0-9]+]]=, $0, 15{{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[L5:[0-9]+]]=, $1, 15{{$}}
; SIMD128-NEXT: i32.const $push[[M4:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[M5:[0-9]+]]=, $pop[[L5]], $pop[[M4]]{{$}}
; SIMD128-NEXT: i32.shr_s $push[[M6:[0-9]+]]=, $pop[[L4]], $pop[[M5]]{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[R:[0-9]+]]=, $pop[[M7:[0-9]+]], 15, $pop[[M6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shr_s_vec_v16i8(<16 x i8> %v, <16 x i8> %x) {
  %a = ashr <16 x i8> %v, %x
  ret <16 x i8> %a
}

; CHECK-LABEL: shr_u_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shr_u_v16i8 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i8x16.shr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shr_u_v16i8(<16 x i8> %v, i8 %x) {
  %t = insertelement <16 x i8> undef, i8 %x, i32 0
  %s = shufflevector <16 x i8> %t, <16 x i8> undef,
    <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0,
                i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %a = lshr <16 x i8> %v, %s
  ret <16 x i8> %a
}

; CHECK-LABEL: shr_u_vec_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype shr_u_vec_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i32.const $push[[M0:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[M1:[0-9]+]]=, $pop[[L1]], $pop[[M0]]{{$}}
; SIMD128-NEXT: i32.shr_u $push[[M2:[0-9]+]]=, $pop[[L0]], $pop[[M1]]
; SIMD128-NEXT: i8x16.splat $push[[M3:[0-9]+]]=, $pop[[M2]]
; Skip 14 lanes
; SIMD128:      i8x16.extract_lane_u $push[[L4:[0-9]+]]=, $0, 15{{$}}
; SIMD128-NEXT: i8x16.extract_lane_u $push[[L5:[0-9]+]]=, $1, 15{{$}}
; SIMD128-NEXT: i32.const $push[[M4:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i32.and $push[[M5:[0-9]+]]=, $pop[[L5]], $pop[[M4]]{{$}}
; SIMD128-NEXT: i32.shr_u $push[[M6:[0-9]+]]=, $pop[[L4]], $pop[[M5]]{{$}}
; SIMD128-NEXT: i8x16.replace_lane $push[[R:[0-9]+]]=, $pop[[M7:[0-9]+]], 15, $pop[[M6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shr_u_vec_v16i8(<16 x i8> %v, <16 x i8> %x) {
  %a = lshr <16 x i8> %v, %x
  ret <16 x i8> %a
}

; CHECK-LABEL: and_v16i8:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype and_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @and_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = and <16 x i8> %x, %y
  ret <16 x i8> %a
}

; CHECK-LABEL: or_v16i8:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype or_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @or_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = or <16 x i8> %x, %y
  ret <16 x i8> %a
}

; CHECK-LABEL: xor_v16i8:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype xor_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.xor $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @xor_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = xor <16 x i8> %x, %y
  ret <16 x i8> %a
}

; CHECK-LABEL: not_v16i8:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype not_v16i8 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @not_v16i8(<16 x i8> %x) {
  %a = xor <16 x i8> %x, <i8 -1, i8 -1, i8 -1, i8 -1,
                          i8 -1, i8 -1, i8 -1, i8 -1,
                          i8 -1, i8 -1, i8 -1, i8 -1,
                          i8 -1, i8 -1, i8 -1, i8 -1>
  ret <16 x i8> %a
}

; CHECK-LABEL: andnot_v16i8:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype andnot_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: v128.andnot $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
; SIMD128-FAST-NEXT: v128.not
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: return
define <16 x i8> @andnot_v16i8(<16 x i8> %x, <16 x i8> %y) {
 %inv_y = xor <16 x i8> %y,
   <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
 %a = and <16 x i8> %x, %inv_y
 ret <16 x i8> %a
}

; CHECK-LABEL: bitselect_v16i8:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype bitselect_v16i8 (v128, v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: v128.not
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: v128.or
; SIMD128-FAST-NEXT: return
define <16 x i8> @bitselect_v16i8(<16 x i8> %c, <16 x i8> %v1, <16 x i8> %v2) {
  %masked_v1 = and <16 x i8> %c, %v1
  %inv_mask = xor <16 x i8> %c,
    <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
     i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %masked_v2 = and <16 x i8> %inv_mask, %v2
  %a = or <16 x i8> %masked_v1, %masked_v2
  ret <16 x i8> %a
}

; ==============================================================================
; 8 x i16
; ==============================================================================
; CHECK-LABEL: add_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype add_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.add $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @add_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = add <8 x i16> %x, %y
  ret <8 x i16> %a
}

; CHECK-LABEL: sub_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype sub_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.sub $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @sub_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = sub <8 x i16> %x, %y
  ret <8 x i16> %a
}

; CHECK-LABEL: mul_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype mul_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.mul $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @mul_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = mul <8 x i16> %x, %y
  ret <8 x i16> %a
}

; CHECK-LABEL: min_s_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype min_s_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.min_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @min_s_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %c = icmp slt <8 x i16> %x, %y
  %a = select <8 x i1> %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %a
}

; CHECK-LABEL: min_u_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype min_u_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.min_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @min_u_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %c = icmp ult <8 x i16> %x, %y
  %a = select <8 x i1> %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %a
}

; CHECK-LABEL: max_s_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype max_s_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.max_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @max_s_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %c = icmp sgt <8 x i16> %x, %y
  %a = select <8 x i1> %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %a
}

; CHECK-LABEL: max_u_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype max_u_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.max_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @max_u_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %c = icmp ugt <8 x i16> %x, %y
  %a = select <8 x i1> %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %a
}

; CHECK-LABEL: avgr_u_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype avgr_u_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.avgr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @avgr_u_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = add nuw <8 x i16> %x, %y
  %b = add nuw <8 x i16> %a, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %c = udiv <8 x i16> %b, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <8 x i16> %c
}

; CHECK-LABEL: avgr_u_v8i16_wrap:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype avgr_u_v8i16_wrap (v128, v128) -> (v128){{$}}
; SIMD128-NOT: i16x8.avgr_u
define <8 x i16> @avgr_u_v8i16_wrap(<8 x i16> %x, <8 x i16> %y) {
  %a = add <8 x i16> %x, %y
  %b = add <8 x i16> %a, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %c = udiv <8 x i16> %b, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <8 x i16> %c
}

; CHECK-LABEL: abs_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype abs_v8i16 (v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.abs $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @abs_v8i16(<8 x i16> %x) {
  %a = sub <8 x i16> zeroinitializer, %x
  %b = icmp slt <8 x i16> %x, zeroinitializer
  %c = select <8 x i1> %b, <8 x i16> %a, <8 x i16> %x
  ret <8 x i16> %c
}

; CHECK-LABEL: neg_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype neg_v8i16 (v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.neg $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @neg_v8i16(<8 x i16> %x) {
  %a = sub <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>,
                     %x
  ret <8 x i16> %a
}

; CHECK-LABEL: shl_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype shl_v8i16 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.shl $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @shl_v8i16(<8 x i16> %v, i16 %x) {
  %t = insertelement <8 x i16> undef, i16 %x, i32 0
  %s = shufflevector <8 x i16> %t, <8 x i16> undef,
    <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %a = shl <8 x i16> %v, %s
  ret <8 x i16> %a
}

; CHECK-LABEL: shl_const_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype shl_const_v8i16 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 5
; SIMD128-NEXT: i16x8.shl $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @shl_const_v8i16(<8 x i16> %v) {
  %a = shl <8 x i16> %v,
    <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  ret <8 x i16> %a
}

; CHECK-LABEL: shl_vec_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype shl_vec_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i32.const $push[[M0:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i32.and $push[[M1:[0-9]+]]=, $pop[[L1]], $pop[[M0]]{{$}}
; SIMD128-NEXT: i32.shl $push[[M2:[0-9]+]]=, $pop[[L0]], $pop[[M1]]{{$}}
; SIMD128-NEXT: i16x8.splat $push[[M3:[0-9]+]]=, $pop[[M2]]{{$}}
; Skip 6 lanes
; SIMD128:      i16x8.extract_lane_u $push[[L4:[0-9]+]]=, $0, 7{{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[L5:[0-9]+]]=, $1, 7{{$}}
; SIMD128-NEXT: i32.const $push[[M4:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i32.and $push[[M5:[0-9]+]]=, $pop[[L5]], $pop[[M4]]{{$}}
; SIMD128-NEXT: i32.shl $push[[M6:[0-9]+]]=, $pop[[L4]], $pop[[M5]]{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[R:[0-9]+]]=, $pop[[M7:[0-9]+]], 7, $pop[[M6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @shl_vec_v8i16(<8 x i16> %v, <8 x i16> %x) {
  %a = shl <8 x i16> %v, %x
  ret <8 x i16> %a
}

; CHECK-LABEL: shr_s_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype shr_s_v8i16 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.shr_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @shr_s_v8i16(<8 x i16> %v, i16 %x) {
  %t = insertelement <8 x i16> undef, i16 %x, i32 0
  %s = shufflevector <8 x i16> %t, <8 x i16> undef,
    <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %a = ashr <8 x i16> %v, %s
  ret <8 x i16> %a
}

; CHECK-LABEL: shr_s_vec_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype shr_s_vec_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.extract_lane_s $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i32.const $push[[M0:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i32.and $push[[M1:[0-9]+]]=, $pop[[L1]], $pop[[M0]]{{$}}
; SIMD128-NEXT: i32.shr_s $push[[M2:[0-9]+]]=, $pop[[L0]], $pop[[M1]]{{$}}
; SIMD128-NEXT: i16x8.splat $push[[M3:[0-9]+]]=, $pop[[M2]]{{$}}
; Skip 6 lanes
; SIMD128:      i16x8.extract_lane_s $push[[L4:[0-9]+]]=, $0, 7{{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[L5:[0-9]+]]=, $1, 7{{$}}
; SIMD128-NEXT: i32.const $push[[M4:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i32.and $push[[M5:[0-9]+]]=, $pop[[L5]], $pop[[M4]]{{$}}
; SIMD128-NEXT: i32.shr_s $push[[M6:[0-9]+]]=, $pop[[L4]], $pop[[M5]]{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[R:[0-9]+]]=, $pop[[M7:[0-9]+]], 7, $pop[[M6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @shr_s_vec_v8i16(<8 x i16> %v, <8 x i16> %x) {
  %a = ashr <8 x i16> %v, %x
  ret <8 x i16> %a
}

; CHECK-LABEL: shr_u_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype shr_u_v8i16 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i16x8.shr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @shr_u_v8i16(<8 x i16> %v, i16 %x) {
  %t = insertelement <8 x i16> undef, i16 %x, i32 0
  %s = shufflevector <8 x i16> %t, <8 x i16> undef,
    <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %a = lshr <8 x i16> %v, %s
  ret <8 x i16> %a
}

; CHECK-LABEL: shr_u_vec_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype shr_u_vec_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i32.const $push[[M0:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i32.and $push[[M1:[0-9]+]]=, $pop[[L1]], $pop[[M0]]{{$}}
; SIMD128-NEXT: i32.shr_u $push[[M2:[0-9]+]]=, $pop[[L0]], $pop[[M1]]{{$}}
; SIMD128-NEXT: i16x8.splat $push[[M3:[0-9]+]]=, $pop[[M2]]{{$}}
; Skip 6 lanes
; SIMD128:      i16x8.extract_lane_u $push[[L4:[0-9]+]]=, $0, 7{{$}}
; SIMD128-NEXT: i16x8.extract_lane_u $push[[L5:[0-9]+]]=, $1, 7{{$}}
; SIMD128-NEXT: i32.const $push[[M4:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i32.and $push[[M5:[0-9]+]]=, $pop[[L5]], $pop[[M4]]{{$}}
; SIMD128-NEXT: i32.shr_u $push[[M6:[0-9]+]]=, $pop[[L4]], $pop[[M5]]{{$}}
; SIMD128-NEXT: i16x8.replace_lane $push[[R:[0-9]+]]=, $pop[[M7:[0-9]+]], 7, $pop[[M6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @shr_u_vec_v8i16(<8 x i16> %v, <8 x i16> %x) {
  %a = lshr <8 x i16> %v, %x
  ret <8 x i16> %a
}

; CHECK-LABEL: and_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype and_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @and_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = and <8 x i16> %x, %y
  ret <8 x i16> %a
}

; CHECK-LABEL: or_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype or_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @or_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = or <8 x i16> %x, %y
  ret <8 x i16> %a
}

; CHECK-LABEL: xor_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype xor_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.xor $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @xor_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = xor <8 x i16> %x, %y
  ret <8 x i16> %a
}

; CHECK-LABEL: not_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype not_v8i16 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @not_v8i16(<8 x i16> %x) {
  %a = xor <8 x i16> %x, <i16 -1, i16 -1, i16 -1, i16 -1,
                          i16 -1, i16 -1, i16 -1, i16 -1>
  ret <8 x i16> %a
}

; CHECK-LABEL: andnot_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype andnot_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: v128.andnot $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
; SIMD128-FAST-NEXT: v128.not
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: return
define <8 x i16> @andnot_v8i16(<8 x i16> %x, <8 x i16> %y) {
 %inv_y = xor <8 x i16> %y,
   <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
 %a = and <8 x i16> %x, %inv_y
 ret <8 x i16> %a
}

; CHECK-LABEL: bitselect_v8i16:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype bitselect_v8i16 (v128, v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: v128.not
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: v128.or
; SIMD128-FAST-NEXT: return
define <8 x i16> @bitselect_v8i16(<8 x i16> %c, <8 x i16> %v1, <8 x i16> %v2) {
  %masked_v1 = and <8 x i16> %v1, %c
  %inv_mask = xor <8 x i16>
    <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>,
    %c
  %masked_v2 = and <8 x i16> %v2, %inv_mask
  %a = or <8 x i16> %masked_v1, %masked_v2
  ret <8 x i16> %a
}

; CHECK-LABEL: extmul_low_s_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extmul_low_s_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i16x8.extmul_low_i8x16_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @extmul_low_s_v8i16(<16 x i8> %v1, <16 x i8> %v2) {
  %low1 = shufflevector <16 x i8> %v1, <16 x i8> undef,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %low2 = shufflevector <16 x i8> %v2, <16 x i8> undef,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %extended1 = sext <8 x i8> %low1 to <8 x i16>
  %extended2 = sext <8 x i8> %low2 to <8 x i16>
  %a = mul <8 x i16> %extended1, %extended2
  ret <8 x i16> %a
}

; CHECK-LABEL: extmul_high_s_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extmul_high_s_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i16x8.extmul_high_i8x16_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @extmul_high_s_v8i16(<16 x i8> %v1, <16 x i8> %v2) {
  %high1 = shufflevector <16 x i8> %v1, <16 x i8> undef,
           <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %high2 = shufflevector <16 x i8> %v2, <16 x i8> undef,
           <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %extended1 = sext <8 x i8> %high1 to <8 x i16>
  %extended2 = sext <8 x i8> %high2 to <8 x i16>
  %a = mul <8 x i16> %extended1, %extended2
  ret <8 x i16> %a
}

; CHECK-LABEL: extmul_low_u_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extmul_low_u_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i16x8.extmul_low_i8x16_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @extmul_low_u_v8i16(<16 x i8> %v1, <16 x i8> %v2) {
  %low1 = shufflevector <16 x i8> %v1, <16 x i8> undef,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %low2 = shufflevector <16 x i8> %v2, <16 x i8> undef,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %extended1 = zext <8 x i8> %low1 to <8 x i16>
  %extended2 = zext <8 x i8> %low2 to <8 x i16>
  %a = mul <8 x i16> %extended1, %extended2
  ret <8 x i16> %a
}

; CHECK-LABEL: extmul_high_u_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype extmul_high_u_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i16x8.extmul_high_i8x16_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @extmul_high_u_v8i16(<16 x i8> %v1, <16 x i8> %v2) {
  %high1 = shufflevector <16 x i8> %v1, <16 x i8> undef,
           <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %high2 = shufflevector <16 x i8> %v2, <16 x i8> undef,
           <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %extended1 = zext <8 x i8> %high1 to <8 x i16>
  %extended2 = zext <8 x i8> %high2 to <8 x i16>
  %a = mul <8 x i16> %extended1, %extended2
  ret <8 x i16> %a
}

; ==============================================================================
; 4 x i32
; ==============================================================================
; CHECK-LABEL: add_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype add_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.add $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @add_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %a = add <4 x i32> %x, %y
  ret <4 x i32> %a
}

; CHECK-LABEL: sub_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype sub_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.sub $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @sub_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %a = sub <4 x i32> %x, %y
  ret <4 x i32> %a
}

; CHECK-LABEL: mul_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype mul_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.mul $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @mul_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %a = mul <4 x i32> %x, %y
  ret <4 x i32> %a
}

; CHECK-LABEL: min_s_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype min_s_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.min_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @min_s_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %c = icmp slt <4 x i32> %x, %y
  %a = select <4 x i1> %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %a
}

; CHECK-LABEL: min_u_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype min_u_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.min_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @min_u_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %c = icmp ult <4 x i32> %x, %y
  %a = select <4 x i1> %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %a
}

; CHECK-LABEL: max_s_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype max_s_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.max_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @max_s_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %c = icmp sgt <4 x i32> %x, %y
  %a = select <4 x i1> %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %a
}

; CHECK-LABEL: max_u_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype max_u_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.max_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @max_u_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %c = icmp ugt <4 x i32> %x, %y
  %a = select <4 x i1> %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %a
}

; CHECK-LABEL: abs_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype abs_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.abs $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @abs_v4i32(<4 x i32> %x) {
  %a = sub <4 x i32> zeroinitializer, %x
  %b = icmp slt <4 x i32> %x, zeroinitializer
  %c = select <4 x i1> %b, <4 x i32> %a, <4 x i32> %x
  ret <4 x i32> %c
}

; CHECK-LABEL: neg_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype neg_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.neg $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @neg_v4i32(<4 x i32> %x) {
  %a = sub <4 x i32> <i32 0, i32 0, i32 0, i32 0>, %x
  ret <4 x i32> %a
}

; CHECK-LABEL: shl_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype shl_v4i32 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.shl $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @shl_v4i32(<4 x i32> %v, i32 %x) {
  %t = insertelement <4 x i32> undef, i32 %x, i32 0
  %s = shufflevector <4 x i32> %t, <4 x i32> undef,
    <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %a = shl <4 x i32> %v, %s
  ret <4 x i32> %a
}

; CHECK-LABEL: shl_const_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype shl_const_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 5
; SIMD128-NEXT: i32x4.shl $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @shl_const_v4i32(<4 x i32> %v) {
  %a = shl <4 x i32> %v, <i32 5, i32 5, i32 5, i32 5>
  ret <4 x i32> %a
}

; CHECK-LABEL: shl_vec_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype shl_vec_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i32.shl $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: i32x4.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; Skip 2 lanes
; SIMD128:      i32x4.extract_lane $push[[L4:[0-9]+]]=, $0, 3{{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[L5:[0-9]+]]=, $1, 3{{$}}
; SIMD128-NEXT: i32.shl $push[[L6:[0-9]+]]=, $pop[[L4]], $pop[[L5]]{{$}}
; SIMD128-NEXT: i32x4.replace_lane $push[[R:[0-9]+]]=, $pop[[L7:[0-9]+]], 3, $pop[[L6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @shl_vec_v4i32(<4 x i32> %v, <4 x i32> %x) {
  %a = shl <4 x i32> %v, %x
  ret <4 x i32> %a
}

; CHECK-LABEL: shr_s_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype shr_s_v4i32 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.shr_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @shr_s_v4i32(<4 x i32> %v, i32 %x) {
  %t = insertelement <4 x i32> undef, i32 %x, i32 0
  %s = shufflevector <4 x i32> %t, <4 x i32> undef,
    <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %a = ashr <4 x i32> %v, %s
  ret <4 x i32> %a
}

; CHECK-LABEL: shr_s_vec_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype shr_s_vec_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i32.shr_s $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: i32x4.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; Skip 2 lanes
; SIMD128:      i32x4.extract_lane $push[[L4:[0-9]+]]=, $0, 3{{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[L5:[0-9]+]]=, $1, 3{{$}}
; SIMD128-NEXT: i32.shr_s $push[[L6:[0-9]+]]=, $pop[[L4]], $pop[[L5]]{{$}}
; SIMD128-NEXT: i32x4.replace_lane $push[[R:[0-9]+]]=, $pop[[L7:[0-9]+]], 3, $pop[[L6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @shr_s_vec_v4i32(<4 x i32> %v, <4 x i32> %x) {
  %a = ashr <4 x i32> %v, %x
  ret <4 x i32> %a
}

; CHECK-LABEL: shr_u_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype shr_u_v4i32 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i32x4.shr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @shr_u_v4i32(<4 x i32> %v, i32 %x) {
  %t = insertelement <4 x i32> undef, i32 %x, i32 0
  %s = shufflevector <4 x i32> %t, <4 x i32> undef,
    <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %a = lshr <4 x i32> %v, %s
  ret <4 x i32> %a
}

; CHECK-LABEL: shr_u_vec_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype shr_u_vec_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i32.shr_u $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: i32x4.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; Skip 2 lanes
; SIMD128:      i32x4.extract_lane $push[[L4:[0-9]+]]=, $0, 3{{$}}
; SIMD128-NEXT: i32x4.extract_lane $push[[L5:[0-9]+]]=, $1, 3{{$}}
; SIMD128-NEXT: i32.shr_u $push[[L6:[0-9]+]]=, $pop[[L4]], $pop[[L5]]{{$}}
; SIMD128-NEXT: i32x4.replace_lane $push[[R:[0-9]+]]=, $pop[[L7:[0-9]+]], 3, $pop[[L6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @shr_u_vec_v4i32(<4 x i32> %v, <4 x i32> %x) {
  %a = lshr <4 x i32> %v, %x
  ret <4 x i32> %a
}

; CHECK-LABEL: and_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype and_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @and_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %a = and <4 x i32> %x, %y
  ret <4 x i32> %a
}

; CHECK-LABEL: or_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype or_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @or_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %a = or <4 x i32> %x, %y
  ret <4 x i32> %a
}

; CHECK-LABEL: xor_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype xor_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.xor $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @xor_v4i32(<4 x i32> %x, <4 x i32> %y) {
  %a = xor <4 x i32> %x, %y
  ret <4 x i32> %a
}

; CHECK-LABEL: not_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype not_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @not_v4i32(<4 x i32> %x) {
  %a = xor <4 x i32> %x, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %a
}

; CHECK-LABEL: andnot_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype andnot_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: v128.andnot $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
; SIMD128-FAST-NEXT: v128.not
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: return
define <4 x i32> @andnot_v4i32(<4 x i32> %x, <4 x i32> %y) {
 %inv_y = xor <4 x i32> %y, <i32 -1, i32 -1, i32 -1, i32 -1>
 %a = and <4 x i32> %x, %inv_y
 ret <4 x i32> %a
}

; CHECK-LABEL: bitselect_v4i32:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype bitselect_v4i32 (v128, v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
; SIMD128-FAST-NEXT: v128.not
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: v128.or
; SIMD128-FAST-NEXT: return
define <4 x i32> @bitselect_v4i32(<4 x i32> %c, <4 x i32> %v1, <4 x i32> %v2) {
  %masked_v1 = and <4 x i32> %c, %v1
  %inv_mask = xor <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, %c
  %masked_v2 = and <4 x i32> %inv_mask, %v2
  %a = or <4 x i32> %masked_v2, %masked_v1
  ret <4 x i32> %a
}

; CHECK-LABEL: extmul_low_s_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype extmul_low_s_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i32x4.extmul_low_i16x8_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @extmul_low_s_v4i32(<8 x i16> %v1, <8 x i16> %v2) {
  %low1 = shufflevector <8 x i16> %v1, <8 x i16> undef,
           <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %low2 = shufflevector <8 x i16> %v2, <8 x i16> undef,
           <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %extended1 = sext <4 x i16> %low1 to <4 x i32>
  %extended2 = sext <4 x i16> %low2 to <4 x i32>
  %a = mul <4 x i32> %extended1, %extended2
  ret <4 x i32> %a
}

; CHECK-LABEL: extmul_high_s_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype extmul_high_s_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i32x4.extmul_high_i16x8_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @extmul_high_s_v4i32(<8 x i16> %v1, <8 x i16> %v2) {
  %high1 = shufflevector <8 x i16> %v1, <8 x i16> undef,
           <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %high2 = shufflevector <8 x i16> %v2, <8 x i16> undef,
           <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %extended1 = sext <4 x i16> %high1 to <4 x i32>
  %extended2 = sext <4 x i16> %high2 to <4 x i32>
  %a = mul <4 x i32> %extended1, %extended2
  ret <4 x i32> %a
}

; CHECK-LABEL: extmul_low_u_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype extmul_low_u_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i32x4.extmul_low_i16x8_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @extmul_low_u_v4i32(<8 x i16> %v1, <8 x i16> %v2) {
  %low1 = shufflevector <8 x i16> %v1, <8 x i16> undef,
           <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %low2 = shufflevector <8 x i16> %v2, <8 x i16> undef,
           <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %extended1 = zext <4 x i16> %low1 to <4 x i32>
  %extended2 = zext <4 x i16> %low2 to <4 x i32>
  %a = mul <4 x i32> %extended1, %extended2
  ret <4 x i32> %a
}

; CHECK-LABEL: extmul_high_u_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype extmul_high_u_v4i32 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i32x4.extmul_high_i16x8_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @extmul_high_u_v4i32(<8 x i16> %v1, <8 x i16> %v2) {
  %high1 = shufflevector <8 x i16> %v1, <8 x i16> undef,
           <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %high2 = shufflevector <8 x i16> %v2, <8 x i16> undef,
           <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %extended1 = zext <4 x i16> %high1 to <4 x i32>
  %extended2 = zext <4 x i16> %high2 to <4 x i32>
  %a = mul <4 x i32> %extended1, %extended2
  ret <4 x i32> %a
}

; ==============================================================================
; 2 x i64
; ==============================================================================
; CHECK-LABEL: add_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype add_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i64x2.add $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @add_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %a = add <2 x i64> %x, %y
  ret <2 x i64> %a
}

; CHECK-LABEL: sub_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype sub_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i64x2.sub $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @sub_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %a = sub <2 x i64> %x, %y
  ret <2 x i64> %a
}

; CHECK-LABEL: mul_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype mul_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128: i64x2.mul $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @mul_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %a = mul <2 x i64> %x, %y
  ret <2 x i64> %a
}

; CHECK-LABEL: abs_v2i64:
; NO-SIMD128-NOT: i64x2:
; SIMD128-NEXT: .functype abs_v2i64 (v128) -> (v128){{$}}
; SIMD128-NEXT: i64x2.abs $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @abs_v2i64(<2 x i64> %x) {
  %a = sub <2 x i64> zeroinitializer, %x
  %b = icmp slt <2 x i64> %x, zeroinitializer
  %c = select <2 x i1> %b, <2 x i64> %a, <2 x i64> %x
  ret <2 x i64> %c
}

; CHECK-LABEL: neg_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype neg_v2i64 (v128) -> (v128){{$}}
; SIMD128-NEXT: i64x2.neg $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @neg_v2i64(<2 x i64> %x) {
  %a = sub <2 x i64> <i64 0, i64 0>, %x
  ret <2 x i64> %a
}

; CHECK-LABEL: shl_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shl_v2i64 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.shl $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shl_v2i64(<2 x i64> %v, i32 %x) {
  %x2 = zext i32 %x to i64
  %t = insertelement <2 x i64> undef, i64 %x2, i32 0
  %s = shufflevector <2 x i64> %t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %a = shl <2 x i64> %v, %s
  ret <2 x i64> %a
}

; CHECK-LABEL: shl_sext_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shl_sext_v2i64 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.shl $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shl_sext_v2i64(<2 x i64> %v, i32 %x) {
  %x2 = sext i32 %x to i64
  %t = insertelement <2 x i64> undef, i64 %x2, i32 0
  %s = shufflevector <2 x i64> %t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %a = shl <2 x i64> %v, %s
  ret <2 x i64> %a
}

; CHECK-LABEL: shl_noext_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shl_noext_v2i64 (v128, i64) -> (v128){{$}}
; SIMD128-NEXT: i32.wrap_i64 $push[[L0:[0-9]+]]=, $1{{$}}
; SIMD128-NEXT: i64x2.shl $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shl_noext_v2i64(<2 x i64> %v, i64 %x) {
  %t = insertelement <2 x i64> undef, i64 %x, i32 0
  %s = shufflevector <2 x i64> %t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %a = shl <2 x i64> %v, %s
  ret <2 x i64> %a
}

; CHECK-LABEL: shl_const_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shl_const_v2i64 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 5{{$}}
; SIMD128-NEXT: i64x2.shl $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shl_const_v2i64(<2 x i64> %v) {
  %a = shl <2 x i64> %v, <i64 5, i64 5>
  ret <2 x i64> %a
}

; CHECK-LABEL: shl_vec_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shl_vec_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i64.shl $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: i64x2.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L4:[0-9]+]]=, $0, 1{{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L5:[0-9]+]]=, $1, 1{{$}}
; SIMD128-NEXT: i64.shl $push[[L6:[0-9]+]]=, $pop[[L4]], $pop[[L5]]{{$}}
; SIMD128-NEXT: i64x2.replace_lane $push[[R:[0-9]+]]=, $pop[[L3]], 1, $pop[[L6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shl_vec_v2i64(<2 x i64> %v, <2 x i64> %x) {
  %a = shl <2 x i64> %v, %x
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_s_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_s_v2i64 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.shr_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_s_v2i64(<2 x i64> %v, i32 %x) {
  %x2 = zext i32 %x to i64
  %t = insertelement <2 x i64> undef, i64 %x2, i32 0
  %s = shufflevector <2 x i64> %t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %a = ashr <2 x i64> %v, %s
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_s_sext_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_s_sext_v2i64 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.shr_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_s_sext_v2i64(<2 x i64> %v, i32 %x) {
  %x2 = sext i32 %x to i64
  %t = insertelement <2 x i64> undef, i64 %x2, i32 0
  %s = shufflevector <2 x i64> %t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %a = ashr <2 x i64> %v, %s
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_s_noext_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_s_noext_v2i64 (v128, i64) -> (v128){{$}}
; SIMD128-NEXT: i32.wrap_i64 $push[[L0:[0-9]+]]=, $1{{$}}
; SIMD128-NEXT: i64x2.shr_s $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_s_noext_v2i64(<2 x i64> %v, i64 %x) {
  %t = insertelement <2 x i64> undef, i64 %x, i32 0
  %s = shufflevector <2 x i64> %t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %a = ashr <2 x i64> %v, %s
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_s_const_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_s_const_v2i64 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 5{{$}}
; SIMD128-NEXT: i64x2.shr_s $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_s_const_v2i64(<2 x i64> %v) {
  %a = ashr <2 x i64> %v, <i64 5, i64 5>
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_s_vec_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_s_vec_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i64.shr_s $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: i64x2.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L4:[0-9]+]]=, $0, 1{{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L5:[0-9]+]]=, $1, 1{{$}}
; SIMD128-NEXT: i64.shr_s $push[[L6:[0-9]+]]=, $pop[[L4]], $pop[[L5]]{{$}}
; SIMD128-NEXT: i64x2.replace_lane $push[[R:[0-9]+]]=, $pop[[L3]], 1, $pop[[L6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_s_vec_v2i64(<2 x i64> %v, <2 x i64> %x) {
  %a = ashr <2 x i64> %v, %x
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_u_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_u_v2i64 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.shr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_u_v2i64(<2 x i64> %v, i32 %x) {
  %x2 = zext i32 %x to i64
  %t = insertelement <2 x i64> undef, i64 %x2, i32 0
  %s = shufflevector <2 x i64> %t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %a = lshr <2 x i64> %v, %s
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_u_sext_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_u_sext_v2i64 (v128, i32) -> (v128){{$}}
; SIMD128-NEXT: i64x2.shr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_u_sext_v2i64(<2 x i64> %v, i32 %x) {
  %x2 = sext i32 %x to i64
  %t = insertelement <2 x i64> undef, i64 %x2, i32 0
  %s = shufflevector <2 x i64> %t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %a = lshr <2 x i64> %v, %s
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_u_noext_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_u_noext_v2i64 (v128, i64) -> (v128){{$}}
; SIMD128-NEXT: i32.wrap_i64 $push[[L0:[0-9]+]]=, $1{{$}}
; SIMD128-NEXT: i64x2.shr_u $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_u_noext_v2i64(<2 x i64> %v, i64 %x) {
  %t = insertelement <2 x i64> undef, i64 %x, i32 0
  %s = shufflevector <2 x i64> %t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %a = lshr <2 x i64> %v, %s
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_u_const_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_u_const_v2i64 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[L0:[0-9]+]]=, 5{{$}}
; SIMD128-NEXT: i64x2.shr_u $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_u_const_v2i64(<2 x i64> %v) {
  %a = lshr <2 x i64> %v, <i64 5, i64 5>
  ret <2 x i64> %a
}

; CHECK-LABEL: shr_u_vec_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype shr_u_vec_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L0:[0-9]+]]=, $0, 0{{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L1:[0-9]+]]=, $1, 0{{$}}
; SIMD128-NEXT: i64.shr_u $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; SIMD128-NEXT: i64x2.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L4:[0-9]+]]=, $0, 1{{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[L5:[0-9]+]]=, $1, 1{{$}}
; SIMD128-NEXT: i64.shr_u $push[[L6:[0-9]+]]=, $pop[[L4]], $pop[[L5]]{{$}}
; SIMD128-NEXT: i64x2.replace_lane $push[[R:[0-9]+]]=, $pop[[L3]], 1, $pop[[L6]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @shr_u_vec_v2i64(<2 x i64> %v, <2 x i64> %x) {
  %a = lshr <2 x i64> %v, %x
  ret <2 x i64> %a
}

; CHECK-LABEL: and_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype and_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.and $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @and_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %a = and <2 x i64> %x, %y
  ret <2 x i64> %a
}

; CHECK-LABEL: or_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype or_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.or $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @or_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %a = or <2 x i64> %x, %y
  ret <2 x i64> %a
}

; CHECK-LABEL: xor_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype xor_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.xor $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @xor_v2i64(<2 x i64> %x, <2 x i64> %y) {
  %a = xor <2 x i64> %x, %y
  ret <2 x i64> %a
}

; CHECK-LABEL: not_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype not_v2i64 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.not $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @not_v2i64(<2 x i64> %x) {
  %a = xor <2 x i64> %x, <i64 -1, i64 -1>
  ret <2 x i64> %a
}

; CHECK-LABEL: andnot_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype andnot_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: v128.andnot $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
; SIMD128-FAST-NEXT: v128.not
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: return
define <2 x i64> @andnot_v2i64(<2 x i64> %x, <2 x i64> %y) {
 %inv_y = xor <2 x i64> %y, <i64 -1, i64 -1>
 %a = and <2 x i64> %x, %inv_y
 ret <2 x i64> %a
}

; CHECK-LABEL: bitselect_v2i64:
; NO-SIMD128-NOT: v128
; SIMD128-NEXT: .functype bitselect_v2i64 (v128, v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $0{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
; SIMD128-FAST-NEXT: v128.not
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: v128.and
; SIMD128-FAST-NEXT: v128.or
; SIMD128-FAST-NEXT: return
define <2 x i64> @bitselect_v2i64(<2 x i64> %c, <2 x i64> %v1, <2 x i64> %v2) {
  %masked_v1 = and <2 x i64> %v1, %c
  %inv_mask = xor <2 x i64> <i64 -1, i64 -1>, %c
  %masked_v2 = and <2 x i64> %v2, %inv_mask
  %a = or <2 x i64> %masked_v2, %masked_v1
  ret <2 x i64> %a
}

; CHECK-LABEL: extmul_low_s_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype extmul_low_s_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i64x2.extmul_low_i32x4_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @extmul_low_s_v2i64(<4 x i32> %v1, <4 x i32> %v2) {
  %low1 = shufflevector <4 x i32> %v1, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %low2 = shufflevector <4 x i32> %v2, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %extended1 = sext <2 x i32> %low1 to <2 x i64>
  %extended2 = sext <2 x i32> %low2 to <2 x i64>
  %a = mul <2 x i64> %extended1, %extended2
  ret <2 x i64> %a
}

; CHECK-LABEL: extmul_high_s_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype extmul_high_s_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i64x2.extmul_high_i32x4_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @extmul_high_s_v2i64(<4 x i32> %v1, <4 x i32> %v2) {
  %high1 = shufflevector <4 x i32> %v1, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %high2 = shufflevector <4 x i32> %v2, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %extended1 = sext <2 x i32> %high1 to <2 x i64>
  %extended2 = sext <2 x i32> %high2 to <2 x i64>
  %a = mul <2 x i64> %extended1, %extended2
  ret <2 x i64> %a
}

; CHECK-LABEL: extmul_low_u_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype extmul_low_u_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i64x2.extmul_low_i32x4_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @extmul_low_u_v2i64(<4 x i32> %v1, <4 x i32> %v2) {
  %low1 = shufflevector <4 x i32> %v1, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %low2 = shufflevector <4 x i32> %v2, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %extended1 = zext <2 x i32> %low1 to <2 x i64>
  %extended2 = zext <2 x i32> %low2 to <2 x i64>
  %a = mul <2 x i64> %extended1, %extended2
  ret <2 x i64> %a
}

; CHECK-LABEL: extmul_high_u_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype extmul_high_u_v2i64 (v128, v128) -> (v128){{$}}
; SIMD128-SLOW-NEXT: i64x2.extmul_high_i32x4_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-SLOW-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @extmul_high_u_v2i64(<4 x i32> %v1, <4 x i32> %v2) {
  %high1 = shufflevector <4 x i32> %v1, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %high2 = shufflevector <4 x i32> %v2, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %extended1 = zext <2 x i32> %high1 to <2 x i64>
  %extended2 = zext <2 x i32> %high2 to <2 x i64>
  %a = mul <2 x i64> %extended1, %extended2
  ret <2 x i64> %a
}

; ==============================================================================
; 4 x float
; ==============================================================================
; CHECK-LABEL: neg_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype neg_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.neg $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @neg_v4f32(<4 x float> %x) {
  ; nsz makes this semantically equivalent to flipping sign bit
  %a = fsub nsz <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>, %x
  ret <4 x float> %a
}

; CHECK-LABEL: abs_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype abs_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.abs $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.fabs.v4f32(<4 x float>) nounwind readnone
define <4 x float> @abs_v4f32(<4 x float> %x) {
  %a = call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  ret <4 x float> %a
}

; CHECK-LABEL: min_unordered_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype min_unordered_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[L0:[0-9]+]]=, 0x1.4p2, 0x1.4p2, 0x1.4p2, 0x1.4p2{{$}}
; SIMD128-NEXT: f32x4.min $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @min_unordered_v4f32(<4 x float> %x) {
  %cmps = fcmp ule <4 x float> %x, <float 5., float 5., float 5., float 5.>
  %a = select <4 x i1> %cmps, <4 x float> %x,
    <4 x float> <float 5., float 5., float 5., float 5.>
  ret <4 x float> %a
}

; CHECK-LABEL: max_unordered_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype max_unordered_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[L0:[0-9]+]]=, 0x1.4p2, 0x1.4p2, 0x1.4p2, 0x1.4p2
; SIMD128-NEXT: f32x4.max $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @max_unordered_v4f32(<4 x float> %x) {
  %cmps = fcmp uge <4 x float> %x, <float 5., float 5., float 5., float 5.>
  %a = select <4 x i1> %cmps, <4 x float> %x,
    <4 x float> <float 5., float 5., float 5., float 5.>
  ret <4 x float> %a
}

; CHECK-LABEL: min_ordered_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype min_ordered_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[L0:[0-9]+]]=, 0x1.4p2, 0x1.4p2, 0x1.4p2, 0x1.4p2{{$}}
; SIMD128-NEXT: f32x4.min $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @min_ordered_v4f32(<4 x float> %x) {
  %cmps = fcmp ole <4 x float> <float 5., float 5., float 5., float 5.>, %x
  %a = select <4 x i1> %cmps,
    <4 x float> <float 5., float 5., float 5., float 5.>, <4 x float> %x
  ret <4 x float> %a
}

; CHECK-LABEL: max_ordered_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype max_ordered_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[L0:[0-9]+]]=, 0x1.4p2, 0x1.4p2, 0x1.4p2, 0x1.4p2{{$}}
; SIMD128-NEXT: f32x4.max $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @max_ordered_v4f32(<4 x float> %x) {
  %cmps = fcmp oge <4 x float> <float 5., float 5., float 5., float 5.>, %x
  %a = select <4 x i1> %cmps,
    <4 x float> <float 5., float 5., float 5., float 5.>, <4 x float> %x
  ret <4 x float> %a
}

; CHECK-LABEL: min_intrinsic_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype min_intrinsic_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.min $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.minimum.v4f32(<4 x float>, <4 x float>)
define <4 x float> @min_intrinsic_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = call <4 x float> @llvm.minimum.v4f32(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %a
}

; CHECK-LABEL: minnum_intrinsic_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype minnum_intrinsic_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.min $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
define <4 x float> @minnum_intrinsic_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = call nnan <4 x float> @llvm.minnum.v4f32(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %a
}

; CHECK-LABEL: max_intrinsic_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype max_intrinsic_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.max $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.maximum.v4f32(<4 x float>, <4 x float>)
define <4 x float> @max_intrinsic_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = call <4 x float> @llvm.maximum.v4f32(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %a
}

; CHECK-LABEL: maxnum_intrinsic_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype maxnum_intrinsic_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.max $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
define <4 x float> @maxnum_intrinsic_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = call nnan <4 x float> @llvm.maxnum.v4f32(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %a
}

; CHECK-LABEL: min_const_intrinsic_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype min_const_intrinsic_v4f32 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=, 0x1.4p2, 0x1.4p2, 0x1.4p2, 0x1.4p2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @min_const_intrinsic_v4f32() {
  %a = call <4 x float> @llvm.minimum.v4f32(
    <4 x float> <float 42., float 42., float 42., float 42.>,
    <4 x float> <float 5., float 5., float 5., float 5.>
  )
  ret <4 x float> %a
}

; CHECK-LABEL: max_const_intrinsic_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype max_const_intrinsic_v4f32 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=, 0x1.5p5, 0x1.5p5, 0x1.5p5, 0x1.5p5{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @max_const_intrinsic_v4f32() {
  %a = call <4 x float> @llvm.maximum.v4f32(
    <4 x float> <float 42., float 42., float 42., float 42.>,
    <4 x float> <float 5., float 5., float 5., float 5.>
  )
  ret <4 x float> %a
}

; CHECK-LABEL: pmin_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype pmin_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.pmin $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @pmin_v4f32(<4 x float> %x, <4 x float> %y) {
  %c = fcmp olt <4 x float> %y, %x
  %a = select <4 x i1> %c, <4 x float> %y, <4 x float> %x
  ret <4 x float> %a
}

; CHECK-LABEL: pmin_int_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype pmin_int_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.pmin $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @pmin_int_v4f32(<4 x i32> %x, <4 x i32> %y) {
  %fx = bitcast <4 x i32> %x to <4 x float>
  %fy = bitcast <4 x i32> %y to <4 x float>
  %c = fcmp olt <4 x float> %fy, %fx
  %a = select <4 x i1> %c, <4 x i32> %y, <4 x i32> %x
  ret <4 x i32> %a
}

; CHECK-LABEL: pmax_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype pmax_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.pmax $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @pmax_v4f32(<4 x float> %x, <4 x float> %y) {
  %c = fcmp olt <4 x float> %x, %y
  %a = select <4 x i1> %c, <4 x float> %y, <4 x float> %x
  ret <4 x float> %a
}

; CHECK-LABEL: pmax_int_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype pmax_int_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.pmax $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @pmax_int_v4f32(<4 x i32> %x, <4 x i32> %y) {
  %fx = bitcast <4 x i32> %x to <4 x float>
  %fy = bitcast <4 x i32> %y to <4 x float>
  %c = fcmp olt <4 x float> %fx, %fy
  %a = select <4 x i1> %c, <4 x i32> %y, <4 x i32> %x
  ret <4 x i32> %a
}

; CHECK-LABEL: add_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype add_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.add $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @add_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = fadd <4 x float> %x, %y
  ret <4 x float> %a
}

; CHECK-LABEL: sub_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype sub_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.sub $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @sub_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = fsub <4 x float> %x, %y
  ret <4 x float> %a
}

; CHECK-LABEL: div_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype div_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.div $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @div_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = fdiv <4 x float> %x, %y
  ret <4 x float> %a
}

; CHECK-LABEL: mul_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype mul_v4f32 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.mul $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x float> @mul_v4f32(<4 x float> %x, <4 x float> %y) {
  %a = fmul <4 x float> %x, %y
  ret <4 x float> %a
}

; CHECK-LABEL: sqrt_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype sqrt_v4f32 (v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.sqrt $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
define <4 x float> @sqrt_v4f32(<4 x float> %x) {
  %a = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  ret <4 x float> %a
}

; ==============================================================================
; 2 x double
; ==============================================================================
; CHECK-LABEL: neg_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype neg_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.neg $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @neg_v2f64(<2 x double> %x) {
  ; nsz makes this semantically equivalent to flipping sign bit
  %a = fsub nsz <2 x double> <double 0., double 0.>, %x
  ret <2 x double> %a
}

; CHECK-LABEL: abs_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype abs_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.abs $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.fabs.v2f64(<2 x double>) nounwind readnone
define <2 x double> @abs_v2f64(<2 x double> %x) {
  %a = call <2 x double> @llvm.fabs.v2f64(<2 x double> %x)
  ret <2 x double> %a
}

; CHECK-LABEL: min_unordered_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype min_unordered_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[L0:[0-9]+]]=, 0x1.4p2, 0x1.4p2{{$}}
; SIMD128-NEXT: f64x2.min $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @min_unordered_v2f64(<2 x double> %x) {
  %cmps = fcmp ule <2 x double> %x, <double 5., double 5.>
  %a = select <2 x i1> %cmps, <2 x double> %x,
    <2 x double> <double 5., double 5.>
  ret <2 x double> %a
}

; CHECK-LABEL: max_unordered_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype max_unordered_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[L0:[0-9]+]]=, 0x1.4p2, 0x1.4p2{{$}}
; SIMD128-NEXT: f64x2.max $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @max_unordered_v2f64(<2 x double> %x) {
  %cmps = fcmp uge <2 x double> %x, <double 5., double 5.>
  %a = select <2 x i1> %cmps, <2 x double> %x,
    <2 x double> <double 5., double 5.>
  ret <2 x double> %a
}

; CHECK-LABEL: min_ordered_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype min_ordered_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[L0:[0-9]+]]=, 0x1.4p2, 0x1.4p2{{$}}
; SIMD128-NEXT: f64x2.min $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @min_ordered_v2f64(<2 x double> %x) {
  %cmps = fcmp ole <2 x double> <double 5., double 5.>, %x
  %a = select <2 x i1> %cmps, <2 x double> <double 5., double 5.>,
    <2 x double> %x
  ret <2 x double> %a
}

; CHECK-LABEL: max_ordered_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype max_ordered_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[L0:[0-9]+]]=, 0x1.4p2, 0x1.4p2{{$}}
; SIMD128-NEXT: f64x2.max $push[[R:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @max_ordered_v2f64(<2 x double> %x) {
  %cmps = fcmp oge <2 x double> <double 5., double 5.>, %x
  %a = select <2 x i1> %cmps, <2 x double> <double 5., double 5.>,
    <2 x double> %x
  ret <2 x double> %a
}

; CHECK-LABEL: min_intrinsic_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype min_intrinsic_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.min $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.minimum.v2f64(<2 x double>, <2 x double>)
define <2 x double> @min_intrinsic_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = call <2 x double> @llvm.minimum.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %a
}

; CHECK-LABEL: max_intrinsic_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype max_intrinsic_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.max $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.maximum.v2f64(<2 x double>, <2 x double>)
define <2 x double> @max_intrinsic_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = call <2 x double> @llvm.maximum.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %a
}

; CHECK-LABEL: min_const_intrinsic_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype min_const_intrinsic_v2f64 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=, 0x1.4p2, 0x1.4p2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @min_const_intrinsic_v2f64() {
  %a = call <2 x double> @llvm.minimum.v2f64(
    <2 x double> <double 42., double 42.>,
    <2 x double> <double 5., double 5.>
  )
  ret <2 x double> %a
}

; CHECK-LABEL: max_const_intrinsic_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype max_const_intrinsic_v2f64 () -> (v128){{$}}
; SIMD128-NEXT: v128.const $push[[R:[0-9]+]]=, 0x1.5p5, 0x1.5p5{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @max_const_intrinsic_v2f64() {
  %a = call <2 x double> @llvm.maximum.v2f64(
    <2 x double> <double 42., double 42.>,
    <2 x double> <double 5., double 5.>
  )
  ret <2 x double> %a
}

; CHECK-LABEL: pmin_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype pmin_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.pmin $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @pmin_v2f64(<2 x double> %x, <2 x double> %y) {
  %c = fcmp olt <2 x double> %y, %x
  %a = select <2 x i1> %c, <2 x double> %y, <2 x double> %x
  ret <2 x double> %a
}

; CHECK-LABEL: pmin_int_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype pmin_int_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.pmin $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @pmin_int_v2f64(<2 x i64> %x, <2 x i64> %y) {
  %fx = bitcast <2 x i64> %x to <2 x double>
  %fy = bitcast <2 x i64> %y to <2 x double>
  %c = fcmp olt <2 x double> %fy, %fx
  %a = select <2 x i1> %c, <2 x i64> %y, <2 x i64> %x
  ret <2 x i64> %a
}

; CHECK-LABEL: pmax_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype pmax_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.pmax $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @pmax_v2f64(<2 x double> %x, <2 x double> %y) {
  %c = fcmp olt <2 x double> %x, %y
  %a = select <2 x i1> %c, <2 x double> %y, <2 x double> %x
  ret <2 x double> %a
}

; CHECK-LABEL: pmax_int_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype pmax_int_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.pmax $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @pmax_int_v2f64(<2 x i64> %x, <2 x i64> %y) {
  %fx = bitcast <2 x i64> %x to <2 x double>
  %fy = bitcast <2 x i64> %y to <2 x double>
  %c = fcmp olt <2 x double> %fx, %fy
  %a = select <2 x i1> %c, <2 x i64> %y, <2 x i64> %x
  ret <2 x i64> %a
}

; CHECK-LABEL: add_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype add_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.add $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @add_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = fadd <2 x double> %x, %y
  ret <2 x double> %a
}

; CHECK-LABEL: sub_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype sub_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.sub $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @sub_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = fsub <2 x double> %x, %y
  ret <2 x double> %a
}

; CHECK-LABEL: div_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype div_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.div $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @div_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = fdiv <2 x double> %x, %y
  ret <2 x double> %a
}

; CHECK-LABEL: mul_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype mul_v2f64 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.mul $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x double> @mul_v2f64(<2 x double> %x, <2 x double> %y) {
  %a = fmul <2 x double> %x, %y
  ret <2 x double> %a
}

; CHECK-LABEL: sqrt_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-NEXT: .functype sqrt_v2f64 (v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.sqrt $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.sqrt.v2f64(<2 x double> %x)
define <2 x double> @sqrt_v2f64(<2 x double> %x) {
  %a = call <2 x double> @llvm.sqrt.v2f64(<2 x double> %x)
  ret <2 x double> %a
}

; RUN: llc -mtriple=arm64-apple-ios7.0 -o - %s | FileCheck %s

@ptr = global i8* null

define <8 x i8> @test_v8i8_pre_load(<8 x i8>* %addr) {
; CHECK-LABEL: test_v8i8_pre_load:
; CHECK: ldr d0, [x0, #40]!
  %newaddr = getelementptr <8 x i8>* %addr, i32 5
  %val = load <8 x i8>* %newaddr, align 8
  store <8 x i8>* %newaddr, <8 x i8>** bitcast(i8** @ptr to <8 x i8>**)
  ret <8 x i8> %val
}

define <8 x i8> @test_v8i8_post_load(<8 x i8>* %addr) {
; CHECK-LABEL: test_v8i8_post_load:
; CHECK: ldr d0, [x0], #40
  %newaddr = getelementptr <8 x i8>* %addr, i32 5
  %val = load <8 x i8>* %addr, align 8
  store <8 x i8>* %newaddr, <8 x i8>** bitcast(i8** @ptr to <8 x i8>**)
  ret <8 x i8> %val
}

define void @test_v8i8_pre_store(<8 x i8> %in, <8 x i8>* %addr) {
; CHECK-LABEL: test_v8i8_pre_store:
; CHECK: str d0, [x0, #40]!
  %newaddr = getelementptr <8 x i8>* %addr, i32 5
  store <8 x i8> %in, <8 x i8>* %newaddr, align 8
  store <8 x i8>* %newaddr, <8 x i8>** bitcast(i8** @ptr to <8 x i8>**)
  ret void
}

define void @test_v8i8_post_store(<8 x i8> %in, <8 x i8>* %addr) {
; CHECK-LABEL: test_v8i8_post_store:
; CHECK: str d0, [x0], #40
  %newaddr = getelementptr <8 x i8>* %addr, i32 5
  store <8 x i8> %in, <8 x i8>* %addr, align 8
  store <8 x i8>* %newaddr, <8 x i8>** bitcast(i8** @ptr to <8 x i8>**)
  ret void
}

define <4 x i16> @test_v4i16_pre_load(<4 x i16>* %addr) {
; CHECK-LABEL: test_v4i16_pre_load:
; CHECK: ldr d0, [x0, #40]!
  %newaddr = getelementptr <4 x i16>* %addr, i32 5
  %val = load <4 x i16>* %newaddr, align 8
  store <4 x i16>* %newaddr, <4 x i16>** bitcast(i8** @ptr to <4 x i16>**)
  ret <4 x i16> %val
}

define <4 x i16> @test_v4i16_post_load(<4 x i16>* %addr) {
; CHECK-LABEL: test_v4i16_post_load:
; CHECK: ldr d0, [x0], #40
  %newaddr = getelementptr <4 x i16>* %addr, i32 5
  %val = load <4 x i16>* %addr, align 8
  store <4 x i16>* %newaddr, <4 x i16>** bitcast(i8** @ptr to <4 x i16>**)
  ret <4 x i16> %val
}

define void @test_v4i16_pre_store(<4 x i16> %in, <4 x i16>* %addr) {
; CHECK-LABEL: test_v4i16_pre_store:
; CHECK: str d0, [x0, #40]!
  %newaddr = getelementptr <4 x i16>* %addr, i32 5
  store <4 x i16> %in, <4 x i16>* %newaddr, align 8
  store <4 x i16>* %newaddr, <4 x i16>** bitcast(i8** @ptr to <4 x i16>**)
  ret void
}

define void @test_v4i16_post_store(<4 x i16> %in, <4 x i16>* %addr) {
; CHECK-LABEL: test_v4i16_post_store:
; CHECK: str d0, [x0], #40
  %newaddr = getelementptr <4 x i16>* %addr, i32 5
  store <4 x i16> %in, <4 x i16>* %addr, align 8
  store <4 x i16>* %newaddr, <4 x i16>** bitcast(i8** @ptr to <4 x i16>**)
  ret void
}

define <2 x i32> @test_v2i32_pre_load(<2 x i32>* %addr) {
; CHECK-LABEL: test_v2i32_pre_load:
; CHECK: ldr d0, [x0, #40]!
  %newaddr = getelementptr <2 x i32>* %addr, i32 5
  %val = load <2 x i32>* %newaddr, align 8
  store <2 x i32>* %newaddr, <2 x i32>** bitcast(i8** @ptr to <2 x i32>**)
  ret <2 x i32> %val
}

define <2 x i32> @test_v2i32_post_load(<2 x i32>* %addr) {
; CHECK-LABEL: test_v2i32_post_load:
; CHECK: ldr d0, [x0], #40
  %newaddr = getelementptr <2 x i32>* %addr, i32 5
  %val = load <2 x i32>* %addr, align 8
  store <2 x i32>* %newaddr, <2 x i32>** bitcast(i8** @ptr to <2 x i32>**)
  ret <2 x i32> %val
}

define void @test_v2i32_pre_store(<2 x i32> %in, <2 x i32>* %addr) {
; CHECK-LABEL: test_v2i32_pre_store:
; CHECK: str d0, [x0, #40]!
  %newaddr = getelementptr <2 x i32>* %addr, i32 5
  store <2 x i32> %in, <2 x i32>* %newaddr, align 8
  store <2 x i32>* %newaddr, <2 x i32>** bitcast(i8** @ptr to <2 x i32>**)
  ret void
}

define void @test_v2i32_post_store(<2 x i32> %in, <2 x i32>* %addr) {
; CHECK-LABEL: test_v2i32_post_store:
; CHECK: str d0, [x0], #40
  %newaddr = getelementptr <2 x i32>* %addr, i32 5
  store <2 x i32> %in, <2 x i32>* %addr, align 8
  store <2 x i32>* %newaddr, <2 x i32>** bitcast(i8** @ptr to <2 x i32>**)
  ret void
}

define <2 x float> @test_v2f32_pre_load(<2 x float>* %addr) {
; CHECK-LABEL: test_v2f32_pre_load:
; CHECK: ldr d0, [x0, #40]!
  %newaddr = getelementptr <2 x float>* %addr, i32 5
  %val = load <2 x float>* %newaddr, align 8
  store <2 x float>* %newaddr, <2 x float>** bitcast(i8** @ptr to <2 x float>**)
  ret <2 x float> %val
}

define <2 x float> @test_v2f32_post_load(<2 x float>* %addr) {
; CHECK-LABEL: test_v2f32_post_load:
; CHECK: ldr d0, [x0], #40
  %newaddr = getelementptr <2 x float>* %addr, i32 5
  %val = load <2 x float>* %addr, align 8
  store <2 x float>* %newaddr, <2 x float>** bitcast(i8** @ptr to <2 x float>**)
  ret <2 x float> %val
}

define void @test_v2f32_pre_store(<2 x float> %in, <2 x float>* %addr) {
; CHECK-LABEL: test_v2f32_pre_store:
; CHECK: str d0, [x0, #40]!
  %newaddr = getelementptr <2 x float>* %addr, i32 5
  store <2 x float> %in, <2 x float>* %newaddr, align 8
  store <2 x float>* %newaddr, <2 x float>** bitcast(i8** @ptr to <2 x float>**)
  ret void
}

define void @test_v2f32_post_store(<2 x float> %in, <2 x float>* %addr) {
; CHECK-LABEL: test_v2f32_post_store:
; CHECK: str d0, [x0], #40
  %newaddr = getelementptr <2 x float>* %addr, i32 5
  store <2 x float> %in, <2 x float>* %addr, align 8
  store <2 x float>* %newaddr, <2 x float>** bitcast(i8** @ptr to <2 x float>**)
  ret void
}

define <1 x i64> @test_v1i64_pre_load(<1 x i64>* %addr) {
; CHECK-LABEL: test_v1i64_pre_load:
; CHECK: ldr d0, [x0, #40]!
  %newaddr = getelementptr <1 x i64>* %addr, i32 5
  %val = load <1 x i64>* %newaddr, align 8
  store <1 x i64>* %newaddr, <1 x i64>** bitcast(i8** @ptr to <1 x i64>**)
  ret <1 x i64> %val
}

define <1 x i64> @test_v1i64_post_load(<1 x i64>* %addr) {
; CHECK-LABEL: test_v1i64_post_load:
; CHECK: ldr d0, [x0], #40
  %newaddr = getelementptr <1 x i64>* %addr, i32 5
  %val = load <1 x i64>* %addr, align 8
  store <1 x i64>* %newaddr, <1 x i64>** bitcast(i8** @ptr to <1 x i64>**)
  ret <1 x i64> %val
}

define void @test_v1i64_pre_store(<1 x i64> %in, <1 x i64>* %addr) {
; CHECK-LABEL: test_v1i64_pre_store:
; CHECK: str d0, [x0, #40]!
  %newaddr = getelementptr <1 x i64>* %addr, i32 5
  store <1 x i64> %in, <1 x i64>* %newaddr, align 8
  store <1 x i64>* %newaddr, <1 x i64>** bitcast(i8** @ptr to <1 x i64>**)
  ret void
}

define void @test_v1i64_post_store(<1 x i64> %in, <1 x i64>* %addr) {
; CHECK-LABEL: test_v1i64_post_store:
; CHECK: str d0, [x0], #40
  %newaddr = getelementptr <1 x i64>* %addr, i32 5
  store <1 x i64> %in, <1 x i64>* %addr, align 8
  store <1 x i64>* %newaddr, <1 x i64>** bitcast(i8** @ptr to <1 x i64>**)
  ret void
}

define <16 x i8> @test_v16i8_pre_load(<16 x i8>* %addr) {
; CHECK-LABEL: test_v16i8_pre_load:
; CHECK: ldr q0, [x0, #80]!
  %newaddr = getelementptr <16 x i8>* %addr, i32 5
  %val = load <16 x i8>* %newaddr, align 8
  store <16 x i8>* %newaddr, <16 x i8>** bitcast(i8** @ptr to <16 x i8>**)
  ret <16 x i8> %val
}

define <16 x i8> @test_v16i8_post_load(<16 x i8>* %addr) {
; CHECK-LABEL: test_v16i8_post_load:
; CHECK: ldr q0, [x0], #80
  %newaddr = getelementptr <16 x i8>* %addr, i32 5
  %val = load <16 x i8>* %addr, align 8
  store <16 x i8>* %newaddr, <16 x i8>** bitcast(i8** @ptr to <16 x i8>**)
  ret <16 x i8> %val
}

define void @test_v16i8_pre_store(<16 x i8> %in, <16 x i8>* %addr) {
; CHECK-LABEL: test_v16i8_pre_store:
; CHECK: str q0, [x0, #80]!
  %newaddr = getelementptr <16 x i8>* %addr, i32 5
  store <16 x i8> %in, <16 x i8>* %newaddr, align 8
  store <16 x i8>* %newaddr, <16 x i8>** bitcast(i8** @ptr to <16 x i8>**)
  ret void
}

define void @test_v16i8_post_store(<16 x i8> %in, <16 x i8>* %addr) {
; CHECK-LABEL: test_v16i8_post_store:
; CHECK: str q0, [x0], #80
  %newaddr = getelementptr <16 x i8>* %addr, i32 5
  store <16 x i8> %in, <16 x i8>* %addr, align 8
  store <16 x i8>* %newaddr, <16 x i8>** bitcast(i8** @ptr to <16 x i8>**)
  ret void
}

define <8 x i16> @test_v8i16_pre_load(<8 x i16>* %addr) {
; CHECK-LABEL: test_v8i16_pre_load:
; CHECK: ldr q0, [x0, #80]!
  %newaddr = getelementptr <8 x i16>* %addr, i32 5
  %val = load <8 x i16>* %newaddr, align 8
  store <8 x i16>* %newaddr, <8 x i16>** bitcast(i8** @ptr to <8 x i16>**)
  ret <8 x i16> %val
}

define <8 x i16> @test_v8i16_post_load(<8 x i16>* %addr) {
; CHECK-LABEL: test_v8i16_post_load:
; CHECK: ldr q0, [x0], #80
  %newaddr = getelementptr <8 x i16>* %addr, i32 5
  %val = load <8 x i16>* %addr, align 8
  store <8 x i16>* %newaddr, <8 x i16>** bitcast(i8** @ptr to <8 x i16>**)
  ret <8 x i16> %val
}

define void @test_v8i16_pre_store(<8 x i16> %in, <8 x i16>* %addr) {
; CHECK-LABEL: test_v8i16_pre_store:
; CHECK: str q0, [x0, #80]!
  %newaddr = getelementptr <8 x i16>* %addr, i32 5
  store <8 x i16> %in, <8 x i16>* %newaddr, align 8
  store <8 x i16>* %newaddr, <8 x i16>** bitcast(i8** @ptr to <8 x i16>**)
  ret void
}

define void @test_v8i16_post_store(<8 x i16> %in, <8 x i16>* %addr) {
; CHECK-LABEL: test_v8i16_post_store:
; CHECK: str q0, [x0], #80
  %newaddr = getelementptr <8 x i16>* %addr, i32 5
  store <8 x i16> %in, <8 x i16>* %addr, align 8
  store <8 x i16>* %newaddr, <8 x i16>** bitcast(i8** @ptr to <8 x i16>**)
  ret void
}

define <4 x i32> @test_v4i32_pre_load(<4 x i32>* %addr) {
; CHECK-LABEL: test_v4i32_pre_load:
; CHECK: ldr q0, [x0, #80]!
  %newaddr = getelementptr <4 x i32>* %addr, i32 5
  %val = load <4 x i32>* %newaddr, align 8
  store <4 x i32>* %newaddr, <4 x i32>** bitcast(i8** @ptr to <4 x i32>**)
  ret <4 x i32> %val
}

define <4 x i32> @test_v4i32_post_load(<4 x i32>* %addr) {
; CHECK-LABEL: test_v4i32_post_load:
; CHECK: ldr q0, [x0], #80
  %newaddr = getelementptr <4 x i32>* %addr, i32 5
  %val = load <4 x i32>* %addr, align 8
  store <4 x i32>* %newaddr, <4 x i32>** bitcast(i8** @ptr to <4 x i32>**)
  ret <4 x i32> %val
}

define void @test_v4i32_pre_store(<4 x i32> %in, <4 x i32>* %addr) {
; CHECK-LABEL: test_v4i32_pre_store:
; CHECK: str q0, [x0, #80]!
  %newaddr = getelementptr <4 x i32>* %addr, i32 5
  store <4 x i32> %in, <4 x i32>* %newaddr, align 8
  store <4 x i32>* %newaddr, <4 x i32>** bitcast(i8** @ptr to <4 x i32>**)
  ret void
}

define void @test_v4i32_post_store(<4 x i32> %in, <4 x i32>* %addr) {
; CHECK-LABEL: test_v4i32_post_store:
; CHECK: str q0, [x0], #80
  %newaddr = getelementptr <4 x i32>* %addr, i32 5
  store <4 x i32> %in, <4 x i32>* %addr, align 8
  store <4 x i32>* %newaddr, <4 x i32>** bitcast(i8** @ptr to <4 x i32>**)
  ret void
}


define <4 x float> @test_v4f32_pre_load(<4 x float>* %addr) {
; CHECK-LABEL: test_v4f32_pre_load:
; CHECK: ldr q0, [x0, #80]!
  %newaddr = getelementptr <4 x float>* %addr, i32 5
  %val = load <4 x float>* %newaddr, align 8
  store <4 x float>* %newaddr, <4 x float>** bitcast(i8** @ptr to <4 x float>**)
  ret <4 x float> %val
}

define <4 x float> @test_v4f32_post_load(<4 x float>* %addr) {
; CHECK-LABEL: test_v4f32_post_load:
; CHECK: ldr q0, [x0], #80
  %newaddr = getelementptr <4 x float>* %addr, i32 5
  %val = load <4 x float>* %addr, align 8
  store <4 x float>* %newaddr, <4 x float>** bitcast(i8** @ptr to <4 x float>**)
  ret <4 x float> %val
}

define void @test_v4f32_pre_store(<4 x float> %in, <4 x float>* %addr) {
; CHECK-LABEL: test_v4f32_pre_store:
; CHECK: str q0, [x0, #80]!
  %newaddr = getelementptr <4 x float>* %addr, i32 5
  store <4 x float> %in, <4 x float>* %newaddr, align 8
  store <4 x float>* %newaddr, <4 x float>** bitcast(i8** @ptr to <4 x float>**)
  ret void
}

define void @test_v4f32_post_store(<4 x float> %in, <4 x float>* %addr) {
; CHECK-LABEL: test_v4f32_post_store:
; CHECK: str q0, [x0], #80
  %newaddr = getelementptr <4 x float>* %addr, i32 5
  store <4 x float> %in, <4 x float>* %addr, align 8
  store <4 x float>* %newaddr, <4 x float>** bitcast(i8** @ptr to <4 x float>**)
  ret void
}


define <2 x i64> @test_v2i64_pre_load(<2 x i64>* %addr) {
; CHECK-LABEL: test_v2i64_pre_load:
; CHECK: ldr q0, [x0, #80]!
  %newaddr = getelementptr <2 x i64>* %addr, i32 5
  %val = load <2 x i64>* %newaddr, align 8
  store <2 x i64>* %newaddr, <2 x i64>** bitcast(i8** @ptr to <2 x i64>**)
  ret <2 x i64> %val
}

define <2 x i64> @test_v2i64_post_load(<2 x i64>* %addr) {
; CHECK-LABEL: test_v2i64_post_load:
; CHECK: ldr q0, [x0], #80
  %newaddr = getelementptr <2 x i64>* %addr, i32 5
  %val = load <2 x i64>* %addr, align 8
  store <2 x i64>* %newaddr, <2 x i64>** bitcast(i8** @ptr to <2 x i64>**)
  ret <2 x i64> %val
}

define void @test_v2i64_pre_store(<2 x i64> %in, <2 x i64>* %addr) {
; CHECK-LABEL: test_v2i64_pre_store:
; CHECK: str q0, [x0, #80]!
  %newaddr = getelementptr <2 x i64>* %addr, i32 5
  store <2 x i64> %in, <2 x i64>* %newaddr, align 8
  store <2 x i64>* %newaddr, <2 x i64>** bitcast(i8** @ptr to <2 x i64>**)
  ret void
}

define void @test_v2i64_post_store(<2 x i64> %in, <2 x i64>* %addr) {
; CHECK-LABEL: test_v2i64_post_store:
; CHECK: str q0, [x0], #80
  %newaddr = getelementptr <2 x i64>* %addr, i32 5
  store <2 x i64> %in, <2 x i64>* %addr, align 8
  store <2 x i64>* %newaddr, <2 x i64>** bitcast(i8** @ptr to <2 x i64>**)
  ret void
}


define <2 x double> @test_v2f64_pre_load(<2 x double>* %addr) {
; CHECK-LABEL: test_v2f64_pre_load:
; CHECK: ldr q0, [x0, #80]!
  %newaddr = getelementptr <2 x double>* %addr, i32 5
  %val = load <2 x double>* %newaddr, align 8
  store <2 x double>* %newaddr, <2 x double>** bitcast(i8** @ptr to <2 x double>**)
  ret <2 x double> %val
}

define <2 x double> @test_v2f64_post_load(<2 x double>* %addr) {
; CHECK-LABEL: test_v2f64_post_load:
; CHECK: ldr q0, [x0], #80
  %newaddr = getelementptr <2 x double>* %addr, i32 5
  %val = load <2 x double>* %addr, align 8
  store <2 x double>* %newaddr, <2 x double>** bitcast(i8** @ptr to <2 x double>**)
  ret <2 x double> %val
}

define void @test_v2f64_pre_store(<2 x double> %in, <2 x double>* %addr) {
; CHECK-LABEL: test_v2f64_pre_store:
; CHECK: str q0, [x0, #80]!
  %newaddr = getelementptr <2 x double>* %addr, i32 5
  store <2 x double> %in, <2 x double>* %newaddr, align 8
  store <2 x double>* %newaddr, <2 x double>** bitcast(i8** @ptr to <2 x double>**)
  ret void
}

define void @test_v2f64_post_store(<2 x double> %in, <2 x double>* %addr) {
; CHECK-LABEL: test_v2f64_post_store:
; CHECK: str q0, [x0], #80
  %newaddr = getelementptr <2 x double>* %addr, i32 5
  store <2 x double> %in, <2 x double>* %addr, align 8
  store <2 x double>* %newaddr, <2 x double>** bitcast(i8** @ptr to <2 x double>**)
  ret void
}

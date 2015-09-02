; check AVX2 instructions that are disabled in case avx512VL/avx512BW present
   
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=core-avx2 -mattr=+avx2                 -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=knl                                    -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=knl  -mattr=+avx512vl                  -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=knl  -mattr=+avx512bw                  -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=knl  -mattr=+avx512vl -mattr=+avx512bw -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=skx                                    -o /dev/null

define <4 x i64> @vpand_256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = and <4 x i64> %a2, %b
  ret <4 x i64> %x
}

define <2 x i64> @vpand_128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %x = and <2 x i64> %a2, %b
  ret <2 x i64> %x
}

define <4 x i64> @vpandn_256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %y = xor <4 x i64> %a2, <i64 -1, i64 -1, i64 -1, i64 -1>
  %x = and <4 x i64> %a, %y
  ret <4 x i64> %x
}

define <2 x i64> @vpandn_128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %y = xor <2 x i64> %a2, <i64 -1, i64 -1>
  %x = and <2 x i64> %a, %y
  ret <2 x i64> %x
}

define <4 x i64> @vpor_256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = or <4 x i64> %a2, %b
  ret <4 x i64> %x
}

define <4 x i64> @vpxor_256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = xor <4 x i64> %a2, %b
  ret <4 x i64> %x
}

define <2 x i64> @vpor_128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %x = or <2 x i64> %a2, %b
  ret <2 x i64> %x
}

define <2 x i64> @vpxor_128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %x = xor <2 x i64> %a2, %b
  ret <2 x i64> %x
}

define <4 x i64> @test_vpaddq_256(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = add <4 x i64> %i, %j
  ret <4 x i64> %x
}

define <8 x i32> @test_vpaddd_256(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = add <8 x i32> %i, %j
  ret <8 x i32> %x
}

define <16 x i16> @test_vpaddw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = add <16 x i16> %i, %j
  ret <16 x i16> %x
}

define <32 x i8> @test_vpaddb_256(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = add <32 x i8> %i, %j
  ret <32 x i8> %x
}

define <4 x i64> @test_vpsubq_256(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = sub <4 x i64> %i, %j
  ret <4 x i64> %x
}

define <8 x i32> @test_vpsubd_256(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = sub <8 x i32> %i, %j
  ret <8 x i32> %x
}

define <16 x i16> @test_vpsubw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = sub <16 x i16> %i, %j
  ret <16 x i16> %x
}

define <32 x i8> @test_vpsubb_256(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = sub <32 x i8> %i, %j
  ret <32 x i8> %x
}

define <16 x i16> @test_vpmullw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = mul <16 x i16> %i, %j
  ret <16 x i16> %x
}

define <8 x i32> @test_vpcmpgtd_256(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %bincmp = icmp slt <8 x i32> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %x
}

define <32 x i8> @test_vpcmpeqb_256(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %bincmp = icmp eq <32 x i8> %i, %j
  %x = sext <32 x i1> %bincmp to <32 x i8>
  ret <32 x i8> %x
}

define <16 x i16> @test_vpcmpeqw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %bincmp = icmp eq <16 x i16> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i16>
  ret <16 x i16> %x
}

define <32 x i8> @test_vpcmpgtb_256(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %bincmp = icmp slt <32 x i8> %i, %j
  %x = sext <32 x i1> %bincmp to <32 x i8>
  ret <32 x i8> %x
}

define <16 x i16> @test_vpcmpgtw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %bincmp = icmp slt <16 x i16> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i16>
  ret <16 x i16> %x
}

define <8 x i32> @test_vpcmpeqd_256(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %bincmp = icmp eq <8 x i32> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %x
}

define <2 x i64> @test_vpaddq_128(<2 x i64> %i, <2 x i64> %j) nounwind readnone {
  %x = add <2 x i64> %i, %j
  ret <2 x i64> %x
}

define <4 x i32> @test_vpaddd_128(<4 x i32> %i, <4 x i32> %j) nounwind readnone {
  %x = add <4 x i32> %i, %j
  ret <4 x i32> %x
}

define <8 x i16> @test_vpaddw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %x = add <8 x i16> %i, %j
  ret <8 x i16> %x
}

define <16 x i8> @test_vpaddb_128(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %x = add <16 x i8> %i, %j
  ret <16 x i8> %x
}

define <2 x i64> @test_vpsubq_128(<2 x i64> %i, <2 x i64> %j) nounwind readnone {
  %x = sub <2 x i64> %i, %j
  ret <2 x i64> %x
}

define <4 x i32> @test_vpsubd_128(<4 x i32> %i, <4 x i32> %j) nounwind readnone {
  %x = sub <4 x i32> %i, %j
  ret <4 x i32> %x
}

define <8 x i16> @test_vpsubw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %x = sub <8 x i16> %i, %j
  ret <8 x i16> %x
}

define <16 x i8> @test_vpsubb_128(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %x = sub <16 x i8> %i, %j
  ret <16 x i8> %x
}

define <8 x i16> @test_vpmullw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %x = mul <8 x i16> %i, %j
  ret <8 x i16> %x
}

define <8 x i16> @test_vpcmpgtw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %bincmp = icmp slt <8 x i16> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i16>
  ret <8 x i16> %x
}

define <16 x i8> @test_vpcmpgtb_128(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %bincmp = icmp slt <16 x i8> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i8>
  ret <16 x i8> %x
}

define <8 x i16> @test_vpcmpeqw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %bincmp = icmp eq <8 x i16> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i16>
  ret <8 x i16> %x
}

define <16 x i8> @test_vpcmpeqb_128(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %bincmp = icmp eq <16 x i8> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i8>
  ret <16 x i8> %x
}

define <8 x i16> @shuffle_v8i16_vpalignr(<8 x i16> %a, <8 x i16> %b) {
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  ret <8 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_vpalignr(<16 x i16> %a, <16 x i16> %b) {
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 31, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <16 x i16> %shuffle
}

define <16 x i8> @shuffle_v16i8_vpalignr(<16 x i8> %a, <16 x i8> %b) {
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <16 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_vpalignr(<32 x i8> %a, <32 x i8> %b) {
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 undef, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 63, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <32 x i8> %shuffle
}

define <2 x i64> @shuffle_v2i64_vpalignr(<2 x i64> %a, <2 x i64> %b) {
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %shuffle
}

define <4 x i32> @shuffle_v4i32_vpalignr(<4 x i32> %a, <4 x i32> %b) {
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 7, i32 0, i32 1, i32 2>
  ret <4 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_vpalignr(<8 x i32> %a, <8 x i32> %b) {
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 11, i32 0, i32 1, i32 2, i32 15, i32 4, i32 5, i32 6>
  ret <8 x i32> %shuffle
}

define <4 x double> @shuffle_v4f64_5163(<4 x double> %a, <4 x double> %b) {
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 5, i32 1, i32 6, i32 3>
  ret <4 x double> %shuffle
}

define <2 x double> @shuffle_v2f64_bitcast_1z(<2 x double> %a) {
  %shuffle64 = shufflevector <2 x double> %a, <2 x double> zeroinitializer, <2 x i32> <i32 2, i32 1>
  %bitcast32 = bitcast <2 x double> %shuffle64 to <4 x float>
  %shuffle32 = shufflevector <4 x float> %bitcast32, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
  %bitcast64 = bitcast <4 x float> %shuffle32 to <2 x double>
  ret <2 x double> %bitcast64
}
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

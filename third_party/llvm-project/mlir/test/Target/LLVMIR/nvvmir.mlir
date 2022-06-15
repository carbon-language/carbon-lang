// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_special_regs
llvm.func @nvvm_special_regs() -> i32 {
  // CHECK: %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %1 = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %2 = nvvm.read.ptx.sreg.tid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %3 = nvvm.read.ptx.sreg.tid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %5 = nvvm.read.ptx.sreg.ntid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %6 = nvvm.read.ptx.sreg.ntid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %7 = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %8 = nvvm.read.ptx.sreg.ctaid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %9 = nvvm.read.ptx.sreg.ctaid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %10 = nvvm.read.ptx.sreg.nctaid.x : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %11 = nvvm.read.ptx.sreg.nctaid.y : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %12 = nvvm.read.ptx.sreg.nctaid.z : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %13 = nvvm.read.ptx.sreg.warpsize : i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %14 = nvvm.read.ptx.sreg.laneid : i32
  llvm.return %1 : i32
}

// CHECK-LABEL: @nvvm_rcp
llvm.func @nvvm_rcp(%0: f32) -> f32 {
  // CHECK: call float @llvm.nvvm.rcp.approx.ftz.f
  %1 = nvvm.rcp.approx.ftz.f %0 : f32
  llvm.return %1 : f32
}

// CHECK-LABEL: @llvm_nvvm_barrier0
llvm.func @llvm_nvvm_barrier0() {
  // CHECK: call void @llvm.nvvm.barrier0()
  nvvm.barrier0
  llvm.return
}

// CHECK-LABEL: @nvvm_shfl
llvm.func @nvvm_shfl(
    %0 : i32, %1 : i32, %2 : i32,
    %3 : i32, %4 : f32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %6 = nvvm.shfl.sync bfly %0, %3, %1, %2 : i32 -> i32
  // CHECK: call float @llvm.nvvm.shfl.sync.bfly.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %7 = nvvm.shfl.sync bfly %0, %4, %1, %2 : f32 -> f32
  // CHECK: call i32 @llvm.nvvm.shfl.sync.up.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %8 = nvvm.shfl.sync up %0, %3, %1, %2 : i32 -> i32
  // CHECK: call float @llvm.nvvm.shfl.sync.up.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %9 = nvvm.shfl.sync up %0, %4, %1, %2 : f32 -> f32
  // CHECK: call i32 @llvm.nvvm.shfl.sync.down.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %10 = nvvm.shfl.sync down %0, %3, %1, %2 : i32 -> i32
  // CHECK: call float @llvm.nvvm.shfl.sync.down.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %11 = nvvm.shfl.sync down %0, %4, %1, %2 : f32 -> f32
  // CHECK: call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %12 = nvvm.shfl.sync idx %0, %3, %1, %2 : i32 -> i32
  // CHECK: call float @llvm.nvvm.shfl.sync.idx.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %13 = nvvm.shfl.sync idx %0, %4, %1, %2 : f32 -> f32
  llvm.return %6 : i32
}

// CHECK-LABEL: @nvvm_shfl_pred
llvm.func @nvvm_shfl_pred(
    %0 : i32, %1 : i32, %2 : i32,
    %3 : i32, %4 : f32) -> !llvm.struct<(i32, i1)> {
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %6 = nvvm.shfl.sync bfly %0, %3, %1, %2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.bfly.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %7 = nvvm.shfl.sync bfly %0, %4, %1, %2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.up.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %8 = nvvm.shfl.sync up %0, %3, %1, %2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.up.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %9 = nvvm.shfl.sync up %0, %4, %1, %2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.down.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %10 = nvvm.shfl.sync down %0, %3, %1, %2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.down.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %11 = nvvm.shfl.sync down %0, %4, %1, %2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.idx.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %12 = nvvm.shfl.sync idx %0, %3, %1, %2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.idx.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %13 = nvvm.shfl.sync idx %0, %4, %1, %2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  llvm.return %6 : !llvm.struct<(i32, i1)>
}

// CHECK-LABEL: @nvvm_vote
llvm.func @nvvm_vote(%0 : i32, %1 : i1) -> i32 {
  // CHECK: call i32 @llvm.nvvm.vote.ballot.sync(i32 %{{.*}}, i1 %{{.*}})
  %3 = nvvm.vote.ballot.sync %0, %1 : i32
  llvm.return %3 : i32
}

// CHECK-LABEL: @nvvm_mma_mn8n8k4_row_col_f32_f32
llvm.func @nvvm_mma_mn8n8k4_row_col_f32_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                    %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
                    %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> {
  // CHECK: call { float, float, float, float, float, float, float, float } @llvm.nvvm.mma.m8n8k4.row.col.f32.f32
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
  {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_f16_f16
llvm.func @nvvm_mma_m16n8k16_f16_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : vector<2xf16>, %c1 : vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f16
  %0 = nvvm.mma.sync A[ %a0, %a1, %a2, %a3 ] B[ %b0, %b1 ] C[ %c0, %c1 ]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 16, n = 8, k = 16>}
     : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// f32 return type, f16 accumulate type
// CHECK-LABEL: @nvvm_mma_m16n8k16_f32_f16
llvm.func @nvvm_mma_m16n8k16_f32_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : vector<2xf16>, %c1 : vector<2xf16>) -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f16
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// f16 return type, f32 accumulate type
// CHECK-LABEL: @nvvm_mma_m16n8k16_f16_f32
llvm.func @nvvm_mma_m16n8k16_f16_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)> {
  // CHECK: call { <2 x half>, <2 x half> } @llvm.nvvm.mma.m16n8k16.row.col.f16.f32
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// f32 return type, f32 accumulate type
// CHECK-LABEL: @nvvm_mma_m16n8k16_f32_f32
llvm.func @nvvm_mma_m16n8k16_f32_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.m16n8k16.row.col.f32.f32
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_s8_s8
llvm.func @nvvm_mma_m16n8k16_s8_s8(%a0 : i32, %a1 : i32,
                                %b0 : i32,
                                %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k16.row.col.s8
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<wrapped>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_s8_u8
llvm.func @nvvm_mma_m16n8k16_s8_u8(%a0 : i32, %a1 : i32,
                                %b0 : i32,
                                %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k16.row.col.satfinite.s8.u8
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<u8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<satfinite>,
     shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k128_b1_b1
llvm.func @nvvm_mma_m16n8k128_b1_b1(%a0 : i32, %a1 : i32,
                                    %b0 : i32,
                                    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) -> !llvm.struct<(i32,i32,i32,i32)> {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.xor.popc.m16n8k128.row.col.b1
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     b1Op = #nvvm.mma_b1op<xor_popc>, shape = #nvvm.shape<m = 16, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k32_s4_s4
llvm.func @nvvm_mma_m16n8k32_s4_s4(%a0 : i32, %a1 : i32,
                               %b0 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) -> !llvm.struct<(i32,i32,i32,i32)> {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.mma.m16n8k32.row.col.satfinite.s4
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s4>, multiplicandBPtxType = #nvvm.mma_type<s4>,
     intOverflowBehavior=#nvvm.mma_int_overflow<satfinite>,
     shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m8n8k4_f64_f64
llvm.func @nvvm_mma_m8n8k4_f64_f64(%a0 : f64,
                                   %b0 : f64,
                                   %c0 : f64, %c1 : f64) -> !llvm.struct<(f64, f64)> {
  // CHECK: call { double, double } @llvm.nvvm.mma.m8n8k4.row.col.f64
  %0 = nvvm.mma.sync A[%a0] B[%b0] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = #nvvm.shape<m = 8, n = 8, k = 4>} : (f64, f64, f64) -> !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k4_tf32_f32
llvm.func @nvvm_mma_m16n8k4_tf32_f32(%a0 : i32, %a1 : i32,
                                     %b0 : i32,
                                     %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: call { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>,
     shape = #nvvm.shape<m = 16, n = 8, k = 4>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// The test below checks the correct mapping of the nvvm.wmma.*.load.* op to the correct intrinsic
// in the LLVM NVPTX backend.
// CHECK-LABEL: @gpu_wmma_load_op
llvm.func @gpu_wmma_load_op(%arg0: !llvm.ptr<i32, 3>, %arg1: i32) {
  // CHECK: call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p3(ptr addrspace(3) %{{.*}}, i32 %{{.*}})
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<f16>, frag = #nvvm.mma_frag<a>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr<i32, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>

  llvm.return
}

// The test below checks the correct mapping of the nvvm.wmma.*.store.* op to the correct intrinsic
// in the LLVM NVPTX backend.
// CHECK-LABEL: @gpu_wmma_store_op
llvm.func @gpu_wmma_store_op(%arg0: !llvm.ptr<i32, 3>, %arg1: i32,
                            %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                            %arg4: vector<2 xf16>, %arg5: vector<2 x f16>) {
  // CHECK: call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p3(ptr addrspace(3) %{{.*}}, <2 x half> {{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}}, i32 %{{.*}})
  nvvm.wmma.store %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
    {eltype = #nvvm.mma_type<f16>, k = 16 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : !llvm.ptr<i32, 3>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>
  llvm.return
}

// The test below checks the correct mapping of the nvvm.wmma.*.mma.* op to the correct intrinsic
// in the LLVM NVPTX backend.
// CHECK-LABEL: @gpu_wmma_mma_op
llvm.func @gpu_wmma_mma_op(%arg0: vector<2 x f16>, %arg1: vector<2 x f16>,
                        %arg2: vector<2 x f16>, %arg3: vector<2 x f16>,
                        %arg4: vector<2 x f16>, %arg5: vector<2 x f16>,
                        %arg6: vector<2 x f16>, %arg7: vector<2 x f16>,
                        %arg8: vector<2 x f16>, %arg9: vector<2 x f16>,
                        %arg10: vector<2 x f16>, %arg11: vector<2 x f16>,
                        %arg12: vector<2 x f16>, %arg13: vector<2 x f16>,
                        %arg14: vector<2 x f16>, %arg15: vector<2 x f16>,
                        %arg16: vector<2 x f16>, %arg17: vector<2 x f16>,
                        %arg18: vector<2 x f16>, %arg19: vector<2 x f16>) {
  // CHECK: call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16(<2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}}, <2 x half> {{.*}})
  %0 = nvvm.wmma.mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19
    {eltypeA = #nvvm.mma_type<f16>, eltypeB = #nvvm.mma_type<f16>, k = 16 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>,
       vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)
      -> !llvm.struct<(vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>)>
  llvm.return
}

// CHECK-LABEL: @nvvm_wmma_load_tf32
llvm.func @nvvm_wmma_load_tf32(%arg0: !llvm.ptr<i32>, %arg1 : i32) {
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.row.stride.tf32.p0(ptr %{{.*}}, i32 %{{.*}})
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<tf32>, frag = #nvvm.mma_frag<a>, k = 8 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr<i32>) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return
}

// CHECK-LABEL: @nvvm_wmma_mma
llvm.func @nvvm_wmma_mma(%0 : i32, %1 : i32, %2 : i32, %3 : i32, %4 : i32, %5 : i32,
                    %6 : i32, %7 : i32, %8 : f32, %9 : f32, %10 : f32,
                    %11 : f32, %12 : f32, %13 : f32, %14 : f32, %15 : f32) {
  // CHECK: { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.row.row.tf32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
  %r = nvvm.wmma.mma %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
    {eltypeA = #nvvm.mma_type<tf32>, eltypeB = #nvvm.mma_type<f32>, k = 8 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, f32, f32, f32, f32)
    -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return
}

// CHECK-LABEL: @cp_async
llvm.func @cp_async(%arg0: !llvm.ptr<i8, 3>, %arg1: !llvm.ptr<i8, 1>) {
// CHECK: call void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}})
  nvvm.cp.async.shared.global %arg0, %arg1, 4
// CHECK: call void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}})
  nvvm.cp.async.shared.global %arg0, %arg1, 8
// CHECK: call void @llvm.nvvm.cp.async.ca.shared.global.16(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}})
  nvvm.cp.async.shared.global %arg0, %arg1, 16
// CHECK: call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}})
  nvvm.cp.async.shared.global %arg0, %arg1, 16 {bypass_l1}
// CHECK: call void @llvm.nvvm.cp.async.commit.group()
  nvvm.cp.async.commit.group
// CHECK: call void @llvm.nvvm.cp.async.wait.group(i32 0)
  nvvm.cp.async.wait.group 0
  llvm.return
}

// CHECK-LABEL: @ld_matrix
llvm.func @ld_matrix(%arg0: !llvm.ptr<i32, 3>) {
  // CHECK: call i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16.p3(ptr addrspace(3) %{{.*}})
  %l1 = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> i32
  // CHECK: call { i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3) %{{.*}})
  %l2 = nvvm.ldmatrix %arg0 {num = 2 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32)>
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %{{.*}})
  %l4 = nvvm.ldmatrix %arg0 {num = 4 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
   // CHECK: call i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16.p3(ptr addrspace(3) %{{.*}})
  %l1t = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<col>} : (!llvm.ptr<i32, 3>) -> i32
  // CHECK: call { i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16.p3(ptr addrspace(3) %{{.*}})
  %l2t = nvvm.ldmatrix %arg0 {num = 2 : i32, layout = #nvvm.mma_layout<col>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32)>
  // CHECK: call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16.p3(ptr addrspace(3) %{{.*}})
  %l4t = nvvm.ldmatrix %arg0 {num = 4 : i32, layout = #nvvm.mma_layout<col>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return
}

// This function has the "kernel" attribute attached and should appear in the
// NVVM annotations after conversion.
llvm.func @kernel_func() attributes {nvvm.kernel} {
  llvm.return
}

// CHECK:     !nvvm.annotations =
// CHECK-NOT: {ptr @nvvm_special_regs, !"kernel", i32 1}
// CHECK:     {ptr @kernel_func, !"kernel", i32 1}

//===-- AMDGPUSearchableTables.td - ------------------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"

//===----------------------------------------------------------------------===//
// Resource intrinsics table.
//===----------------------------------------------------------------------===//

class RsrcIntrinsic<AMDGPURsrcIntrinsic intr> {
  Intrinsic Intr = !cast<Intrinsic>(intr);
  bits<8> RsrcArg = intr.RsrcArg;
  bit IsImage = intr.IsImage;
}

def RsrcIntrinsics : GenericTable {
  let FilterClass = "RsrcIntrinsic";
  let Fields = ["Intr", "RsrcArg", "IsImage"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "lookupRsrcIntrinsic";
}

foreach intr = !listconcat(AMDGPUBufferIntrinsics,
                           AMDGPUImageIntrinsics,
                           AMDGPUImageDimIntrinsics,
                           AMDGPUImageDimGatherIntrinsics,
                           AMDGPUImageDimGetResInfoIntrinsics,
                           AMDGPUImageDimAtomicIntrinsics) in {
  def : RsrcIntrinsic<!cast<AMDGPURsrcIntrinsic>(intr)>;
}

class SourceOfDivergence<Intrinsic intr> {
  Intrinsic Intr = intr;
}

def SourcesOfDivergence : GenericTable {
  let FilterClass = "SourceOfDivergence";
  let Fields = ["Intr"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "lookupSourceOfDivergence";
}

def : SourceOfDivergence<int_amdgcn_workitem_id_x>;
def : SourceOfDivergence<int_amdgcn_workitem_id_y>;
def : SourceOfDivergence<int_amdgcn_workitem_id_z>;
def : SourceOfDivergence<int_amdgcn_interp_mov>;
def : SourceOfDivergence<int_amdgcn_interp_p1>;
def : SourceOfDivergence<int_amdgcn_interp_p2>;
def : SourceOfDivergence<int_amdgcn_mbcnt_hi>;
def : SourceOfDivergence<int_amdgcn_mbcnt_lo>;
def : SourceOfDivergence<int_r600_read_tidig_x>;
def : SourceOfDivergence<int_r600_read_tidig_y>;
def : SourceOfDivergence<int_r600_read_tidig_z>;
def : SourceOfDivergence<int_amdgcn_atomic_inc>;
def : SourceOfDivergence<int_amdgcn_atomic_dec>;
def : SourceOfDivergence<int_amdgcn_ds_fadd>;
def : SourceOfDivergence<int_amdgcn_ds_fmin>;
def : SourceOfDivergence<int_amdgcn_ds_fmax>;
def : SourceOfDivergence<int_amdgcn_image_atomic_swap>;
def : SourceOfDivergence<int_amdgcn_image_atomic_add>;
def : SourceOfDivergence<int_amdgcn_image_atomic_sub>;
def : SourceOfDivergence<int_amdgcn_image_atomic_smin>;
def : SourceOfDivergence<int_amdgcn_image_atomic_umin>;
def : SourceOfDivergence<int_amdgcn_image_atomic_smax>;
def : SourceOfDivergence<int_amdgcn_image_atomic_umax>;
def : SourceOfDivergence<int_amdgcn_image_atomic_and>;
def : SourceOfDivergence<int_amdgcn_image_atomic_or>;
def : SourceOfDivergence<int_amdgcn_image_atomic_xor>;
def : SourceOfDivergence<int_amdgcn_image_atomic_inc>;
def : SourceOfDivergence<int_amdgcn_image_atomic_dec>;
def : SourceOfDivergence<int_amdgcn_image_atomic_cmpswap>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_swap>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_add>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_sub>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_smin>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_umin>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_smax>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_umax>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_and>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_or>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_xor>;
def : SourceOfDivergence<int_amdgcn_buffer_atomic_cmpswap>;
def : SourceOfDivergence<int_amdgcn_ps_live>;
def : SourceOfDivergence<int_amdgcn_ds_swizzle>;

foreach intr = AMDGPUImageDimAtomicIntrinsics in
def : SourceOfDivergence<intr>;

class D16ImageDimIntrinsic<AMDGPUImageDimIntrinsic intr> {
  Intrinsic Intr = intr;
  code D16HelperIntr =
      !cast<code>("AMDGPUIntrinsic::SI_image_d16helper_" # intr.P.OpMod # intr.P.Dim.Name);
}

def D16ImageDimIntrinsics : GenericTable {
  let FilterClass = "D16ImageDimIntrinsic";
  let Fields = ["Intr", "D16HelperIntr"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "lookupD16ImageDimIntrinsic";
}

foreach intr = !listconcat(AMDGPUImageDimIntrinsics,
                           AMDGPUImageDimGatherIntrinsics) in {
  def : D16ImageDimIntrinsic<intr>;
}

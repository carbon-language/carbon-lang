//===-- AMDGPURegisterInfo.td - AMDGPU register info -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tablegen register definitions common to all hw codegen targets.
//
//===----------------------------------------------------------------------===//

let Namespace = "AMDGPU" in {
  def sel_x : SubRegIndex;
  def sel_y : SubRegIndex;
  def sel_z : SubRegIndex;
  def sel_w : SubRegIndex;
}

include "R600RegisterInfo.td"
include "SIRegisterInfo.td"

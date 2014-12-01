(*===-- llvm_transform_utils.ml - LLVM OCaml Interface --------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

external clone_module : Llvm.llmodule -> Llvm.llmodule = "llvm_clone_module"

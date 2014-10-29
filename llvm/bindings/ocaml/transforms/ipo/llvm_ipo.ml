(*===-- llvm_ipo.ml - LLVM OCaml Interface --------------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

external add_argument_promotion
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_argument_promotion"
external add_constant_merge
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_constant_merge"
external add_dead_arg_elimination
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_dead_arg_elimination"
external add_function_attrs
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_function_attrs"
external add_function_inlining
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_function_inlining"
external add_always_inliner
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_always_inliner"
external add_global_dce
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_global_dce"
external add_global_optimizer
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_global_optimizer"
external add_ipc_propagation
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_ip_constant_propagation"
external add_prune_eh
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_prune_eh"
external add_ipsccp
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_ipsccp"
external add_internalize
  : [ `Module ] Llvm.PassManager.t -> all_but_main:bool -> unit
  = "llvm_add_internalize"
external add_strip_dead_prototypes
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_strip_dead_prototypes"
external add_strip_symbols
  : [ `Module ] Llvm.PassManager.t -> unit
  = "llvm_add_strip_symbols"

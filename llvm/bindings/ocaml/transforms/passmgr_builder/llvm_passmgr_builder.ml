(*===-- llvm_passmgr_builder.ml - LLVM OCaml Interface --------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

type t

external create : unit -> t
  = "llvm_pmbuilder_create"
external set_opt_level : int -> t -> unit
  = "llvm_pmbuilder_set_opt_level"
external set_size_level : int -> t -> unit
  = "llvm_pmbuilder_set_size_level"
external set_disable_unit_at_a_time : bool -> t -> unit
  = "llvm_pmbuilder_set_disable_unit_at_a_time"
external set_disable_unroll_loops : bool -> t -> unit
  = "llvm_pmbuilder_set_disable_unroll_loops"
external use_inliner_with_threshold : int -> t -> unit
  = "llvm_pmbuilder_use_inliner_with_threshold"
external populate_function_pass_manager
  : [ `Function ] Llvm.PassManager.t -> t -> unit
  = "llvm_pmbuilder_populate_function_pass_manager"
external populate_module_pass_manager
  : [ `Module ] Llvm.PassManager.t -> t -> unit
  = "llvm_pmbuilder_populate_module_pass_manager"
external populate_lto_pass_manager
  : [ `Module ] Llvm.PassManager.t -> internalize:bool -> run_inliner:bool -> t -> unit
  = "llvm_pmbuilder_populate_lto_pass_manager"
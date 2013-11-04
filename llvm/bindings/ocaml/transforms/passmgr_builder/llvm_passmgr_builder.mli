(*===-- llvm_passmgr_builder.mli - LLVM OCaml Interface -------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Pass Manager Builder.

    This interface provides an OCaml API for LLVM pass manager builder
    from the [LLVMCore] library. *)

type t

(** See [llvm::PassManagerBuilder]. *)
external create : unit -> t
  = "llvm_pmbuilder_create"

(** See [llvm::PassManagerBuilder::OptLevel]. *)
external set_opt_level : int -> t -> unit
  = "llvm_pmbuilder_set_opt_level"

(** See [llvm::PassManagerBuilder::SizeLevel]. *)
external set_size_level : int -> t -> unit
  = "llvm_pmbuilder_set_size_level"

(** See [llvm::PassManagerBuilder::DisableUnitAtATime]. *)
external set_disable_unit_at_a_time : bool -> t -> unit
  = "llvm_pmbuilder_set_disable_unit_at_a_time"

(** See [llvm::PassManagerBuilder::DisableUnrollLoops]. *)
external set_disable_unroll_loops : bool -> t -> unit
  = "llvm_pmbuilder_set_disable_unroll_loops"

(** See [llvm::PassManagerBuilder::Inliner]. *)
external use_inliner_with_threshold : int -> t -> unit
  = "llvm_pmbuilder_use_inliner_with_threshold"

(** See [llvm::PassManagerBuilder::populateFunctionPassManager]. *)
external populate_function_pass_manager
  : [ `Function ] Llvm.PassManager.t -> t -> unit
  = "llvm_pmbuilder_populate_function_pass_manager"

(** See [llvm::PassManagerBuilder::populateModulePassManager]. *)
external populate_module_pass_manager
  : [ `Module ] Llvm.PassManager.t -> t -> unit
  = "llvm_pmbuilder_populate_module_pass_manager"

(** See [llvm::PassManagerBuilder::populateLTOPassManager]. *)
external populate_lto_pass_manager
  : [ `Module ] Llvm.PassManager.t -> internalize:bool -> run_inliner:bool -> t -> unit
  = "llvm_pmbuilder_populate_lto_pass_manager"
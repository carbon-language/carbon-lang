(*===-- llvm_executionengine.ml - LLVM OCaml Interface --------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

exception Error of string

let () = Callback.register_exception "Llvm_executionengine.Error" (Error "")

external initialize : unit -> bool
  = "llvm_ee_initialize"

type llexecutionengine

type llcompileroptions = {
  opt_level: int;
  code_model: Llvm_target.CodeModel.t;
  no_framepointer_elim: bool;
  enable_fast_isel: bool;
}

let default_compiler_options = {
  opt_level = 0;
  code_model = Llvm_target.CodeModel.JITDefault;
  no_framepointer_elim = false;
  enable_fast_isel = false }

external create : ?options:llcompileroptions -> Llvm.llmodule -> llexecutionengine
  = "llvm_ee_create"
external dispose : llexecutionengine -> unit
  = "llvm_ee_dispose"
external add_module : Llvm.llmodule -> llexecutionengine -> unit
  = "llvm_ee_add_module"
external remove_module : Llvm.llmodule -> llexecutionengine -> unit
  = "llvm_ee_remove_module"
external run_static_ctors : llexecutionengine -> unit
  = "llvm_ee_run_static_ctors"
external run_static_dtors : llexecutionengine -> unit
  = "llvm_ee_run_static_dtors"
external data_layout : llexecutionengine -> Llvm_target.DataLayout.t
  = "llvm_ee_get_data_layout"
external add_global_mapping_ : Llvm.llvalue -> int64 -> llexecutionengine -> unit
  = "llvm_ee_add_global_mapping"
external get_pointer_to_global_ : Llvm.llvalue -> llexecutionengine -> int64
  = "llvm_ee_get_pointer_to_global"

let add_global_mapping llval ptr ee =
  add_global_mapping_ llval (Ctypes.raw_address_of_ptr (Ctypes.to_voidp ptr)) ee

let get_pointer_to_global llval typ ee =
  Ctypes.coerce (let open Ctypes in ptr void) typ
                (Ctypes.ptr_of_raw_address (get_pointer_to_global_ llval ee))

(* The following are not bound. Patches are welcome.
target_machine : llexecutionengine -> Llvm_target.TargetMachine.t
 *)

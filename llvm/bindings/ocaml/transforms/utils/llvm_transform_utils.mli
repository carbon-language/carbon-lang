(*===-- llvm_transform_utils.mli - LLVM OCaml Interface -------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Transform Utilities.

    This interface provides an OCaml API for LLVM transform utilities, the
    classes in the [LLVMTransformUtils] library. *)

(** [clone_module m] returns an exact copy of module [m].
    See the [llvm::CloneModule] function. *)
external clone_module : Llvm.llmodule -> Llvm.llmodule = "llvm_clone_module"

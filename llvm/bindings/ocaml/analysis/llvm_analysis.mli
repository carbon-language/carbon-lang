(*===-- llvm_analysis.mli - LLVM Ocaml Interface ----------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by Gordon Henriksen and is distributed under the
 * University of Illinois Open Source License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===
 *
 * This interface provides an ocaml API for LLVM IR analyses, the classes in
 * the Analysis library.
 *
 *===----------------------------------------------------------------------===*)


(** [verify_module m] returns [None] if the module [m] is valid, and
    [Some reason] if it is invalid. [reason] is a string containing a
    human-readable validation report. See [llvm::verifyModule]. **)
external verify_module : Llvm.llmodule -> string option = "llvm_verify_module"

(** [verify_function f] returns [None] if the function [f] is valid, and
    [Some reason] if it is invalid. [reason] is a string containing a
    human-readable validation report. See [llvm::verifyFunction]. **)
external verify_function : Llvm.llvalue -> bool = "llvm_verify_function"

(** [verify_module m] returns if the module [m] is valid, but prints a
    validation report to [stderr] and aborts the program if it is invalid. See 
    [llvm::verifyModule]. **)
external assert_valid_module : Llvm.llmodule -> unit
                             = "llvm_assert_valid_module"

(** [verify_function f] returns if the function [f] is valid, but prints a
    validation report to [stderr] and aborts the program if it is invalid. See 
    [llvm::verifyFunction]. **)
external assert_valid_function : Llvm.llvalue -> unit
                               = "llvm_assert_valid_function"

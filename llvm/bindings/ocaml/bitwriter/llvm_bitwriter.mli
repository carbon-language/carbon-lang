(*===-- llvm_bitwriter.mli - LLVM Ocaml Interface ---------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file was developed by Gordon Henriksen and is distributed under the
 * University of Illinois Open Source License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===
 *
 * This interface provides an ocaml API for the LLVM bitcode writer, the
 * classes in the Bitwriter library.
 *
 *===----------------------------------------------------------------------===*)


(* Writes the bitcode for module the given path. Returns true if successful. *)
external write_bitcode_file : Llvm.llmodule -> string -> bool
                            = "llvm_write_bitcode_file"

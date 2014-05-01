(*===-- llvm_all_backends.ml - LLVM OCaml Interface -----------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

external initialize : unit -> unit = "llvm_initialize_all"

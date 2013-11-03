(*===-- llvm_linker.ml - LLVM OCaml Interface ------------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

exception Error of string

external register_exns : exn -> unit = "llvm_register_linker_exns"
let _ = register_exns (Error "")

module Mode = struct
  type t =
  | DestroySource
  | PreserveSource
end

external link_modules : Llvm.llmodule -> Llvm.llmodule -> Mode.t -> unit
                      = "llvm_link_modules"
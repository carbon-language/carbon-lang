(*===-- llvm_bitreader.ml - LLVM OCaml Interface --------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

exception Error of string

let () = Callback.register_exception "Llvm_bitreader.Error" (Error "")

external get_module
  : Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule
  = "llvm_get_module"
external parse_bitcode
  : Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule
  = "llvm_parse_bitcode"

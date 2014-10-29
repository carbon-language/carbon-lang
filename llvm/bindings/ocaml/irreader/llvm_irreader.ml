(*===-- llvm_irreader.ml - LLVM OCaml Interface ---------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)


exception Error of string

let _ = Callback.register_exception "Llvm_irreader.Error" (Error "")

external parse_ir : Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule
                  = "llvm_parse_ir"

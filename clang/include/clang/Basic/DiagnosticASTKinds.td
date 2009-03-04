//==--- DiagnosticASTKinds.td - libast diagnostics ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

let Component = "AST" in {

def note_comma_in_ice : Note<
  "C does not permit evaluated commas in an integer constant expression">;
def note_expr_divide_by_zero : Note<"division by zero">;

}

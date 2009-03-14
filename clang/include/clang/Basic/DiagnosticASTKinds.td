//==--- DiagnosticASTKinds.td - libast diagnostics ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

let Component = "AST" in {

//def note_comma_in_ice : Note<
//  "C does not permit evaluated commas in an integer constant expression">;
def note_expr_divide_by_zero : Note<"division by zero">;

// inline asm related.
def err_asm_invalid_escape : Error<
  "invalid %% escape in inline assembly string">;
def err_asm_unknown_symbolic_operand_name : Error<
  "unknown symbolic operand name in inline assembly string">;

def err_asm_unterminated_symbolic_operand_name : Error<
  "unterminated symbolic operand name in inline assembly string">;
def err_asm_empty_symbolic_operand_name : Error<
  "empty symbolic operand name in inline assembly string">;
def err_asm_invalid_operand_number : Error<
  "invalid operand number in inline asm string">;

}

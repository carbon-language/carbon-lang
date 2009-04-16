//==--- DiagnosticGroups.td - Diagnostic Group Definitions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def ImplicitFunctionDeclare : DiagGroup<"implicit-function-declaration">;
def Trigraphs : DiagGroup<"trigraphs">;

// Empty DiagGroups: these are recognized by clang but ignored.
def : DiagGroup<"aggregate-return">;
def : DiagGroup<"bad-function-cast">;
def : DiagGroup<"cast-align">;
def : DiagGroup<"cast-qual">;
def : DiagGroup<"char-align">;
def : DiagGroup<"char-subscripts">;
def : DiagGroup<"declaration-after-statement">;
def : DiagGroup<"error-implicit-function-declaration">;
def : DiagGroup<"format-security">;
def : DiagGroup<"format=2">;
def : DiagGroup<"format">;
def : DiagGroup<"four-char-constants">;
def : DiagGroup<"init-self">;
def : DiagGroup<"inline">;
def : DiagGroup<"int-to-pointer-cast">;
def : DiagGroup<"missing-braces">;
def : DiagGroup<"missing-declarations">;
def : DiagGroup<"missing-format-attribute">;
def : DiagGroup<"nested-externs">;
def : DiagGroup<"newline-eof">;
def : DiagGroup<"no-#warnings">;
def : DiagGroup<"no-comment">;
def : DiagGroup<"format-y2k">;
def : DiagGroup<"long-long">;
def : DiagGroup<"missing-field-initializers">;
def : DiagGroup<"nonportable-cfstrings">;
def : DiagGroup<"old-style-definition">;
def : DiagGroup<"packed">;
def : DiagGroup<"parentheses">;
def : DiagGroup<"pointer-arith">;
def : DiagGroup<"pointer-to-int-cast">;
def : DiagGroup<"redundant-decls">;
def : DiagGroup<"return-type">;
def : DiagGroup<"shadow">;
def : DiagGroup<"shorten-64-to-32">;
def : DiagGroup<"sign-compare">;
def : DiagGroup<"strict-overflow=">;
def : DiagGroup<"strict-overflow">;
def : DiagGroup<"strict-prototypes">;
def : DiagGroup<"strict-selector-match">;
def Switch         : DiagGroup<"switch">;
def Uninitialized  : DiagGroup<"uninitialized">;
def UnknownPragmas : DiagGroup<"unknown-pragmas">;
def : DiagGroup<"unused-function">;
def : DiagGroup<"unused-label">;
def : DiagGroup<"unused-parameter">;
def UnusedValue    : DiagGroup<"unused-value">;
def UnusedVariable : DiagGroup<"unused-variable">;
def : DiagGroup<"variadic-macros">;
def VolatileRegisterVar : DiagGroup<"volatile-register-var">;
def : DiagGroup<"write-strings">;

def : DiagGroup<"extra">;
def : DiagGroup<"">;  // -W
def : DiagGroup<"most">;

// Aggregation warning settings.
def : DiagGroup<"all", [
    ImplicitFunctionDeclare,
    Switch,
    Trigraphs,
    Uninitialized,
    UnknownPragmas,
    UnusedValue,
    UnusedVariable,
    VolatileRegisterVar
 ]>;


//==--- DiagnosticGroups.td - Diagnostic Group Definitions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def Trigraphs : DiagGroup<"trigraphs">;

def ImplicitFunctionDeclare : DiagGroup<"implicit-function-declaration">;
def ImplicitInt : DiagGroup<"implicit-int">;

// Aggregation warning settings.
def Implicit : DiagGroup<"implicit", [
    ImplicitFunctionDeclare,
    ImplicitInt
]>;
    


// Empty DiagGroups: these are recognized by clang but ignored.
def : DiagGroup<"aggregate-return">;
def : DiagGroup<"bad-function-cast">;
def : DiagGroup<"cast-align">;
def : DiagGroup<"cast-qual">;
def : DiagGroup<"char-align">;
def : DiagGroup<"char-subscripts">;
def Comment : DiagGroup<"comment">;
def : DiagGroup<"conversion">;
def : DiagGroup<"declaration-after-statement">;
def : DiagGroup<"disabled-optimization">;
def : DiagGroup<"discard-qual">;
def ExtraTokens : DiagGroup<"extra-tokens">;

def FormatExtraArgs : DiagGroup<"format-extra-args">;
def FormatZeroLength : DiagGroup<"format-zero-length">;

def Format : DiagGroup<"format", [FormatExtraArgs, FormatZeroLength]>;
def FormatSecurity : DiagGroup<"format-security", [Format]>;
def FormatNonLiteral : DiagGroup<"format-nonliteral", [FormatSecurity]>;
def FormatY2K : DiagGroup<"format-y2k", [Format]>;
def Format2 : DiagGroup<"format=2",
                        [FormatNonLiteral, FormatSecurity, FormatY2K]>;

def FourByteMultiChar : DiagGroup<"four-char-constants">;
def : DiagGroup<"init-self">;
def : DiagGroup<"inline">;
def : DiagGroup<"int-to-pointer-cast">;
def : DiagGroup<"missing-braces">;
def : DiagGroup<"missing-declarations">;
def : DiagGroup<"missing-format-attribute">;
def : DiagGroup<"missing-noreturn">;
def MultiChar : DiagGroup<"multichar">;
def : DiagGroup<"nested-externs">;
def : DiagGroup<"newline-eof">;
def : DiagGroup<"long-long">;
def : DiagGroup<"missing-field-initializers">;
def : DiagGroup<"nonportable-cfstrings">;
def : DiagGroup<"old-style-definition">;
def : DiagGroup<"packed">;
def Parentheses : DiagGroup<"parentheses">;
def : DiagGroup<"pointer-arith">;
def : DiagGroup<"pointer-to-int-cast">;
def : DiagGroup<"redundant-decls">;
def ReturnType : DiagGroup<"return-type">;
def : DiagGroup<"sequence-point">;
def : DiagGroup<"shadow">;
def : DiagGroup<"shorten-64-to-32">;
def : DiagGroup<"sign-compare">;

// Just silence warnings about common forms of -Wstrict-aliasing for now.
def : DiagGroup<"strict-aliasing=0">;
def : DiagGroup<"strict-aliasing=1">;
def : DiagGroup<"strict-aliasing=2">;
def : DiagGroup<"strict-aliasing">;

// Just silence warnings about common forms of -Wstrict-aliasing for now.
def : DiagGroup<"strict-overflow=0">;
def : DiagGroup<"strict-overflow=1">;
def : DiagGroup<"strict-overflow=2">;
def : DiagGroup<"strict-overflow">;

def InvalidOffsetof : DiagGroup<"invalid-offsetof">;
def : DiagGroup<"strict-prototypes">;
def : DiagGroup<"strict-selector-match">;
def Switch         : DiagGroup<"switch">;
def : DiagGroup<"type-limits">;
def Uninitialized  : DiagGroup<"uninitialized">;
def UnknownPragmas : DiagGroup<"unknown-pragmas">;
def : DiagGroup<"unused-function">;
def : DiagGroup<"unused-label">;
def : DiagGroup<"unused-parameter">;
def UnusedValue    : DiagGroup<"unused-value">;
def UnusedVariable : DiagGroup<"unused-variable">;
def : DiagGroup<"variadic-macros">;
def VectorConversions : DiagGroup<"vector-conversions">;      // clang specific
def VolatileRegisterVar : DiagGroup<"volatile-register-var">;
def : DiagGroup<"write-strings">;

// Aggregation warning settings.

def Extra : DiagGroup<"extra">;

def Most : DiagGroup<"most", [
    Comment,
    Format,
    Implicit,
    MultiChar,
    Switch,
    Trigraphs,
    Uninitialized,
    UnknownPragmas,
    UnusedValue,
    UnusedVariable,
    VectorConversions,
    VolatileRegisterVar
 ]>;

// -Wall is -Wmost -Wparentheses
def : DiagGroup<"all", [Most, Parentheses]>;

// Aliases.
def : DiagGroup<"", [Extra]>;  // -W = -Wextra
def : DiagGroup<"endif-labels", [ExtraTokens]>; // endif-labels = endif-tokens


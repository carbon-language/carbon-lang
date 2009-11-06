//==--- DiagnosticGroups.td - Diagnostic Group Definitions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def ImplicitFunctionDeclare : DiagGroup<"implicit-function-declaration">;
def ImplicitInt : DiagGroup<"implicit-int">;

// Aggregation warning settings.
def Implicit : DiagGroup<"implicit", [
    ImplicitFunctionDeclare,
    ImplicitInt
]>;
    


// Empty DiagGroups: these are recognized by clang but ignored.
def : DiagGroup<"address">;
def : DiagGroup<"aggregate-return">;
def : DiagGroup<"attributes">;
def : DiagGroup<"bad-function-cast">;
def : DiagGroup<"c++-compat">;
def : DiagGroup<"cast-align">;
def : DiagGroup<"cast-qual">;
def : DiagGroup<"char-align">;
def Comment : DiagGroup<"comment">;
def : DiagGroup<"conversion">;
def : DiagGroup<"declaration-after-statement">;
def : DiagGroup<"disabled-optimization">;
def : DiagGroup<"discard-qual">;
def : DiagGroup<"div-by-zero">;
def EmptyBody : DiagGroup<"empty-body">;
def ExtraTokens : DiagGroup<"extra-tokens">;

def FormatExtraArgs : DiagGroup<"format-extra-args">;
def FormatZeroLength : DiagGroup<"format-zero-length">;

def FourByteMultiChar : DiagGroup<"four-char-constants">;
def : DiagGroup<"import">;
def : DiagGroup<"init-self">;
def : DiagGroup<"inline">;
def : DiagGroup<"int-to-pointer-cast">;
def : DiagGroup<"invalid-pch">;
def : DiagGroup<"missing-braces">;
def : DiagGroup<"missing-declarations">;
def : DiagGroup<"missing-format-attribute">;
def : DiagGroup<"missing-include-dirs">;
def : DiagGroup<"missing-noreturn">;
def MultiChar : DiagGroup<"multichar">;
def : DiagGroup<"nested-externs">;
def : DiagGroup<"newline-eof">;
def LongLong : DiagGroup<"long-long">;
def MismatchedTags : DiagGroup<"mismatched-tags">;
def : DiagGroup<"missing-field-initializers">;
def NonNull : DiagGroup<"nonnull">;
def : DiagGroup<"nonportable-cfstrings">;
def : DiagGroup<"old-style-definition">;
def : DiagGroup<"overflow">;
def : DiagGroup<"overloaded-virtual">;
def : DiagGroup<"packed">;
def Parentheses : DiagGroup<"parentheses">;
def PointerArith : DiagGroup<"pointer-arith">;
def : DiagGroup<"pointer-to-int-cast">;
def : DiagGroup<"redundant-decls">;
def ReturnType : DiagGroup<"return-type">;
def : DiagGroup<"sequence-point">;
def : DiagGroup<"shadow">;
def : DiagGroup<"shorten-64-to-32">;
def SignCompare : DiagGroup<"sign-compare">;

// Preprocessor warnings.
def : DiagGroup<"builtin-macro-redefined">;

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
def Trigraphs : DiagGroup<"trigraphs">;

def : DiagGroup<"type-limits">;
def Uninitialized  : DiagGroup<"uninitialized">;
def UnknownPragmas : DiagGroup<"unknown-pragmas">;
def UnusedArgument : DiagGroup<"unused-argument">;
def UnusedFunction : DiagGroup<"unused-function">;
def UnusedLabel : DiagGroup<"unused-label">;
def UnusedParameter : DiagGroup<"unused-parameter">;
def UnusedValue    : DiagGroup<"unused-value">;
def UnusedVariable : DiagGroup<"unused-variable">;
def ReadOnlySetterAttrs : DiagGroup<"readonly-setter-attrs">;
def Reorder : DiagGroup<"reorder">;
def UndeclaredSelector : DiagGroup<"undeclared-selector">;
def SuperSubClassMismatch : DiagGroup<"super-class-method-mismatch">;
def : DiagGroup<"variadic-macros">;
def VectorConversions : DiagGroup<"vector-conversions">;      // clang specific
def VolatileRegisterVar : DiagGroup<"volatile-register-var">;
def : DiagGroup<"write-strings">;
def CharSubscript : DiagGroup<"char-subscripts">;

// Aggregation warning settings.


def Unused : DiagGroup<"unused",
                       [UnusedArgument, UnusedFunction, UnusedLabel,
                        UnusedParameter, UnusedValue, UnusedVariable]>;

// Format settings.
def Format : DiagGroup<"format", [FormatExtraArgs, FormatZeroLength, NonNull]>;
def FormatSecurity : DiagGroup<"format-security", [Format]>;
def FormatNonLiteral : DiagGroup<"format-nonliteral", [FormatSecurity]>;
def FormatY2K : DiagGroup<"format-y2k", [Format]>;
def Format2 : DiagGroup<"format=2",
                        [FormatNonLiteral, FormatSecurity, FormatY2K]>;


def Extra : DiagGroup<"extra", [
    SignCompare,
    UnusedParameter
  ]>;

def Most : DiagGroup<"most", [
    Comment,
    Format,
    Implicit,
    MismatchedTags,
    MultiChar,
    ReturnType,
    Switch,
    Trigraphs,
    Uninitialized,
    UnknownPragmas,
    UnusedValue,
    UnusedVariable,
    VectorConversions,
    VolatileRegisterVar,
    Reorder,
    CharSubscript
 ]>;

// -Wall is -Wmost -Wparentheses
def : DiagGroup<"all", [Most, Parentheses]>;

// Aliases.
def : DiagGroup<"", [Extra]>;  // -W = -Wextra
def : DiagGroup<"endif-labels", [ExtraTokens]>; // endif-labels = endif-tokens

// A warning group for warnings that we want to have on by default in clang,
// but which aren't no by default in GCC.
def NonGCC : DiagGroup<"non-gcc",
    [SignCompare]>;

//===--- OptParser.td - Common Option Parsing Interfaces ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the common interfaces used by the option parsing TableGen
//  backend.
//
//===----------------------------------------------------------------------===//

// Define the kinds of options.

class OptionKind<string name, int predecence = 0> {
  string Name = name;
  // The kind precedence, kinds with lower precedence are matched first.
  int Precedence = predecence;
}

// An option group.
def KIND_GROUP : OptionKind<"Group">;
// A flag with no values.
def KIND_FLAG : OptionKind<"Flag">;
// An option which prefixes its (single) value.
def KIND_JOINED : OptionKind<"Joined", 1>;
// An option which is followed by its value.
def KIND_SEPARATE : OptionKind<"Separate">;
// An option followed by its values, which are separated by commas.
def KIND_COMMAJOINED : OptionKind<"CommaJoined">;
// An option which is which takes multiple (separate) arguments.
def KIND_MULTIARG : OptionKind<"MultiArg">;
// An option which is either joined to its (non-empty) value, or followed by its
// value.
def KIND_JOINED_OR_SEPARATE : OptionKind<"JoinedOrSeparate">;
// An option which is both joined to its (first) value, and followed by its
// (second) value.
def KIND_JOINED_AND_SEPARATE : OptionKind<"JoinedAndSeparate">;

// Define the option flags.

class OptionFlag {}

// DriverOption - The option is a "driver" option, and should not be forwarded
// to gcc.
def DriverOption : OptionFlag;

// LinkerInput - The option is a linker input.
def LinkerInput : OptionFlag;

// NoArgumentUnused - Don't report argument unused warnings for this option; this
// is useful for options like -static or -dynamic which a user may always end up
// passing, even if the platform defaults to (or only supports) that option.
def NoArgumentUnused : OptionFlag;

// RenderAsInput - The option should not render the name when rendered as an
// input (i.e., the option is rendered as values).
def RenderAsInput : OptionFlag;

// RenderJoined - The option should be rendered joined, even if separate (only
// sensible on single value separate options).
def RenderJoined : OptionFlag;

// RenderSeparate - The option should be rendered separately, even if joined
// (only sensible on joined options).
def RenderSeparate : OptionFlag;

// Unsupported - The option is unsupported, and the driver will reject command
// lines that use it.
def Unsupported : OptionFlag;

// Define the option group class.

class OptionGroup<string name> {
  string EnumName = ?; // Uses the def name if undefined.
  string Name = name;
  OptionGroup Group = ?;
}

// Define the option class.

class Option<string name, OptionKind kind> {
  string EnumName = ?; // Uses the def name if undefined.
  string Name = name;
  OptionKind Kind = kind;
  // Used by MultiArg option kind.
  int NumArgs = 0;
  string HelpText = ?;
  string MetaVarName = ?;
  list<OptionFlag> Flags = [];
  OptionGroup Group = ?;
  Option Alias = ?;
}

// Helpers for defining options.

class Flag<string name> : Option<name, KIND_FLAG>;
class Joined<string name> : Option<name, KIND_JOINED>;
class Separate<string name> : Option<name, KIND_SEPARATE>;
class CommaJoined<string name> : Option<name, KIND_COMMAJOINED>;
class MultiArg<string name, int numargs> : Option<name, KIND_MULTIARG> {
  int NumArgs = numargs;
}
class JoinedOrSeparate<string name> : Option<name, KIND_JOINED_OR_SEPARATE>;
class JoinedAndSeparate<string name> : Option<name, KIND_JOINED_AND_SEPARATE>;

// Mix-ins for adding optional attributes.

class Alias<Option alias> { Option Alias = alias; }
class EnumName<string name> { string EnumName = name; }
class Flags<list<OptionFlag> flags> { list<OptionFlag> Flags = flags; }
class Group<OptionGroup group> { OptionGroup Group = group; }
class HelpText<string text> { string HelpText = text; }
class MetaVarName<string name> { string MetaVarName = name; }

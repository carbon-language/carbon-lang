//===--- CLCompatOptions.td - Options for clang-cl ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by clang-cl.
//
//===----------------------------------------------------------------------===//

def cl_Group : OptionGroup<"<clang-cl options>">,
  HelpText<"CL.EXE COMPATIBILITY OPTIONS">;

def cl_ignored_Group : OptionGroup<"<clang-cl ignored options>">,
  Group<cl_Group>;

class CLFlag<string name> : Option<["/", "-"], name, KIND_FLAG>,
  Group<cl_Group>, Flags<[CLOption, DriverOption]>;

class CLIgnoredFlag<string name> : Option<["/", "-"], name, KIND_FLAG>,
  Group<cl_ignored_Group>, Flags<[CLOption, DriverOption]>;

class CLJoined<string name> : Option<["/", "-"], name, KIND_JOINED>,
  Group<cl_Group>, Flags<[CLOption, DriverOption]>;

class CLIgnoredJoined<string name> : Option<["/", "-"], name, KIND_JOINED>,
  Group<cl_ignored_Group>, Flags<[CLOption, DriverOption]>;

class CLJoinedOrSeparate<string name> : Option<["/", "-"], name,
  KIND_JOINED_OR_SEPARATE>, Group<cl_Group>, Flags<[CLOption, DriverOption]>;

// Aliases:

def _SLASH_c : CLFlag<"c">, HelpText<"Compile only">, Alias<c>;
def _SLASH_D : CLJoinedOrSeparate<"D">, HelpText<"Define macro">,
  MetaVarName<"<macro[=value]>">, Alias<D>;
def _SLASH_GR : CLFlag<"GR">, HelpText<"Enable RTTI">, Alias<frtti>;
def _SLASH_GR_ : CLFlag<"GR-">, HelpText<"Disable RTTI">, Alias<fno_rtti>;
def _SLASH_help : CLFlag<"help">, Alias<help>,
  HelpText<"Display available options">;
def _SLASH_I : CLJoinedOrSeparate<"I">,
  HelpText<"Add directory to include search path">, MetaVarName<"<dir>">,
  Alias<I>;
def _SLASH_J : CLFlag<"J">, HelpText<"Make char type unsigned">,
  Alias<funsigned_char>;
def _SLASH_O : CLJoined<"O">, HelpText<"Optimization level">,
  MetaVarName<"<n>">, Alias<O>;
def _SLASH_Ob0 : CLFlag<"Ob0">, HelpText<"Disable inlining">,
  Alias<fno_inline>;
def _SLASH_Od : CLFlag<"Od">, HelpText<"Disable optimization">, Alias<O0>;
def _SLASH_Os : CLFlag<"Os">, HelpText<"Optimize for size">, Alias<O>,
  AliasArgs<["s"]>;
def _SLASH_Ot : CLFlag<"Ot">, HelpText<"Optimize for speed">, Alias<O>,
  AliasArgs<["2"]>;
def _SLASH_Ox : CLFlag<"Ox">, HelpText<"Maximum optimization">, Alias<O>,
  AliasArgs<["3"]>;
def _SLASH_Oy : CLFlag<"Oy">, HelpText<"Enable frame pointer omission">,
  Alias<fomit_frame_pointer>;
def _SLASH_Oy_ : CLFlag<"Oy-">, HelpText<"Disable frame pointer omission">,
  Alias<fno_omit_frame_pointer>;
def _SLASH_P : CLFlag<"P">, HelpText<"Only run the preprocessor">, Alias<E>;
def _SLASH_QUESTION : CLFlag<"?">, Alias<help>,
  HelpText<"Display available options">;
def _SLASH_U : CLJoinedOrSeparate<"U">, HelpText<"Undefine macro">,
  MetaVarName<"<macro>">, Alias<U>;
def _SLASH_W0 : CLFlag<"W0">, HelpText<"Disable all warnings">, Alias<w>;
def _SLASH_W1 : CLFlag<"W1">, HelpText<"Enable -Wall">, Alias<Wall>;
def _SLASH_W2 : CLFlag<"W2">, HelpText<"Enable -Wall">, Alias<Wall>;
def _SLASH_W3 : CLFlag<"W3">, HelpText<"Enable -Wall">, Alias<Wall>;
def _SLASH_W4 : CLFlag<"W4">, HelpText<"Enable -Wall">, Alias<Wall>;
def _SLASH_Wall : CLFlag<"Wall">, HelpText<"Enable -Wall">, Alias<Wall>;
def _SLASH_WX : CLFlag<"WX">, HelpText<"Treat warnings as errors">,
  Alias<W_Joined>, AliasArgs<["error"]>;
def _SLASH_WX_ : CLFlag<"WX-">, HelpText<"Do not treat warnings as errors">,
  Alias<W_Joined>, AliasArgs<["no-error"]>;
def _SLASH_w_flag : CLFlag<"w">, HelpText<"Disable all warnings">, Alias<w>;
def _SLASH_Zs : CLFlag<"Zs">, HelpText<"Syntax-check only">,
  Alias<fsyntax_only>;


// Ignored:

def _SLASH_analyze_ : CLIgnoredFlag<"analyze-">;
def _SLASH_errorReport : CLIgnoredJoined<"errorReport">;
def _SLASH_nologo : CLIgnoredFlag<"nologo">;
def _SLASH_Ob1 : CLIgnoredFlag<"Ob1">;
def _SLASH_Ob2 : CLIgnoredFlag<"Ob2">;
def _SLASH_Zc_forScope : CLIgnoredFlag<"Zc:forScope">;
def _SLASH_Zc_wchar_t : CLIgnoredFlag<"Zc:wchar_t">;


// Unsupported:

def _SLASH_EH : CLJoined<"EH">;
def _SLASH_Fd : CLJoined<"Fd">;
def _SLASH_Fo : CLJoined<"Fo">;
def _SLASH_fp : CLJoined<"fp">;
def _SLASH_Gd : CLFlag<"Gd">;
def _SLASH_GL : CLFlag<"GL">;
def _SLASH_GL_ : CLFlag<"GL-">;
def _SLASH_Gm : CLFlag<"Gm">;
def _SLASH_Gm_ : CLFlag<"Gm-">;
def _SLASH_GS : CLFlag<"GS">;
def _SLASH_Gy : CLFlag<"Gy">;
def _SLASH_Gy_ : CLFlag<"Gy-">;
def _SLASH_GZ : CLFlag<"GZ">;
def _SLASH_MD : CLFlag<"MD">;
def _SLASH_MT : CLFlag<"MT">;
def _SLASH_MDd : CLFlag<"MDd">;
def _SLASH_MTd : CLFlag<"MTd">;
def _SLASH_Oi : CLFlag<"Oi">;
def _SLASH_RTC : CLJoined<"RTC">;
def _SLASH_showIncludes : CLJoined<"showIncludes">;
def _SLASH_Tc : CLJoined<"Tc">;
def _SLASH_TC : CLFlag<"TC">;
def _SLASH_Tp : CLJoined<"Tp">;
def _SLASH_TP : CLFlag<"TP">;
def _SLASH_w : CLJoined<"w">;
def _SLASH_Zc : CLJoined<"Zc:">;
def _SLASH_ZI : CLFlag<"ZI">;
def _SLASH_Zi : CLFlag<"Zi">;

// Check that LanguageMap is processed properly.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[
(switch_option "dummy1", (help "none"))
]>;

def dummy_tool : Tool<[
(command "dummy_cmd"),
(in_language "dummy_lang"),
(out_language "dummy_lang"),
(actions (case
         (switch_on "dummy1"), (forward "dummy1")))
]>;

def lang_map : LanguageMap<[
    // CHECK: langMap["dummy"] = "dummy_lang"
    // CHECK: langMap["DUM"] = "dummy_lang"
    (lang_to_suffixes "dummy_lang", ["dummy", "DUM"]),
    // CHECK: langMap["DUM2"] = "dummy_lang_2"
    (lang_to_suffixes "dummy_lang_2", "DUM2")
]>;

def DummyGraph : CompilationGraph<[(edge "root", "dummy_tool")]>;

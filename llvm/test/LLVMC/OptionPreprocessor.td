// Test for the OptionPreprocessor and related functionality.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -Qunused-arguments -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[
(switch_option "foo", (help "dummy")),
(switch_option "bar", (help "dummy")),
(switch_option "baz", (help "dummy")),
(parameter_option "foo_p", (help "dummy")),
(parameter_option "bar_p", (help "dummy")),
(parameter_option "baz_p", (help "dummy")),
(parameter_list_option "foo_l", (help "dummy"))
]>;

def Preprocess : OptionPreprocessor<
(case
      // CHECK: W1
      // CHECK: foo = false;
      // CHECK: foo_p = "";
      // CHECK: foo_l.clear();
      (and (switch_on "foo"), (any_switch_on ["bar", "baz"])),
           [(warning "W1"), (unset_option "foo"),
                            (unset_option "foo_p"), (unset_option "foo_l")],
      // CHECK: W2
      // CHECK: foo = true;
      // CHECK: bar = true;
      // CHECK: baz = false;
      // CHECK: foo_p = "asdf";
      // CHECK: foo_l.clear();
      // CHECK: foo_l.push_back("qwert");
      // CHECK: foo_l.push_back("yuiop");
      // CHECK: foo_l.push_back("asdf");
      (and (switch_on ["foo", "bar"]), (any_empty ["foo_p", "bar_p"])),
           [(warning "W2"), (set_option "foo"),
                            (set_option "bar", true),
                            (set_option "baz", false),
                            (set_option "foo_p", "asdf"),
                            (set_option "foo_l", ["qwert", "yuiop", "asdf"])],
      // CHECK: W3
      // CHECK: foo = true;
      // CHECK: bar = true;
      // CHECK: baz = true;
      (and (empty ["foo_p", "bar_p"]), (any_not_empty ["baz_p"])),
           [(warning "W3"), (set_option ["foo", "bar", "baz"])])
>;

// Shut up warnings...
def dummy : Tool<
[(in_language "dummy"),
 (out_language "dummy"),
 (output_suffix "d"),
 (cmd_line "dummy $INFILE -o $OUTFILE"),
 (actions (case (switch_on "foo"), (error),
                (switch_on "bar"), (error),
                (switch_on "baz"), (error),
                (not_empty "foo_p"), (error),
                (not_empty "bar_p"), (error),
                (not_empty "baz_p"), (error),
                (not_empty "foo_l"), (error)))
]>;

def Graph : CompilationGraph<[Edge<"root", "dummy">]>;


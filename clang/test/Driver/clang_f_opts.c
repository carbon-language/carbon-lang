// REQUIRES: clang-driver

// RUN: %clang -### -S -fasm -fblocks -fbuiltin -fno-math-errno -fcommon -fpascal-strings -fno-blocks -fno-builtin -fmath-errno -fno-common -fno-pascal-strings -fblocks -fbuiltin -fmath-errno -fcommon -fpascal-strings -fsplit-stack %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS1 %s
// RUN: %clang -### -S -fasm -fblocks -fbuiltin -fno-math-errno -fcommon -fpascal-strings -fno-asm -fno-blocks -fno-builtin -fmath-errno -fno-common -fno-pascal-strings -fno-show-source-location -fshort-enums -fshort-wchar %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS2 %s

// CHECK-OPTIONS1: -split-stacks
// CHECK-OPTIONS1: -fgnu-keywords
// CHECK-OPTIONS1: -fblocks
// CHECK-OPTIONS1: -fpascal-strings

// CHECK-OPTIONS2: -fmath-errno
// CHECK-OPTIONS2: -fno-gnu-keywords
// CHECK-OPTIONS2: -fno-builtin
// CHECK-OPTIONS2: -fshort-enums
// CHECK-OPTIONS2: -fshort-wchar
// CHECK-OPTIONS2: -fno-common
// CHECK-OPTIONS2: -fno-show-source-location

// RUN: %clang -### -S -Wwrite-strings %s 2>&1 | FileCheck -check-prefix=WRITE-STRINGS1 %s
// WRITE-STRINGS1: -fconst-strings
// RUN: %clang -### -S -Wwrite-strings -Wno-write-strings %s 2>&1 | FileCheck -check-prefix=WRITE-STRINGS2 %s
// WRITE-STRINGS2-NOT: -fconst-strings
// RUN: %clang -### -S -Wwrite-strings -w %s 2>&1 | FileCheck -check-prefix=WRITE-STRINGS3 %s
// WRITE-STRINGS3-NOT: -fconst-strings

// RUN: %clang -### -x c++ -c %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -x c++ -c -Wdeprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -x c++ -c -Wno-deprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-OFF-CHECK %s
// RUN: %clang -### -x c++ -c -Wno-deprecated -Wdeprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -x c++ -c -w %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -c %s 2>&1 | FileCheck -check-prefix=DEPRECATED-OFF-CHECK %s
// RUN: %clang -### -c -Wdeprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-OFF-CHECK %s
// DEPRECATED-ON-CHECK: -fdeprecated-macro
// DEPRECATED-OFF-CHECK-NOT: -fdeprecated-macro

// RUN: %clang -### -S -ffp-contract=fast %s 2>&1 | FileCheck -check-prefix=FP-CONTRACT-FAST-CHECK %s
// RUN: %clang -### -S -ffast-math %s 2>&1 | FileCheck -check-prefix=FP-CONTRACT-FAST-CHECK %s
// RUN: %clang -### -S -ffp-contract=off %s 2>&1 | FileCheck -check-prefix=FP-CONTRACT-OFF-CHECK %s
// FP-CONTRACT-FAST-CHECK: -ffp-contract=fast
// FP-CONTRACT-OFF-CHECK: -ffp-contract=off

// RUN: %clang -### -S -funroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-UNROLL-LOOPS %s
// RUN: %clang -### -S -fno-unroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-NO-UNROLL-LOOPS %s
// RUN: %clang -### -S -fno-unroll-loops -funroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-UNROLL-LOOPS %s
// RUN: %clang -### -S -funroll-loops -fno-unroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-NO-UNROLL-LOOPS %s
// CHECK-UNROLL-LOOPS: "-funroll-loops"
// CHECK-NO-UNROLL-LOOPS: "-fno-unroll-loops"

// RUN: %clang -### -S -freroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-REROLL-LOOPS %s
// RUN: %clang -### -S -fno-reroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-NO-REROLL-LOOPS %s
// RUN: %clang -### -S -fno-reroll-loops -freroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-REROLL-LOOPS %s
// RUN: %clang -### -S -freroll-loops -fno-reroll-loops %s 2>&1 | FileCheck -check-prefix=CHECK-NO-REROLL-LOOPS %s
// CHECK-REROLL-LOOPS: "-freroll-loops"
// CHECK-NO-REROLL-LOOPS-NOT: "-freroll-loops"

// RUN: %clang -### -S -fprofile-sample-use=%S/Inputs/file.prof %s 2>&1 | FileCheck -check-prefix=CHECK-SAMPLE-PROFILE %s
// CHECK-SAMPLE-PROFILE: "-fprofile-sample-use={{.*}}/file.prof"

// RUN: %clang -### -S -fauto-profile=%S/Inputs/file.prof %s 2>&1 | FileCheck -check-prefix=CHECK-AUTO-PROFILE %s
// CHECK-AUTO-PROFILE: "-fprofile-sample-use={{.*}}/file.prof"

// RUN: %clang -### -S -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-vectorize -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -fvectorize -fno-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -ftree-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-vectorize -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -ftree-vectorize -fno-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -O %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-vectorize -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -O1 -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -Ofast %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -Oz %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// CHECK-VECTORIZE: "-vectorize-loops"
// CHECK-NO-VECTORIZE-NOT: "-vectorize-loops"

// RUN: %clang -### -S -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-slp-vectorize -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -fslp-vectorize -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -ftree-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-slp-vectorize -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -ftree-slp-vectorize -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -O %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -Oz %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-slp-vectorize -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -O1 -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -Ofast %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// CHECK-SLP-VECTORIZE: "-vectorize-slp"
// CHECK-NO-SLP-VECTORIZE-NOT: "-vectorize-slp"

// RUN: %clang -### -S -fslp-vectorize-aggressive %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE-AGG %s
// RUN: %clang -### -S -fno-slp-vectorize-aggressive -fslp-vectorize-aggressive %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE-AGG %s
// RUN: %clang -### -S -fno-slp-vectorize-aggressive %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE-AGG %s
// RUN: %clang -### -S -fslp-vectorize-aggressive -fno-slp-vectorize-aggressive %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE-AGG %s
// CHECK-SLP-VECTORIZE-AGG: "-vectorize-slp-aggressive"
// CHECK-NO-SLP-VECTORIZE-AGG-NOT: "-vectorize-slp-aggressive"

// RUN: %clang -### -S -fextended-identifiers %s 2>&1 | FileCheck -check-prefix=CHECK-EXTENDED-IDENTIFIERS %s
// RUN: %clang -### -S -fno-extended-identifiers %s 2>&1 | FileCheck -check-prefix=CHECK-NO-EXTENDED-IDENTIFIERS %s
// CHECK-EXTENDED-IDENTIFIERS: "-cc1"
// CHECK-EXTENDED-IDENTIFIERS-NOT: "-fextended-identifiers"
// CHECK-NO-EXTENDED-IDENTIFIERS: error: unsupported option '-fno-extended-identifiers'

// RUN: %clang -### -S -fno-pascal-strings -mpascal-strings %s 2>&1 | FileCheck -check-prefix=CHECK-M-PASCAL-STRINGS %s
// CHECK-M-PASCAL-STRINGS: "-fpascal-strings"

// RUN: %clang -### -S -fpascal-strings -mno-pascal-strings %s 2>&1 | FileCheck -check-prefix=CHECK-NO-M-PASCAL-STRINGS %s
// CHECK-NO-M-PASCAL-STRINGS-NOT: "-fpascal-strings"

// RUN: %clang -### -S -O4 %s 2>&1 | FileCheck -check-prefix=CHECK-MAX-O %s
// CHECK-MAX-O: warning: -O4 is equivalent to -O3
// CHECK-MAX-O: -O3

// RUN: %clang -S -O20 -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-INVALID-O %s
// CHECK-INVALID-O: warning: optimization level '-O20' is not supported; using '-O3' instead

// RUN: %clang -### -S -finput-charset=iso-8859-1 -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-INVALID-CHARSET %s
// CHECK-INVALID-CHARSET: error: invalid value 'iso-8859-1' in '-finput-charset=iso-8859-1'

// Test that we don't error on these.
// RUN: %clang -### -S -Werror                                                \
// RUN:     -falign-functions -falign-functions=2 -fno-align-functions        \
// RUN:     -fasynchronous-unwind-tables -fno-asynchronous-unwind-tables      \
// RUN:     -fbuiltin -fno-builtin                                            \
// RUN:     -fdiagnostics-show-location=once                                  \
// RUN:     -ffloat-store -fno-float-store                                    \
// RUN:     -feliminate-unused-debug-types -fno-eliminate-unused-debug-types  \
// RUN:     -fgcse -fno-gcse                                                  \
// RUN:     -fident -fno-ident                                                \
// RUN:     -fimplicit-templates -fno-implicit-templates                      \
// RUN:     -finput-charset=UTF-8                                             \
// RUN:     -fivopts -fno-ivopts                                              \
// RUN:     -fnon-call-exceptions -fno-non-call-exceptions                    \
// RUN:     -fpermissive -fno-permissive                                      \
// RUN:     -fdefer-pop -fno-defer-pop                                        \
// RUN:     -fprefetch-loop-arrays -fno-prefetch-loop-arrays                  \
// RUN:     -fprofile-correction -fno-profile-correction                      \
// RUN:     -fprofile-dir=bar                                                 \
// RUN:     -fprofile-use -fprofile-use=zed -fno-profile-use                  \
// RUN:     -fprofile-values -fno-profile-values                              \
// RUN:     -frounding-math -fno-rounding-math                                \
// RUN:     -fsee -fno-see                                                    \
// RUN:     -ftracer -fno-tracer                                              \
// RUN:     -funroll-all-loops -fno-unroll-all-loops                          \
// RUN:     -fuse-ld=gold                                                     \
// RUN:     -fno-builtin-foobar                                               \
// RUN:     -fno-builtin-strcat -fno-builtin-strcpy                           \
// RUN:     -fno-var-tracking                                                 \
// RUN:     -fno-unsigned-char                                                \
// RUN:     -fno-signed-char                                                  \
// RUN:     -fstrength-reduce -fno-strength-reduce                            \
// RUN:     -finline-limit=1000                                               \
// RUN:     -finline-limit                                                    \
// RUN:     %s 2>&1 | FileCheck --check-prefix=IGNORE %s
// IGNORE-NOT: error: unknown argument

// Test that the warning is displayed on these.
// RUN: %clang -###                                                           \
// RUN: -finline-limit=1000                                                   \
// RUN: -finline-limit                                                        \
// RUN: -fexpensive-optimizations                                             \
// RUN: -fno-expensive-optimizations                                          \
// RUN: -fno-defer-pop                                                        \
// RUN: -finline-functions                                                    \
// RUN: -fno-keep-inline-functions                                            \
// RUN: -freorder-blocks                                                      \
// RUN: -fprofile-dir=/rand/dir                                               \
// RUN: -fprofile-use                                                         \
// RUN: -fprofile-use=/rand/dir                                               \
// RUN: -falign-functions                                                     \
// RUN: -falign-functions=1                                                   \
// RUN: -ffloat-store                                                         \
// RUN: -fgcse                                                                \
// RUN: -fivopts                                                              \
// RUN: -fprefetch-loop-arrays                                                \
// RUN: -fprofile-correction                                                  \
// RUN: -fprofile-values                                                      \
// RUN: -frounding-math                                                       \
// RUN: -fschedule-insns                                                      \
// RUN: -fsignaling-nans                                                      \
// RUN: -fstrength-reduce                                                     \
// RUN: -ftracer                                                              \
// RUN: -funroll-all-loops                                                    \
// RUN: -funswitch-loops                                                      \
// RUN: %s 2>&1 | FileCheck --check-prefix=CHECK-WARNING %s
// CHECK-WARNING-DAG: optimization flag '-finline-limit=1000' is not supported
// CHECK-WARNING-DAG: optimization flag '-finline-limit' is not supported
// CHECK-WARNING-DAG: optimization flag '-fexpensive-optimizations' is not supported
// CHECK-WARNING-DAG: optimization flag '-fno-expensive-optimizations' is not supported
// CHECK-WARNING-DAG: optimization flag '-fno-defer-pop' is not supported
// CHECK-WARNING-DAG: optimization flag '-finline-functions' is not supported
// CHECK-WARNING-DAG: optimization flag '-fno-keep-inline-functions' is not supported
// CHECK-WARNING-DAG: optimization flag '-freorder-blocks' is not supported
// CHECK-WARNING-DAG: optimization flag '-fprofile-dir=/rand/dir' is not supported
// CHECK-WARNING-DAG: optimization flag '-fprofile-use' is not supported
// CHECK-WARNING-DAG: optimization flag '-fprofile-use=/rand/dir' is not supported
// CHECK-WARNING-DAG: optimization flag '-falign-functions' is not supported
// CHECK-WARNING-DAG: optimization flag '-falign-functions=1' is not supported
// CHECK-WARNING-DAG: optimization flag '-ffloat-store' is not supported
// CHECK-WARNING-DAG: optimization flag '-fgcse' is not supported
// CHECK-WARNING-DAG: optimization flag '-fivopts' is not supported
// CHECK-WARNING-DAG: optimization flag '-fprefetch-loop-arrays' is not supported
// CHECK-WARNING-DAG: optimization flag '-fprofile-correction' is not supported
// CHECK-WARNING-DAG: optimization flag '-fprofile-values' is not supported
// CHECK-WARNING-DAG: optimization flag '-frounding-math' is not supported
// CHECK-WARNING-DAG: optimization flag '-fschedule-insns' is not supported
// CHECK-WARNING-DAG: optimization flag '-fsignaling-nans' is not supported
// CHECK-WARNING-DAG: optimization flag '-fstrength-reduce' is not supported
// CHECK-WARNING-DAG: optimization flag '-ftracer' is not supported
// CHECK-WARNING-DAG: optimization flag '-funroll-all-loops' is not supported
// CHECK-WARNING-DAG: optimization flag '-funswitch-loops' is not supported

// Test that we mute the warning on these
// RUN: %clang -### -finline-limit=1000 -Wno-invalid-command-line-argument              \
// RUN:     %s 2>&1 | FileCheck --check-prefix=CHECK-NO-WARNING1 %s
// RUN: %clang -### -finline-limit -Wno-invalid-command-line-argument                   \
// RUN:     %s 2>&1 | FileCheck --check-prefix=CHECK-NO-WARNING2 %s
// CHECK-NO-WARNING1-NOT: optimization flag '-finline-limit=1000' is not supported
// CHECK-NO-WARNING2-NOT: optimization flag '-finline-limit' is not supported


// RUN: %clang -### -fshort-wchar -fno-short-wchar %s 2>&1 | FileCheck -check-prefix=CHECK-WCHAR1 %s
// RUN: %clang -### -fno-short-wchar -fshort-wchar %s 2>&1 | FileCheck -check-prefix=CHECK-WCHAR2 %s
// CHECK-WCHAR1: -fno-short-wchar
// CHECK-WCHAR1-NOT: -fshort-wchar
// CHECK-WCHAR2: -fshort-wchar
// CHECK-WCHAR2-NOT: -fno-short-wchar

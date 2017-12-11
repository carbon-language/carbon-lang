// The automatic preinclude of stdc-predef.h should not occur if
// the source filename indicates a preprocessed file.
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s

int i;
// The automatic preinclude of stdc-predef.h should not occur if
// the source filename indicates a preprocessed file.
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s

int i;

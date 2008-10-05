#!/bin/sh
set -ex

if [ -z "$CC" ]; then
    CC="gcc"
fi

SRCLANG=c
MACROLIST=macro-list.txt
CCDEFS=cc-definitions.txt
CLANGDEFS=clang-definitions.txt

# Gather list of macros as "NAME" = NAME.
$CC -dM -E -x $SRCLANG /dev/null -o - | \
grep "#define" | sort -f | sed -e "s/#define \([^ ]*\) .*/\"\1\" = \1/" > $MACROLIST

$CC -E -x $SRCLANG $MACROLIST > $CCDEFS

clang -E -x $SRCLANG $MACROLIST > $CLANGDEFS

diff $CCDEFS $CLANGDEFS



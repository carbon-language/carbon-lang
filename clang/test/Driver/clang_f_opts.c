// RUN: clang -### -S -x c /dev/null -fblocks -fbuiltin -fmath-errno -fcommon -fpascal-strings -fno-blocks -fno-builtin -fno-math-errno -fno-common -fno-pascal-strings -fblocks -fbuiltin -fmath-errno -fcommon -fpascal-strings %s 2> %t
// RUN: grep -F '"-fblocks"' %t
// RUN: grep -F '"-fpascal-strings"' %t
// RUN: clang -### -S -x c /dev/null -fblocks -fbuiltin -fmath-errno -fcommon -fpascal-strings -fno-blocks -fno-builtin -fno-math-errno -fno-common -fno-pascal-strings -fno-show-source-location -fshort-wchar %s 2> %t
// RUN: grep -F '"-fno-builtin"' %t
// RUN: grep -F '"-fno-common"' %t
// RUN: grep -F '"-fno-math-errno"' %t
// RUN: grep -F '"-fno-show-source-location"' %t
// RUN: grep -F '"-fshort-wchar"' %t

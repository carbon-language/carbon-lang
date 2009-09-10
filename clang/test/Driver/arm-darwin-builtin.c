// RUN: clang -ccc-host-triple x86_64-apple-darwin9 -arch arm -### -fsyntax-only %s 2> %t &&
// RUN: grep -- "-fno-builtin-strcat" %t &&
// RUN: grep -- "-fno-builtin-strcpy" %t &&

// RUN: clang -ccc-host-triple x86_64-apple-darwin9 -arch arm -### -fsyntax-only %s -fbuiltin-strcat -fbuiltin-strcpy 2> %t &&
// RUN: not grep -- "-fno-builtin-strcat" %t &&
// RUN: not grep -- "-fno-builtin-strcpy" %t &&

// RUN: clang -ccc-no-clang -ccc-host-triple x86_64-apple-darwin9 -arch arm -### -fsyntax-only %s -fbuiltin-strcat -fbuiltin-strcpy 2> %t &&
// RUN: not grep -- "-fno-builtin-strcat" %t &&
// RUN: not grep -- "-fno-builtin-strcpy" %t


// RUN: xcc -ccc-no-clang -### -S --all-warnings %s &> %t &&
// RUN: grep -- '"-Wall"' %t &&

// RUN: xcc -ccc-no-clang -### -S --ansi %s &> %t &&
// RUN: grep -- '"-ansi"' %t &&

// RUN: xcc -ccc-no-clang -### -S --assert foo --assert=foo %s &> %t &&
// RUN: grep -- '"-A" "foo" "-A" "foo"' %t &&

// RUN: xcc -ccc-no-clang -### -S --classpath foo --classpath=foo %s &> %t &&
// RUN: grep -- '"-fclasspath=foo" "-fclasspath=foo"' %t &&

// RUN: true
